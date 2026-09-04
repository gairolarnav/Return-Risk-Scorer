"""
Single-record + batch scoring (ARCHITECTURE.md §3, the decision layer).

Takes raw return record(s), applies the *same* feature engineering the model
was trained with, and routes each to a recommended intervention using the
cost-calibrated decision rule from src/evaluate.py. `score_record` and
`score_batch` share one code path (`prepare_frame` + `_route`) rather than
being two implementations that happen to agree today — `scripts/score.py`
calls these, it does not reimplement them.

Two things this module deliberately does not do:

  * It does not re-implement feature engineering. It calls
    `src.features.add_transaction_level_features`, the same function
    `src/features.py` uses to build the training set. Two divergent
    implementations of the same feature builder is the failure mode
    ARCHITECTURE.md §8 calls out, and it silently invalidates every number
    the model produces at serving time.

  * It does not take argmax. Argmax is the cost-blind decision, and a
    cost-calibrated policy is the centerpiece of this project — a scorer that
    ignores it at the point of actual use would make the whole §6.2 analysis
    decorative. The action comes from
    `expected_cost_decision`, under explicit, caller-visible postures.

Both cost axes are exposed, not just one. An earlier version took a `posture`
setting C_fp : C_fn and left `build_cost_matrix`'s friction cell at its
default, which meant the served decision could not be moved along the
approve/soft-friction axis at all — the axis the §6.2 correction identifies
as the only one that moves on this data (§6.2, and sweep_friction_curve's
docstring). Scoring a record on the `full` track under all three C_fp:C_fn
postures returned byte-identical actions for 12,000 of 12,000 test rows. The
`friction_posture` argument fixes that; `posture` is kept and still reported,
because showing which axis *doesn't* move is part of the honest account.

Nothing here executes an action. It returns a recommendation and the
probabilities behind it, per the defense-only constraint (§7).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.evaluate import (
    ACTIONS,
    DEFAULT_POSTURES,
    FRICTION_POSTURES,
    build_cost_matrix,
    expected_cost_decision,
)
from src.features import add_transaction_level_features

# The posture each entry point falls back to. Named here rather than repeated
# as a literal in four signatures: "balanced (1:2)" is friction_cost=1.0, the
# value build_cost_matrix has always defaulted to, so scoring without flags
# reproduces every previously committed artifact exactly.
DEFAULT_POSTURE = "loss-neutral (1:1)"
DEFAULT_FRICTION_POSTURE = "balanced (1:2)"

# What each routed action means operationally (§1/§3). Keyed by ACTION, not by
# class — an earlier version keyed this by predicted class and misspelled
# "Policy Abuser" as "Policy Abuse", so that entire class silently fell
# through to an "unmapped" string. Keying on the three actions the decision
# layer can actually emit removes the whole category of error.
INTERVENTION = {
    "approve": "Approve — no friction",
    "soft_friction": "Soft friction — return fee, condition inspection, pattern flag",
    "hard_block": "Hard block — hold refund, route to manual review",
}


def _verify_bundle_digest(joblib_path: Path, sidecar: Path) -> None:
    """Check the .joblib against the digest its JSON sidecar recorded, before
    anything unpickles it.

    joblib is pickle-backed: loading a bundle runs whatever is inside it. The
    honest description of this check is narrow -- the digest lives in the same
    repository as the file it covers, so whoever can rewrite one can rewrite the
    other, and it stops no deliberate attacker. What it does stop is the failure
    that actually occurs: a truncated or corrupted artifact, and a .joblib and
    .json that drifted apart because a track was retrained and only one of the
    two was committed. Both otherwise surface as a confusing model error much
    further downstream.

    A sidecar with no digest is a pre-check artifact and is rejected rather than
    waved through, for the same reason a bundle without "categories" is: a
    silent fallback is how the bug it guards against gets back in.
    """
    if not sidecar.exists():
        return  # nothing to check against; the key check below still applies
    recorded = json.loads(sidecar.read_text()).get("bundle_sha256")
    if recorded is None:
        raise ValueError(
            f"{sidecar} records no 'bundle_sha256'. It predates the integrity "
            "check — retrain with `python -m src.model` rather than loading a "
            "bundle whose contents cannot be confirmed."
        )
    digest = hashlib.sha256()
    with open(joblib_path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != recorded:
        raise ValueError(
            f"{joblib_path} does not match the digest in {sidecar.name}: "
            f"expected {recorded[:16]}..., got {actual[:16]}.... The bundle is "
            "corrupt, or it and its metrics file came from different runs. "
            "Retrain with `python -m src.model`."
        )


def load_run(run_path: str | Path) -> dict:
    """Load a saved run bundle. Accepts the .json or .joblib path."""
    run_path = Path(run_path)
    joblib_path = run_path.with_suffix(".joblib")
    _verify_bundle_digest(joblib_path, run_path.with_suffix(".json"))
    bundle = joblib.load(joblib_path)
    # "categories" is checked here, not just the three original keys: a bundle
    # missing it is a pre-rewrite artifact, and scoring with one reintroduces
    # the categorical-schema mismatch (a one-row inference frame defines its
    # own categories and LightGBM rejects it). Silently defaulting to {} in
    # prepare_frame would let that bug back in for any old bundle; failing
    # loudly here forces a retrain instead.
    for key in ("model", "label_encoder", "feature_cols", "categories"):
        if key not in bundle:
            raise KeyError(
                f"Run bundle at {run_path} is missing {key!r}. It was probably "
                "written by an older version of src/model.py — retrain with "
                "`python -m src.model`."
            )
    return bundle


# Values a field cannot legitimately hold. Checked at inference only.
#
# The bounds are read off the training data rather than invented: every numeric
# column in data/processed/train.parquet has min >= 0 (account_age_days 1,
# avg_order_value_usd 15.01, total_orders_lifetime 1, days_to_return 1), and
# return_rate_pct is a percentage. So none of these can fire on a record drawn
# from the real distribution -- they exist for what a caller sends, not for what
# the dataset contains.
#
# Deliberately NOT here: cross-field invariants, of which the obvious one is
# total_returns_lifetime <= total_orders_lifetime. It is real and it is
# violable, but a two-column rule has no unambiguous answer to "which of the two
# do I null", and guessing would substitute a worse lie for the one this is
# fixing. Left out knowingly rather than overlooked.
BINARY_FLAGS = (
    "multiple_accounts_flag",
    "tracking_number_valid",
    "photo_evidence_provided",
    "item_returned_opened",
    "return_packaging_intact",
    "address_change_before_delivery",
    "refund_to_different_account",
    "review_left_after_return",
    "discount_used",
    "is_high_value_item",
)

# column -> (inclusive lower bound, inclusive upper bound or None)
DOMAIN_RULES = {"return_rate_pct": (0.0, 100.0)}
DOMAIN_RULES.update({flag: (0.0, 1.0) for flag in BINARY_FLAGS})
DEFAULT_NUMERIC_BOUNDS = (0.0, None)


def _coerce_numeric_columns(
    records: pd.DataFrame, bundle: dict
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Force object-dtype columns that the model treats as numeric into numbers,
    reporting the ones that held something unconvertible.

    Why this runs before `add_transaction_level_features`: the engineered
    features divide raw columns by each other, so a string reaching that point
    fails as `TypeError: unsupported operand type(s) for /: 'str' and 'str'`
    inside pandas, with no indication of which column or row was at fault. By
    then the useful context is gone.

    Only `object`-dtype columns are touched. A column that already parsed as
    numeric is left exactly as it was, so a well-formed record takes an
    unchanged path through scoring -- this must not perturb any published
    number.

    Categorical columns are skipped: their values are *supposed* to be strings,
    and `prepare_frame` pins them to the training levels further down.
    """
    numeric_cols = [
        col
        for col in bundle["feature_cols"]
        if col not in bundle.get("categories", {}) and col in records.columns
    ]
    invalid: list[str] = []
    out_of_range: list[str] = []
    coerced = records

    def _writable():
        nonlocal coerced
        if coerced is records:
            coerced = records.copy()
        return coerced

    for col in numeric_cols:
        if coerced[col].dtype == object:
            converted = pd.to_numeric(coerced[col], errors="coerce")
            # Only report a column whose values were actually lost. An object
            # column of clean numeric strings converts silently and is not a
            # caller error.
            if converted.isna().sum() > coerced[col].isna().sum():
                invalid.append(col)
            _writable()[col] = converted

        low, high = DOMAIN_RULES.get(col, DEFAULT_NUMERIC_BOUNDS)
        values = coerced[col]
        violates = values.notna() & (values < low)
        if high is not None:
            violates |= values.notna() & (values > high)
        if violates.any():
            out_of_range.append(col)
            # Null the offending cells only. A batch where one row carries a
            # negative refund keeps scoring the other rows on real values --
            # and, more to the point, the impossible number stops contributing
            # to a recommendation instead of driving one.
            _writable().loc[violates, col] = np.nan

    return coerced, invalid, out_of_range


def prepare_frame(records: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Build a model frame matching the training schema exactly, for one row
    or many. `prepare_record` is a one-row convenience wrapper over this —
    both single-record and batch scoring go through the same transform.

    A caller that omits a feature gets NaN for it, not a hard failure: LightGBM
    handles missing values natively and a merchant integration will legitimately
    not carry every column. But the omission is never silent — the names land in
    `frame.attrs["missing_features"]` and are surfaced in the scoring output, so
    a recommendation made on a partial record says so.

    A value that should be numeric but is not -- "N/A", "", "unknown", the
    string a spreadsheet export leaves behind -- is coerced to NaN rather than
    allowed to reach feature engineering, where dividing it raised a TypeError
    from two frames deep inside pandas. Those column names land in
    `frame.attrs["invalid_features"]` and are surfaced the same way missing ones
    are: NaN is what the model was trained to handle, but a caller must never
    have to guess that a field was dropped.

    Raises ValueError when the record supplies *none* of the expected features,
    which is a caller error (wrong schema, empty payload) rather than a partial
    record worth scoring.
    """
    records, invalid, out_of_range = _coerce_numeric_columns(records, bundle)
    frame = add_transaction_level_features(records)

    feature_cols = bundle["feature_cols"]
    missing = [col for col in feature_cols if col not in frame.columns]
    if len(missing) == len(feature_cols):
        raise ValueError(
            f"Record supplies none of the {len(feature_cols)} features this "
            f"'{bundle.get('track', 'unknown')}' model was trained on. Got columns "
            f"{sorted(records.columns)[:10]}...; expected e.g. {feature_cols[:5]}. "
            "Check the input schema against runs/model_<track>.json 'feature_cols'."
        )
    for col in missing:
        # np.nan, not None: None makes the column `object` dtype, and LightGBM
        # rejects that with a bare "pandas dtypes must be int, float or bool"
        # traceback. NaN is the value the model was actually trained to handle.
        frame[col] = np.nan
    frame = frame[feature_cols].copy()

    # Reapply the *training* categorical levels. A frame built from scratch
    # would otherwise define its own categories from whatever values happen
    # to be present (a single-value set, for a one-row frame) and LightGBM
    # would reject the schema. A level never seen in training becomes NaN,
    # which is the honest representation of "this value was never in
    # training" rather than a silently shifted integer code.
    for col, levels in bundle.get("categories", {}).items():
        if col in frame.columns:
            frame[col] = pd.Categorical(frame[col], categories=levels)

    # Derive what the model actually could not see, rather than trusting the
    # bookkeeping above to have caught every cause.
    #
    # `missing`, `invalid` and `out_of_range` each explain some NaNs. Anything
    # left NaN after those is a feature that arrived present, parseable and in
    # range and still could not be computed -- in practice the engineered ratios
    # whose denominator was zero (src/features.py nulls those denominators, so
    # avg_order_value_usd=0 yields refund_to_avg_order_ratio=NaN). Reporting
    # only the named causes let a record be scored on three silently dropped
    # features while claiming `features_missing: []`.
    #
    # Subtracting the explained sets from the observed NaNs means a cause nobody
    # anticipated still surfaces here instead of disappearing.
    nan_mask = frame.isna()
    explained = set(missing) | set(invalid) | set(out_of_range)
    degraded = [
        col for col in frame.columns if col not in explained and bool(nan_mask[col].any())
    ]

    frame.attrs["missing_features"] = missing
    frame.attrs["invalid_features"] = invalid
    frame.attrs["out_of_range_features"] = out_of_range
    frame.attrs["degraded_features"] = degraded
    # Per row, not per frame: in a batch, row 0 carrying a bad cell says nothing
    # about row 1, and a count that pretends otherwise is the same class of
    # dishonesty this block exists to remove.
    frame.attrs["n_features_not_seen"] = nan_mask.sum(axis=1).tolist()
    return frame


def prepare_record(record: dict, bundle: dict) -> pd.DataFrame:
    """Build a one-row model frame matching the training schema exactly."""
    return prepare_frame(pd.DataFrame([record]), bundle)


def _route(
    proba,
    class_names: list[str],
    posture: str,
    friction_posture: str = DEFAULT_FRICTION_POSTURE,
) -> pd.DataFrame:
    """Shared decision step for score_record and score_batch: cost-calibrated
    action per row (never argmax) under two explicit postures.

    `posture` sets C_fp : C_fn (hard-block an honest customer vs. approve real
    fraud). `friction_posture` sets friction cost vs. missed-recovery cost
    (approve vs. soft-friction). Both are exposed because on this dataset only
    the second one moves: the C_fp:C_fn sweep is byte-identical across every
    posture on `full` and differs on 29 of 12,000 rows on `testbed`, while the
    friction axis spans 2.78%-24.79% of legitimate customers. Routing under a
    hardcoded friction cost — as this function used to — meant the served
    decision could not be moved along the very axis §6.2 calls the deliverable.
    """
    if posture not in DEFAULT_POSTURES:
        raise ValueError(
            f"Unknown posture {posture!r}. Expected one of {sorted(DEFAULT_POSTURES)}."
        )
    if friction_posture not in FRICTION_POSTURES:
        raise ValueError(
            f"Unknown friction posture {friction_posture!r}. "
            f"Expected one of {sorted(FRICTION_POSTURES)}."
        )

    c_fp, c_fn = DEFAULT_POSTURES[posture]
    cost_matrix = build_cost_matrix(
        class_names,
        c_fp=c_fp,
        c_fn=c_fn,
        friction_cost=FRICTION_POSTURES[friction_posture],
    )
    action_idx = expected_cost_decision(proba, cost_matrix)
    most_likely_idx = proba.argmax(axis=1)

    actions = [ACTIONS[i] for i in action_idx]
    most_likely = [class_names[i] for i in most_likely_idx]

    out = pd.DataFrame(
        {
            "most_likely_class": most_likely,
            "recommended_action": actions,
            "recommended_intervention": [INTERVENTION[a] for a in actions],
            # Surfaced so a reviewer can see the decision was cost-driven
            # rather than argmax — these differ exactly when the record is
            # near a boundary, which is the interesting case.
            "argmax_would_route_to": [_class_default_action(c) for c in most_likely],
        }
    )
    for i, cls in enumerate(class_names):
        out[f"proba_{cls}"] = proba[:, i]
    out["posture"] = posture
    out["friction_posture"] = friction_posture
    return out


def score_record(
    record: dict,
    bundle: dict,
    posture: str = DEFAULT_POSTURE,
    friction_posture: str = DEFAULT_FRICTION_POSTURE,
    explain: bool = False,
) -> dict:
    """Score one raw return record and route it to an intervention.

    `posture` selects the C_fp : C_fn stance from
    src.evaluate.DEFAULT_POSTURES; `friction_posture` selects the
    approve-vs-soft-friction stance from src.evaluate.FRICTION_POSTURES. Both
    are explicit arguments because both change the recommendation and hiding
    either inside the scorer would defeat the point of §6.2 — but they are not
    equally powerful on this data, and the caller should know which is which:
    on the `full` track the C_fp:C_fn axis moves nothing at all, while the
    friction axis moves roughly a fifth of legitimate customers across its
    range. See FRICTION_POSTURES for the measured spans.
    """
    clf = bundle["model"]
    label_encoder = bundle["label_encoder"]
    class_names = list(label_encoder.classes_)

    frame = prepare_record(record, bundle)
    proba = clf.predict_proba(frame)
    routed = _route(proba, class_names, posture, friction_posture).iloc[0]

    result = {
        "most_likely_class": routed["most_likely_class"],
        "class_probabilities": {c: float(p) for c, p in zip(class_names, proba[0])},
        "posture": posture,
        "friction_posture": friction_posture,
        "recommended_action": routed["recommended_action"],
        "recommended_intervention": routed["recommended_intervention"],
        "argmax_would_route_to": routed["argmax_would_route_to"],
        "track": bundle.get("track", "unknown"),
        # Surfaced, not swallowed: a recommendation built on a partial record
        # names the features it did not have.
        "features_missing": list(frame.attrs.get("missing_features", [])),
        # Same principle for a field that was present but unusable: coercing it
        # to NaN keeps the record scorable, and naming it here keeps that
        # substitution from being invisible.
        "features_invalid": list(frame.attrs.get("invalid_features", [])),
        # Present and parseable, but impossible -- a negative refund, a flag
        # that is neither 0 nor 1. Nulled rather than allowed to drive a
        # recommendation; see DOMAIN_RULES.
        "features_out_of_range": list(frame.attrs.get("out_of_range_features", [])),
        # Present, valid, in range, and still not computable -- the zero
        # denominator case.
        "features_degraded": list(frame.attrs.get("degraded_features", [])),
        # The number that cannot be gamed by categorising causes: how many of
        # the trained features the model saw as missing for this row.
        "n_features_not_seen": int(frame.attrs.get("n_features_not_seen", [0])[0]),
    }

    if explain:
        # Imported here, not at module scope: shap pulls in a slow dependency
        # tree, and the default scoring path should not pay for a feature it is
        # not using. src.explain owns the TreeExplainer so the per-record
        # explanation and the per-class study in runs/shap_*.json cannot drift.
        from src.explain import explain_row

        result["explanation"] = explain_row(bundle, frame, result["most_likely_class"])

    return result


def score_batch(
    records: pd.DataFrame,
    bundle: dict,
    posture: str = DEFAULT_POSTURE,
    friction_posture: str = DEFAULT_FRICTION_POSTURE,
) -> pd.DataFrame:
    """Score many raw return records at once (`scripts/score.py --csv`).

    Same feature engineering and cost-calibrated decision rule as
    `score_record`, under the same two postures, vectorized over the whole
    frame instead of looping a Python-level call per row. Returns one output
    row per input row, with the original columns preserved alongside the
    scoring columns so results can be traced back to the source record.
    """
    clf = bundle["model"]
    label_encoder = bundle["label_encoder"]
    class_names = list(label_encoder.classes_)

    frame = prepare_frame(records, bundle)
    proba = clf.predict_proba(frame)
    routed = _route(proba, class_names, posture, friction_posture)
    routed["track"] = bundle.get("track", "unknown")

    # A missing column is missing for the whole batch, so this is one number per
    # run rather than per row — but it belongs in the CSV, where a reader who
    # never saw the console output can still see the scores were partial.
    missing = list(frame.attrs.get("missing_features", []))
    invalid = list(frame.attrs.get("invalid_features", []))
    out_of_range = list(frame.attrs.get("out_of_range_features", []))
    degraded = list(frame.attrs.get("degraded_features", []))

    # A missing column is missing for the whole batch, so this one stays a
    # single number per run.
    routed["n_features_missing"] = len(missing)
    # These are per row. A frame-level count repeated on every row would report
    # row 1 as damaged because row 0 was.
    nan_mask = frame.isna().reset_index(drop=True)
    for name, cols in (
        ("n_features_invalid", invalid),
        ("n_features_out_of_range", out_of_range),
        ("n_features_degraded", degraded),
    ):
        routed[name] = (
            nan_mask[cols].sum(axis=1).to_numpy() if cols else 0
        )
    routed["n_features_not_seen"] = nan_mask.sum(axis=1).to_numpy()

    out = pd.concat([records.reset_index(drop=True), routed.reset_index(drop=True)], axis=1)
    out.attrs["missing_features"] = missing
    out.attrs["invalid_features"] = invalid
    out.attrs["out_of_range_features"] = out_of_range
    out.attrs["degraded_features"] = degraded
    return out


def _class_default_action(class_name: str) -> str:
    from src.evaluate import CLASS_TARGET_ACTION

    return CLASS_TARGET_ACTION.get(class_name, "unmapped")
