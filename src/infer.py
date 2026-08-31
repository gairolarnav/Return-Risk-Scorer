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
approve/soft-friction axis at all — the axis the Day 4 correction identifies
as the only one that moves on this data (§6.2, and sweep_friction_curve's
docstring). Scoring a record on the `full` track under all three C_fp:C_fn
postures returned byte-identical actions for 12,000 of 12,000 test rows. The
`friction_posture` argument fixes that; `posture` is kept and still reported,
because showing which axis *doesn't* move is part of the honest account.

Nothing here executes an action. It returns a recommendation and the
probabilities behind it, per the defense-only constraint (§7).
"""

from __future__ import annotations

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


def load_run(run_path: str | Path) -> dict:
    """Load a saved run bundle. Accepts the .json or .joblib path."""
    run_path = Path(run_path)
    bundle = joblib.load(run_path.with_suffix(".joblib"))
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


def prepare_frame(records: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Build a model frame matching the training schema exactly, for one row
    or many. `prepare_record` is a one-row convenience wrapper over this —
    both single-record and batch scoring go through the same transform.

    A caller that omits a feature gets NaN for it, not a hard failure: LightGBM
    handles missing values natively and a merchant integration will legitimately
    not carry every column. But the omission is never silent — the names land in
    `frame.attrs["missing_features"]` and are surfaced in the scoring output, so
    a recommendation made on a partial record says so.

    Raises ValueError when the record supplies *none* of the expected features,
    which is a caller error (wrong schema, empty payload) rather than a partial
    record worth scoring.
    """
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

    frame.attrs["missing_features"] = missing
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

    return {
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
    }


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
    routed["n_features_missing"] = len(missing)

    out = pd.concat([records.reset_index(drop=True), routed.reset_index(drop=True)], axis=1)
    out.attrs["missing_features"] = missing
    return out


def _class_default_action(class_name: str) -> str:
    from src.evaluate import CLASS_TARGET_ACTION

    return CLASS_TARGET_ACTION.get(class_name, "unmapped")
