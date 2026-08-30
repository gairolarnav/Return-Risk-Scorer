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
    `expected_cost_decision`, under an explicit, caller-visible posture.

Nothing here executes an action. It returns a recommendation and the
probabilities behind it, per the defense-only constraint (§7).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.evaluate import (
    ACTIONS,
    DEFAULT_POSTURES,
    build_cost_matrix,
    expected_cost_decision,
)
from src.features import add_transaction_level_features

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
    # "categories" is included here, not just the three original keys: a
    # bundle missing it is exactly the pre-rewrite artifact bug 1 above was
    # fixed for (a one-row inference frame defines its own categorical
    # schema and LightGBM rejects it). Silently defaulting to {} in
    # prepare_record would let that bug back in for any old bundle instead
    # of failing loudly here, which is what "Old bundles are incompatible"
    # promises.
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
    both single-record and batch scoring go through the same transform."""
    frame = add_transaction_level_features(records)

    feature_cols = bundle["feature_cols"]
    for col in feature_cols:
        if col not in frame.columns:
            frame[col] = None
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

    return frame


def prepare_record(record: dict, bundle: dict) -> pd.DataFrame:
    """Build a one-row model frame matching the training schema exactly."""
    return prepare_frame(pd.DataFrame([record]), bundle)


def _route(proba, class_names: list[str], posture: str) -> pd.DataFrame:
    """Shared decision step for score_record and score_batch: cost-calibrated
    action per row (never argmax) under an explicit posture."""
    if posture not in DEFAULT_POSTURES:
        raise ValueError(
            f"Unknown posture {posture!r}. Expected one of {sorted(DEFAULT_POSTURES)}."
        )

    c_fp, c_fn = DEFAULT_POSTURES[posture]
    cost_matrix = build_cost_matrix(class_names, c_fp=c_fp, c_fn=c_fn)
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
    return out


def score_record(
    record: dict,
    bundle: dict,
    posture: str = "loss-neutral (1:1)",
) -> dict:
    """Score one raw return record and route it to an intervention.

    `posture` selects the merchant cost stance from
    src.evaluate.DEFAULT_POSTURES. It is an explicit argument with no silent
    default beyond the neutral one, because the chosen posture changes the
    recommendation and hiding that choice inside the scorer would defeat the
    point of §6.2.
    """
    clf = bundle["model"]
    label_encoder = bundle["label_encoder"]
    class_names = list(label_encoder.classes_)

    frame = prepare_record(record, bundle)
    proba = clf.predict_proba(frame)
    routed = _route(proba, class_names, posture).iloc[0]

    return {
        "most_likely_class": routed["most_likely_class"],
        "class_probabilities": {c: float(p) for c, p in zip(class_names, proba[0])},
        "posture": posture,
        "recommended_action": routed["recommended_action"],
        "recommended_intervention": routed["recommended_intervention"],
        "argmax_would_route_to": routed["argmax_would_route_to"],
        "track": bundle.get("track", "unknown"),
    }


def score_batch(
    records: pd.DataFrame,
    bundle: dict,
    posture: str = "loss-neutral (1:1)",
) -> pd.DataFrame:
    """Score many raw return records at once (`scripts/score.py --csv`).

    Same feature engineering and cost-calibrated decision rule as
    `score_record`, vectorized over the whole frame instead of looping a
    Python-level call per row. Returns one output row per input row, with the
    original columns preserved alongside the scoring columns so results can
    be traced back to the source record.
    """
    clf = bundle["model"]
    label_encoder = bundle["label_encoder"]
    class_names = list(label_encoder.classes_)

    frame = prepare_frame(records, bundle)
    proba = clf.predict_proba(frame)
    routed = _route(proba, class_names, posture)
    routed.index = records.index
    routed["track"] = bundle.get("track", "unknown")

    return pd.concat([records.reset_index(drop=True), routed.reset_index(drop=True)], axis=1)


def _class_default_action(class_name: str) -> str:
    from src.evaluate import CLASS_TARGET_ACTION

    return CLASS_TARGET_ACTION.get(class_name, "unmapped")
