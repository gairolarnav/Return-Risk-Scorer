"""
SMOTE vs. the class-weighted baseline on `testbed` (docs/ARCHITECTURE.md §5).

Tests SMOTE against the class-weighted baseline and produces a documented
keep/discard verdict with the numbers behind it — the decision criteria are
fixed in code below (KEEP_MARGIN, PRECISION_REGRESSION_TOL) before the result
is known, so the verdict cannot be written to fit whichever arm won.

Restricted to `testbed`, on purpose. src/model.py's own docstring notes that
on `full`, class weighting is "very nearly a no-op ... a model at 0.999
macro-F1 has no minority-class recall left to recover." Running an imbalance
intervention against a track with essentially no imbalance-driven error left
to fix would produce a "no measurable effect" result that tells a reviewer
nothing about SMOTE — only that `full` doesn't need it. `testbed` (rung G)
is where real minority-class confusion exists (see runs/shap_interpretation.md),
so it is the only track where this comparison means anything.

Uses SMOTENC, not plain SMOTE: the testbed feature set has 8 categorical
columns (customer_segment, country, platform, device_type, payment_method,
return_reason, shipping_carrier, product_category) that plain SMOTE's
continuous-only interpolation cannot handle. Confirmed directly against the
real training frame (dtype check + a trial fit_resample) before committing
to this rather than assumed from the imbalanced-learn docs.

Both arms are trained and scored inside this one run, on the same
train/test split and the same LightGBM hyperparameters src/model.py uses —
a fair paired comparison, not a comparison against a possibly-stale
runs/model_testbed.json from a different code path.

Run as:
    python -m src.smote_experiment
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

from src.data_gate import NEEDS_FEATURES, require_artifacts
from src.features import PROCESSED_DIR, TARGET_COL, feature_columns
from src.model import (
    LEARNING_RATE,
    N_ESTIMATORS,
    RANDOM_STATE,
    as_model_frame,
    compute_class_weights,
)

RUNS_DIR = Path("runs")
TRACK = "testbed"  # SMOTE is evaluated on testbed only -- see module docstring.

# A track this size producing a macro-F1 improvement smaller than this is
# noise, not a reason to add a resampling step + its complexity/runtime cost
# to the pipeline. Chosen before looking at the result, not fit to it.
KEEP_MARGIN = 0.01
# A drop bigger than this in any single class's precision means SMOTE is
# manufacturing minority recall by making majority-class predictions worse
# -- exactly the false-positive-cost tradeoff this project reports rather than
# hiding behind a headline macro-F1 number (docs/ARCHITECTURE.md §6.2).
PRECISION_REGRESSION_TOL = 0.02


def _fit_and_evaluate(
    x_fit: pd.DataFrame,
    y_fit,
    x_test: pd.DataFrame,
    y_test,
    class_names: list[str],
    sample_weight=None,
) -> dict:
    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(class_names),
        random_state=RANDOM_STATE,
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        verbosity=-1,
        n_jobs=-1,
    )
    clf.fit(x_fit, y_fit, sample_weight=sample_weight)
    pred = clf.predict(x_test)
    report = classification_report(
        y_test, pred, target_names=class_names, output_dict=True, zero_division=0
    )
    return {
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "accuracy": float(accuracy_score(y_test, pred)),
        "per_class_precision": {c: report[c]["precision"] for c in class_names},
        "per_class_recall": {c: report[c]["recall"] for c in class_names},
        "per_class_f1": {c: report[c]["f1-score"] for c in class_names},
    }


def _verdict(baseline: dict, smote_result: dict, class_names: list[str]) -> tuple[str, str]:
    f1_delta = smote_result["macro_f1"] - baseline["macro_f1"]
    precision_deltas = {
        c: smote_result["per_class_precision"][c] - baseline["per_class_precision"][c]
        for c in class_names
    }
    worst_class, worst_delta = min(precision_deltas.items(), key=lambda kv: kv[1])

    if f1_delta < KEEP_MARGIN:
        return (
            "DISCARD",
            f"macro-F1 moved by {f1_delta:+.4f} (< {KEEP_MARGIN} keep margin) — SMOTENC does not "
            "measurably improve on class weighting alone on testbed. Class weighting is already "
            "in the pipeline at zero extra inference-time cost and no synthetic-neighbor step to "
            "explain to a panelist; SMOTE would add both for no measured benefit.",
        )
    if worst_delta < -PRECISION_REGRESSION_TOL:
        return (
            "DISCARD",
            f"macro-F1 improved by {f1_delta:+.4f}, but {worst_class} precision dropped by "
            f"{worst_delta:+.4f} (> {PRECISION_REGRESSION_TOL} regression tolerance) — the gain is "
            "manufactured minority recall at the cost of more false positives on another class, "
            "which is exactly the false-positive-cost tradeoff this project reports honestly rather "
            "than hiding behind a headline macro-F1 number.",
        )
    return (
        "KEEP",
        f"macro-F1 improved by {f1_delta:+.4f} (>= {KEEP_MARGIN} keep margin) with no per-class "
        f"precision regression worse than {worst_delta:+.4f} ({worst_class}) — a real, not "
        "manufactured, improvement.",
    )


def run() -> dict:
    """CLI entrypoint: train the class-weighted baseline and the SMOTENC
    arm on `testbed` (see module docstring for why only that track), score
    both, compute the keep/discard verdict, and write
    runs/smote_testbed.json.

    Returns: the full result dict that gets written, in case a caller
    wants it in memory. Reads data/processed/*.parquet directly (must
    already exist — run src.features first) rather than the trained
    model.joblib, since this needs to refit from scratch for a fair
    paired comparison (see module docstring).
    """
    require_artifacts(
        [PROCESSED_DIR / "train.parquet", PROCESSED_DIR / "test.parquet"], NEEDS_FEATURES
    )
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    cols = feature_columns(train, track=TRACK)
    x_train, x_test = as_model_frame(train, test, cols)

    # Train only -- see the note in src.model.train_track.
    label_encoder = LabelEncoder().fit(train[TARGET_COL])
    y_train = label_encoder.transform(train[TARGET_COL])
    y_test = label_encoder.transform(test[TARGET_COL])
    class_names = list(label_encoder.classes_)

    cat_idx = [i for i, c in enumerate(cols) if isinstance(x_train[c].dtype, pd.CategoricalDtype)]

    smote = SMOTENC(categorical_features=cat_idx, random_state=RANDOM_STATE, k_neighbors=5)
    x_res, y_res = smote.fit_resample(x_train, y_train)
    # Pin the resampled categorical columns back to the exact training
    # categories -- SMOTENC's synthetic rows are valid category values, but
    # the dtype needs re-pinning or LightGBM sees a schema drift from the
    # baseline arm. Same trap as the train/test categorical pinning in
    # src.model.as_model_frame, hit at a different step.
    for c in cols:
        if isinstance(x_train[c].dtype, pd.CategoricalDtype):
            x_res[c] = pd.Categorical(x_res[c], categories=x_train[c].cat.categories)

    # Class-weighted baseline: src.model.compute_class_weights itself, not a
    # copy of its arithmetic. The model is re-fit here (rather than read from
    # runs/model_testbed.json) so both arms come from the same code in the same
    # run -- a fair paired comparison. Re-fitting the model does not require
    # reimplementing the weighting, and a second copy that drifted would make
    # the "baseline" this experiment measures against quietly not the baseline.
    weights = compute_class_weights(train[TARGET_COL])
    sample_weight = train[TARGET_COL].map(weights).to_numpy()
    baseline = _fit_and_evaluate(x_train, y_train, x_test, y_test, class_names, sample_weight)

    smote_result = _fit_and_evaluate(x_res, y_res, x_test, y_test, class_names, sample_weight=None)

    verdict, reason = _verdict(baseline, smote_result, class_names)

    payload = {
        "track": TRACK,
        "random_state": RANDOM_STATE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_train_baseline": len(x_train),
        "n_train_smote_nc": len(x_res),
        "categorical_features_smoted": [cols[i] for i in cat_idx],
        "class_weighted_baseline": baseline,
        "smote_nc": smote_result,
        "verdict": verdict,
        "reason": reason,
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / "smote_testbed.json"
    out.write_text(json.dumps(payload, indent=2))

    print(f"class-weighted baseline: macro-F1={baseline['macro_f1']:.4f}  acc={baseline['accuracy']:.4f}")
    print(f"SMOTENC:                 macro-F1={smote_result['macro_f1']:.4f}  acc={smote_result['accuracy']:.4f}")
    print("per-class recall (baseline -> SMOTENC):")
    for c in class_names:
        print(
            f"  {c:<20} {baseline['per_class_recall'][c]:.4f} -> {smote_result['per_class_recall'][c]:.4f}"
        )
    print(f"verdict: {verdict} — {reason}")
    print(f"-> {out}")
    return payload


if __name__ == "__main__":
    run()
