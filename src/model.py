"""
Model training (ARCHITECTURE.md §5), dual-track.

Two tracks are trained from the same code path, and which one produced a
number is always recorded alongside it:

    full      The honest model — every legitimate feature. Reported as "the
              model", together with the leakage finding and the flat
              cost sweep it produces.

    testbed   Rung G of src/ablation.py. NOT a model. A deliberately
              handicapped variant that exists so the §6.2 cost-calibration
              method has a non-degenerate region to operate on.

See the FEATURE_SETS block in src/features.py for why the distinction exists
and why the testbed is never presented as a result.

Class weights are inverse-frequency (§5). Note for the writeup: at a
70/12/10/8 split this is a mild intervention, and on the `full` track it is
very nearly a no-op, because a model at 0.999 macro-F1 has no minority-class
recall left to recover. The weighting matters on `testbed`, which is the
only track where it can.

RANDOM_STATE is fixed here and reused everywhere per §8.1.

Run as:
    python -m src.model              # trains both tracks
    python -m src.model full         # trains one
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder

from src.data_gate import (
    NEEDS_FEATURES,
    NEEDS_RAW_CSV,
    RANDOM_STATE,
    RAW_PATH,
    require_artifacts,
)
from src.features import FEATURE_SETS, PROCESSED_DIR, TARGET_COL, feature_columns

RUNS_DIR = Path("runs")

# Fixed, not tuned. There is no validation split, no search and no early
# stopping in this build, and that is deliberate — see docs/ARCHITECTURE.md
# §5.3. Briefly: the full track sits at 0.9988 macro-F1 with 99.9% of
# predictions above p=0.99, so a search would optimise in the fourth decimal
# by fitting the synthetic generator harder, and with only one temporal test
# split it would have to select against the test set to do it. The tuning
# effort in this project went into the cost matrix (§6.2), which is swept.
N_ESTIMATORS = 400
LEARNING_RATE = 0.05


def compute_class_weights(y: pd.Series) -> dict:
    """Inverse-frequency class weights (§5).

    Each class gets its own weight rather than a single global multiplier, so a
    later pass can move one class without disturbing the others. No such pass
    was run — see the note on N_ESTIMATORS/LEARNING_RATE above and §5.3.
    """
    counts = y.value_counts()
    n_classes = len(counts)
    return {cls: len(y) / (n_classes * count) for cls, count in counts.items()}


def as_model_frame(
    train: pd.DataFrame, test: pd.DataFrame, cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cast non-numeric columns to categorical, pinning test categories to the
    training set's. A category seen only at test time becomes NaN rather than
    silently shifting the integer codes the model was fitted against — the
    latter is a quiet correctness bug that shows up as unexplained test-set
    degradation."""
    x_train, x_test = train[cols].copy(), test[cols].copy()
    for col in cols:
        if not pd.api.types.is_numeric_dtype(x_train[col]):
            x_train[col] = x_train[col].astype("category")
            x_test[col] = pd.Categorical(x_test[col], categories=x_train[col].cat.categories)
    return x_train, x_test


def strawman_metrics(y_train: pd.Series, y_test: pd.Series) -> dict:
    """Always-predict-the-majority-class baseline (§6.1).

    Recorded specifically to justify rejecting accuracy as the headline
    metric, not as a result. On this dataset it scores ~0.70 accuracy and
    ~0.21 macro-F1 — the gap between those two numbers is the argument.
    """
    majority = y_train.value_counts().idxmax()
    pred = pd.Series([majority] * len(y_test), index=y_test.index)
    return {
        "strawman_class": str(majority),
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
    }


def train_track(
    train: pd.DataFrame,
    test: pd.DataFrame,
    track: str,
    class_weighted: bool = True,
) -> dict:
    """Train one LightGBM classifier for `track` ("full" or "testbed") and
    return everything a caller needs to save and later score with it.

    Args: already-split train/test frames; track selects the feature set
        via src.features.feature_columns; class_weighted toggles inverse-
        frequency weighting (on by default, see compute_class_weights).
    Returns: a dict with the fitted model, label_encoder, feature_cols,
    the training categorical `categories` (the non-obvious part — these
    must be persisted and reapplied at inference time, or a single-row
    frame builds its own category set and LightGBM rejects it), `proba`/
    `y_test` for downstream evaluation, and the `metrics` dict that gets
    written to runs/model_{track}.json.
    """
    cols = feature_columns(train, track=track)
    x_train, x_test = as_model_frame(train, test, cols)

    # Fit on TRAIN ONLY. Fitting on train+test would be harmless in effect here
    # (both splits carry all four classes, so the encoder is identical either
    # way) but it is still fitting a transformer on the test set, which this
    # project has no business doing. A class present only at test time would
    # now raise on transform rather than silently entering the label space --
    # the correct behaviour: a class the model never trained on cannot be a
    # prediction target.
    label_encoder = LabelEncoder().fit(train[TARGET_COL])
    y_train = label_encoder.transform(train[TARGET_COL])
    y_test = label_encoder.transform(test[TARGET_COL])

    sample_weight = None
    weights = None
    if class_weighted:
        weights = compute_class_weights(train[TARGET_COL])
        sample_weight = train[TARGET_COL].map(weights).to_numpy()

    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(label_encoder.classes_),
        random_state=RANDOM_STATE,
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        verbosity=-1,
        n_jobs=-1,
    )
    clf.fit(x_train, y_train, sample_weight=sample_weight)

    proba = clf.predict_proba(x_test)
    pred = proba.argmax(axis=1)
    top_prob = proba.max(axis=1)

    metrics = {
        "track": track,
        "n_features": len(cols),
        "feature_cols": cols,
        "class_weighted": class_weighted,
        "class_weights": {str(k): float(v) for k, v in (weights or {}).items()},
        "n_train": len(train),
        "n_test": len(test),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "accuracy": float(accuracy_score(y_test, pred)),
        "mean_top_prob": float(top_prob.mean()),
        "threshold_sensitive_frac": float((top_prob < 0.90).mean()),
        "strawman": strawman_metrics(train[TARGET_COL], test[TARGET_COL]),
    }

    # Persist the training set's categorical levels. Without these, a
    # single-record inference frame builds its own categories from one row,
    # LightGBM sees a different categorical schema than it was fitted on, and
    # prediction dies with "train and valid dataset categorical_feature do not
    # match". Rebuilding them from the raw CSV at inference time would be the
    # two-implementations failure mode §8 warns about.
    categories = {
        col: list(x_train[col].cat.categories)
        for col in cols
        if isinstance(x_train[col].dtype, pd.CategoricalDtype)
    }

    return {
        "model": clf,
        "label_encoder": label_encoder,
        "feature_cols": cols,
        "categories": categories,
        "proba": proba,
        "y_test": y_test,
        "metrics": metrics,
    }


def apply_hand_written_rule(df: pd.DataFrame) -> pd.Series:
    """The four-rule, zero-training baseline that motivates this project's
    headline finding (docs/LEAKAGE_FINDING.md) — read straight off the
    per-class range tables there, not fit to anything."""

    def rule(r):
        """One row -> one predicted class label, per the four thresholds
        read straight off docs/LEAKAGE_FINDING.md's per-class range tables."""
        if r.wishlist_to_cart_time_hrs <= 5.0:
            return "Wardrobing" if r.days_to_return >= 25 else "Fraudulent Return"
        return "Policy Abuser" if r.return_rate_pct > 15 else "Legitimate"

    return df.apply(rule, axis=1)


def rule_baseline_metrics(raw_path: Path = RAW_PATH) -> dict:
    """Accuracy/macro-F1 of the hand-written rule against the full raw
    dataset — not the train/test split, because the point is that the
    generator itself is separable, not a generalisation claim.

    This is the runnable source for the "0.9425 accuracy / 0.9188 macro-F1"
    figure quoted throughout README.md and docs/, which previously existed
    only as prose with no committed code behind it (every
    quoted figure must be regenerable). tests/test_baseline_rule.py pins the
    four thresholds, so an edit here cannot silently falsify those documents.
    """
    df = pd.read_csv(raw_path)
    pred = apply_hand_written_rule(df)
    y_true = df[TARGET_COL]
    # Union of true and predicted, not `y_true.unique()`. Taking labels from the
    # true column alone drops any class the rule predicts but that is absent
    # from y_true, so those predictions vanish from the matrix and the cells no
    # longer sum to n_rows — the same silent-divergence bug
    # evaluate.plot_confusion_matrix pins `labels=` to avoid.
    labels = sorted(set(y_true.unique()) | set(pred.unique()))
    return {
        "n_rows": len(df),
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(y_true, pred, labels=labels).tolist(),
        },
    }


def _sha256(path: Path) -> str:
    """SHA-256 of a file, read in chunks so a large bundle is not slurped."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_run(result: dict, run_name: str) -> Path:
    """One joblib artifact + one JSON metrics file per run (§8.1) — no
    experiment-tracking service at this scale."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": result["model"],
            "label_encoder": result["label_encoder"],
            "feature_cols": result["feature_cols"],
            "categories": result["categories"],
            "track": result["metrics"]["track"],
        },
        RUNS_DIR / f"{run_name}.joblib",
    )
    np.save(RUNS_DIR / f"{run_name}_proba.npy", result["proba"])
    np.save(RUNS_DIR / f"{run_name}_ytest.npy", result["y_test"])

    payload = {
        "run_name": run_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        # Digest of the .joblib this JSON describes. joblib is pickle-backed, so
        # loading one executes whatever it contains; src.infer.load_run checks
        # this before unpickling.
        #
        # What it does and does not buy, stated plainly: the digest sits in the
        # repository beside the file it covers, so anyone able to replace the
        # bundle can update the digest too. It is not a defence against a
        # deliberate attacker. It does catch the cases that actually happen --
        # a truncated checkout, a corrupted transfer, and a pair that drifted
        # because one track was retrained and only the .joblib was committed.
        "bundle_sha256": _sha256(RUNS_DIR / f"{run_name}.joblib"),
        **result["metrics"],
    }
    out = RUNS_DIR / f"{run_name}.json"
    out.write_text(json.dumps(payload, indent=2, default=str))
    return out


def run(tracks: list[str] | None = None) -> None:
    """CLI entrypoint: train_track for each of `tracks` (both, if None),
    save each to runs/model_{track}.*, then compute and save the track-
    independent rule baseline. Reads data/processed/*.parquet, which must
    already exist (run src.features first)."""
    require_artifacts(
        [PROCESSED_DIR / "train.parquet", PROCESSED_DIR / "test.parquet"], NEEDS_FEATURES
    )
    # Checked up front rather than after training: rule_baseline_metrics reads
    # the raw CSV at the very end of this function, and discovering it is
    # missing there would throw away both tracks' fits.
    require_artifacts([RAW_PATH], NEEDS_RAW_CSV)

    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")

    for track in tracks or list(FEATURE_SETS):
        result = train_track(train, test, track)
        m = result["metrics"]
        out = save_run(result, f"model_{track}")
        print(
            f"[{track:>7}] n_feat={m['n_features']:>3}  "
            f"macro-F1={m['macro_f1']:.4f}  acc={m['accuracy']:.4f}  "
            f"threshold-sensitive={m['threshold_sensitive_frac']:.1%}"
        )
        print(
            f"          strawman ({m['strawman']['strawman_class']}): "
            f"macro-F1={m['strawman']['macro_f1']:.4f} acc={m['strawman']['accuracy']:.4f}"
        )
        print(f"          -> {out}")

    # Track-independent: uses the raw CSV directly, not train/test. Written
    # to its own file rather than folded into model_{track}.json, which is
    # frozen output other docs already cite by exact hash.
    rule = rule_baseline_metrics()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    rule_out = RUNS_DIR / "baseline_rule.json"
    rule_out.write_text(
        json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), **rule}, indent=2)
    )
    print(f"[   rule] macro-F1={rule['macro_f1']:.4f}  acc={rule['accuracy']:.4f}  -> {rule_out}")


if __name__ == "__main__":
    run(sys.argv[1:] or None)
