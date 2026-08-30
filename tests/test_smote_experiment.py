"""
Tests for src/smote_experiment.py — SMOTE against the class-weighted
baseline on testbed, with a documented keep/discard verdict and the numbers
behind it (docs/ARCHITECTURE.md §5).

Synthetic in-memory data under tmp_path, same approach as
tests/test_score.py and tests/test_explain.py — deliberately imbalanced
(one minority class) so SMOTENC has real oversampling work to do, with each
class's training fold still comfortably above the default k_neighbors=5
floor SMOTENC needs to find neighbors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.model import RANDOM_STATE
from src.smote_experiment import _verdict, run

CLASSES = ["Legitimate", "Wardrobing", "Policy Abuser", "Fraudulent Return"]
# Imbalanced on purpose: majority class gets far more rows than the rest.
N_PER_CLASS = {"Legitimate": 60, "Wardrobing": 18, "Policy Abuser": 18, "Fraudulent Return": 18}


def _synthetic_frame(seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i, cls in enumerate(CLASSES):
        # Give each class a shifted center so the classifier has *something*
        # real to learn, rather than pure label noise.
        center = i * 40
        for _ in range(N_PER_CLASS[cls]):
            rows.append(
                {
                    "avg_order_value_usd": float(rng.normal(center + 50, 10)),
                    "refund_amount_requested_usd": float(rng.normal(center + 20, 8)),
                    "total_orders_lifetime": int(rng.integers(1, 30)),
                    "total_returns_lifetime": int(rng.integers(0, 10)),
                    "account_age_days": int(rng.integers(1, 900)),
                    "days_to_return": int(rng.integers(1, 60)),
                    "product_category": rng.choice(["Apparel", "Electronics", "Home"]),
                    "customer_segment": rng.choice(["Gold", "Silver", "Bronze"]),
                    "order_date": "2022-01-01",
                    "return_date": "2022-06-15",
                    "abuse_type": cls,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def wired_smote(tmp_path, monkeypatch):
    from src.features import add_transaction_level_features

    frame = add_transaction_level_features(_synthetic_frame())
    train_parts, test_parts = [], []
    for cls in CLASSES:
        cls_rows = frame[frame["abuse_type"] == cls]
        cut = int(len(cls_rows) * 0.7)
        train_parts.append(cls_rows.iloc[:cut])
        test_parts.append(cls_rows.iloc[cut:])
    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    train.to_parquet(processed_dir / "train.parquet", index=False)
    test.to_parquet(processed_dir / "test.parquet", index=False)

    runs_dir = tmp_path / "runs"

    import src.smote_experiment as smote_module

    monkeypatch.setattr(smote_module, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(smote_module, "RUNS_DIR", runs_dir)
    # This tiny synthetic set doesn't have a "testbed" ablation split -- the
    # module's TRACK constant just needs to name a valid track in
    # FEATURE_SETS; "full" works identically here since feature_columns only
    # excludes DROP_COLS/TARGET_COL for a track with an empty exclusion list.
    monkeypatch.setattr(smote_module, "TRACK", "full")
    return runs_dir


def test_run_writes_a_verdict_json_with_real_numbers(wired_smote):
    payload = run()
    assert payload["track"] == "full"
    assert payload["random_state"] == RANDOM_STATE
    assert payload["verdict"] in ("KEEP", "DISCARD")
    assert payload["reason"]  # non-empty prose, not a placeholder

    baseline = payload["class_weighted_baseline"]
    smote_result = payload["smote_nc"]
    for result in (baseline, smote_result):
        assert 0.0 <= result["macro_f1"] <= 1.0
        assert 0.0 <= result["accuracy"] <= 1.0
        assert set(result["per_class_recall"]) == set(CLASSES)

    out_path = wired_smote / "smote_testbed.json"
    assert out_path.exists()
    import json

    on_disk = json.loads(out_path.read_text())
    assert on_disk == payload


def test_smote_resampling_balances_the_training_classes(wired_smote):
    payload = run()
    # The training set is imbalanced 60/18/18/18; SMOTENC must have actually
    # run (not silently no-opped) for n_train_smote_nc to exceed the
    # original imbalanced count by roughly what balancing 4 classes implies.
    assert payload["n_train_smote_nc"] > payload["n_train_baseline"]


def test_categorical_columns_are_reported_and_nonempty(wired_smote):
    payload = run()
    assert set(payload["categorical_features_smoted"]) >= {"product_category", "customer_segment"}


def test_verdict_discards_below_the_keep_margin():
    class_names = ["A", "B"]
    baseline = {
        "macro_f1": 0.80,
        "per_class_precision": {"A": 0.80, "B": 0.80},
    }
    smote_result = {
        "macro_f1": 0.803,  # +0.003, below the 0.01 keep margin
        "per_class_precision": {"A": 0.80, "B": 0.80},
    }
    verdict, reason = _verdict(baseline, smote_result, class_names)
    assert verdict == "DISCARD"
    assert "keep margin" in reason


def test_verdict_discards_on_precision_regression_even_with_f1_gain():
    class_names = ["A", "B"]
    baseline = {
        "macro_f1": 0.80,
        "per_class_precision": {"A": 0.80, "B": 0.80},
    }
    smote_result = {
        "macro_f1": 0.83,  # comfortably above the keep margin
        "per_class_precision": {"A": 0.80, "B": 0.70},  # -0.10 on B, over tolerance
    }
    verdict, reason = _verdict(baseline, smote_result, class_names)
    assert verdict == "DISCARD"
    assert "regression" in reason or "false positives" in reason


def test_verdict_keeps_a_real_improvement():
    class_names = ["A", "B"]
    baseline = {
        "macro_f1": 0.80,
        "per_class_precision": {"A": 0.80, "B": 0.80},
    }
    smote_result = {
        "macro_f1": 0.82,  # +0.02, above keep margin
        "per_class_precision": {"A": 0.81, "B": 0.79},  # tiny, within tolerance
    }
    verdict, reason = _verdict(baseline, smote_result, class_names)
    assert verdict == "KEEP"
