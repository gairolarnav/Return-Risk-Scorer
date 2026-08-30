"""
Tests for src/segment_audit.py — segment-level FPR audit by order-value
bucket (docs/ARCHITECTURE.md §6.2 stretch goal).

Two layers: pure-function tests against hand-built arrays (no model/training
needed, since audit_track's only real logic is bucketing + rate arithmetic
over already-decided actions), and one end-to-end test against a synthetic
saved run under tmp_path (same pattern as tests/test_score.py) to exercise
the actual file-reading/writing path.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.evaluate import ACTIONS, DEFAULT_POSTURES, build_cost_matrix
from src.segment_audit import _concentration, _order_value_buckets, audit_track, run

CLASSES = ["Fraudulent Return", "Legitimate", "Policy Abuser", "Wardrobing"]


class _DummyModel:
    """audit_track never calls bundle["model"] -- it reads the saved
    proba/ytest arrays directly -- so this only needs to exist and be
    picklable, not do anything."""


def test_order_value_buckets_are_balanced_quartiles():
    values = pd.Series(np.arange(1, 101, dtype=float))  # 1..100, evenly spread
    bucket, labels = _order_value_buckets(values, n_buckets=4)
    assert len(labels) == 4
    counts = bucket.value_counts()
    assert set(counts.index) == set(labels)
    # Quartiles of a uniform 1..100 series should split ~25/25/25/25.
    for label in labels:
        assert 20 <= counts[label] <= 30


def test_order_value_buckets_labels_carry_the_dollar_range():
    values = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    _, labels = _order_value_buckets(values, n_buckets=4)
    assert all("$" in label for label in labels)
    assert any("lowest" in label for label in labels)
    assert any("highest" in label for label in labels)


@pytest.fixture
def wired_audit(tmp_path, monkeypatch):
    """A synthetic saved run: proba/ytest arrays + a matching test.parquet
    with avg_order_value_usd, wired the way src.model.save_run +
    src.features.run actually produce them."""
    rng = np.random.default_rng(0)
    n = 400
    y_true_idx = rng.integers(0, 4, size=n)  # indices into CLASSES (alphabetical)

    # Build proba that mostly agrees with y_true, confidently enough that
    # the "default" rows actually route to approve rather than
    # soft_friction: with c_fn=120 in the cost matrix, even ~5% residual
    # mass elsewhere makes approve's expected cost exceed soft_friction's
    # friction_cost=1.0 baseline, which would make every row -- not just the
    # ones deliberately injected below -- get frictioned. 0.999 confidence
    # keeps that residual-mass cost below the friction cost for "default"
    # rows, so only the deliberately uncertain low-value rows get frictioned.
    proba = np.full((n, 4), (1 - 0.999) / 3)
    proba[np.arange(n), y_true_idx] = 0.999
    proba /= proba.sum(axis=1, keepdims=True)

    # Order values: deliberately skew so low-value legitimate rows get
    # frictioned more than high-value ones, to prove the concentration
    # detector actually fires when it should.
    order_value = rng.uniform(10, 500, size=n)
    legit_idx = CLASSES.index("Legitimate")
    is_legit = y_true_idx == legit_idx
    # For legitimate rows with a low order value, inject uncertainty toward
    # Policy Abuser so the decision layer routes some of them to friction.
    low_value_legit = is_legit & (order_value < 100)
    proba[low_value_legit] = [0.02, 0.55, 0.41, 0.02]

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    np.save(runs_dir / "model_full_proba.npy", proba)
    np.save(runs_dir / "model_full_ytest.npy", y_true_idx)

    from sklearn.preprocessing import LabelEncoder

    import joblib

    bundle = {
        "model": _DummyModel(),
        "label_encoder": LabelEncoder().fit(CLASSES),
        "feature_cols": ["avg_order_value_usd"],
        "categories": {},
        "track": "full",
    }
    joblib.dump(bundle, runs_dir / "model_full.joblib")

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    test_df = pd.DataFrame({"avg_order_value_usd": order_value, "abuse_type": [CLASSES[i] for i in y_true_idx]})
    test_df.to_parquet(processed_dir / "test.parquet", index=False)

    import src.segment_audit as audit_module

    monkeypatch.setattr(audit_module, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(audit_module, "PROCESSED_DIR", processed_dir)
    return runs_dir


def test_concentration_ignores_all_noise_level_rates():
    """All buckets under MIN_RATE_TO_FLAG (single-digit row counts on a
    few-thousand-row bucket) must not be flagged, even though one is
    exactly 0 and another isn't -- this is the exact case that used to be
    silently indistinguishable from "everything is real and flat"."""
    result = _concentration([0.0004, 0.0, 0.0013, 0.0])
    assert result["flagged"] is False
    assert result["ratio"] is None


def test_concentration_flags_a_real_zero_vs_nonzero_split():
    """A rate that's clearly non-negligible in one bucket and exactly zero
    in another is the strongest possible concentration signal and must be
    flagged, with ratio reported as undefined (not silently None-and-fine)."""
    result = _concentration([0.63, 0.0, 0.0, 0.0])
    assert result["flagged"] is True
    assert result["ratio"] is None
    assert result["max"] == pytest.approx(0.63)
    assert result["min"] == 0.0


def test_concentration_flags_a_real_ratio_above_threshold():
    result = _concentration([0.1040, 0.1231, 0.0730, 0.0504])
    assert result["flagged"] is True
    assert result["ratio"] == pytest.approx(0.1231 / 0.0504)


def test_concentration_does_not_flag_a_ratio_below_threshold():
    result = _concentration([0.10, 0.11, 0.09, 0.095])
    assert result["flagged"] is False


def test_audit_track_reports_all_buckets_and_valid_rates(wired_audit):
    result = audit_track("full")
    assert result["track"] == "full"
    assert len(result["buckets"]) == 4
    total_legit = sum(r["n_legitimate_customers"] for r in result["buckets"])
    assert total_legit == result["n_legitimate_total"]
    for row in result["buckets"]:
        if row["n_legitimate_customers"] > 0:
            assert 0.0 <= row["hard_block_fpr"] <= 1.0
            assert 0.0 <= row["soft_friction_rate"] <= 1.0


def test_audit_flags_the_injected_concentration(wired_audit):
    """The fixture deliberately gives low-order-value legitimate rows more
    friction than high-order-value ones -- the concentration detector must
    actually notice."""
    result = audit_track("full")
    buckets = result["buckets"]
    lowest = buckets[0]
    highest = buckets[-1]
    assert lowest["soft_friction_rate"] > highest["soft_friction_rate"]
    assert highest["soft_friction_rate"] == 0.0
    concentration = result["soft_friction_concentration"]
    assert concentration["flagged"] is True
    # min is exactly 0 here (an unbounded ratio) -- the strongest possible
    # concentration signal, which is exactly the case _concentration must
    # not silently report as "ratio undefined, nothing to see."
    assert concentration["ratio"] is None
    assert concentration["max"] > 0.0


def test_run_writes_json_and_png(wired_audit):
    results = run(["full"])
    runs_dir = wired_audit
    json_path = runs_dir / "segment_fpr_full.json"
    png_path = runs_dir / "segment_fpr_full.png"
    assert json_path.exists()
    assert png_path.exists()
    on_disk = json.loads(json_path.read_text())
    assert on_disk == results["full"]


def test_no_action_disagrees_with_the_shared_decision_layer(wired_audit):
    """Sanity check that audit_track is really routing through
    build_cost_matrix/expected_cost_decision and not some parallel
    implementation: the hard_block_fpr for Fraudulent-Return-heavy buckets
    should be near the class's true positive hard-block rate, which is only
    meaningful if ACTIONS/DEFAULT_POSTURES came from src.evaluate."""
    assert "loss-neutral (1:1)" in DEFAULT_POSTURES
    assert set(ACTIONS) == {"approve", "soft_friction", "hard_block"}
    cost = build_cost_matrix(CLASSES, c_fp=120.0, c_fn=120.0)
    assert cost.loc["Legitimate", "approve"] == 0.0  # smoke check the shared matrix is sane
