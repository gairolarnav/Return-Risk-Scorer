"""
Tests for src/infer.py — regression coverage for the four bugs the Day 5
rewrite of this module was meant to close. Each test below is named for the
bug it guards against, not just the function it calls.

Uses synthetic in-memory bundles/records rather than the trained runs/*.joblib
artifacts, matching tests/test_features.py's approach — the Kaggle CSV and
trained bundles are gitignored, so a fresh clone must be able to run this
file before ever running the pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from src.evaluate import ACTIONS
from src.infer import INTERVENTION, load_run, prepare_record, score_record

CLASSES = ["Fraudulent Return", "Legitimate", "Policy Abuser", "Wardrobing"]


class _StubClassifier:
    """A fake LightGBM-shaped model: returns a fixed probability row for
    every record, regardless of features. Lets a test pin the exact
    probability vector the decision layer sees, instead of depending on
    where a real trained model happens to place its boundary."""

    def __init__(self, proba_row):
        self._proba_row = np.asarray(proba_row, dtype=float)

    def predict_proba(self, frame):
        return np.tile(self._proba_row, (len(frame), 1))


def _label_encoder() -> LabelEncoder:
    return LabelEncoder().fit(CLASSES)


def _bundle(model, feature_cols, categories=None, track="unit-test") -> dict:
    return {
        "model": model,
        "label_encoder": _label_encoder(),
        "feature_cols": feature_cols,
        "categories": categories or {},
        "track": track,
    }


# Bug 1 — categorical schema mismatch on a one-row inference frame


def test_prepare_record_reapplies_training_categories():
    """A single-record frame must not define its own one-value category set —
    it must be pinned to the categories the model was trained on, or LightGBM
    raises 'train and valid dataset categorical_feature do not match'."""
    bundle = _bundle(
        model=_StubClassifier([0.25, 0.25, 0.25, 0.25]),
        feature_cols=["product_category"],
        categories={"product_category": ["Apparel", "Electronics", "Home"]},
    )
    frame = prepare_record({"product_category": "Electronics"}, bundle)
    assert isinstance(frame["product_category"].dtype, pd.CategoricalDtype)
    assert list(frame["product_category"].cat.categories) == ["Apparel", "Electronics", "Home"]
    assert frame["product_category"].iloc[0] == "Electronics"


def test_prepare_record_maps_unseen_category_to_nan_not_a_shifted_code():
    """A category never seen in training must become NaN — the honest
    representation — not silently reindex onto a training code it doesn't
    mean (the same failure mode src.model.as_model_frame guards against on the
    train/test split, hit here at serving time instead)."""
    bundle = _bundle(
        model=_StubClassifier([0.25, 0.25, 0.25, 0.25]),
        feature_cols=["product_category"],
        categories={"product_category": ["Apparel", "Electronics", "Home"]},
    )
    frame = prepare_record({"product_category": "Toys"}, bundle)
    assert pd.isna(frame["product_category"].iloc[0])


def test_load_run_rejects_a_bundle_missing_categories(tmp_path):
    """Bundles written before the Day 5 rewrite lack the `categories` key, and
    load_run must reject them with a clear "retrain" message rather than
    loading them. A bundle missing categories must fail loudly here, not
    fall through to prepare_record silently skipping reapplication (bug 1
    reappearing under a different name)."""
    import joblib

    old_style = {
        "model": _StubClassifier([0.25, 0.25, 0.25, 0.25]),
        "label_encoder": _label_encoder(),
        "feature_cols": ["product_category"],
        # no "categories" key — this is what a pre-rewrite bundle looks like.
    }
    path = tmp_path / "old_bundle.joblib"
    joblib.dump(old_style, path)
    with pytest.raises(KeyError, match="categories"):
        load_run(tmp_path / "old_bundle")


# Bug 2 — INTERVENTION keyed by class name ("Policy Abuse" typo) instead of
# by action, silently routing Policy Abuser to "Unmapped"


def test_intervention_is_keyed_by_action_not_class_name():
    assert set(INTERVENTION) == set(ACTIONS)


def test_policy_abuser_never_routes_to_unmapped():
    """The exact regression: a confidently-Policy-Abuser row must resolve to
    a real intervention string, not fall through a class-name key mismatch."""
    # [Fraudulent, Legitimate, Policy Abuser, Wardrobing] — matches the
    # alphabetically-sorted LabelEncoder class order.
    bundle = _bundle(
        model=_StubClassifier([0.01, 0.02, 0.95, 0.02]),
        feature_cols=["avg_order_value_usd"],
    )
    result = score_record({"avg_order_value_usd": 100.0}, bundle)
    assert result["most_likely_class"] == "Policy Abuser"
    assert result["recommended_intervention"] != "Unmapped"
    assert "unmapped" not in result["recommended_intervention"].lower()


# Bug 3 — inference took argmax, bypassing the cost-calibrated decision layer


def test_score_record_routes_via_cost_decision_not_argmax():
    """A near-boundary probability vector must be able to disagree with
    argmax. Here the most-likely class is Legitimate (0.55), but the mass on
    Policy Abuser (0.40) is expensive enough under the loss-neutral cost
    matrix that the Bayes-optimal action is soft_friction, not approve.
    argmax-based inference would report 'approve' and be wrong about what
    the project's own decision layer would actually do."""
    proba = [0.02, 0.55, 0.40, 0.03]  # Fraud, Legit, PolicyAbuser, Wardrobing
    bundle = _bundle(model=_StubClassifier(proba), feature_cols=["avg_order_value_usd"])
    result = score_record({"avg_order_value_usd": 100.0}, bundle, posture="loss-neutral (1:1)")

    assert result["most_likely_class"] == "Legitimate"
    assert result["argmax_would_route_to"] == "approve"
    assert result["recommended_action"] == "soft_friction"
    assert result["recommended_action"] != result["argmax_would_route_to"]


def test_posture_choice_changes_the_routed_action():
    """The posture argument must actually reach the decision. For this
    probability vector, expected cost favors soft_friction under a
    loss-neutral posture (c_fn=120) but flips to hard_block once false
    negatives get expensive enough (c_fn=960, loss-averse) — verified by
    hand against build_cost_matrix's cell assignments, not asserted blind."""
    proba = [0.35, 0.30, 0.20, 0.15]  # Fraud, Legit, PolicyAbuser, Wardrobing
    bundle = _bundle(model=_StubClassifier(proba), feature_cols=["avg_order_value_usd"])

    neutral = score_record({"avg_order_value_usd": 100.0}, bundle, posture="loss-neutral (1:1)")
    averse = score_record({"avg_order_value_usd": 100.0}, bundle, posture="loss-averse (1:8)")

    assert neutral["recommended_action"] == "soft_friction"
    assert averse["recommended_action"] == "hard_block"
    # Same probabilities, different posture label surfaced faithfully:
    assert neutral["posture"] != averse["posture"]
    assert neutral["class_probabilities"] == averse["class_probabilities"]


def test_unknown_posture_is_rejected_explicitly():
    bundle = _bundle(model=_StubClassifier([0.25] * 4), feature_cols=["avg_order_value_usd"])
    with pytest.raises(ValueError, match="Unknown posture"):
        score_record({"avg_order_value_usd": 100.0}, bundle, posture="does-not-exist")


# Bug 4 — inference skipped feature engineering, so served records had a
# different schema than trained ones


def test_prepare_record_applies_the_same_feature_engineering_as_training():
    """A raw record with no precomputed derived columns must come out of
    prepare_record with them populated by src.features.
    add_transaction_level_features — the same function src/features.py uses
    to build the training set, called on a record that only carries the raw
    inputs a real caller would send."""
    bundle = _bundle(
        model=_StubClassifier([0.25] * 4),
        feature_cols=["refund_to_avg_order_ratio", "returns_per_order"],
    )
    record = {
        "avg_order_value_usd": 100.0,
        "refund_amount_requested_usd": 50.0,
        "total_orders_lifetime": 10,
        "total_returns_lifetime": 2,
    }
    frame = prepare_record(record, bundle)
    assert frame["refund_to_avg_order_ratio"].iloc[0] == pytest.approx(0.5)
    assert frame["returns_per_order"].iloc[0] == pytest.approx(0.2)


def test_prepare_record_never_reimplements_feature_engineering(monkeypatch):
    """If src.features.add_transaction_level_features is not called at all,
    prepare_record must fail to produce derived columns — proving the two
    code paths are the same function, not two implementations that happen to
    agree today."""
    import src.infer as infer_module

    monkeypatch.setattr(infer_module, "add_transaction_level_features", lambda df: df)
    bundle = _bundle(
        model=_StubClassifier([0.25] * 4),
        feature_cols=["refund_to_avg_order_ratio"],
    )
    record = {"avg_order_value_usd": 100.0, "refund_amount_requested_usd": 50.0}
    frame = prepare_record(record, bundle)
    # Column still exists (prepare_record backfills missing feature_cols with
    # None) but is NOT the computed ratio — proving the value came from the
    # backfill, not from feature engineering that silently ran twice.
    assert pd.isna(frame["refund_to_avg_order_ratio"].iloc[0])


# load_run — required-key validation in general


def test_load_run_accepts_a_complete_bundle(tmp_path):
    import joblib

    complete = _bundle(model=_StubClassifier([0.25] * 4), feature_cols=["x"])
    path = tmp_path / "good_bundle.joblib"
    joblib.dump(complete, path)
    loaded = load_run(tmp_path / "good_bundle")
    assert loaded["track"] == "unit-test"
    assert loaded["categories"] == {}
