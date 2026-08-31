"""
Tests for src/infer.py — regression coverage for the four bugs the Day 5
rewrite of this module was meant to close, plus the partial-record handling
added afterwards. Each test below is named for the bug it guards against, not
just the function it calls.

The four original bugs:
  1. a one-row inference frame built its own categorical schema, so LightGBM
     rejected it ("train and valid dataset categorical_feature do not match")
  2. the intervention map was keyed by class name and misspelled "Policy
     Abuser", silently routing that class to an unmapped string
  3. inference took argmax, bypassing the cost-calibrated decision layer
  4. inference skipped feature engineering, so served records had a different
     schema than trained ones

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
from src.infer import INTERVENTION, load_run, prepare_record, score_batch, score_record

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
    loading them. A bundle missing categories that loads anyway falls through
    to prepare_record silently skipping reapplication — bug 1 reappearing
    under a different trigger."""
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


# The friction axis — the one the Day 4 correction identifies as the only axis
# that moves. It was reachable from src.evaluate's sweeps but NOT from the
# serving path, which hardcoded build_cost_matrix's friction_cost default.


def test_friction_posture_changes_the_routed_action():
    """The friction argument must reach the decision, not just be echoed back.

    For this vector the record is mostly Legitimate (0.70) with real
    Policy-Abuser mass (0.20). Expected costs, verified by hand against
    build_cost_matrix's cells rather than asserted blind:

        approve       = .02(120) + .20(2) + .08(2)      = 2.96  (all postures)
        soft_friction = .02(48)  + .70(friction_cost)
                      = 1.03 / 1.66 / 6.56  at fc = 0.1 / 1.0 / 8.0

    so friction stays cheaper than approving until friction gets expensive,
    and the action flips soft_friction -> approve at the approve-first end.
    """
    proba = [0.02, 0.70, 0.20, 0.08]  # Fraud, Legit, PolicyAbuser, Wardrobing
    bundle = _bundle(model=_StubClassifier(proba), feature_cols=["avg_order_value_usd"])
    record = {"avg_order_value_usd": 100.0}

    recovery = score_record(record, bundle, friction_posture="recovery-first (1:20)")
    approve = score_record(record, bundle, friction_posture="approve-first (4:1)")

    assert recovery["recommended_action"] == "soft_friction"
    assert approve["recommended_action"] == "approve"
    # Same probabilities, different friction label surfaced faithfully.
    assert recovery["friction_posture"] != approve["friction_posture"]
    assert recovery["class_probabilities"] == approve["class_probabilities"]


def test_friction_default_reproduces_the_historical_operating_point():
    """`balanced (1:2)` is friction_cost=1.0 — the value build_cost_matrix has
    always defaulted to, which is what the decision layer used before the
    friction axis was exposed at all.

    This test is what makes the new flag safe to add: every previously
    committed artifact (runs/segment_fpr_*.json in particular) was generated at
    that operating point, so scoring with no friction argument must still land
    there. Pinned against the explicitly-passed posture rather than against a
    recorded string, so it fails if the default is ever repointed.
    """
    proba = [0.02, 0.70, 0.20, 0.08]
    bundle = _bundle(model=_StubClassifier(proba), feature_cols=["avg_order_value_usd"])
    record = {"avg_order_value_usd": 100.0}

    implicit = score_record(record, bundle)
    explicit = score_record(record, bundle, friction_posture="balanced (1:2)")

    assert implicit["friction_posture"] == "balanced (1:2)"
    assert implicit["recommended_action"] == explicit["recommended_action"]
    assert implicit["recommended_action"] == "soft_friction"


def test_unknown_friction_posture_is_rejected_explicitly():
    """Same failure mode as an unknown --posture: a typo must raise, not fall
    back to the default and silently score under a policy nobody chose."""
    bundle = _bundle(model=_StubClassifier([0.25] * 4), feature_cols=["avg_order_value_usd"])
    with pytest.raises(ValueError, match="Unknown friction posture"):
        score_record({"avg_order_value_usd": 100.0}, bundle, friction_posture="does-not-exist")


def test_batch_carries_both_posture_columns():
    """A batch CSV must record which policy produced it. Both axes, because
    reading the output later without knowing the friction posture makes the
    recommendation uninterpretable."""
    proba = [0.02, 0.70, 0.20, 0.08]
    bundle = _bundle(model=_StubClassifier(proba), feature_cols=["avg_order_value_usd"])
    records = pd.DataFrame({"avg_order_value_usd": [100.0, 250.0]})

    out = score_batch(records, bundle, friction_posture="approve-first (4:1)")

    assert list(out["posture"]) == ["loss-neutral (1:1)"] * 2
    assert list(out["friction_posture"]) == ["approve-first (4:1)"] * 2
    assert list(out["recommended_action"]) == ["approve"] * 2


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
    # `avg_order_value_usd` is in feature_cols purely so the record still
    # supplies *some* recognised feature with engineering stubbed out —
    # otherwise prepare_frame's "supplies none of the features" guard fires
    # first and this test stops measuring what it is named for.
    bundle = _bundle(
        model=_StubClassifier([0.25] * 4),
        feature_cols=["avg_order_value_usd", "refund_to_avg_order_ratio"],
    )
    record = {"avg_order_value_usd": 100.0, "refund_amount_requested_usd": 50.0}
    frame = prepare_record(record, bundle)
    # Column still exists (prepare_record backfills missing feature_cols with
    # None) but is NOT the computed ratio — proving the value came from the
    # backfill, not from feature engineering that silently ran twice.
    assert pd.isna(frame["refund_to_avg_order_ratio"].iloc[0])


# Partial and malformed records — a caller will not always send all 35 columns


def test_partial_record_scores_instead_of_raising_a_dtype_traceback():
    """A record missing one trained feature must score. The regression: missing
    columns were backfilled with None, which makes the column `object` dtype,
    and LightGBM rejects that with a bare 'pandas dtypes must be int, float or
    bool' traceback — an internal error surfaced to a caller who simply didn't
    send an optional field."""
    bundle = _bundle(
        model=_StubClassifier([0.01, 0.95, 0.02, 0.02]),
        feature_cols=["avg_order_value_usd", "review_left_after_return"],
    )
    frame = prepare_record({"avg_order_value_usd": 100.0}, bundle)

    # The absent column must be float-NaN, not object-None.
    assert frame["review_left_after_return"].isna().all()
    assert frame["review_left_after_return"].dtype != object

    result = score_record({"avg_order_value_usd": 100.0}, bundle)
    assert result["recommended_action"] in ACTIONS


def test_partial_record_names_the_features_it_did_not_have():
    """Scoring on a partial record is allowed but never silent — the omission
    is reported so a recommendation is not read as if it were made on a
    complete record."""
    bundle = _bundle(
        model=_StubClassifier([0.25] * 4),
        feature_cols=["avg_order_value_usd", "review_left_after_return", "age"],
    )
    result = score_record({"avg_order_value_usd": 100.0}, bundle)
    assert sorted(result["features_missing"]) == ["age", "review_left_after_return"]


def test_complete_record_reports_nothing_missing():
    bundle = _bundle(
        model=_StubClassifier([0.25] * 4),
        feature_cols=["avg_order_value_usd"],
    )
    result = score_record({"avg_order_value_usd": 100.0}, bundle)
    assert result["features_missing"] == []


def test_record_with_no_recognisable_features_is_rejected_clearly():
    """An empty or wrong-schema payload is a caller error, not a partial
    record. It must raise a message naming the expected schema rather than
    scoring 35 NaNs and returning a confident-looking probability vector."""
    bundle = _bundle(
        model=_StubClassifier([0.25] * 4),
        feature_cols=["avg_order_value_usd", "age"],
    )
    with pytest.raises(ValueError, match="supplies none of the"):
        score_record({}, bundle)


def test_batch_with_a_missing_column_scores_and_counts_it():
    bundle = _bundle(
        model=_StubClassifier([0.25] * 4),
        feature_cols=["avg_order_value_usd", "review_left_after_return"],
    )
    records = pd.DataFrame({"avg_order_value_usd": [100.0, 200.0, 300.0]})
    out = score_batch(records, bundle)

    assert len(out) == 3
    assert (out["n_features_missing"] == 1).all()
    assert out.attrs["missing_features"] == ["review_left_after_return"]


# load_run — required-key validation in general


def test_load_run_accepts_a_complete_bundle(tmp_path):
    import joblib

    complete = _bundle(model=_StubClassifier([0.25] * 4), feature_cols=["x"])
    path = tmp_path / "good_bundle.joblib"
    joblib.dump(complete, path)
    loaded = load_run(tmp_path / "good_bundle")
    assert loaded["track"] == "unit-test"
    assert loaded["categories"] == {}
