"""
Tests for the four-rule, zero-training baseline in src/model.py.

This baseline produces the single most important number in the project — the
0.9425 accuracy / 0.9188 macro-F1 that the README, docs/EVALUATION.md,
docs/PITCH.md and docs/LEAKAGE_FINDING.md all lead with as evidence that the
dataset is degenerate. Until these tests existed, nothing guarded it: an edited
threshold in `apply_hand_written_rule` left the suite green while silently
falsifying every document in the repo.

The thresholds are pinned branch by branch rather than by asserting a score,
because the Kaggle CSV is gitignored and a fresh clone must be able to run this
file before ever downloading the data. `test_rule_thresholds_match_the_published
_baseline` is the tripwire: it fails if any of the four constants read off
docs/LEAKAGE_FINDING.md's per-class range tables ever move.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.model import apply_hand_written_rule, rule_baseline_metrics

# The four thresholds, exactly as docs/LEAKAGE_FINDING.md reports them. If a
# change to the rule is deliberate, these move *and* so does every quoted
# number in README.md, docs/EVALUATION.md, docs/PITCH.md and
# docs/LEAKAGE_FINDING.md — that is the point of restating them here.
WISHLIST_HRS_CUT = 5.0
DAYS_TO_RETURN_CUT = 25
RETURN_RATE_PCT_CUT = 15


def _row(wishlist_hrs: float, days_to_return: float, return_rate_pct: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "wishlist_to_cart_time_hrs": wishlist_hrs,
                "days_to_return": days_to_return,
                "return_rate_pct": return_rate_pct,
            }
        ]
    )


def _classify(wishlist_hrs: float, days_to_return: float, return_rate_pct: float) -> str:
    return apply_hand_written_rule(_row(wishlist_hrs, days_to_return, return_rate_pct)).iloc[0]


# Each of the four branches reaches its own class


def test_fast_wishlist_and_slow_return_is_wardrobing():
    assert _classify(1.0, 30, 5.0) == "Wardrobing"


def test_fast_wishlist_and_fast_return_is_fraudulent():
    assert _classify(1.0, 3, 5.0) == "Fraudulent Return"


def test_slow_wishlist_and_high_return_rate_is_policy_abuser():
    assert _classify(50.0, 30, 40.0) == "Policy Abuser"


def test_slow_wishlist_and_low_return_rate_is_legitimate():
    assert _classify(50.0, 30, 2.0) == "Legitimate"


def test_all_four_classes_are_reachable():
    """A rule that can only emit three classes cannot score 0.9188 macro-F1 on
    a 4-class target — macro-F1 would cap near 0.75. Guards against a branch
    being collapsed or a class name being misspelled."""
    emitted = {
        _classify(1.0, 30, 5.0),
        _classify(1.0, 3, 5.0),
        _classify(50.0, 30, 40.0),
        _classify(50.0, 30, 2.0),
    }
    assert emitted == {"Wardrobing", "Fraudulent Return", "Policy Abuser", "Legitimate"}


# The exact thresholds and their inclusivity


@pytest.mark.parametrize(
    "wishlist_hrs, expected_branch_is_fast",
    [
        (WISHLIST_HRS_CUT - 0.01, True),
        (WISHLIST_HRS_CUT, True),  # <= is inclusive at the cut
        (WISHLIST_HRS_CUT + 0.01, False),
    ],
)
def test_wishlist_cut_is_inclusive_at_the_boundary(wishlist_hrs, expected_branch_is_fast):
    fast_branch_classes = {"Wardrobing", "Fraudulent Return"}
    result = _classify(wishlist_hrs, 30, 2.0)
    assert (result in fast_branch_classes) is expected_branch_is_fast


@pytest.mark.parametrize(
    "days, expected",
    [
        (DAYS_TO_RETURN_CUT - 1, "Fraudulent Return"),
        (DAYS_TO_RETURN_CUT, "Wardrobing"),  # >= is inclusive at the cut
        (DAYS_TO_RETURN_CUT + 1, "Wardrobing"),
    ],
)
def test_days_to_return_cut_is_inclusive_at_the_boundary(days, expected):
    assert _classify(1.0, days, 5.0) == expected


@pytest.mark.parametrize(
    "rate, expected",
    [
        (RETURN_RATE_PCT_CUT - 0.01, "Legitimate"),
        (RETURN_RATE_PCT_CUT, "Legitimate"),  # > is strict at the cut
        (RETURN_RATE_PCT_CUT + 0.01, "Policy Abuser"),
    ],
)
def test_return_rate_cut_is_strict_at_the_boundary(rate, expected):
    assert _classify(50.0, 30, rate) == expected


def test_rule_thresholds_match_the_published_baseline():
    """Tripwire for the documented rule. If this fails, the four constants in
    `apply_hand_written_rule` have moved and the 0.9425 / 0.9188 figures quoted
    across README.md and docs/ are no longer what the code produces —
    regenerate them with `python -m src.model` before editing any document."""
    # Probe each threshold from both sides; a moved cut flips at least one.
    assert _classify(WISHLIST_HRS_CUT, 30, 2.0) == "Wardrobing"
    assert _classify(WISHLIST_HRS_CUT + 0.01, 30, 2.0) == "Legitimate"
    assert _classify(1.0, DAYS_TO_RETURN_CUT, 2.0) == "Wardrobing"
    assert _classify(1.0, DAYS_TO_RETURN_CUT - 1, 2.0) == "Fraudulent Return"
    assert _classify(50.0, 30, RETURN_RATE_PCT_CUT) == "Legitimate"
    assert _classify(50.0, 30, RETURN_RATE_PCT_CUT + 0.01) == "Policy Abuser"


# rule_baseline_metrics — the function that writes runs/baseline_rule.json


@pytest.fixture
def separable_csv(tmp_path):
    """A tiny frame the rule classifies perfectly, so the metrics arithmetic is
    checked against a known answer rather than against whatever the real CSV
    happens to produce (which a fresh clone cannot read — it is gitignored)."""
    rows = []
    for _ in range(5):
        rows.append({"wishlist_to_cart_time_hrs": 1.0, "days_to_return": 30,
                     "return_rate_pct": 5.0, "abuse_type": "Wardrobing"})
        rows.append({"wishlist_to_cart_time_hrs": 1.0, "days_to_return": 3,
                     "return_rate_pct": 5.0, "abuse_type": "Fraudulent Return"})
        rows.append({"wishlist_to_cart_time_hrs": 50.0, "days_to_return": 30,
                     "return_rate_pct": 40.0, "abuse_type": "Policy Abuser"})
        rows.append({"wishlist_to_cart_time_hrs": 50.0, "days_to_return": 30,
                     "return_rate_pct": 2.0, "abuse_type": "Legitimate"})
    path = tmp_path / "returns.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_rule_baseline_metrics_scores_a_separable_frame_perfectly(separable_csv):
    result = rule_baseline_metrics(separable_csv)
    assert result["n_rows"] == 20
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["macro_f1"] == pytest.approx(1.0)


def test_rule_baseline_metrics_reports_a_full_square_confusion_matrix(separable_csv):
    """The published finding cites per-class behaviour, so the matrix must be
    labelled and 4x4 — not silently collapsed to the classes that happen to be
    predicted."""
    result = rule_baseline_metrics(separable_csv)
    labels = result["confusion_matrix"]["labels"]
    matrix = result["confusion_matrix"]["matrix"]

    assert sorted(labels) == ["Fraudulent Return", "Legitimate", "Policy Abuser", "Wardrobing"]
    assert len(matrix) == len(labels)
    assert all(len(row) == len(labels) for row in matrix)
    assert sum(sum(row) for row in matrix) == result["n_rows"]


def test_rule_baseline_metrics_penalises_a_frame_the_rule_gets_wrong(separable_csv):
    """Sanity check that the metric is measuring agreement rather than always
    returning 1.0: relabel every row and the score must collapse."""
    df = pd.read_csv(separable_csv)
    df["abuse_type"] = "Legitimate"
    df.to_csv(separable_csv, index=False)

    result = rule_baseline_metrics(separable_csv)
    assert result["accuracy"] == pytest.approx(0.25)
    assert result["macro_f1"] < 0.5


def test_confusion_matrix_keeps_predictions_for_classes_absent_from_the_truth(separable_csv):
    """Regression: labels were taken from `y_true.unique()` alone, so any class
    the rule predicted but that never appeared as a true label was dropped from
    the matrix entirely and the cells stopped summing to n_rows. Here every true
    label is Legitimate while the rule still emits all four classes."""
    df = pd.read_csv(separable_csv)
    df["abuse_type"] = "Legitimate"
    df.to_csv(separable_csv, index=False)

    result = rule_baseline_metrics(separable_csv)
    labels = result["confusion_matrix"]["labels"]
    matrix = result["confusion_matrix"]["matrix"]

    assert sorted(labels) == ["Fraudulent Return", "Legitimate", "Policy Abuser", "Wardrobing"]
    assert sum(sum(row) for row in matrix) == result["n_rows"]
