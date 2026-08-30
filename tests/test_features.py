"""
Tests for src/features.py — synthetic in-memory fixtures matching the
confirmed real schema, so they run without the Kaggle CSV (which is
gitignored and not part of the repo).
"""

import numpy as np
import pandas as pd
import pytest

from src.features import DROP_COLS, add_transaction_level_features, build_and_split


@pytest.fixture
def synthetic_df():
    """Mirrors the real column names/dtypes for the fields features.py touches."""
    return pd.DataFrame(
        {
            "order_id": ["ORD1", "ORD2", "ORD3", "ORD4"],
            "customer_id": ["CUST1", "CUST2", "CUST3", "CUST4"],
            "account_age_days": [100, 200, 0, 50],
            "order_date": ["2022-01-01", "2022-02-01", "2022-03-01", "2022-04-01"],
            "return_date": ["2022-01-15", "2022-02-10", "2022-03-05", "2022-04-20"],
            "days_to_return": [14, 9, 4, 19],
            "avg_order_value_usd": [100.0, 200.0, 0.0, 50.0],
            "refund_amount_requested_usd": [50.0, 100.0, 25.0, 50.0],
            "total_orders_lifetime": [10, 20, 5, 0],
            "total_returns_lifetime": [2, 10, 1, 0],
            "abuse_label": [0, 1, 2, 3],
            "abuse_type": ["Legitimate", "Policy Abuser", "Fraudulent Return", "Wardrobing"],
        }
    )


def test_refund_ratio_computed(synthetic_df):
    out = add_transaction_level_features(synthetic_df)
    assert out["refund_to_avg_order_ratio"].iloc[0] == pytest.approx(0.5)


def test_refund_ratio_handles_zero_denominator(synthetic_df):
    """avg_order_value_usd == 0 must yield NaN, never inf — an inf silently
    becomes the model's most extreme split point."""
    out = add_transaction_level_features(synthetic_df)
    assert pd.isna(out["refund_to_avg_order_ratio"].iloc[2])
    assert not np.isinf(out["refund_to_avg_order_ratio"].dropna()).any()


def test_returns_per_order_and_kept_orders(synthetic_df):
    out = add_transaction_level_features(synthetic_df)
    assert out["returns_per_order"].iloc[0] == pytest.approx(0.2)
    assert out["orders_kept_lifetime"].iloc[1] == 10
    # zero lifetime orders -> NaN, not a divide-by-zero
    assert pd.isna(out["returns_per_order"].iloc[3])


def test_orders_per_account_day_handles_zero_age(synthetic_df):
    out = add_transaction_level_features(synthetic_df)
    assert pd.isna(out["orders_per_account_day"].iloc[2])


def test_return_seasonality_features(synthetic_df):
    out = add_transaction_level_features(synthetic_df)
    assert out["return_month"].iloc[0] == 1
    assert out["return_dayofweek"].iloc[0] == pd.Timestamp("2022-01-15").dayofweek


def test_days_to_return_not_recomputed(synthetic_df):
    """The raw column is exact for 100% of real rows; recomputing it would add
    a perfectly collinear duplicate."""
    out = add_transaction_level_features(synthetic_df)
    assert list(out["days_to_return"]) == [14, 9, 4, 19]


def test_leaking_and_id_columns_are_dropped(tmp_path, synthetic_df):
    """abuse_label is a 1:1 encoding of the target (docs/LEAKAGE_FINDING.md).
    It must never survive into a modelling frame."""
    csv = tmp_path / "returns.csv"
    synthetic_df.to_csv(csv, index=False)
    train, test = build_and_split(csv)
    for frame in (train, test):
        for col in DROP_COLS:
            assert col not in frame.columns
        assert "abuse_type" in frame.columns


def test_split_is_temporal_on_return_date(tmp_path, synthetic_df):
    """Every training return must precede every test return — otherwise the
    split leaks future outcomes backwards."""
    csv = tmp_path / "returns.csv"
    synthetic_df.to_csv(csv, index=False)
    train, test = build_and_split(csv)
    assert len(train) + len(test) == len(synthetic_df)
    assert len(test) > 0
    # return_date is dropped from the output, so re-derive order via the
    # month feature, which is monotonic within this fixture.
    assert train["return_month"].max() <= test["return_month"].min()


@pytest.fixture
def repeat_customer_df():
    """Two rows for the same customer, with total_orders_lifetime /
    total_returns_lifetime that go *down* between the earlier and later
    return — this is not a fabricated edge case. Inspecting the real
    Kaggle CSV's 1,945 repeat customers shows exactly this: these "lifetime"
    columns are independent per-row snapshots from the generator, not a
    running ledger (e.g. one real customer's total_orders_lifetime goes
    78 -> 57 -> 12 -> 14 across their four rows, non-monotonic in return_date
    order). So there is no genuine cross-row customer history in this
    dataset for a trailing aggregate to leak from in the first place."""
    return pd.DataFrame(
        {
            "order_id": ["ORD1", "ORD2"],
            "customer_id": ["CUST_REPEAT", "CUST_REPEAT"],
            "account_age_days": [300, 300],
            "order_date": ["2022-01-01", "2022-06-01"],
            "return_date": ["2022-01-10", "2022-06-15"],
            "days_to_return": [9, 14],
            "avg_order_value_usd": [100.0, 100.0],
            "refund_amount_requested_usd": [50.0, 50.0],
            "total_orders_lifetime": [78, 12],  # later row has FEWER lifetime orders
            "total_returns_lifetime": [10, 1],  # and fewer lifetime returns
            "abuse_label": [0, 0],
            "abuse_type": ["Legitimate", "Legitimate"],
        }
    )


def test_no_cross_row_customer_aggregation_exists(repeat_customer_df):
    """The temporal-leakage risk this test class exists to catch is a
    trailing aggregate for row n computed from other rows of the same
    customer — which would leak if it ever included row n itself.
    src/features.py has no such computation (no groupby/rolling/expanding/
    shift over customer_id anywhere in add_transaction_level_features); the
    "lifetime" columns are used exactly as given, per row. This test pins
    that down: each row's derived features must depend only on that row's
    own values, never on its customer's other row(s) — proven here by two
    rows of the same customer whose per-row values are NOT a monotonic
    running total (matching what the real data actually looks like)."""
    out = add_transaction_level_features(repeat_customer_df)

    # Row 0 (earlier return) has the larger raw lifetime values; if any
    # cross-row aggregation were happening, row 1's derived ratio would be
    # pulled toward row 0's. It isn't -- each row's ratio matches only its
    # own total_orders_lifetime / total_returns_lifetime.
    assert out["returns_per_order"].iloc[0] == pytest.approx(10 / 78)
    assert out["returns_per_order"].iloc[1] == pytest.approx(1 / 12)
    assert out["orders_kept_lifetime"].iloc[0] == 78 - 10
    assert out["orders_kept_lifetime"].iloc[1] == 12 - 1

    # The raw lifetime columns themselves must pass through untouched --
    # not smoothed, not replaced by a running max/sum across the customer's
    # rows.
    assert list(out["total_orders_lifetime"]) == [78, 12]
    assert list(out["total_returns_lifetime"]) == [10, 1]
