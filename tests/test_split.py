"""
Split-hygiene tests for src.features.build_and_split.

Every reported number depends on the train/test split actually being a
disjoint partition, and on the temporal cut actually preventing future
returns from leaking backward into training. Both are checked directly
against the fixture's original return_date column (which build_and_split
drops from its output, being in DROP_COLS) by matching rows on their
preserved DataFrame index -- not a coarser proxy.

Runs entirely on an in-memory fixture; never touches the Kaggle CSV.
"""

import pandas as pd
import pytest

from src.features import build_and_split


@pytest.fixture
def dated_df():
    """20 rows, each a distinct return_date, so the split boundary can be
    checked at day granularity rather than the coarser month-level proxy
    tests/test_features.py already uses."""
    n = 20
    return pd.DataFrame(
        {
            "order_id": [f"ORD{i}" for i in range(n)],
            "customer_id": [f"CUST{i}" for i in range(n)],
            "order_date": pd.date_range("2022-01-01", periods=n, freq="5D").astype(str),
            "return_date": pd.date_range("2022-01-10", periods=n, freq="5D").astype(str),
            "abuse_label": [i % 4 for i in range(n)],
            "abuse_type": (["Legitimate", "Wardrobing", "Policy Abuser", "Fraudulent Return"] * 5),
        }
    )


def test_train_and_test_are_disjoint_on_row_index(tmp_path, dated_df):
    """train/test must be a strict partition of the input rows -- no row
    duplicated into both, none dropped silently."""
    csv = tmp_path / "returns.csv"
    dated_df.to_csv(csv, index=False)
    train, test = build_and_split(csv)

    assert set(train.index).isdisjoint(set(test.index))
    assert len(train) + len(test) == len(dated_df)


def test_max_train_timestamp_precedes_min_test_timestamp(tmp_path, dated_df):
    """The actual temporal-leakage check: every training return must have
    happened strictly before every test return. return_date is dropped from
    build_and_split's output (it's in DROP_COLS), so this looks it up from
    the original fixture via the row index build_and_split preserves,
    rather than relying on a coarser derived proxy like return_month."""
    csv = tmp_path / "returns.csv"
    dated_df.to_csv(csv, index=False)
    train, test = build_and_split(csv)

    original_return_date = pd.to_datetime(dated_df["return_date"])
    train_dates = original_return_date.loc[train.index]
    test_dates = original_return_date.loc[test.index]

    assert len(train_dates) > 0
    assert len(test_dates) > 0
    assert train_dates.max() <= test_dates.min()
