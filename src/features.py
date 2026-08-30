"""
Feature engineering (ARCHITECTURE.md §4).

Written against the CONFIRMED schema (60,000 x 35), after the Day 1 gate ran
against the real file. See docs/DATA_NOTES.md for the findings this module
is built on. Three of them shape the code here:

1. `abuse_label` is a 1:1 integer encoding of the target `abuse_type`
   (macro-F1 = 1.000 on its own). It is dropped, not modelled. See DROP_COLS.

2. Median rows-per-customer is 1.0 (56,061 of 58,006 customers appear once).
   The §4.2 per-customer aggregates cannot be *derived* — there is no history
   to aggregate over. What the dataset does ship is the same behavioural
   signal pre-computed per row: `total_orders_lifetime`,
   `total_returns_lifetime`, `return_rate_pct`, `previous_dispute_count`,
   `multiple_accounts_flag`, `account_age_days`. So the signal §4.2 wanted is
   present, but as a given, not as our engineering. That distinction is
   stated plainly in the writeup — claiming credit for engineering it would
   be the dishonest version.

3. `days_to_return` already exists in the raw data and equals
   (return_date - order_date) exactly for 100% of rows. Recomputing it would
   add a perfectly collinear duplicate column, so we don't.

Run as:
    python -m src.features

Reads data/raw/returns.csv, writes:
    data/processed/train.parquet
    data/processed/test.parquet
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_gate import RAW_PATH

PROCESSED_DIR = Path("data/processed")

TARGET_COL = "abuse_type"

# Never fed to the model.
#   abuse_label  — 1:1 encoding of the target (Day 1 leakage finding)
#   order_id     — unique per row, pure identifier
#   customer_id  — 58,006 near-unique values, no repeat history to exploit
#   order_date / return_date — used to build the temporal split and
#                  order_month/dow features, then dropped as raw strings
DROP_COLS = ["abuse_label", "order_id", "customer_id", "order_date", "return_date"]

# The split is ordered by RETURN date, not order date. The label describes the
# return, so return_date is when it materialises and when a deployed scorer
# would actually see the record. Splitting on order_date instead leaves 1,399
# training rows whose return (and therefore label) lands inside the test
# window — a subtle temporal leak for no benefit. Class balance is stable
# across the return_date split (see docs/DATA_NOTES.md).
SPLIT_DATE_COL = "return_date"
TEST_FRACTION = 0.2


# Dual-track feature sets, decided Day 2 after the leakage finding below.
#
# docs/LEAKAGE_FINDING.md establishes that this dataset's classes are
# box-separated by the generator: four hand-written if/else rules score 0.919
# macro-F1 with no training at all. That has a consequence the original
# finding missed, and which src/ablation.py now measures directly — on the
# full feature set 99.9% of test predictions land above p=0.99, so the
# `C_fp : C_fn` sweep that ARCHITECTURE.md §6.2 names as the centerpiece
# deliverable is mathematically flat. Every merchant posture produces the
# same thresholds and the same precision/recall, because there are ~12 rows
# in 12,000 near any boundary.
#
# The build therefore runs two tracks, and the distinction between them is
# stated in the writeup rather than blurred:
#
#   FULL      The honest model. Every legitimate feature the dataset ships.
#             This is what gets reported as "the model," together with the
#             leakage finding and the flat cost sweep. The flat sweep is not
#             a failure to be hidden — it is the positive evidence that the
#             dataset is degenerate.
#
#   TESTBED   Rung G of the ablation ladder. NOT a model, and never presented
#             as one. It is a deliberately handicapped variant constructed so
#             the decision layer has a non-degenerate region to operate on
#             (~18% of rows threshold-sensitive vs. 0.1% on FULL), letting the
#             §6.2 cost-calibration method be demonstrated and audited.
#
# Why TESTBED cannot be the headline model: the ablation ladder degrades
# smoothly from 0.999 to 0.856 with no natural cut point, so any choice of
# rung is arbitrary. "We dropped features until the task got hard" is not a
# modelling result. Rung G is chosen for one stated reason only — it is the
# first rung with enough boundary mass for a threshold sweep to move
# decisions at all.
# A note on derived proxies, found the hard way. The first version of this
# list excluded only the raw artifact columns, and the ablation ladder came
# out almost flat (0.999 -> 0.978 and no further movement). The reason is
# that this module's own engineered features silently reconstruct the columns
# being dropped: `returns_per_order` IS `return_rate_pct` up to a factor of
# 100, and `orders_kept_lifetime` recovers `total_returns_lifetime` given
# `total_orders_lifetime`. An ablation that removes a feature but keeps an
# algebraic restatement of it measures nothing. Every excluded raw column
# therefore drags its derived proxies out with it.
TESTBED_EXCLUDED = [
    # Per-class ranges are near-disjoint; see the range tables in
    # docs/LEAKAGE_FINDING.md. These four carry most of the artifact.
    "wishlist_to_cart_time_hrs",
    "days_to_return",
    "return_rate_pct",
    "total_returns_lifetime",
    # Weaker but same character.
    "customer_support_contacts",
    "previous_dispute_count",
    # Derived proxies for the above, engineered in this module.
    "returns_per_order",  # == return_rate_pct / 100
    "orders_kept_lifetime",  # total_orders_lifetime - total_returns_lifetime
]

FEATURE_SETS: dict[str, list[str]] = {
    "full": [],
    "testbed": TESTBED_EXCLUDED,
}


def feature_columns(df: pd.DataFrame, track: str = "full") -> list[str]:
    """Model feature columns for a track. Raises on an unknown track rather
    than silently falling back to `full`, which would misreport the run."""
    if track not in FEATURE_SETS:
        raise ValueError(f"Unknown track {track!r}. Expected one of {sorted(FEATURE_SETS)}.")
    excluded = set(FEATURE_SETS[track]) | set(DROP_COLS) | {TARGET_COL}
    return [c for c in df.columns if c not in excluded]


def add_transaction_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """§4.1 — derived from a single return record, no cross-row history needed."""
    df = df.copy()

    # Refund as a share of the customer's typical order value. A refund far
    # above their normal basket is a different signal than one at par with it,
    # and neither raw column carries that on its own.
    if {"refund_amount_requested_usd", "avg_order_value_usd"}.issubset(df.columns):
        denom = df["avg_order_value_usd"].replace(0, np.nan)
        ratio = df["refund_amount_requested_usd"] / denom
        df["refund_to_avg_order_ratio"] = ratio.replace([np.inf, -np.inf], np.nan)

    # Absolute count of lifetime returns says little without the order count
    # behind it; 5 returns on 6 orders and 5 on 500 are opposite situations.
    # (`return_rate_pct` ships in the data and encodes the same idea; this is
    # the un-rounded form plus the raw gap.)
    if {"total_returns_lifetime", "total_orders_lifetime"}.issubset(df.columns):
        denom = df["total_orders_lifetime"].replace(0, np.nan)
        df["returns_per_order"] = (df["total_returns_lifetime"] / denom).replace(
            [np.inf, -np.inf], np.nan
        )
        df["orders_kept_lifetime"] = df["total_orders_lifetime"] - df["total_returns_lifetime"]

    # Tenure-normalised ordering rate — separates a heavy buyer from a new
    # account that ordered heavily in a short window.
    if {"total_orders_lifetime", "account_age_days"}.issubset(df.columns):
        denom = df["account_age_days"].replace(0, np.nan)
        df["orders_per_account_day"] = (df["total_orders_lifetime"] / denom).replace(
            [np.inf, -np.inf], np.nan
        )

    # Seasonality: post-holiday return waves are a real merchant pattern and
    # month/day-of-week are the cheap way to let the model see them.
    if "return_date" in df.columns:
        rd = pd.to_datetime(df["return_date"], errors="coerce")
        df["return_month"] = rd.dt.month
        df["return_dayofweek"] = rd.dt.dayofweek

    return df


def build_and_split(raw_path: Path = RAW_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the raw CSV, engineer §4.1 features, and split temporally on
    SPLIT_DATE_COL (return_date, not order_date — see the module-level
    comment on why).

    Args: raw_path to the Kaggle CSV.
    Returns: (train, test) frames with DROP_COLS already removed and
    `abuse_type` still present as the target. The non-obvious part: the sort
    uses `kind="mergesort"` specifically because it's stable — ties on
    return_date (same-day returns) keep their original row order instead of
    being shuffled across the train/test boundary nondeterministically.
    """
    df = pd.read_csv(raw_path)
    df = add_transaction_level_features(df)

    split_dates = pd.to_datetime(df[SPLIT_DATE_COL], errors="coerce")
    df = df.assign(_split_ts=split_dates).sort_values("_split_ts", kind="mergesort")

    cut_idx = int(len(df) * (1 - TEST_FRACTION))
    train = df.iloc[:cut_idx].drop(columns=["_split_ts"])
    test = df.iloc[cut_idx:].drop(columns=["_split_ts"])

    train = train.drop(columns=[c for c in DROP_COLS if c in train.columns])
    test = test.drop(columns=[c for c in DROP_COLS if c in test.columns])

    return train, test


def run(raw_path: Path = RAW_PATH) -> None:
    """CLI entrypoint: build_and_split, then write both frames to
    data/processed/{train,test}.parquet. src/model.py reads from these
    files rather than recomputing the split; src/ablation.py calls
    build_and_split directly instead, since its ladder needs the full
    unfiltered column set fresh each time."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train, test = build_and_split(raw_path)
    train.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    test.to_parquet(PROCESSED_DIR / "test.parquet", index=False)
    print(f"Wrote {len(train):,} train rows and {len(test):,} test rows to {PROCESSED_DIR}")
    print(f"Feature columns ({train.shape[1] - 1}): {[c for c in train.columns if c != TARGET_COL]}")


if __name__ == "__main__":
    run()
