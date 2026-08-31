"""
Day 1 data gate (ARCHITECTURE.md §9.1).

Answers three questions from the raw CSV before any modeling work begins,
and writes the findings + their consequences to docs/DATA_NOTES.md:

    1. Is there a usable repeat-customer identifier?
       (median rows-per-customer -> viability of §4.2 behavioral aggregates)
    2. Is there a usable timestamp?
       (temporal vs. random split, §2)
    3. Leakage sweep: mutual information per feature against the label,
       plus a single-feature decision tree per feature. Any single feature
       reaching ~0.6 macro-F1 alone is treated as a generation artifact,
       not signal.

CORRECTION TO ARCHITECTURE.md §9.1 (made Day 1, after the first real run):
The architecture doc specifies a *depth-1* tree per feature. That test is
invalid for a 4-class target and produces false negatives. A depth-1 tree
makes one split and has two leaves, so it can name at most 2 of the 4
classes — its macro-F1 is capped near ~0.45 no matter how perfectly the
feature encodes the label. Run as specified, this gate PASSED `abuse_label`,
a column that is a 1:1 integer encoding of `abuse_type` and scores macro-F1
= 1.000 on its own at depth 3.

The sweep therefore fits at depth `ceil(log2(n_classes))` at minimum — here
`n_classes - 1` — so the tree can address every class, and reports the
depth-1 number alongside it to make the correction auditable. This is
recorded in docs/DATA_NOTES.md rather than silently patched.

Run as:
    python -m src.data_gate

Expects the raw Kaggle CSV at data/raw/returns.csv (see README Quickstart).
This script deliberately does not hardcode column names beyond the target,
since the exact schema is confirmed on first run against the real file —
it searches for plausible customer-id / timestamp columns and reports what
it found so a human can confirm or correct the guess in DATA_NOTES.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42

RAW_PATH = Path("data/raw/returns.csv")
DATA_NOTES_PATH = Path("docs/DATA_NOTES.md")

# Standard remedies, so every stage names the same command for the same missing
# prerequisite instead of each phrasing it slightly differently.
NEEDS_RAW_CSV = (
    "Place the Kaggle CSV there (see README Quickstart) and re-run.\n"
    "    Dataset: https://www.kaggle.com/datasets/sarveshchhetri/"
    "e-commerce-return-abuse-detection-dataset"
)
NEEDS_FEATURES = "Build the processed splits first:\n    python -m src.features"
NEEDS_MODEL = "Train the tracks first:\n    python -m src.model"


def require_artifacts(paths, remedy: str) -> None:
    """Exit with a remedy instead of a traceback when a pipeline stage's inputs
    are missing.

    Every stage after `src.data_gate` depends on an artifact an earlier stage
    writes, and most of those artifacts are gitignored by design (the raw CSV,
    the parquet splits, the model bundles). So "the file isn't there yet" is the
    single most likely thing to happen to someone running this repo for the
    first time, and it was producing a bare FileNotFoundError from library depth
    in seven of eight entry points — the reviewer's first impression of the
    project being a stack trace.

    Args: `paths` — the files the stage actually reads, checked together so the
        message lists everything missing rather than failing on the first one;
        `remedy` — the command that produces them (see the NEEDS_* constants).
    Raises SystemExit(1) after printing to stderr. Never returns non-None.
    """
    missing = [Path(p) for p in paths if not Path(p).exists()]
    if not missing:
        return
    listed = "\n".join(f"  - {p}" for p in missing)
    print(
        f"ERROR: {len(missing)} required input(s) not found:\n{listed}\n\n{remedy}",
        file=sys.stderr,
    )
    sys.exit(1)

TARGET_COL_CANDIDATES = ["abuse_type", "return_type", "label", "class", "target"]
CUSTOMER_ID_CANDIDATES = [
    "customer_id", "customerid", "user_id", "userid", "cust_id", "buyer_id",
]
TIMESTAMP_CANDIDATES = [
    "order_date", "return_date", "purchase_date", "timestamp", "date",
    "created_at", "order_datetime", "return_datetime",
]

LEAKAGE_MACRO_F1_THRESHOLD = 0.6

# High-cardinality row identifiers. Excluded from the leakage sweep because
# ordinal-encoding 58k unique strings produces a meaningless split point and
# costs minutes; they are excluded from the model in src/model.py for the
# same reason. Confirmed against the real schema: order_id is unique per row,
# customer_id is near-unique (see Q1).
ID_COLS = ["order_id", "customer_id"]


def _find_column(columns: list[str], candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    # loose substring fallback
    for col_lower, col in lower_map.items():
        for cand in candidates:
            if cand in col_lower:
                return col
    return None


def _detect_target(df: pd.DataFrame) -> str:
    target = _find_column(list(df.columns), TARGET_COL_CANDIDATES)
    if target is None:
        raise ValueError(
            "Could not auto-detect the target column. Columns present: "
            f"{list(df.columns)}. Set the correct name in TARGET_COL_CANDIDATES "
            "or pass it explicitly, then re-run the gate."
        )
    return target


def check_customer_viability(df: pd.DataFrame) -> dict:
    """Gate 1: is there a customer-id column with enough repeat rows per
    customer to build §4.2 behavioral aggregates?

    Args:
        df: the raw dataframe (any column set; the id column is detected,
            not assumed).

    Returns: a dict with a `verdict` string and the median/mean/max rows-
    per-customer that produced it. Viability is `median > 1.0`, not merely
    "an id column exists" — a customer id that never repeats (as on this
    dataset) still resolves to NOT VIABLE.
    """
    cust_col = _find_column(list(df.columns), CUSTOMER_ID_CANDIDATES)
    if cust_col is None:
        return {
            "customer_id_column": None,
            "verdict": "NO USABLE CUSTOMER ID FOUND",
            "median_rows_per_customer": None,
            "detail": (
                "No column matched known customer-id patterns "
                f"{CUSTOMER_ID_CANDIDATES}. §4.2 behavioral-aggregate features "
                "are undefined; fall back to §4.1 transaction-level features only."
            ),
        }
    counts = df[cust_col].value_counts()
    median_rows = float(counts.median())
    viable = median_rows > 1.0
    return {
        "customer_id_column": cust_col,
        "verdict": "VIABLE" if viable else "NOT VIABLE (median = 1 row/customer)",
        "median_rows_per_customer": median_rows,
        "mean_rows_per_customer": float(counts.mean()),
        "max_rows_per_customer": int(counts.max()),
        "n_unique_customers": int(df[cust_col].nunique()),
        "detail": (
            "Median rows-per-customer > 1 -> §4.2 behavioral aggregates are "
            "buildable." if viable else
            "Median rows-per-customer == 1 -> every §4.2 feature is undefined; "
            "Day 2 scope changes to the §4.1 fallback (see ARCHITECTURE.md §9.2)."
        ),
    }


def check_timestamp(df: pd.DataFrame) -> dict:
    """Gate 2: is there a timestamp column reliable enough to split on
    chronologically, rather than falling back to a random split?

    Args:
        df: the raw dataframe (timestamp column is detected, not assumed).

    Returns: a dict with a `verdict` and `split_strategy`. Unparseable rows
    below 5% still count as usable — the threshold is deliberately generous
    because a temporal split is strictly more honest than a random one
    whenever it's available at all (see the `order_date` vs `return_date`
    discussion in src/features.py).
    """
    ts_col = _find_column(list(df.columns), TIMESTAMP_CANDIDATES)
    if ts_col is None:
        return {
            "timestamp_column": None,
            "verdict": "NO USABLE TIMESTAMP FOUND",
            "split_strategy": "random (stratified 80/20)",
            "detail": (
                f"No column matched known timestamp patterns {TIMESTAMP_CANDIDATES}. "
                "Falling back to a stratified random split."
            ),
        }
    parsed = pd.to_datetime(df[ts_col], errors="coerce")
    n_bad = int(parsed.isna().sum())
    usable = n_bad < 0.05 * len(df)
    return {
        "timestamp_column": ts_col,
        "verdict": "USABLE" if usable else f"UNRELIABLE ({n_bad} unparseable rows)",
        "unparseable_rows": n_bad,
        "date_range": (
            f"{parsed.min()} to {parsed.max()}" if usable else None
        ),
        "split_strategy": "temporal (chronological 80/20)" if usable else "random (stratified 80/20)",
        "detail": (
            "Timestamp parses cleanly -> use a temporal split so evaluation "
            "reflects realistic future-prediction conditions."
            if usable else
            "Timestamp too unreliable to split on -> stratified random split instead."
        ),
    }


def leakage_sweep(df: pd.DataFrame, target_col: str) -> dict:
    """Gate 3: fit a single-feature decision tree per column and flag any
    feature that alone reaches >= LEAKAGE_MACRO_F1_THRESHOLD macro-F1.

    Args:
        df: the raw dataframe. ID_COLS are excluded from the sweep (high-
            cardinality identifiers produce a meaningless split point).
        target_col: the label column name.

    Returns: a dict with `sweep_depth`, the full per-feature results table
    (both `depth1_macro_f1` and `single_feature_macro_f1`), the mutual-
    information ranking, and the `suspects` subset. The non-obvious
    assumption: `sweep_depth = max(2, n_classes - 1)`, not depth-1 as
    ARCHITECTURE.md §9.1 originally specified — see the module docstring
    for why depth-1 is invalid on a 4-class target.
    """
    y_raw = df[target_col]
    y = LabelEncoder().fit_transform(y_raw)
    feature_cols = [c for c in df.columns if c != target_col and c not in ID_COLS]

    # Depth needed for the tree to be able to address every class at all.
    # ceil(log2(n_classes)) is the floor; n_classes - 1 gives headroom to
    # isolate each class with axis-aligned splits on a single feature.
    n_classes = len(np.unique(y))
    sweep_depth = max(2, n_classes - 1)

    results = []
    for col in feature_cols:
        x = df[[col]].copy()
        if pd.api.types.is_numeric_dtype(x[col]):
            x = x.fillna(x[col].median()).to_numpy()
        else:
            # Covers object, pandas 2.x `str`/ArrowString, category and datetime
            # columns — all encoded ordinally purely so a depth-1 tree can split
            # on them. The encoding is meaningless as an ordering; it only needs
            # to preserve distinctness for the leakage test.
            x = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            ).fit_transform(x[[col]].astype(str))

        # Single-feature decision tree macro-F1 via 3-fold CV, at two depths:
        # depth-1 as ARCHITECTURE.md §9.1 originally specified (kept only so
        # the correction is auditable), and `sweep_depth`, which is deep enough
        # to address all n_classes. See the module docstring for why depth-1
        # alone is an invalid leakage test on a multiclass target.
        row = {"feature": col}
        for label, depth in (("depth1_macro_f1", 1), ("single_feature_macro_f1", sweep_depth)):
            try:
                clf = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
                scores = cross_val_score(clf, x, y, cv=3, scoring="f1_macro")
                row[label] = float(np.mean(scores))
            except ValueError:
                # A single-feature fit can fail (e.g. all-NaN or constant column
                # after encoding) — recorded as NaN rather than aborting the
                # whole sweep over one bad column.
                row[label] = float("nan")

        results.append(row)

    results_df = pd.DataFrame(results).sort_values(
        "single_feature_macro_f1", ascending=False, na_position="last"
    )

    # Mutual information across all features together (needs a single numeric matrix)
    x_all = df[feature_cols].copy()
    for col in x_all.columns:
        if pd.api.types.is_numeric_dtype(x_all[col]):
            x_all[col] = x_all[col].fillna(x_all[col].median())
        else:
            x_all[col] = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            ).fit_transform(x_all[[col]].astype(str))
    mi = mutual_info_classif(x_all, y, random_state=RANDOM_STATE)
    mi_df = pd.DataFrame({"feature": feature_cols, "mutual_information": mi}).sort_values(
        "mutual_information", ascending=False
    )

    suspects = results_df[results_df["single_feature_macro_f1"] >= LEAKAGE_MACRO_F1_THRESHOLD]

    return {
        "sweep_depth": sweep_depth,
        "single_feature_f1": results_df,
        "mutual_information": mi_df,
        "suspects": suspects,
        "verdict": (
            f"{len(suspects)} SUSPECT FEATURE(S) >= {LEAKAGE_MACRO_F1_THRESHOLD} macro-F1 alone"
            if len(suspects) > 0 else "NO SINGLE-FEATURE LEAKAGE DETECTED"
        ),
    }


def _fmt_table(df: pd.DataFrame, n: int = 10) -> str:
    return df.head(n).to_markdown(index=False)


def write_data_notes(
    df: pd.DataFrame,
    target_col: str,
    customer_result: dict,
    timestamp_result: dict,
    leakage_result: dict,
) -> None:
    """Render the three gate results (from check_customer_viability,
    check_timestamp, leakage_sweep) to docs/DATA_NOTES.md as markdown.

    Args: df/target_col for the class-balance table; the three gate result
        dicts, rendered verbatim rather than re-summarized, so the written
        file is traceable to exactly what the gate computed.
    Returns: None — writes to DATA_NOTES_PATH as a side effect.
    """
    DATA_NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)

    class_balance = df[target_col].value_counts(normalize=True).round(4) * 100

    lines = []
    lines.append("# Data Notes — Day 1 Gate Findings\n")
    lines.append(
        "Generated by `src/data_gate.py` (ARCHITECTURE.md §9.1). "
        "These findings and their consequences are referenced in the final writeup.\n"
    )

    lines.append("## Dataset shape\n")
    lines.append(f"- Rows: {len(df):,}\n- Columns: {df.shape[1]}\n- Target column detected: `{target_col}`\n")

    lines.append("\n## Class balance\n")
    lines.append(class_balance.to_string() + "\n")

    lines.append("\n## Q1 — Usable repeat-customer identifier?\n")
    for k, v in customer_result.items():
        lines.append(f"- **{k}**: {v}")

    lines.append("\n\n## Q2 — Usable timestamp / split strategy?\n")
    for k, v in timestamp_result.items():
        lines.append(f"- **{k}**: {v}")

    depth = leakage_result["sweep_depth"]
    lines.append("\n\n## Q3 — Leakage sweep\n")
    lines.append(f"- **verdict**: {leakage_result['verdict']}\n")
    lines.append(
        "\n**Method correction (Day 1).** ARCHITECTURE.md §9.1 specifies a *depth-1* "
        "tree per feature. That test is invalid on a 4-class target: one split gives "
        "two leaves, so the tree can name at most 2 of 4 classes and its macro-F1 is "
        "capped near ~0.45 however perfectly the feature encodes the label. Run as "
        "specified, this gate **passed** `abuse_label`. The sweep below fits at "
        f"depth {depth} (enough to address all classes) and reports the depth-1 number "
        "alongside it so the correction is auditable.\n"
    )
    lines.append(
        f"\n### Top single-feature macro-F1 (depth-{depth} tree, 3-fold CV)\n"
        f"`depth1_macro_f1` is the original §9.1 test, shown for comparison only.\n"
    )
    lines.append(_fmt_table(leakage_result["single_feature_f1"]))
    lines.append("\n\n### Top mutual information vs. target\n")
    lines.append(_fmt_table(leakage_result["mutual_information"]))
    if len(leakage_result["suspects"]) > 0:
        lines.append(
            "\n\n**Action required:** the feature(s) above the "
            f"{LEAKAGE_MACRO_F1_THRESHOLD} macro-F1 threshold must be inspected "
            "before modeling proceeds (§6.4 suspiciously-good-result protocol). "
            "Either drop/transform the leaking feature with a documented reason, "
            "or record here why it is legitimate signal rather than a generation artifact."
        )
    else:
        lines.append("\n\nNo single feature clears the leakage threshold on its own.")

    lines.append("\n\n## Consequences for the build plan\n")
    lines.append(f"- Split strategy adopted: **{timestamp_result['split_strategy']}**")
    if customer_result["median_rows_per_customer"] in (None, 1.0):
        lines.append(
            "- §4.2 behavioral-aggregate features are **not viable as specified** — "
            "Day 2 scope falls back to §4.1 transaction-level features only. "
            "This is a headline limitation, not a hidden one."
        )
    else:
        lines.append("- §4.2 behavioral-aggregate features are viable and proceed as planned.")

    DATA_NOTES_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote {DATA_NOTES_PATH}")


def run(raw_path: Path = RAW_PATH) -> None:
    """CLI entrypoint: run all three gates against `raw_path` and write
    docs/DATA_NOTES.md. Exits with a clear message (not a traceback) if the
    raw CSV isn't present, since that's the first thing a fresh clone hits."""
    require_artifacts([raw_path], NEEDS_RAW_CSV)

    df = pd.read_csv(raw_path)
    target_col = _detect_target(df)

    print(f"Loaded {raw_path} — {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"Detected target column: {target_col}")

    customer_result = check_customer_viability(df)
    print(f"Q1 customer viability: {customer_result['verdict']}")

    timestamp_result = check_timestamp(df)
    print(f"Q2 timestamp / split: {timestamp_result['verdict']}")

    leakage_result = leakage_sweep(df, target_col)
    print(f"Q3 leakage sweep: {leakage_result['verdict']}")

    write_data_notes(df, target_col, customer_result, timestamp_result, leakage_result)


if __name__ == "__main__":
    run()
