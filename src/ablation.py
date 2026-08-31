"""
Degeneracy evidence — the ablation ladder (Day 2).

This module exists because of the Day 1 leakage finding (docs/LEAKAGE_FINDING.md).
It is not a feature-selection tool and must not be read as one. Its job is to
measure, reproducibly, *how much* of this dataset's separability is an artifact
of the synthetic generator, and to produce the evidence table that the writeup
leads with.

Two quantities are reported per feature set:

  macro_f1
      Standard held-out macro-F1 on the temporal test split.

  threshold_sensitive_frac
      Share of test rows whose top predicted probability is below 0.90.
      This is the number that matters for ARCHITECTURE.md §6.2. A cost-ratio
      sweep can only ever change a decision for a row near a class boundary;
      rows the model calls at p>0.99 are decided identically under every
      merchant posture. When this fraction is ~0, the cost sweep is
      mathematically guaranteed to be flat and the §6.2 centerpiece has
      nothing to analyse — regardless of how carefully the cost matrix is
      specified.

The headline result (see docs/LEAKAGE_FINDING.md for the full write-up):
the full feature set scores ~0.999 macro-F1 with ~0.1% of rows threshold-
sensitive, and performance degrades *smoothly* as artifact features are
removed, with no natural cut point. That smoothness is itself the diagnosis —
the generator drew every feature from per-class bounded ranges, so leakage is
a dataset-wide property rather than three bad columns.

Run as:
    python -m src.ablation
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

from src.data_gate import RANDOM_STATE
from src.features import DROP_COLS, TARGET_COL, build_and_split
from src.model import as_model_frame

RUNS_DIR = Path("runs")

# A prediction whose top class probability is at or above this value is
# decided the same way under every cost posture, so it contributes nothing
# to the §6.2 sweep. 0.90 is deliberately generous — the conclusion holds
# at any reasonable value, and a looser bar makes the degeneracy finding
# harder to accuse of being manufactured by a tight threshold.
DECIDED_PROB = 0.90

# The ablation ladder, ordered by descending leakage contribution as measured
# on Day 1. Each rung removes one more generator artifact. Ordering was
# derived from the single-feature and greedy-forward-selection sweeps in
# src/data_gate.py, not chosen to produce a pleasing curve.
#
# IMPORTANT: this ladder is diagnostic evidence, not a model-selection
# procedure. "Drop features until the model gets worse" would be indefensible
# as a way to pick a model. The point of running it is to show that no cut
# point is principled — which is the finding.
#
# Each rung drags out the *derived proxies* of the columns it removes as well
# as the raw columns. This is not cosmetic: the first version of this ladder
# dropped only raw columns and came out nearly flat, because src/features.py
# engineers `returns_per_order` (== return_rate_pct / 100) and
# `orders_kept_lifetime` (== total_orders - total_returns), which restate the
# dropped columns algebraically. An ablation that leaves a restatement of the
# feature it just removed measures nothing at all.
_PROXIES = {
    "return_rate_pct": ["returns_per_order"],
    "total_returns_lifetime": ["orders_kept_lifetime", "returns_per_order"],
    "avg_order_value_usd": ["refund_to_avg_order_ratio"],
    "refund_amount_requested_usd": ["refund_to_avg_order_ratio"],
    "days_to_return": [],
}


def _with_proxies(raw_cols: list[str]) -> list[str]:
    out = list(raw_cols)
    for col in raw_cols:
        out.extend(_PROXIES.get(col, []))
    return sorted(set(out))


_RUNGS: list[tuple[str, list[str]]] = [
    ("A. all features", []),
    ("B. -wishlist_to_cart_time_hrs", ["wishlist_to_cart_time_hrs"]),
    ("C. B -days_to_return", ["wishlist_to_cart_time_hrs", "days_to_return"]),
    (
        "D. C -return_rate_pct",
        ["wishlist_to_cart_time_hrs", "days_to_return", "return_rate_pct"],
    ),
    (
        "E. D -total_returns_lifetime",
        [
            "wishlist_to_cart_time_hrs",
            "days_to_return",
            "return_rate_pct",
            "total_returns_lifetime",
        ],
    ),
    (
        "F. E -customer_support_contacts",
        [
            "wishlist_to_cart_time_hrs",
            "days_to_return",
            "return_rate_pct",
            "total_returns_lifetime",
            "customer_support_contacts",
        ],
    ),
    (
        "G. F -previous_dispute_count  <- TESTBED",
        [
            "wishlist_to_cart_time_hrs",
            "days_to_return",
            "return_rate_pct",
            "total_returns_lifetime",
            "customer_support_contacts",
            "previous_dispute_count",
        ],
    ),
    (
        "H. G -avg_order_value, -refund_amount",
        [
            "wishlist_to_cart_time_hrs",
            "days_to_return",
            "return_rate_pct",
            "total_returns_lifetime",
            "customer_support_contacts",
            "previous_dispute_count",
            "avg_order_value_usd",
            "refund_amount_requested_usd",
        ],
    ),
]

ABLATION_LADDER: list[tuple[str, list[str]]] = [
    (name, _with_proxies(cols)) for name, cols in _RUNGS
]


def evaluate_feature_set(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: list[str],
    label_encoder: LabelEncoder,
    n_estimators: int = 150,
    learning_rate: float = 0.08,
) -> dict:
    """Fit one rung of the ladder and report both quantities.

    Uses a lighter model than the final one in src/model.py on purpose: this
    is a comparative sweep across eight feature sets, and the *relative*
    picture is what carries the finding. The absolute headline numbers come
    from src/model.py.
    """
    x_train, x_test = as_model_frame(train, test, cols)
    y_train = label_encoder.transform(train[TARGET_COL])
    y_test = label_encoder.transform(test[TARGET_COL])

    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(label_encoder.classes_),
        random_state=RANDOM_STATE,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        verbosity=-1,
        n_jobs=-1,
    )
    clf.fit(x_train, y_train)

    proba = clf.predict_proba(x_test)
    top_prob = proba.max(axis=1)

    return {
        "n_features": len(cols),
        "macro_f1": float(f1_score(y_test, proba.argmax(axis=1), average="macro")),
        "mean_top_prob": float(top_prob.mean()),
        "threshold_sensitive_frac": float((top_prob < DECIDED_PROB).mean()),
    }


def run_ladder(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Fit every rung in ABLATION_LADDER and return one row of results per
    rung. Args: already-split train/test frames (see src.features). A rung
    is skipped (not zero-filled) if dropping its columns leaves none —
    that's a bug in a rung definition, not a result worth recording as 0."""
    # Train only -- see the note in src.model.train_track.
    label_encoder = LabelEncoder().fit(train[TARGET_COL])
    base_cols = [c for c in train.columns if c != TARGET_COL and c not in DROP_COLS]

    rows = []
    for name, dropped in ABLATION_LADDER:
        cols = [c for c in base_cols if c not in dropped]
        if not cols:
            continue
        result = evaluate_feature_set(train, test, cols, label_encoder)
        rows.append({"feature_set": name, **result})
        print(
            f"{name:<40}{result['n_features']:>4}"
            f"{result['macro_f1']:>10.4f}{result['threshold_sensitive_frac']:>10.1%}"
        )

    return pd.DataFrame(rows)


def run() -> None:
    """CLI entrypoint: run the full ladder and write
    runs/ablation_ladder.{json,md}. Reads the raw CSV via
    src.features.build_and_split (no cached parquet path)."""
    train, test = build_and_split()
    print(f"{'feature set':<40}{'n':>4}{'macro-F1':>10}{'thr-sens':>10}")
    ladder = run_ladder(train, test)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RUNS_DIR / "ablation_ladder.json"
    out_json.write_text(
        json.dumps(
            {
                "decided_prob_threshold": DECIDED_PROB,
                "note": (
                    "Diagnostic evidence for the Day 1 leakage finding, not a "
                    "feature-selection procedure. See src/ablation.py docstring "
                    "and docs/LEAKAGE_FINDING.md."
                ),
                "ladder": ladder.to_dict(orient="records"),
            },
            indent=2,
        )
    )
    print(f"\nWrote {out_json}")

    out_md = RUNS_DIR / "ablation_ladder.md"
    out_md.write_text(ladder.to_markdown(index=False, floatfmt=".4f") + "\n")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    run()
