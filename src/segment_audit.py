"""
Segment-level FPR audit by order-value bucket (docs/ARCHITECTURE.md §6.2
stretch goal).

Checks that the decision policy is not concentrating false blocks on one
customer segment — an aggregate false-block rate can look acceptable while
being paid entirely by one slice of customers.

Reuses the exact decision layer src/evaluate.py and src/infer.py use --
build_cost_matrix + expected_cost_decision under DEFAULT_POSTURES -- against
the saved runs/model_<track>_proba.npy / _ytest.npy arrays, rather than
recomputing predictions a second way. Those arrays are row-aligned with
data/processed/test.parquet (src/model.py builds x_test = test[cols] with no
reshuffling; src/evaluate.py already relies on the same alignment), so
avg_order_value_usd for the bucketing comes from the parquet file by
position.

Reports two rates per bucket, not one:
  - hard_block_fpr    legitimate customers hard-blocked -- the literal "FPR"
                       the checklist item names, and evaluate.py's own
                       false_block_rate_on_legit metric, segmented.
  - soft_friction_rate legitimate customers given friction -- included
                       because docs/LEAKAGE_FINDING.md's Day 4 finding is
                       that hard-block is nearly inert on this data and
                       approve<->soft-friction is the axis that actually
                       moves. Auditing only hard_block_fpr on testbed would
                       report ~0% everywhere and miss where segment
                       concentration could actually show up.

Run as:
    python -m src.segment_audit              # both tracks, loss-neutral posture
    python -m src.segment_audit full          # one track
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_gate import NEEDS_FEATURES, NEEDS_MODEL, require_artifacts
from src.evaluate import ACTIONS, DEFAULT_POSTURES, build_cost_matrix, expected_cost_decision
from src.features import FEATURE_SETS, PROCESSED_DIR

RUNS_DIR = Path("runs")
N_BUCKETS = 4
BUCKET_NAMES = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
# A segment more than this many times more affected than the least-affected
# segment is flagged as concentration worth a second look. Chosen as a round
# number before looking at the result, not fit to it.
CONCENTRATION_RATIO_FLAG = 2.0
# Rates below this are single-digit row counts on a few-thousand-row bucket
# -- noise, not a concentration finding, even if one bucket is exactly 0 and
# another isn't.
MIN_RATE_TO_FLAG = 0.01


def _order_value_buckets(
    values: pd.Series, n_buckets: int = N_BUCKETS
) -> tuple[pd.Series, list[str]]:
    """Quartile buckets on avg_order_value_usd, labelled with the actual
    dollar range so the report is readable without cross-referencing edges
    elsewhere.

    Returns (bucket_label_per_row, ordered_label_list). The label list is
    returned rather than re-derived by the caller so bucket iteration order is
    fixed by the quantile edges, not by whatever order the values happen to
    appear in.
    """
    _, edges = pd.qcut(values, q=n_buckets, retbins=True, duplicates="drop")
    labels = [
        f"{BUCKET_NAMES[i]} (${edges[i]:.0f}-${edges[i + 1]:.0f})" for i in range(len(edges) - 1)
    ]
    codes = pd.qcut(values, q=n_buckets, labels=False, duplicates="drop")
    return codes.map(dict(enumerate(labels))), labels


def _concentration(rates: list[float | None]) -> dict:
    """How unevenly one routed rate is spread across segments. A module-
    level function (not a closure inside audit_track) so it has its own
    test coverage independent of a real/synthetic model run."""
    present = [r for r in rates if r is not None]
    if not present or max(present) < MIN_RATE_TO_FLAG:
        # Below MIN_RATE_TO_FLAG, a "0 in one bucket, nonzero in another"
        # split is a single-digit row count either way -- noise on a
        # few-thousand-row bucket, not a concentration finding.
        return {
            "max": max(present) if present else 0.0,
            "min": min(present) if present else 0.0,
            "ratio": None,
            "flagged": False,
        }
    lo = min(present)
    hi = max(present)
    if lo == 0:
        # Unbounded ratio: the strongest possible concentration signal (one
        # segment gets this outcome at a meaningful rate, another gets it
        # never) -- must be flagged, not silently treated as "ratio
        # undefined, therefore nothing to see" the way a naive hi/lo would.
        return {"max": hi, "min": lo, "ratio": None, "flagged": True}
    ratio = hi / lo
    return {
        "max": hi,
        "min": lo,
        "ratio": ratio,
        "flagged": bool(ratio >= CONCENTRATION_RATIO_FLAG),
    }


def audit_track(track: str, posture: str = "loss-neutral (1:1)") -> dict:
    """Does the decision policy concentrate false blocks / friction on one
    order-value segment of legitimate customers, rather than spreading
    evenly? Bins legitimate test customers into order-value quartiles and
    reports hard-block/soft-friction rate per bucket plus a concentration
    flag.

    Args: track ("full"/"testbed"); posture — a key into
        src.evaluate.DEFAULT_POSTURES, applied via the same
        expected_cost_decision the real decision layer uses (not a
        reimplementation).
    Returns: dict with per-bucket rates and `hard_block_concentration`/
    `soft_friction_concentration` flags (see `_concentration`). Raises
    ValueError if test.parquet and the saved proba array have drifted out
    of row-alignment, rather than silently mismatching buckets to the
    wrong rows.
    """
    require_artifacts(
        [
            RUNS_DIR / f"model_{track}_proba.npy",
            RUNS_DIR / f"model_{track}_ytest.npy",
            RUNS_DIR / f"model_{track}.joblib",
        ],
        NEEDS_MODEL,
    )
    require_artifacts([PROCESSED_DIR / "test.parquet"], NEEDS_FEATURES)

    proba = np.load(RUNS_DIR / f"model_{track}_proba.npy")
    y_true_idx = np.load(RUNS_DIR / f"model_{track}_ytest.npy")
    bundle = joblib.load(RUNS_DIR / f"model_{track}.joblib")
    class_names = list(bundle["label_encoder"].classes_)

    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    if len(test) != len(proba):
        raise ValueError(
            f"test.parquet has {len(test)} rows but model_{track}_proba.npy has {len(proba)} -- "
            "they must be row-aligned. Retrain with `python -m src.model` after `python -m src.features`."
        )

    c_fp, c_fn = DEFAULT_POSTURES[posture]
    cost_matrix = build_cost_matrix(class_names, c_fp=c_fp, c_fn=c_fn)
    action_idx = expected_cost_decision(proba, cost_matrix)
    actions = np.array(ACTIONS)[action_idx]

    legit_idx = class_names.index("Legitimate")
    is_legit = y_true_idx == legit_idx

    bucket, labels = _order_value_buckets(test["avg_order_value_usd"])
    bucket = bucket.to_numpy()

    rows = []
    for label in labels:
        mask = is_legit & (bucket == label)
        n = int(mask.sum())
        row = {
            "bucket": label,
            "n_legitimate_customers": n,
            "hard_block_fpr": float((actions[mask] == "hard_block").mean()) if n else None,
            "soft_friction_rate": float((actions[mask] == "soft_friction").mean()) if n else None,
        }
        rows.append(row)

    return {
        "track": track,
        "posture": posture,
        "n_legitimate_total": int(is_legit.sum()),
        "buckets": rows,
        "hard_block_concentration": _concentration([r["hard_block_fpr"] for r in rows]),
        "soft_friction_concentration": _concentration([r["soft_friction_rate"] for r in rows]),
    }


def plot_audit(result: dict, out_path: Path) -> Path:
    """Render the per-bucket hard-block/soft-friction bar chart from
    audit_track's output. Args: `result` dict as returned by audit_track;
    `out_path` to save to. Returns `out_path` for convenience."""
    # matplotlib treats a pair of "$" as mathtext delimiters, which silently
    # eats the literal dollar signs in bucket labels like "$15-$93" -- escape
    # them for display only; JSON/console output keeps the plain "$" labels.
    buckets = [r["bucket"].replace("$", r"\$") for r in result["buckets"]]
    hard = [r["hard_block_fpr"] or 0.0 for r in result["buckets"]]
    soft = [r["soft_friction_rate"] or 0.0 for r in result["buckets"]]

    x = np.arange(len(buckets))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, hard, width, label="hard_block_fpr", color="#c44e52")
    ax.bar(x + width / 2, soft, width, label="soft_friction_rate", color="#4c72b0")
    ax.set_xticks(x)
    ax.set_xticklabels(buckets, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("rate among legitimate customers")
    ax.set_title(f"Segment FPR audit by order-value bucket — {result['track']} ({result['posture']})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def run(tracks: list[str] | None = None) -> dict[str, dict]:
    """CLI entrypoint: audit_track + plot_audit for each of `tracks` (both,
    if None) at the default posture, writing runs/segment_fpr_{track}.
    {json,png}. Reads model/proba artifacts from src.model.run() and
    test.parquet from src.features.run() — both must already exist."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for track in tracks or list(FEATURE_SETS):
        result = audit_track(track)
        json_path = RUNS_DIR / f"segment_fpr_{track}.json"
        json_path.write_text(json.dumps(result, indent=2))
        png_path = plot_audit(result, RUNS_DIR / f"segment_fpr_{track}.png")
        results[track] = result

        print(f"[{track:>7}] posture={result['posture']}  n_legitimate={result['n_legitimate_total']}")
        for row in result["buckets"]:
            hb = row["hard_block_fpr"]
            sf = row["soft_friction_rate"]
            print(
                f"          {row['bucket']:<28} n={row['n_legitimate_customers']:>5}  "
                f"hard_block_fpr={'n/a' if hb is None else f'{hb:.4f}'}  "
                f"soft_friction_rate={'n/a' if sf is None else f'{sf:.4f}'}"
            )
        hc = result["hard_block_concentration"]
        sc = result["soft_friction_concentration"]
        print(
            f"          hard_block ratio(max/min)={hc['ratio']}  flagged={hc['flagged']}  |  "
            f"soft_friction ratio(max/min)={sc['ratio']}  flagged={sc['flagged']}"
        )
        print(f"          -> {json_path}, {png_path}")
    return results


if __name__ == "__main__":
    run(sys.argv[1:] or None)
