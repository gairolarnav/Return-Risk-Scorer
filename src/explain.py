"""
Per-class SHAP explainability (docs/ARCHITECTURE.md §5).

Deliverable: per-class SHAP on both tracks, saved to runs/, paired with a
written interpretation of *why* the confusable classes confuse
(runs/shap_interpretation.md) — attributions without that reading are a
picture, not an explanation.

Uses shap.TreeExplainer directly on the trained LightGBM booster from each
track's run bundle — no reimplementation of feature engineering or model
loading; both come from src.infer.load_run and the same processed test
parquet every other evaluation in this repo reads.

For this shap==0.46.0 / lightgbm==4.5.0 pairing, TreeExplainer.shap_values on
a multiclass sklearn model returns one array of shape
(n_rows, n_features, n_classes) — confirmed by running it, not assumed from
docs, since this exact shape has changed across shap versions.

Run as:
    python -m src.explain              # both tracks
    python -m src.explain full         # one track
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.features import PROCESSED_DIR
from src.infer import load_run

RUNS_DIR = Path("runs")
TOP_N = 10


def load_test_frame(bundle: dict) -> pd.DataFrame:
    """Same categorical pinning as src.infer.prepare_frame, applied to the
    whole test set rather than one record — the model-ready frame SHAP needs
    to explain, built the one way this repo builds it."""
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    feature_cols = bundle["feature_cols"]
    frame = test[feature_cols].copy()
    for col, levels in bundle.get("categories", {}).items():
        if col in frame.columns:
            frame[col] = pd.Categorical(frame[col], categories=levels)
    return frame


def compute_shap_values(track: str) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Returns (shap_values[n_rows, n_features, n_classes], X, class_names)."""
    bundle = load_run(RUNS_DIR / f"model_{track}")
    frame = load_test_frame(bundle)
    class_names = list(bundle["label_encoder"].classes_)

    explainer = shap.TreeExplainer(bundle["model"])
    shap_values = explainer.shap_values(frame)
    if shap_values.ndim != 3:
        raise ValueError(
            f"Expected shap_values shape (n_rows, n_features, n_classes), got {shap_values.shape}. "
            "The shap/lightgbm version pairing changed its output layout — update this module."
        )
    return shap_values, frame, class_names


def top_features_per_class(
    shap_values: np.ndarray, feature_cols: list[str], class_names: list[str], top_n: int = TOP_N
) -> dict[str, list[dict]]:
    """Mean |SHAP| per feature, per class, ranked — the numbers behind both
    the plot and the written interpretation."""
    result: dict[str, list[dict]] = {}
    for c_idx, cls in enumerate(class_names):
        mean_abs = np.abs(shap_values[:, :, c_idx]).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:top_n]
        result[cls] = [
            {"feature": feature_cols[i], "mean_abs_shap": float(mean_abs[i])} for i in order
        ]
    return result


def plot_per_class_bars(
    shap_values: np.ndarray, feature_cols: list[str], class_names: list[str], track: str
) -> Path:
    """One figure, one subplot per class, top-N features by mean |SHAP|.
    Built with plain matplotlib against values computed above rather than
    shap's own plotting, which calls plt.show() and fights headless/Agg use."""
    n_classes = len(class_names)
    ncols = 2
    nrows = -(-n_classes // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.2 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for c_idx, cls in enumerate(class_names):
        ax = axes[c_idx]
        mean_abs = np.abs(shap_values[:, :, c_idx]).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:TOP_N]
        feats = [feature_cols[i] for i in order][::-1]
        vals = mean_abs[order][::-1]
        ax.barh(feats, vals, color="#4c72b0")
        ax.set_title(cls, fontsize=10)
        ax.set_xlabel("mean |SHAP value|", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)

    for ax in axes[n_classes:]:
        ax.axis("off")

    fig.suptitle(f"Per-class SHAP feature importance — track: {track}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = RUNS_DIR / f"shap_{track}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def write_summary_json(top_features: dict, track: str) -> Path:
    """Write the per-class top-features dict to runs/shap_{track}.json.
    Args: top_features from top_features_per_class; track name for the
    filename. Returns the path written, so callers can print it."""
    import json

    out_path = RUNS_DIR / f"shap_{track}.json"
    out_path.write_text(json.dumps(top_features, indent=2))
    return out_path


def run(tracks: list[str] | None = None) -> dict[str, dict]:
    """CLI entrypoint: compute per-class SHAP for each of `tracks` (both,
    if None), writing runs/shap_{track}.{json,png}. Reads
    runs/model_{track}.joblib, so src.model.run() must already have run.
    Returns {track: top_features} for both tracks, in case a caller wants
    the values in memory rather than re-reading the JSON."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    from src.features import FEATURE_SETS

    all_top = {}
    for track in tracks or list(FEATURE_SETS):
        shap_values, frame, class_names = compute_shap_values(track)
        top_features = top_features_per_class(shap_values, list(frame.columns), class_names)
        json_path = write_summary_json(top_features, track)
        png_path = plot_per_class_bars(shap_values, list(frame.columns), class_names, track)
        all_top[track] = top_features
        print(f"[{track:>7}] wrote {json_path} and {png_path}")
        for cls, feats in top_features.items():
            top3 = ", ".join(f"{f['feature']}={f['mean_abs_shap']:.3f}" for f in feats[:3])
            print(f"          {cls:<20} top: {top3}")
    return all_top


if __name__ == "__main__":
    run(sys.argv[1:] or None)
