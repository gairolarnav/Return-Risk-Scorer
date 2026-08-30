"""
Tests for src/explain.py — per-class SHAP on both tracks, saved to runs/
(docs/ARCHITECTURE.md §5).

Builds a tiny real bundle + processed test parquet under tmp_path (same
approach as tests/test_score.py) and monkeypatches src.explain's RUNS_DIR /
PROCESSED_DIR so the module's real compute path runs end to end without
touching the gitignored dataset or trained artifacts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.explain import (
    compute_shap_values,
    plot_per_class_bars,
    run,
    top_features_per_class,
)
from src.features import add_transaction_level_features
from src.model import save_run, train_track

CLASSES = ["Legitimate", "Wardrobing", "Policy Abuser", "Fraudulent Return"]


def _synthetic_frame(n_per_class=20, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for cls in CLASSES:
        for _ in range(n_per_class):
            rows.append(
                {
                    "avg_order_value_usd": float(rng.uniform(20, 200)),
                    "refund_amount_requested_usd": float(rng.uniform(10, 100)),
                    "total_orders_lifetime": int(rng.integers(1, 30)),
                    "total_returns_lifetime": int(rng.integers(0, 10)),
                    "account_age_days": int(rng.integers(1, 900)),
                    "days_to_return": int(rng.integers(1, 60)),
                    "product_category": rng.choice(["Apparel", "Electronics", "Home"]),
                    "order_date": "2022-01-01",
                    "return_date": "2022-06-15",
                    "abuse_type": cls,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def wired_explain(tmp_path, monkeypatch):
    """Trains + saves a tiny bundle and a matching test.parquet under
    tmp_path, then points src.explain at them."""
    frame = add_transaction_level_features(_synthetic_frame())
    train_parts, test_parts = [], []
    for cls in CLASSES:
        cls_rows = frame[frame["abuse_type"] == cls]
        cut = int(len(cls_rows) * 0.7)
        train_parts.append(cls_rows.iloc[:cut])
        test_parts.append(cls_rows.iloc[cut:])
    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    test.to_parquet(processed_dir / "test.parquet", index=False)

    runs_dir = tmp_path / "runs"
    result = train_track(train, test, track="full", class_weighted=True)

    import src.explain as explain_module
    import src.model as model_module

    monkeypatch.setattr(model_module, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(explain_module, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(explain_module, "PROCESSED_DIR", processed_dir)
    save_run(result, "model_full")
    return runs_dir


def test_compute_shap_values_shape(wired_explain):
    shap_values, frame, class_names = compute_shap_values("full")
    assert shap_values.shape == (len(frame), len(frame.columns), len(class_names))
    assert set(class_names) == set(CLASSES)


def test_top_features_per_class_is_ranked_and_bounded(wired_explain):
    shap_values, frame, class_names = compute_shap_values("full")
    top = top_features_per_class(shap_values, list(frame.columns), class_names, top_n=5)
    assert set(top) == set(class_names)
    for cls, feats in top.items():
        assert len(feats) <= 5
        assert len(feats) <= len(frame.columns)
        scores = [f["mean_abs_shap"] for f in feats]
        assert scores == sorted(scores, reverse=True)
        assert all(f["feature"] in frame.columns for f in feats)


def test_plot_per_class_bars_writes_a_real_image(wired_explain):
    shap_values, frame, class_names = compute_shap_values("full")
    out_path = plot_per_class_bars(shap_values, list(frame.columns), class_names, "full")
    assert out_path.exists()
    assert out_path.stat().st_size > 1000  # not an empty/blank figure


def test_run_writes_json_and_png_for_the_track(wired_explain):
    result = run(["full"])
    runs_dir = wired_explain
    json_path = runs_dir / "shap_full.json"
    png_path = runs_dir / "shap_full.png"
    assert json_path.exists()
    assert png_path.exists()

    import json

    payload = json.loads(json_path.read_text())
    assert set(payload) == set(CLASSES)
    assert set(result) == {"full"}
    assert result["full"] == payload
