"""
Tests for scripts/score.py — the CLI that replaced the cut app/demo.py
(docs/ARCHITECTURE.md §11, correction log).

Trains a tiny real bundle via src.model.train_track on synthetic data (same
code path the real pipeline uses, so the categorical/feature-engineering
machinery is exercised for real) and saves it with src.model.save_run into a
tmp_path runs dir, rather than depending on the gitignored trained artifacts
or the Kaggle CSV — consistent with tests/test_features.py and
tests/test_infer.py.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.score import run
from src.evaluate import ACTIONS
from src.features import add_transaction_level_features
from src.model import save_run, train_track

CLASSES = ["Legitimate", "Wardrobing", "Policy Abuser", "Fraudulent Return"]


def _synthetic_frame(n_per_class=15, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i, cls in enumerate(CLASSES):
        for j in range(n_per_class):
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
                    "return_date": f"2022-{1 + (i * n_per_class + j) % 9:02d}-15",
                    "abuse_type": cls,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def run_dir(tmp_path):
    """A tmp_path runs/ directory holding one real trained bundle per track,
    built the same way `python -m src.model` builds them."""
    frame = add_transaction_level_features(_synthetic_frame())
    # Stratified per-class split, not a contiguous slice: the synthetic frame
    # is grouped by class, so slicing by position risks leaving a class out
    # of train entirely, which silently changes the number of classes
    # LightGBM fits (predict_proba then returns too few columns for the cost
    # matrix, an unrelated-looking matmul shape error).
    train_parts, test_parts = [], []
    for cls in CLASSES:
        cls_rows = frame[frame["abuse_type"] == cls]
        cut = int(len(cls_rows) * 0.7)
        train_parts.append(cls_rows.iloc[:cut])
        test_parts.append(cls_rows.iloc[cut:])
    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)

    out_dir = tmp_path / "runs"
    for track in ("full", "testbed"):
        # testbed's normal exclusion list doesn't apply to this tiny synthetic
        # schema, so just reuse "full" for both — the CLI only cares that a
        # bundle exists at model_<track>.joblib with the right shape.
        result = train_track(train, test, track="full", class_weighted=True)
        result["metrics"]["track"] = track
        import importlib

        model_module = importlib.import_module("src.model")
        old_runs_dir = model_module.RUNS_DIR
        model_module.RUNS_DIR = out_dir
        try:
            save_run(result, f"model_{track}")
        finally:
            model_module.RUNS_DIR = old_runs_dir
    return out_dir


@pytest.fixture
def sample_record() -> dict:
    return {
        "avg_order_value_usd": 80.0,
        "refund_amount_requested_usd": 40.0,
        "total_orders_lifetime": 12,
        "total_returns_lifetime": 3,
        "account_age_days": 300,
        "days_to_return": 10,
        "product_category": "Apparel",
        "return_date": "2022-03-15",
    }


def test_single_record_scoring_writes_valid_json(run_dir, sample_record, tmp_path):
    record_path = tmp_path / "record.json"
    record_path.write_text(json.dumps(sample_record))
    out_path = tmp_path / "out.json"

    code = run(
        [
            "--record-file",
            str(record_path),
            "--track",
            "full",
            "--run-dir",
            str(run_dir),
            "--out",
            str(out_path),
        ]
    )
    assert code == 0
    result = json.loads(out_path.read_text())
    assert result["most_likely_class"] in CLASSES
    assert result["recommended_action"] in ACTIONS
    assert result["track"] == "full"
    assert set(result["class_probabilities"]) == set(CLASSES)


def test_inline_record_json_matches_record_file(run_dir, sample_record, tmp_path):
    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"
    record_path = tmp_path / "record.json"
    record_path.write_text(json.dumps(sample_record))

    run(["--record", json.dumps(sample_record), "--run-dir", str(run_dir), "--out", str(out_a)])
    run(["--record-file", str(record_path), "--run-dir", str(run_dir), "--out", str(out_b)])

    assert json.loads(out_a.read_text()) == json.loads(out_b.read_text())


def test_batch_csv_scores_every_row_and_preserves_input_columns(run_dir, tmp_path):
    batch = pd.DataFrame(
        [
            {
                "avg_order_value_usd": 80.0 + i,
                "refund_amount_requested_usd": 20.0,
                "total_orders_lifetime": 5,
                "total_returns_lifetime": 1,
                "account_age_days": 200,
                "days_to_return": 15,
                "product_category": "Home",
                "return_date": "2022-05-01",
            }
            for i in range(6)
        ]
    )
    csv_path = tmp_path / "batch.csv"
    out_path = tmp_path / "scored.csv"
    batch.to_csv(csv_path, index=False)

    code = run(
        ["--csv", str(csv_path), "--track", "testbed", "--run-dir", str(run_dir), "--out", str(out_path)]
    )
    assert code == 0
    scored = pd.read_csv(out_path)
    assert len(scored) == len(batch)
    assert "avg_order_value_usd" in scored.columns  # original columns preserved
    assert "recommended_action" in scored.columns
    assert set(scored["recommended_action"]).issubset(set(ACTIONS))
    assert (scored["track"] == "testbed").all()


def test_missing_bundle_exits_with_a_clear_message(sample_record, tmp_path, capsys):
    with pytest.raises(SystemExit) as excinfo:
        run(
            [
                "--record",
                json.dumps(sample_record),
                "--run-dir",
                str(tmp_path / "no_such_dir"),
            ]
        )
    assert "python -m src.model" in str(excinfo.value)


def test_malformed_json_record_exits_cleanly(run_dir):
    with pytest.raises(SystemExit) as excinfo:
        run(["--record", "not valid json", "--run-dir", str(run_dir)])
    assert "JSON" in str(excinfo.value)


def test_record_and_csv_are_mutually_exclusive(run_dir, sample_record):
    with pytest.raises(SystemExit):
        run(
            [
                "--record",
                json.dumps(sample_record),
                "--csv",
                "whatever.csv",
                "--run-dir",
                str(run_dir),
            ]
        )


def test_unknown_track_is_rejected_by_argparse(run_dir, sample_record):
    with pytest.raises(SystemExit):
        run(
            [
                "--record",
                json.dumps(sample_record),
                "--track",
                "bogus",
                "--run-dir",
                str(run_dir),
            ]
        )


def test_unknown_friction_is_rejected_by_argparse(run_dir, sample_record):
    with pytest.raises(SystemExit):
        run(
            [
                "--record",
                json.dumps(sample_record),
                "--friction",
                "bogus",
                "--run-dir",
                str(run_dir),
            ]
        )


def test_both_postures_reach_the_cli_output(run_dir, sample_record, tmp_path):
    """The CLI must pass BOTH axes through to the scorer and record them in the
    output. `--friction` was previously unreachable from the command line —
    src.infer hardcoded build_cost_matrix's friction cell — so a user could
    not move the one axis the project calls its centerpiece."""
    out_path = tmp_path / "out.json"
    code = run(
        [
            "--record",
            json.dumps(sample_record),
            "--run-dir",
            str(run_dir),
            "--posture",
            "loss-averse (1:8)",
            "--friction",
            "approve-first (4:1)",
            "--out",
            str(out_path),
        ]
    )
    assert code == 0
    result = json.loads(out_path.read_text())
    assert result["posture"] == "loss-averse (1:8)"
    assert result["friction_posture"] == "approve-first (4:1)"


def test_full_track_warns_that_the_postures_cannot_move_anything(
    run_dir, sample_record, tmp_path, capsys
):
    """Running the demo on `full` under two postures and getting identical
    output is the headline finding, not a broken flag — so the CLI says so,
    the same way src/evaluate.py banners a degenerate sweep. On stderr, so it
    never contaminates the JSON on stdout."""
    run(
        [
            "--record",
            json.dumps(sample_record),
            "--track",
            "full",
            "--run-dir",
            str(run_dir),
            "--out",
            str(tmp_path / "out.json"),
        ]
    )
    assert "LEAKAGE_FINDING" in capsys.readouterr().err
