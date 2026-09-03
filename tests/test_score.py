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


def test_record_matching_no_features_exits_cleanly_not_with_a_traceback(run_dir, capsys):
    """A payload with the wrong schema is a *caller* error, and must read like
    one. src/infer.py::prepare_frame already raises a ValueError naming the
    track, the columns received and the ones expected; the CLI's job is to
    present it as a single line, the same way every other bad-input path here
    does, rather than letting it escape as a Python traceback that looks like
    the tool crashed."""
    with pytest.raises(SystemExit) as excinfo:
        run(["--record", json.dumps({"foo": 1}), "--track", "full", "--run-dir", str(run_dir)])

    message = str(excinfo.value)
    assert "supplies none of" in message
    assert "feature_cols" in message
    # The posture NOTE is a caveat on a recommendation. There is no
    # recommendation here, so it must not have been printed above the error.
    assert "LEAKAGE_FINDING" not in capsys.readouterr().err


def test_malformed_json_is_not_preceded_by_the_posture_note(run_dir, capsys):
    """Ordering matters for the first thing a reviewer sees. Before this, a
    typo'd --record on --track full printed a paragraph about cost postures
    *above* the actual parse error, burying the only line that says what went
    wrong."""
    with pytest.raises(SystemExit):
        run(["--record", "{oops}", "--track", "full", "--run-dir", str(run_dir)])

    assert capsys.readouterr().err == ""


# --- audit findings F1, F2, F3 ------------------------------------------------
#
# All three were the same shape: an ordinary user mistake reaching a library and
# failing there, so the traceback named pandas or LightGBM internals instead of
# what the caller did wrong. The repo argues elsewhere that a caller error must
# read like one; these pin that down at the boundary.


def test_non_numeric_value_scores_and_names_the_field(run_dir, sample_record, tmp_path):
    """F1. A currency column holding "N/A" used to raise
    `TypeError: unsupported operand type(s) for /: 'str' and 'str'` from inside
    pandas, because feature engineering divides those columns before anything
    validates them. It now coerces to NaN -- one bad cell must not take a whole
    batch down -- and says which field it lost."""
    record = dict(sample_record, avg_order_value_usd="N/A")
    out_path = tmp_path / "out.json"

    code = run(["--record", json.dumps(record), "--run-dir", str(run_dir), "--out", str(out_path)])

    assert code == 0
    result = json.loads(out_path.read_text())
    assert result["features_invalid"] == ["avg_order_value_usd"]
    assert result["recommended_action"] in ACTIONS  # still a usable recommendation


def test_clean_record_reports_no_invalid_fields(run_dir, sample_record, tmp_path):
    """The coercion must not fire on well-formed input. If this ever fails, a
    published number was computed on a silently altered frame."""
    out_path = tmp_path / "out.json"
    run(["--record", json.dumps(sample_record), "--run-dir", str(run_dir), "--out", str(out_path)])
    assert json.loads(out_path.read_text())["features_invalid"] == []


def test_batch_counts_unusable_values_per_row(run_dir, tmp_path):
    """F1 through the batch path, which is where a real spreadsheet export --
    the actual source of junk in a numeric column -- would arrive.

    Note the value: "unknown", not "N/A". `pd.read_csv` already maps "N/A",
    "NA", "null" and "" to NaN through its default `na_values`, so those never
    reached the division. The strings that did are the ones pandas has no
    opinion about, which is most of what a human types into a spreadsheet."""
    batch = pd.DataFrame(
        [
            {
                "avg_order_value_usd": "unknown" if i == 0 else 80.0,
                "refund_amount_requested_usd": 20.0,
                "total_orders_lifetime": 5,
                "total_returns_lifetime": 1,
                "account_age_days": 200,
                "days_to_return": 15,
                "product_category": "Home",
                "return_date": "2022-05-01",
            }
            for i in range(3)
        ]
    )
    csv_path = tmp_path / "batch.csv"
    out_path = tmp_path / "scored.csv"
    batch.to_csv(csv_path, index=False)

    code = run(["--csv", str(csv_path), "--run-dir", str(run_dir), "--out", str(out_path)])

    assert code == 0
    scored = pd.read_csv(out_path)
    assert len(scored) == 3, "one bad cell must not drop rows"
    # Per row, not per frame. This asserted `.all() == 1` when the count was a
    # frame-level constant repeated onto every row -- which reported rows 1 and
    # 2 as damaged because row 0 was, the same class of dishonesty as F6.
    assert scored["n_features_invalid"].tolist() == [1, 0, 0]


def test_missing_csv_path_exits_cleanly(run_dir, tmp_path):
    """F2."""
    missing = tmp_path / "no_such.csv"
    with pytest.raises(SystemExit) as excinfo:
        run(["--csv", str(missing), "--run-dir", str(run_dir)])
    assert str(missing) in str(excinfo.value)


def test_missing_record_file_path_exits_cleanly(run_dir, tmp_path):
    """F2. `_load_bundle` guarded the model path; the input paths did not, so a
    typo -- the likeliest mistake there is -- produced a stack trace."""
    missing = tmp_path / "no_such.json"
    with pytest.raises(SystemExit) as excinfo:
        run(["--record-file", str(missing), "--run-dir", str(run_dir)])
    assert str(missing) in str(excinfo.value)


def test_empty_csv_exits_cleanly(run_dir, tmp_path):
    """F3. A zero-byte file raised pandas' EmptyDataError verbatim."""
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(SystemExit) as excinfo:
        run(["--csv", str(empty), "--run-dir", str(run_dir)])
    assert "empty" in str(excinfo.value).lower()


# --- audit findings F5, F6 ----------------------------------------------------


def test_impossible_values_are_discarded_and_named(run_dir, sample_record, tmp_path):
    """F5. A negative refund used to produce "Fraudulent Return" / hard_block
    with p ~= 1: the most punitive action available, computed from a number that
    cannot exist. Impossible input must not drive a confident recommendation."""
    record = dict(
        sample_record,
        avg_order_value_usd=-500.0,
        refund_amount_requested_usd=-9999.0,
        account_age_days=-1,
    )
    out_path = tmp_path / "out.json"

    code = run(["--record", json.dumps(record), "--run-dir", str(run_dir), "--out", str(out_path)])

    assert code == 0
    result = json.loads(out_path.read_text())
    assert set(result["features_out_of_range"]) == {
        "avg_order_value_usd",
        "refund_amount_requested_usd",
        "account_age_days",
    }
    assert result["recommended_action"] != "hard_block", (
        "impossible input must not route to the harshest action"
    )


def test_zero_denominator_is_reported_not_silently_dropped(run_dir, sample_record, tmp_path):
    """F6. src/features.py nulls a zero denominator, so avg_order_value_usd=0
    yields refund_to_avg_order_ratio=NaN. The result still said
    `features_missing: []`, claiming full data on a record scored without it."""
    record = dict(sample_record, avg_order_value_usd=0.0, total_orders_lifetime=0)
    out_path = tmp_path / "out.json"

    run(["--record", json.dumps(record), "--run-dir", str(run_dir), "--out", str(out_path)])

    result = json.loads(out_path.read_text())
    assert "refund_to_avg_order_ratio" in result["features_degraded"]
    assert "returns_per_order" in result["features_degraded"]
    assert result["features_out_of_range"] == [], "0 is a legal value, not out of range"
    assert result["n_features_not_seen"] >= len(result["features_degraded"])


def test_clean_record_reports_nothing_unusable(run_dir, sample_record, tmp_path):
    """The counterpart that makes the three above meaningful: on well-formed
    input every one of these fields must stay empty. If this fails, the
    validation is firing on real data and the published numbers are suspect."""
    out_path = tmp_path / "out.json"
    run(["--record", json.dumps(sample_record), "--run-dir", str(run_dir), "--out", str(out_path)])
    result = json.loads(out_path.read_text())
    assert result["features_out_of_range"] == []
    assert result["features_degraded"] == []
    assert result["features_invalid"] == []


def test_batch_counts_are_per_row_not_frame_level(run_dir, tmp_path):
    """F6, batch shape. Row 0 carrying an impossible value says nothing about
    row 1, and a count that claims otherwise is the defect being fixed."""
    batch = pd.DataFrame(
        [
            {
                "avg_order_value_usd": -80.0 if i == 0 else 80.0,
                "refund_amount_requested_usd": 20.0,
                "total_orders_lifetime": 5,
                "total_returns_lifetime": 1,
                "account_age_days": 200,
                "days_to_return": 15,
                "product_category": "Home",
                "return_date": "2022-05-01",
            }
            for i in range(3)
        ]
    )
    csv_path = tmp_path / "b.csv"
    out_path = tmp_path / "s.csv"
    batch.to_csv(csv_path, index=False)

    run(["--csv", str(csv_path), "--run-dir", str(run_dir), "--out", str(out_path)])

    scored = pd.read_csv(out_path)
    assert scored["n_features_out_of_range"].tolist() == [1, 0, 0]
    assert scored["n_features_not_seen"].iloc[0] > scored["n_features_not_seen"].iloc[1]


# --- audit findings F7, F8 ----------------------------------------------------


def test_out_refuses_to_replace_an_existing_file(run_dir, sample_record, tmp_path):
    """F7. A scored batch is evidence someone may be working from. Re-running
    with a different --track or --friction and silently replacing it is how the
    wrong numbers reach a report."""
    out_path = tmp_path / "out.json"
    out_path.write_text("previous results")

    with pytest.raises(SystemExit) as excinfo:
        run(["--record", json.dumps(sample_record), "--run-dir", str(run_dir), "--out", str(out_path)])

    assert "--overwrite" in str(excinfo.value)
    assert out_path.read_text() == "previous results", "the existing file must survive"


def test_overwrite_flag_allows_replacing(run_dir, sample_record, tmp_path):
    out_path = tmp_path / "out.json"
    out_path.write_text("previous results")

    code = run(
        [
            "--record",
            json.dumps(sample_record),
            "--run-dir",
            str(run_dir),
            "--out",
            str(out_path),
            "--overwrite",
        ]
    )

    assert code == 0
    assert json.loads(out_path.read_text())["most_likely_class"] in CLASSES


def test_explain_is_refused_on_a_batch(run_dir, tmp_path):
    """F8. Supported on one record only -- a batch would rebuild the SHAP
    explainer per row. Refusing beats quietly taking minutes."""
    csv_path = tmp_path / "b.csv"
    pd.DataFrame([{"avg_order_value_usd": 80.0, "return_date": "2022-05-01"}]).to_csv(
        csv_path, index=False
    )
    with pytest.raises(SystemExit) as excinfo:
        run(["--csv", str(csv_path), "--run-dir", str(run_dir), "--explain"])
    assert "single record" in str(excinfo.value)


def test_result_carries_no_explanation_unless_asked(run_dir, sample_record, tmp_path):
    """The default path must not import or run shap."""
    out_path = tmp_path / "out.json"
    run(["--record", json.dumps(sample_record), "--run-dir", str(run_dir), "--out", str(out_path)])
    assert "explanation" not in json.loads(out_path.read_text())
