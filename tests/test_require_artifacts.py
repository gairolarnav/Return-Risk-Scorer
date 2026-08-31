"""
Tests for src.data_gate.require_artifacts — the guard every pipeline stage runs
before it touches a file.

Why this is worth guarding. Every stage after src.data_gate reads something an
earlier stage wrote, and most of those artifacts are gitignored by design (the
raw CSV, the parquet splits, the model bundles). "It isn't there yet" is
therefore the single most likely thing to happen to a first-time reader, and it
used to surface as a bare FileNotFoundError from inside pandas or numpy in seven
of eight entry points.

src/evaluate.py had the sharpest version of the bug: it checked for
runs/model_{track}.json, which IS committed, and then called np.load on the
gitignored _proba.npy — so the helpful branch could never execute. The last test
here pins that specific shape so a future edit cannot reintroduce it.

Runs entirely on tmp_path; never touches the real runs/ or data/ directories.
"""

from __future__ import annotations

import pytest

from src.data_gate import (
    NEEDS_FEATURES,
    NEEDS_MODEL,
    NEEDS_RAW_CSV,
    require_artifacts,
)


def test_present_artifacts_pass_through_silently(tmp_path, capsys):
    """The happy path must not print or exit — the guard is on every stage's
    hot path, so a spurious message would be noise on every successful run."""
    present = tmp_path / "here.parquet"
    present.write_text("x")

    require_artifacts([present], NEEDS_FEATURES)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_missing_artifact_exits_1_instead_of_raising_filenotfound(tmp_path):
    """SystemExit(1), not a traceback. A reviewer's first run of this repo is
    the most likely place to hit a missing input, and a stack trace there reads
    as a broken project rather than an unmet prerequisite."""
    with pytest.raises(SystemExit) as excinfo:
        require_artifacts([tmp_path / "absent.parquet"], NEEDS_FEATURES)
    assert excinfo.value.code == 1


def test_message_names_the_command_that_produces_the_artifact(tmp_path, capsys):
    """The message has to be actionable: a reader who has never seen this repo
    must learn what to run next, not merely which path was empty."""
    with pytest.raises(SystemExit):
        require_artifacts([tmp_path / "train.parquet"], NEEDS_FEATURES)
    err = capsys.readouterr().err
    assert "train.parquet" in err
    assert "python -m src.features" in err


def test_all_missing_paths_are_listed_not_just_the_first(tmp_path, capsys):
    """Checked as a set rather than one at a time, so someone who has run
    nothing yet learns everything they need in one pass instead of rediscovering
    it one failed command at a time."""
    with pytest.raises(SystemExit):
        require_artifacts(
            [tmp_path / "a.npy", tmp_path / "b.npy", tmp_path / "c.joblib"], NEEDS_MODEL
        )
    err = capsys.readouterr().err
    assert "a.npy" in err and "b.npy" in err and "c.joblib" in err
    assert "3 required input(s)" in err


def test_a_present_file_does_not_mask_a_missing_sibling(tmp_path, capsys):
    """The src/evaluate.py bug, pinned.

    That stage guarded on model_{track}.json — a committed file, so the check
    always passed — and then loaded the gitignored _proba.npy, which is what
    actually blew up. Any guard listing a mix of committed and generated
    artifacts must fail on the generated one rather than be satisfied by the
    committed one.
    """
    committed = tmp_path / "model_full.json"
    committed.write_text("{}")
    generated = tmp_path / "model_full_proba.npy"

    with pytest.raises(SystemExit):
        require_artifacts([committed, generated], NEEDS_MODEL)

    err = capsys.readouterr().err
    assert "model_full_proba.npy" in err
    assert "1 required input(s)" in err
    # The committed file is not reported missing — only the real gap is.
    assert "model_full.json" not in err


def test_raw_csv_remedy_points_at_the_dataset_not_a_pipeline_command(tmp_path, capsys):
    """The raw CSV is the one prerequisite no command in this repo can produce,
    so its remedy has to be the download link rather than a `python -m` line."""
    with pytest.raises(SystemExit):
        require_artifacts([tmp_path / "returns.csv"], NEEDS_RAW_CSV)
    err = capsys.readouterr().err
    assert "kaggle.com" in err
    assert "python -m src." not in err
