"""
Tests for the leakage tripwire this project's whole finding rests on
(docs/LEAKAGE_FINDING.md). Two independent lines of defense:

  1. `src/data_gate.py::leakage_sweep` must actually catch a 1:1-encoded
     label at the corrected depth (`max(2, n_classes - 1)`) -- and, since
     the corrected depth exists specifically because depth-1 missed this
     exact bug, the test also demonstrates that failure directly rather
     than just asserting the fix.
  2. Even if the sweep were ever bypassed, `abuse_label` (and the rest of
     DROP_COLS) must never reach the modelling frame `src/features.py`
     produces.

Runs entirely on an in-memory fixture; never touches the Kaggle CSV.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_gate import LEAKAGE_MACRO_F1_THRESHOLD, leakage_sweep
from src.features import DROP_COLS, build_and_split

CLASSES = ["Legitimate", "Wardrobing", "Policy Abuser", "Fraudulent Return"]
N_PER_CLASS = 6  # >= 3 per class so 3-fold CV gets a full fold of every class


@pytest.fixture
def leaky_df():
    """`abuse_label` is a 1:1 encoding of `abuse_type` -- the actual
    leak. `innocuous_feature` is pure noise, a negative control. `return_date`
    is included only so build_and_split (which requires it) can run."""
    rng = np.random.default_rng(0)
    rows = []
    for label_idx, cls in enumerate(CLASSES):
        for _ in range(N_PER_CLASS):
            rows.append({"abuse_type": cls, "abuse_label": label_idx})
    df = pd.DataFrame(rows)
    df["innocuous_feature"] = rng.normal(size=len(df))
    df["return_date"] = pd.date_range("2022-01-01", periods=len(df), freq="D").astype(str)
    return df


def test_leakage_sweep_flags_a_1to1_encoded_label(leaky_df):
    """The exact gate bug: a column that perfectly determines the label must
    be flagged a suspect at the corrected depth (n_classes - 1)."""
    result = leakage_sweep(leaky_df, target_col="abuse_type")
    suspects = set(result["suspects"]["feature"])
    assert "abuse_label" in suspects

    row = result["single_feature_f1"].set_index("feature").loc["abuse_label"]
    assert row["single_feature_macro_f1"] >= LEAKAGE_MACRO_F1_THRESHOLD


def test_depth1_gate_would_have_missed_the_same_label(leaky_df):
    """Demonstrates the failure this project corrected: at depth 1 (as
    ARCHITECTURE.md §9.1 originally specified), one split has two leaves and
    cannot address all 4 classes, so even a perfect 1:1 label encoding scores
    far below the leakage threshold. The corrected sweep_depth is what
    actually catches it (previous test) -- this is why depth-1 alone is not
    a valid gate on this target."""
    result = leakage_sweep(leaky_df, target_col="abuse_type")
    row = result["single_feature_f1"].set_index("feature").loc["abuse_label"]
    assert row["depth1_macro_f1"] < LEAKAGE_MACRO_F1_THRESHOLD
    assert row["depth1_macro_f1"] < row["single_feature_macro_f1"]


def test_innocuous_feature_is_not_flagged(leaky_df):
    """Negative control: a feature carrying no signal must not be a false
    positive, or the gate would be alarming on noise."""
    result = leakage_sweep(leaky_df, target_col="abuse_type")
    suspects = set(result["suspects"]["feature"])
    assert "innocuous_feature" not in suspects


def test_abuse_label_absent_from_feature_matrix(tmp_path, leaky_df):
    """The other half of the tripwire: even if leakage_sweep were somehow
    bypassed, abuse_label (and the rest of DROP_COLS) must never reach the
    frame src/model.py trains on."""
    csv = tmp_path / "returns.csv"
    leaky_df.to_csv(csv, index=False)
    train, test = build_and_split(csv)
    for frame in (train, test):
        for col in DROP_COLS:
            assert col not in frame.columns
        assert "abuse_type" in frame.columns
