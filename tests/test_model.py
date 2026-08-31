"""
Determinism test for src.model.train_track (underwrites the reproducibility
claim in docs/ARCHITECTURE.md §8.1: RANDOM_STATE fixed everywhere means the
same input produces identical output, not merely "close" output, across
repeated runs).

Runs on a small in-memory fixture; never touches the Kaggle CSV.
"""

import numpy as np
import pandas as pd

from src.model import train_track

CLASSES = ["Legitimate", "Wardrobing", "Policy Abuser", "Fraudulent Return"]
N_PER_CLASS = 15


def _synthetic_frame(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for label_idx, cls in enumerate(CLASSES):
        for _ in range(N_PER_CLASS):
            rows.append(
                {
                    "abuse_type": cls,
                    # feature values correlated with the class so the model
                    # has something real to learn, not pure noise
                    "feat_a": rng.normal(loc=label_idx, scale=0.5),
                    "feat_b": rng.normal(loc=-label_idx, scale=0.5),
                }
            )
    return pd.DataFrame(rows)


def test_train_track_is_deterministic_given_the_same_random_state():
    """Same train/test frames, same track, called twice -> bit-identical
    predicted probabilities. If this ever fails, RANDOM_STATE has stopped
    actually pinning every source of randomness in the training path."""
    train = _synthetic_frame(seed=0)
    test = _synthetic_frame(seed=1)

    result_a = train_track(train.copy(), test.copy(), track="full")
    result_b = train_track(train.copy(), test.copy(), track="full")

    np.testing.assert_array_equal(result_a["proba"], result_b["proba"])
    assert result_a["metrics"]["macro_f1"] == result_b["metrics"]["macro_f1"]
