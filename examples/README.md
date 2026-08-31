# Example inputs for `scripts/score.py`

Two files, so the scoring CLI runs immediately after `git clone` and
`pip install -r requirements.txt` — no Kaggle download, no training run.
`runs/model_full.joblib` is committed for the same reason.

```bash
# one record -> class probabilities + routed intervention
python -m scripts.score --record-file examples/record.json

# a batch -> one scored row per input row, input columns preserved
python -m scripts.score --csv examples/sample_returns.csv --out scored.csv

# the axis that actually moves (needs --track testbed; see the note below)
python -m scripts.score --csv examples/sample_returns.csv --track testbed \
    --friction "recovery-first (1:20)"
```

## What these are

| file | contents |
|---|---|
| `record.json` | One return record, raw schema, 33 columns. A true `Policy Abuser` — the class on the Legitimate/Policy-Abuser boundary that `docs/EVALUATION.md` reports as the model's weakest. |
| `sample_returns.csv` | 20 records, 5 per class, in `return_date` order. `abuse_type` is kept so you can check the predictions against ground truth; `scripts/score.py` ignores it. |

Both are verbatim rows from the Kaggle dataset the project is built on
([E-Commerce Return Abuse Detection](https://www.kaggle.com/datasets/sarveshchhetri/e-commerce-return-abuse-detection-dataset),
synthetic, 60,000 × 35), drawn **from the test window only** — the last 20% by
`return_date` — so they are records the committed `full` bundle never trained
on. `abuse_label` is dropped, since it is a 1:1 encoding of the target and
`src/features.py::DROP_COLS` quarantines it everywhere else.

## Two things you will notice

**`--track full` ignores both posture flags.** Every `--posture` and
`--friction` value returns identical actions on that track. That is the
project's headline finding rather than a broken flag — the model puts 99.9% of
its probability mass above p=0.99, so no row sits near a decision boundary for
a cost matrix to move. The CLI says so on stderr. `docs/LEAKAGE_FINDING.md` has
the evidence.

**`--track testbed` needs a training run.** Only the `full` bundle is
committed; `runs/model_testbed.joblib` is not, because the testbed is a
diagnostic ablation rung rather than a model (`docs/ARCHITECTURE.md` §5.2) and
it is the larger artifact. Run `python -m src.model testbed` after placing the
Kaggle CSV to get it.
