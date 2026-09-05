# Return-Risk Scorer

[![CI](https://github.com/gairolarnav/Return-Risk-Scorer/actions/workflows/ci.yml/badge.svg)](https://github.com/gairolarnav/Return-Risk-Scorer/actions/workflows/ci.yml)

Multiclass return-risk scorer — **legitimate / wardrobing / policy abuse /
fraudulent return** .
Most return-fraud tooling collapses these into one binary `is_fraud` flag; this
scores the abuse *type*, so the merchant's response is proportionate — approve,
soft friction, or hard block. Strictly **defense-only**: it scores and routes,
never executing an irreversible action on a customer account.

## The headline result is a negative one

The dataset is synthetic and **degenerate** — each class is drawn from
near-disjoint bounded ranges. Four hand-written `if/else` rules, with zero
training, score 0.9188 macro-F1:

| | accuracy | macro-F1 |
|---|---|---|
| Always-predict-Legitimate strawman | 0.6954 | 0.2051 |
| Four hand-written rules, no model at all | 0.9425 | 0.9188 |
| LightGBM, all features | 0.9995 | **0.9988** |

The sequence is the argument: a model at 0.9988 here is reverse-engineering the
generator, not detecting abuse — evidence in
**[`docs/LEAKAGE_FINDING.md`](docs/LEAKAGE_FINDING.md)**. Hence two tracks:
`full` is the model, and `testbed` is a handicapped diagnostic rung that gives
the cost-calibrated decision layer something non-degenerate to work on, never
reported as a result ([architecture §5.2](docs/ARCHITECTURE.md)).

## Quickstart

```bash
git clone https://github.com/gairolarnav/Return-Risk-Scorer.git
cd Return-Risk-Scorer
git config core.hooksPath .githooks   # local config; does not survive a clone

python3.11 -m venv venv && source venv/bin/activate
brew install libomp                   # Apple Silicon: LightGBM fails at import without it
pip install -r requirements.txt

# scoring needs no dataset and no training run — the model bundle is committed
python -m scripts.score --record-file examples/record.json --track full
python -m scripts.score --csv examples/sample_returns.csv --out scored.csv
```

`pytest -q` (139 tests) and `ruff check src/ scripts/ tests/` run on committed
fixtures alone. The full pipeline needs the Kaggle
[E-Commerce Return Abuse Detection](https://www.kaggle.com/datasets/sarveshchhetri/e-commerce-return-abuse-detection-dataset)
CSV at `data/raw/returns.csv` — commands in [architecture §8.2](docs/ARCHITECTURE.md).

## Documentation

**[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) is the full design** — feature
plan, dual-track build, evaluation framework and cost policy, repository
layout, test plan, status checklist (§10, two items open), and a correction log
of every place the plan was wrong (§11). Then:
[`LEAKAGE_FINDING.md`](docs/LEAKAGE_FINDING.md) (the headline result),
[`EVALUATION.md`](docs/EVALUATION.md) (per-class metrics, both cost axes,
failure-mode disclosure), [`DATA_NOTES.md`](docs/DATA_NOTES.md) (the data gate).

## Limitations

The dataset is fully synthetic and close to trivially separable. Every absolute
number here is a proof of concept against synthetic ground truth, not a claim
about real merchant data — where these distributions overlap heavily and no
threshold cleanly separates a wardrober from an honest customer.
