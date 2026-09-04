# Evaluation

Every number below is read directly from `runs/evaluation_{full,testbed}.json`,
`runs/model_{full,testbed}.json`, and the confusion matrices recomputed from
`runs/model_{track}_{proba,ytest}.npy` (`sklearn.metrics.confusion_matrix`,
same test set, same row order `src/evaluate.py` already relies on) —
reproduced by running `python -m src.evaluate` (and, for the confusion
matrices, the one-liner in each track's section below), not hand-typed. Every
number carries the track that produced it, without exception — a figure
reported without its track is not interpretable. **Accuracy is never the headline** — see "Why accuracy is
rejected" below before any other number in this document.

Two tracks, defined in `src/features.py::FEATURE_SETS`:

| track | what it is |
|---|---|
| `full` | **the honest model** — every legitimate feature the dataset ships |
| `testbed` | **not a model** — ablation rung G, kept only to give the decision layer a non-degenerate region to demonstrate on |

`testbed` is never the headline result. See "Task-triviality caveat" below
for why.

---

## Why accuracy is rejected

| | accuracy | macro-F1 |
|---|---|---|
| Always-predict-Legitimate strawman | 0.6954 | 0.2051 |

70.1% of the test set is `Legitimate`. A model that never looks at a single
feature and always predicts `Legitimate` scores 69.5% accuracy while
correctly identifying zero abuse of any kind — 0.2051 macro-F1 is what that
uselessness actually looks like. Every model number below is reported
alongside its macro-F1 and full per-class breakdown for exactly this reason.

---

## `full` track — per-class metrics

**macro-F1 = 0.9988, accuracy = 0.9995.**

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| Fraudulent Return | 0.9982 | 0.9991 | 0.9987 | 1,112 |
| Legitimate | 1.0000 | 1.0000 | 1.0000 | 8,345 |
| Policy Abuser | 0.9993 | 0.9965 | 0.9979 | 1,414 |
| Wardrobing | 0.9973 | 1.0000 | 0.9987 | 1,129 |

**This is not a result to be proud of.** See "Task-triviality caveat" below —
0.9988 macro-F1 on this dataset is a leakage signal, and the project's
central finding is explaining why, not reporting the number as an
achievement.

### Confusion matrix

```
predicted ->        Fraudulent  Legitimate  Policy Abuser  Wardrobing
Fraudulent Return         1111           0              1           0
Legitimate                   0        8345              0           0
Policy Abuser                 2           0           1409           3
Wardrobing                    0           0              0        1129
```

6 misclassified rows out of 12,000. Weakest boundary (still essentially
noise): **Policy Abuser → Wardrobing, 3 cases.** Chart: `runs/confusion_full.png`.
PR curves (all four classes saturate near the top-left corner):
`runs/pr_curves_full.png`.

### Both cost axes

**`C_fp : C_fn` (block-an-honest-customer vs. let-fraud-through):**

| posture | ratio | false-block on legit | fraud hard-blocked |
|---|---|---|---|
| loss-neutral (1:1) | 1.0× | 0.00% | 100.00% |
| retention-weighted (8:1) | 8.0× | 0.00% | 100.00% |
| loss-averse (1:8) | 0.125× | 0.00% | 100.00% |

Swept continuously from 0.03× to 32×: false-block stays 0.00%→0.00%,
fraud-hard-blocked stays 100.00%→100.00%. **Byte-identical decisions across
the entire posture range.** `src/evaluate.py::sweep_is_degenerate` fires:
`cost_sweep_is_degenerate = true`. This is not a bug — with 0.04% of rows
threshold-sensitive (`threshold_sensitive_frac = 0.0004`), there is
essentially no probability mass near any decision boundary for a cost
posture to act on. It is reported as positive evidence of how degenerate
the task is on this feature set.

**Approve ↔ soft-friction (the axis `docs/LEAKAGE_FINDING.md`'s second
finding identifies as the real one):** also flat on `full` — legitimate
customers frictioned stays 0.00%, abusers caught stays 100.00%, for the same
reason. `full` has no live axis at all; the friction axis only becomes
interesting on `testbed` below.

---

## `testbed` track — per-class metrics

**macro-F1 = 0.8967, accuracy = 0.9297.** (Rung G of the ablation ladder —
never presented as a model; see docs/ARCHITECTURE.md §5.2, "Dual-track design.")

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| Fraudulent Return | 0.9917 | 0.9676 | 0.9795 | 1,112 |
| Legitimate | 0.9759 | 0.9409 | 0.9581 | 8,345 |
| Policy Abuser | 0.7775 | 0.8600 | 0.8167 | 1,414 |
| Wardrobing | 0.7762 | 0.8973 | 0.8324 | 1,129 |

Policy Abuser and Wardrobing are visibly the hardest classes to precision-
separate (0.78 precision each) — both are recall-favored relative to
precision, meaning the model over-predicts them relative to how often
they're actually the right answer, at the expense of Legitimate's recall
(0.9409, the lowest of any class on this track).

### Confusion matrix

```
predicted ->        Fraudulent  Legitimate  Policy Abuser  Wardrobing
Fraudulent Return          1076          13             10          13
Legitimate                    8        7852            274         211
Policy Abuser                  1         129           1216          68
Wardrobing                     0          52             64        1013
```

| actual → predicted | count | share of actual class |
|---|---|---|
| Legitimate → Policy Abuser | 274 | 3.3% of 8,345 Legitimate |
| Legitimate → Wardrobing | 211 | 2.5% |
| Policy Abuser → Legitimate | 129 | 9.1% of 1,414 Policy Abuser |
| Policy Abuser → Wardrobing | 68 | 4.8% |
| Wardrobing → Policy Abuser | 64 | 5.7% of 1,129 Wardrobing |
| Wardrobing → Legitimate | 52 | 4.6% |
| *(Fraudulent Return, any)* | 36 total | 3.2% of 1,112 — still the cleanest class |

**Weakest boundary: Legitimate → Policy Abuser, 274 cases** — see "Failure-mode
disclosure" below for the mechanistic explanation, not just the count. Chart:
`runs/confusion_testbed.png`. PR curves (Policy Abuser and Wardrobing visibly
lower and further right than the other two): `runs/pr_curves_testbed.png`.

### Both cost axes

**`C_fp : C_fn`:** nearly inert even here.

| posture | ratio | false-block on legit | fraud hard-blocked |
|---|---|---|---|
| loss-neutral (1:1) | 1.0× | 0.05% | 97.21% |
| retention-weighted (8:1) | 8.0× | 0.00% | 96.85% |
| loss-averse (1:8) | 0.125× | 0.16% | 97.48% |

Continuous sweep (0.03×–32×): false-block on legitimate customers spans only
**0.00% → 0.23%**; fraud hard-blocked spans only **96.76% → 97.39%**. The
reason: of 592 test rows with a top-two probability margin below 0.3 (the
genuinely ambiguous ones), only 14 involve Fraudulent Return at all — 342 are
Legitimate vs. Policy Abuser. **Fraudulent Return is the easiest class to
separate**, so the hard-block cost parameter has almost nothing left to act
on.

**Approve ↔ soft-friction — the live axis:**

| | span across the posture range |
|---|---|
| legitimate customers given friction | **2.78% → 24.79%** |
| wardrobers / policy abusers caught | **85.49% → 99.57%** |

An order of magnitude of real movement, on the axis a merchant would
actually argue over — how aggressively to fee/flag customers who might just
be heavy returners, not whether to hard-block them. Chart:
`runs/friction_tradeoff_testbed.png`. This is the cost-calibration
centerpiece; the flat `C_fp:C_fn` axis above is reported alongside it because
showing which axis *doesn't* move is part of the honest account, not
something to omit once a live axis was found.

---

## Failure-mode disclosure

**Measured, not predicted.** `docs/ARCHITECTURE.md` §6.3 expected wardrobing
vs. policy-abuse confusion going in; what was actually measured on `testbed`
is Legitimate ↔ Policy Abuser (403 rows combined), with Wardrobing confusion
close behind (211 + 64 = 275 rows touching Wardrobing). The mechanism, from
`runs/shap_interpretation.md`:

- On `full`, SHAP's top feature per class is 1–2 of the same near-disjoint
  generator artifacts (`days_to_return`, `wishlist_to_cart_time_hrs`,
  `return_rate_pct`) `docs/LEAKAGE_FINDING.md` already identified by greedy
  forward selection — which is why `full`'s confusion matrix is nearly the
  identity. This corroborates the leakage finding; it is not a separate,
  positive result about the model's understanding of abuse.
- On `testbed`, with those features removed, Legitimate, Policy Abuser, and
  Wardrobing's top SHAP drivers *overlap* — `refund_to_avg_order_ratio` is
  top-3 for all three classes, `total_orders_lifetime` for two of three.
  These are continuous signals the whole customer population shares to some
  degree, with no single threshold that cleanly separates the three, unlike
  `full`'s disjoint generator ranges. That overlap is the confusion.

**Segment check.** `runs/segment_fpr_audit.md`: the `testbed` soft-friction
rate is not evenly spread across order-value segments (12.31% in the
second-lowest quartile vs. 5.04% in the highest, 2.44×) — but in the
direction that protects high-value legitimate customers, not the direction
`docs/ARCHITECTURE.md` §1 opens by warning against. `hard_block_fpr` stays at
noise level in every segment on both tracks.

**SMOTE was tested as a remedy and discarded.** `runs/smote_verdict.md`:
SMOTENC on `testbed` scored *below* the class-weighted baseline (0.8944 vs.
0.8967 macro-F1), with Wardrobing recall dropping 3.6 points and no
offsetting gain — the confusion above is not an imbalance artifact that
resampling fixes.

---

## Task-triviality caveat (verbatim from `docs/LEAKAGE_FINDING.md`)

> The gradient-boosted model is not detecting return abuse. It is
> **reverse-engineering a rule-based synthetic data generator.** The 0.998 is
> a measure of how cleanly the generator drew its class boundaries, not of
> how well the method would find abuse in merchant data, where these
> distributions overlap heavily and no threshold on "hours from wishlist to
> cart" cleanly separates a wardrober from an honest customer.
>
> Reporting 0.998 as a headline result — even with the standard
> synthetic-data caveat attached — would be misleading. The caveat in
> ARCHITECTURE.md §2 was written to cover "absolute numbers may not
> transfer." This is a stronger problem: the task as posed is close to
> trivial, and any model at all will score near-perfectly on it.
>
> This is the finding the project should lead with, not bury.

Four hand-written `if/else` rules, with **zero training**, score 0.9425
accuracy / 0.9188 macro-F1 — read straight off the per-class range tables in
`docs/LEAKAGE_FINDING.md`. `full`'s 0.9988 is not a large improvement over a
model with no learning in it at all.

## Synthetic-data caveat (verbatim from `docs/ARCHITECTURE.md` §2)

> This dataset is fully synthetic. It is useful for building and honestly
> evaluating a modeling *methodology*, but any absolute performance numbers
> should be presented as a proof of concept against synthetic ground truth,
> not as a claim about real-world merchant data.

Nothing in this document should be read as a claim about real merchant
return data. It is a claim about this methodology, measured honestly against
one synthetic dataset — including the parts where that measurement came out
worse than the plan expected.

---

## Untuned-model caveat

**Every number in this document comes from an untuned model.** There is no
validation split, no hyperparameter search and no early stopping; both tracks
use `N_ESTIMATORS = 400`, `LEARNING_RATE = 0.05`, fixed in `src/model.py` and
never moved. The reasoning is in `docs/ARCHITECTURE.md` §5.3, and the short
version is that a model at 0.9988 macro-F1 with 99.9% of predictions above
p=0.99 offers nothing to tune toward except a tighter fit to the synthetic
generator — which is the behaviour this project's headline finding argues
against rewarding.

So these are not "the best LightGBM can do on this data," and are not offered
as such. They are what a reasonable default does. On a non-degenerate dataset
that gap would be the first thing to close, with a three-way
train/validation/test split and the test set touched once.

---

## Reproduce this document

```bash
python -m src.model             # both tracks -> runs/model_{full,testbed}.{json,joblib,...}
python -m src.evaluate          # -> runs/evaluation_{full,testbed}.json, confusion/PR-curve/cost-sweep charts
python -m src.explain           # -> runs/shap_{full,testbed}.{json,png}, runs/shap_interpretation.md
python -m src.smote_experiment  # -> runs/smote_testbed.json, runs/smote_verdict.md
python -m src.segment_audit     # -> runs/segment_fpr_{full,testbed}.{json,png}, runs/segment_fpr_audit.md
```

`RANDOM_STATE = 42` throughout — every number above reproduces exactly (float
noise at ~1e-15 in cost-sweep CSVs aside; confirmed by retraining during this
document's own preparation).
