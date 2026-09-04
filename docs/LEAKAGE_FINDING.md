# Leakage Finding

**Status: RESOLVED — dual-track build adopted (see "Decision" below).**
Triggered the §6.4 suspiciously-good-result protocol.
Recorded here in full because it is the single most important honest-metrics
result the project has, and because "our model scores 0.998" with no account
of why is the easiest claim on earth for a panelist to dismantle.

---

## What happened

The first end-to-end LightGBM baseline, unweighted, on a temporal split,
with `abuse_label` already removed:

| | Test macro-F1 | Test accuracy |
|---|---|---|
| Always-predict-Legitimate strawman | 0.205 | 0.695 |
| LightGBM baseline | **0.9986** | 0.9994 |

Per ARCHITECTURE.md §6.4, anything above ~0.95 is treated as a leakage
signal, not a result, and blocks progress until explained. It did.

> **Why this says 0.9986 and the README says 0.9988.** They are different
> measurements, not a typo. This document narrates the *unweighted* baseline
> that tripped the gate — reproduce it with
> `train_track(train, test, "full", class_weighted=False)`. The shipped `full`
> track adds inverse-frequency class weighting (ARCHITECTURE.md §5) and scores
> 0.9988 / 0.9995; that is what `runs/model_full.json` holds and what
> `README.md` and `docs/EVALUATION.md` report. The 0.0002 gap is the point:
> class weighting is very nearly a no-op on a model this saturated, because
> there is no minority-class recall left to recover.

---

## Finding 1 — `abuse_label` is a 1:1 encoding of the target

`abuse_label` is an integer column mapping perfectly onto `abuse_type`:

```
abuse_label            0     1     2     3
abuse_type
Legitimate         42060     0     0     0
Policy Abuser          0  7192     0     0
Fraudulent Return      0     0  6112     0
Wardrobing             0     0     0  4636
```

Single-feature macro-F1 = **1.000**. Dropped in `src/features.py::DROP_COLS`.

### The gate as originally specified did not catch this

ARCHITECTURE.md §9.1 specifies a **depth-1** decision tree per feature. That
test is invalid on a 4-class target: one split gives two leaves, so the tree
can name at most 2 of the 4 classes, and its macro-F1 is capped near ~0.45
however perfectly the feature encodes the label. `abuse_label` scored 0.393
at depth 1 — comfortably "clean" — and 1.000 at depth 3.

Mutual information *did* flag it (0.935, far above every other feature), which
is why the sweep runs both tests rather than either alone.

**Correction applied:** the sweep now fits at depth `max(2, n_classes - 1)`
and reports the depth-1 number alongside it. See `src/data_gate.py`.

---

## Finding 2 — the remaining features are box-separated by class

Dropping `abuse_label` moved the baseline from 1.000 to 0.9986. Not a fix.

No *single* remaining feature is strong on its own — the best,
`return_rate_pct`, reaches 0.62. The leakage is **combinatorial**. Greedy
forward selection on the test set:

| Features | Cumulative macro-F1 |
|---|---|
| `return_rate_pct` | 0.621 |
| `+ days_to_return` | 0.944 |
| `+ wishlist_to_cart_time_hrs` | **0.995** |
| `+ total_returns_lifetime` | 0.998 |

**Three features reconstruct the label almost perfectly.** The reason is
visible in the per-class ranges — the generator sampled each feature from
hard-bounded, near-disjoint uniform intervals per class:

`days_to_return`
| class | min | max |
|---|---|---|
| Fraudulent Return | 1 | **5** |
| Legitimate | 1 | 30 |
| Policy Abuser | 1 | 30 |
| Wardrobing | **25** | 55 |

`wishlist_to_cart_time_hrs`
| class | min | max |
|---|---|---|
| Fraudulent Return | 0.1 | **5.0** |
| Wardrobing | 0.1 | **5.0** |
| Legitimate | 0.1 | 72.0 |
| Policy Abuser | 0.1 | 72.0 |

`return_rate_pct`
| class | min | max |
|---|---|---|
| Legitimate | 0.0 | **14.9** |
| Wardrobing | **14.3** | 59.5 |
| Fraudulent Return | 0.0 | 69.7 |
| Policy Abuser | **33.3** | 84.7 |

### The decisive test

Four hand-written if/else rules, read straight off the table above, **with no
training whatsoever**:

```python
def rule(r):
    if r.wishlist_to_cart_time_hrs <= 5.0:
        return 'Wardrobing' if r.days_to_return >= 25 else 'Fraudulent Return'
    return 'Policy Abuser' if r.return_rate_pct > 15 else 'Legitimate'
```

| | accuracy | macro-F1 |
|---|---|---|
| Hand-written rules, zero training | 0.9425 | **0.9188** |

Confusion matrix:

```
predicted ->       Fraudulent  Legitimate  Policy Abuser  Wardrobing
Fraudulent Return        6112           0              0           0
Legitimate               2376       39104              0         580
Policy Abuser             401           0           6698          93
Wardrobing                  0           0              0        4636
```

`Fraudulent Return` and `Wardrobing` are recovered at **100% recall by two
threshold comparisons.**

Runnable source: `src.model.rule_baseline_metrics()`, run against the full
raw CSV (not the train/test split — the point is generator separability,
not generalisation). Output committed at `runs/baseline_rule.json`. This
number existed only as prose until a later correction, which required every
quoted figure to be regenerable from committed code. `tests/test_baseline_rule.py`
now pins the four thresholds, so an edit to the rule cannot silently falsify
this document.

---

## What this means

The gradient-boosted model is not detecting return abuse. It is
**reverse-engineering a rule-based synthetic data generator.** The 0.998 is a
measure of how cleanly the generator drew its class boundaries, not of how
well the method would find abuse in merchant data, where these distributions
overlap heavily and no threshold on "hours from wishlist to cart" cleanly
separates a wardrober from an honest customer.

Reporting 0.998 as a headline result — even with the standard synthetic-data
caveat attached — would be misleading. The caveat in ARCHITECTURE.md §2 was
written to cover "absolute numbers may not transfer." This is a stronger
problem: the task as posed is close to trivial, and any model at all will
score near-perfectly on it.

This is the finding the project should lead with, not bury.

---

## Correction — the cost centerpiece *is* affected

An earlier version of this document concluded that the §6.2 cost-calibration
work was "not invalidated by this finding — a decision policy over predicted
probabilities is just as demonstrable on an easy task as a hard one."

**That was wrong, and the error is worth recording rather than editing out.**

A cost posture can only change a decision for a row whose predicted
probabilities sit near a boundary between two actions. On the full-feature
model, essentially no such rows exist:

| feature set | n | macro-F1 | rows a cost sweep can move (top-prob < 0.90) |
|---|---|---|---|
| A. all features | 35 | 0.9990 | **0.1%** |
| B. − wishlist_to_cart_time_hrs | 34 | 0.9936 | 0.2% |
| C. B − days_to_return | 33 | 0.9779 | 1.6% |
| D. C − return_rate_pct | 31 | 0.9767 | 2.0% |
| E. D − total_returns_lifetime | 29 | 0.9586 | 3.7% |
| F. E − customer_support_contacts | 28 | 0.9342 | 8.4% |
| **G. F − previous_dispute_count** | **27** | **0.8993** | **16.1%** |
| H. G − avg_order_value, refund_amount | 24 | 0.8581 | 23.9% |

> **Rung G reads 0.8993 here; the `testbed` track reads 0.8967.** Same 27
> features, different fit — not a discrepancy. This ladder is a comparative
> sweep across eight feature sets, so `src/ablation.py` fits a lighter model
> (150 trees at lr 0.08, unweighted); only the relative shape of the curve
> carries the finding. The shipped `testbed` track is fit by `src/model.py`
> with the same hyperparameters as `full` (400 trees at lr 0.05,
> class-weighted), so the two tracks stay comparable to each other. Ladder
> numbers live in `runs/ablation_ladder.json`, track numbers in
> `runs/model_testbed.json`.

Measured directly (`src/evaluate.py`, `runs/evaluation_full.json`): on the
full model **every merchant posture from `C_fp:C_fn` = 0.03 to 32 produces
byte-identical decisions.** False-block rate on legitimate customers is
0.00% across the entire range; fraud hard-blocked is 100.00% across the
entire range. The sweep that ARCHITECTURE.md §6.2 names as "the actual
deliverable of the project" returns a flat line.

So the finding invalidates more than the classifier claim. It removes the
cost centerpiece too, unless the build responds to it.

### A note on the ablation table above

Producing it exposed a second-order trap worth recording. The first version
dropped only the raw artifact columns and came out nearly flat (0.999 → 0.978,
no further movement). The reason: `src/features.py` engineers
`returns_per_order` (== `return_rate_pct` / 100) and `orders_kept_lifetime`
(== `total_orders_lifetime` − `total_returns_lifetime`), which restate the
dropped columns algebraically. **An ablation that removes a feature but keeps
a derived restatement of it measures nothing.** Each rung now drags its
proxies out with it.

---

## Decision — dual-track

Neither "report 0.998 and move on" nor "quietly drop features until the task
looks hard" is defensible. The build runs two explicitly-labelled tracks:

**Track `full` — the honest model.** Every legitimate feature. Reported as
the model, with this finding and the flat sweep attached. The flat sweep is
presented as *positive evidence* that the dataset is degenerate, not as a
failed experiment.

**Track `testbed` — rung G. Not a model, and never presented as one.** A
deliberately handicapped variant whose only purpose is to give the decision
layer a non-degenerate region so the §6.2 method can be demonstrated and
audited. Rung G is chosen for one stated reason: it is the first rung with
enough probability mass near a boundary for a sweep to move decisions at all.
It is *not* chosen because it scores well.

Why the testbed cannot be promoted to headline model: the ladder degrades
smoothly from 0.999 to 0.858 with no natural cut point, so any rung is
arbitrary. "We dropped features until the task got hard" is not a result.

---

## Second finding — §6.2 was measuring the wrong axis

Running the calibration on the testbed produced a second correction to the
architecture doc.

ARCHITECTURE.md §6.2 assumes the decisive tension is `C_fp : C_fn` — blocking
an honest customer versus letting real fraud through. **On this data that axis
is nearly inert even on the testbed:** sweeping it across three orders of
magnitude moves the false-block rate only from 0.00% to 0.23%, and fraud
hard-blocked only from 96.76% to 97.39%.

The reason is visible in where the model is actually uncertain. Of 592 test
rows with a top-two margin below 0.3:

| ambiguous pair | count |
|---|---|
| Legitimate vs **Policy Abuser** | 342 |
| Legitimate vs Wardrobing | 119 |
| Policy Abuser vs Wardrobing | 117 |
| *anything* vs Fraudulent Return | **14** |

Runnable source: `src.evaluate.ambiguous_class_pairs()` on the `testbed`
predicted probabilities. Output committed at `runs/ambiguity_testbed.json`
(the `full` track's equivalent, `runs/ambiguity_full.json`, has only 1
ambiguous row of 12,000 — consistent with its 0.1% threshold-sensitive
figure). Also a later correction: this table existed only as prose before.

**Fraudulent Return is the easiest class to separate.** The hard-block call is
not where the difficulty lives, so the cost parameter governing it has almost
nothing to act on.

The real tension is the **approve ↔ soft-friction** boundary — how aggressively
to apply return fees and pattern flags to customers who might just be heavy
returners. Sweeping friction cost against missed-recovery cost moves the
outcome by an order of magnitude:

| | span across the posture range |
|---|---|
| legitimate customers given friction | **2.78% → 24.79%** |
| wardrobers / policy abusers caught | **85.49% → 99.57%** |

That is the operating curve a merchant would actually argue over, and it is
now the centerpiece (`runs/friction_tradeoff_testbed.png`). The inert
`C_fp:C_fn` axis is still reported — showing which axis *doesn't* move is part
of the honest account, and reporting only the inert one would have satisfied
the letter of §6.2 while measuring the wrong thing.
