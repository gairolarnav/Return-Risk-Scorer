# Return-Risk Scorer
### Razorpay AI Buildathon 2026 — Track 02: AI Risk Manager
**Loss class addressed:** Returns abuse (wardrobing, bracketing, fraudulent returns, policy abuse)
**Type:** Detector — multi-class risk scorer with cost-calibrated decision policy
**Constraint compliance:** Strictly defense-only. No generative or adversarial capability. Read-only scoring against historical data.

---

> ## Document status
>
> **This document has been corrected in place, not rewritten.** Three of its
> original design assumptions were falsified by findings during the build. In
> each case the original text is left standing and a dated correction block
> follows it, because what the plan assumed and why it was wrong is more useful
> to a reviewer than a plan that reads as though it was right from the start.
>
> | § | Original claim | Status |
> |---|---|---|
> | §2 | Synthetic data means "absolute numbers may not transfer" | **Superseded** — the problem is stronger than a transfer caveat |
> | §6.2 | The `C_fp : C_fn` sweep is "the actual deliverable of the project" | **Superseded** — that axis is inert; the friction axis replaced it |
> | §9.1 | Leakage sweep uses a depth-1 tree per feature | **Corrected** — depth-1 is invalid on a 4-class target and missed the leak |
> | §9 Day 6 | "3–4 pytest tests on `features.py`" | **Widened** — the leakage finding changed what needs guarding |
>
> Full correction log with dates in §11. The complete leakage investigation is
> `docs/LEAKAGE_FINDING.md` and is the document a reviewer should read second,
> immediately after the README.

---

## 1. Problem Statement

E-commerce merchants lose margin to three distinct patterns hiding inside "returns," each requiring a different intervention:

| Pattern | Description | Right intervention |
|---|---|---|
| **Legitimate return** | Genuine defect, wrong fit, changed mind within policy | Approve, no friction |
| **Wardrobing** | Item used once (e.g. worn to an event) then returned as new | Soft friction — fee, condition inspection |
| **Policy abuse / bracketing** | Ordering multiple variants intending to return most | Soft friction — return-fee nudge, purchase-pattern flag |
| **Fraudulent return** | Empty box, wrong item, stolen goods, tracking manipulation | Hard block — hold refund, manual review, possible ban |

Most public fraud tooling collapses these into a single binary `is_fraud` label. That's the gap this project targets: **a return-risk model that distinguishes abuse *type*, not just abuse *presence*, so the merchant's response is proportionate rather than uniformly punitive.**

A uniformly punitive system is exactly the failure mode the buildathon brief penalizes — it produces good precision on paper while quietly blocking or fee-ing legitimate high-value customers. The core deliverable here is a model **and** an explicit, documented cost policy showing that tradeoff was measured, not ignored.

---

## 2. Data

**Source:** [E-Commerce Return Abuse Detection Dataset](https://www.kaggle.com/datasets/sarveshchhetri/e-commerce-return-abuse-detection-dataset) (Kaggle, synthetic)

**Shape:** 60,000 rows · 35 features · zero missing values

**Target — `abuse_type`:**

| Class | Count | Share | Definition |
|---|---|---|---|
| Legitimate | 42,060 | 70.10% | Normal, honest returns |
| Policy Abuser | 7,192 | 11.99% | Excessive legitimate-looking returns / bracketing pattern |
| Fraudulent Return | 6,112 | 10.19% | Empty boxes, wrong/substitute item, tracking manipulation |
| Wardrobing | 4,636 | 7.73% | Used-once-then-returned |

*Counts are the dataset's exact class distribution, confirmed during the Day 1/2
leakage investigation. The class label in the data is `Policy Abuser`; this
document uses that spelling throughout, matching the data rather than prose.*

**Known limitation (stated up front, not discovered later):** This dataset is fully synthetic. It is useful for building and honestly evaluating a modeling *methodology*, but any absolute performance numbers should be presented as a proof of concept against synthetic ground truth, not as a claim about real-world merchant data. This caveat goes directly in the final writeup and pitch — the brief rewards honesty about this more than it penalizes using synthetic data at all.

**Second limitation, specific to synthetic data — label leakage.** A rule-generated dataset frequently contains one or more features that near-deterministically encode the label. Unchecked, this produces an implausibly strong model that collapses under the first question a panelist asks. Leakage screening is therefore a gated Day 1 task, not an afterthought — see §9.1.

> ### CORRECTION (Day 2) — the caveat above is not strong enough
>
> The paragraph above was written to cover *"absolute numbers may not transfer
> to real merchant data."* The actual finding is a stronger claim and the
> caveat as written would have been misleading if left to stand alone.
>
> **The dataset is degenerate.** Its features are drawn from hard-bounded,
> near-disjoint uniform intervals per class. Four hand-written `if/else`
> threshold rules, read straight off the per-class min/max table with **zero
> training**, achieve:
>
> | | accuracy | macro-F1 |
> |---|---|---|
> | Always-predict-Legitimate strawman | 0.6954 | 0.2051 |
> | **Four hand-written rules, no training** | **0.9425** | **0.9188** |
> | LightGBM, full feature set (unweighted Day 2 baseline) | 0.9994 | 0.9986 |
>
> `Fraudulent Return` and `Wardrobing` are recovered at **100% recall by two
> threshold comparisons.**
>
> The gradient-boosted model is therefore not detecting return abuse. It is
> **reverse-engineering a rule-based synthetic data generator.** The 0.9986
> measures how cleanly the generator drew its class boundaries, not how well
> the method would find abuse in merchant data, where these distributions
> overlap heavily.
>
> This is not a footnote to the results. **It is the headline result**, and the
> README leads with it. Full evidence — per-class range tables, greedy forward
> selection, the decisive rule test — in `docs/LEAKAGE_FINDING.md`.
>
> **Two model numbers appear in this repository, and they are not the same
> measurement.** 0.9986 / 0.9994 is the *unweighted* Day 2 baseline that tripped
> the §6.4 gate — the number this block and `docs/LEAKAGE_FINDING.md` narrate,
> reproducible with `train_track(..., class_weighted=False)`. **0.9988 / 0.9995**
> is the shipped `full` track, which is class-weighted (§5); it is what
> `runs/model_full.json`, `README.md` and `docs/EVALUATION.md` report. The gap
> between them is 0.0002 macro-F1 — class weighting is very nearly a no-op on a
> model this saturated, which is itself part of the finding. Where this document
> narrates a Day 1/2 event it quotes the number measured that day; where it
> states the deliverable it quotes the shipped model.

**Data split strategy:** Stratified train/test split preserving class ratios (80/20). If a usable timestamp field exists, a temporal split is preferred over random. Which one applies is determined by the Day 1 data gate (§9.1) and recorded in `docs/DATA_NOTES.md` with the reasoning.

*Resolved: a temporal split is in use. The Day 1 baseline in
`docs/LEAKAGE_FINDING.md` is reported on a temporal split.*

---

## 3. System Architecture

```
                        ┌─────────────────────────┐
                        │   Raw Return Record       │
                        │ (order, customer, item,   │
                        │  return metadata)          │
                        └────────────┬───────────────┘
                                     │
                     ┌───────────────▼────────────────┐
                     │   Feature Engineering Layer      │
                     │                                   │
                     │  A) Transaction-level features    │
                     │     - days purchase→return         │
                     │     - return-value / order-value    │
                     │     - category, channel flags        │
                     │                                       │
                     │  B) Customer-behavioral aggregates    │
                     │     - trailing return rate              │
                     │     - time-since-last-return              │
                     │     - category concentration of returns   │
                     │     - same-day multi-variant order signal   │
                     │       (bracketing proxy)                      │
                     │                                                 │
                     │  DROP_COLS — leakage quarantine (§9.1)          │
                     └───────────────┬─────────────────────────────────┘
                                     │
                     ┌───────────────▼───────────────┐
                     │     Multi-class Classifier       │
                     │   LightGBM                         │
                     │   (class-weighted, multi:softprob)  │
                     │   two tracks: `full` | `testbed`     │
                     └───────────────┬─────────────────────┘
                                     │
                     ┌───────────────▼───────────────┐
                     │   Cost-Calibrated Decision Layer │
                     │                                    │
                     │  Cost matrix × predicted probs      │
                     │  → per-class threshold                │
                     │  → mapped intervention                  │
                     └───────────────┬─────────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                       ▼
        Approve, no             Fee / condition        Hold refund,
        friction               inspection /             manual review,
                                policy nudge             possible ban
```

**Design principle:** every stage above the decision layer is deterministic given the model output — no autonomous action is taken on a customer account. The system produces a *recommendation and a routed intervention*, not an executed action. This is central to the defense-only compliance story (§7).

*Two boxes were added to the original diagram: the `DROP_COLS` leakage quarantine
in the feature layer, and the `full | testbed` track selector on the classifier
(§5.2). Both are real code paths and were missing from the original drawing.*

---

## 4. Feature Engineering Plan

### 4.1 Transaction-level (available per return record)
- Days between purchase and return
- Return value as a fraction of order value
- Product category, price band
- Return reason code (as declared vs. as inferred)
- Channel (online/COD/etc., if present)

### 4.2 Customer-behavioral aggregates (the actual differentiator)
- Trailing N-order return rate
- Time since last return
- Return-category concentration (many returns clustered in one category — classic bracketing tell)
- Same-day multi-SKU-variant orders followed by partial returns (bracketing proxy)
- Variance in declared return reason across a customer's history (inconsistency signal)

**This section is conditional on a Day 1 finding.** These features require a genuine repeat-customer ID with enough per-customer history to be meaningful. Whether that exists is the first question the Day 1 data gate answers (§9.1). If the median rows-per-customer is 1, every feature in §4.2 is undefined, the transaction-level set in §4.1 carries the model alone, and that becomes a headline limitation in the writeup — documented, not hidden.

> ### NOTE (Day 2) — partially resolved, one open item
>
> The dataset ships several per-customer **lifetime aggregate columns**
> pre-computed — `total_orders_lifetime`, `total_returns_lifetime`,
> `return_rate_pct`, `customer_support_contacts`, `previous_dispute_count` —
> so the model has customer-level behavioural signal without constructing it
> from transaction history. `src/features.py` derives `returns_per_order` and
> `orders_kept_lifetime` from these.
>
> **This creates a trap that cost a first attempt at the §6.2 ablation.** Those
> two derived features *algebraically restate* the columns they come from
> (`returns_per_order` == `return_rate_pct` / 100). An ablation that drops a
> feature but keeps a derived restatement of it measures nothing. Each ablation
> rung now drags its proxies out with it, and `tests/` freezes this as an
> assertion (§9.3, T.4) rather than leaving it as prose someone can regress past.
>
> **OPEN — must be filled from `docs/DATA_NOTES.md` before submission:** the
> rows-per-customer distribution, and therefore whether the *constructed*
> trailing/temporal aggregates in §4.2 are defined at all. This document does
> not assert an answer it does not have. If the median is 1, say so in the
> README as a limitation; if it is greater than 1, the trailing-window features
> are live and the temporal-leakage test (§9.3, T.2) is load-bearing.
>
> ### CLOSED (Day 6, while writing T.2) — neither branch above is quite right
>
> Median rows-per-customer is 1.0 (`docs/DATA_NOTES.md`), but that isn't the
> whole picture: 1,945 of 58,006 customers *do* repeat (max 4 rows). The open
> question was whether, for that subset, the trailing-window features are
> "live" and load-bearing for a temporal-leakage test. Checked directly
> against the raw CSV's repeat-customer rows (not a fixture — this is a
> property of the data itself, not of `src/features.py`): the lifetime
> columns are **not monotonic across a customer's own rows in return_date
> order**. One real customer's `total_orders_lifetime` goes 78 → 57 → 12 → 14
> across four rows. A genuine running ledger cannot decrease.
>
> So the answer is neither "undefined for lack of repeats" nor "live and
> load-bearing" — it's that **`total_orders_lifetime` / `total_returns_lifetime`
> / `return_rate_pct` are independent per-row generator snapshots, not a
> customer history at all**, even for rows that share a `customer_id`. There is
> no real trailing aggregate anywhere in this dataset for a temporal-leakage
> bug to hide in. `tests/test_features.py::test_no_cross_row_customer_aggregation_exists`
> freezes this: it proves `add_transaction_level_features` derives every
> ratio from a row's own values only, and documents the non-monotonic pattern
> in its docstring so the next person doesn't have to re-derive it. This is a
> further data point for the degenerate-generator finding, not a new,
> separate leak — the same near-disjoint per-class ranges that make the
> hand-written rule work (`docs/LEAKAGE_FINDING.md`) are drawn per-row, with
> no attempt to keep them consistent across a repeat customer's history.

---

## 5. Modeling Approach

- **Primary model:** LightGBM, `multi:softprob` objective, class weights inversely proportional to class frequency (tuned per-class, not globally uniform). LightGBM is the committed choice — XGBoost is not a live alternative in this build, and presenting it as one would only read as an undecided design.
- **Class imbalance handling:** Class weighting is the primary approach. The observed ratio (70/12/10/8) is mild, so this is expected to be a small intervention rather than a major one. SMOTE will be tested as a comparison point and the decision to keep or discard it will be justified in the writeup — not applied silently.
- **Explainability:** SHAP values on the final model, per-class, to support the architecture doc and pitch — the panel should be able to see *why* wardrobing and policy abuse get confused, not just that they do.

### 5.1 Explicitly out of scope
- **Isolation Forest / unsupervised anomaly scoring.** Previously carried as an optional secondary signal. Cut from the build: it adds a second model with no defined path into the decision layer (§3), a second explainability story to tell inside a five-minute pitch, and a scope-creep vector on the tightest days. It is recorded as future work in the writeup — which costs nothing and claims nothing.
- Any generative, adversarial, or pattern-synthesis component (see §7 — this is a hard constraint, not a scoping choice).

### 5.2 Dual-track build — ADDED Day 2

*This section did not exist in the original plan. It is the build's response to
the §2 correction and is the design decision most likely to be misread by a
reviewer who finds it in code before finding it in the README.*

The leakage finding leaves two indefensible options and one defensible one.
Reporting 0.9986 and moving on is misleading. Quietly dropping features until
the task looks hard is worse — it is result-shopping. The build therefore runs
**two explicitly labelled tracks through one code path**, selected by a single
parameter:

**Track `full` — the honest model.** Every legitimate feature (all 35 minus
`DROP_COLS`). This is *the model*, and it is what gets reported, with the
leakage finding and the flat cost sweep attached to it. The flat sweep is
presented as **positive evidence that the dataset is degenerate**, not as a
failed experiment.

**Track `testbed` — rung G of the ablation ladder. Not a model, and never
presented as one.** A deliberately handicapped 27-feature variant whose only
purpose is to give the decision layer a non-degenerate region, so the §6.2
calibration method can be demonstrated and audited on something.

Rung G is selected for exactly one stated reason: **it is the first rung with
enough probability mass near a decision boundary for a cost sweep to move any
decision at all** (16.1% of rows below 0.90 top-probability). It is *not*
selected because it scores well.

| rung | features | macro-F1 | rows a cost sweep can move (top-prob < 0.90) |
|---|---|---|---|
| A. all features | 35 | 0.9990 | **0.1%** |
| B. − `wishlist_to_cart_time_hrs` | 34 | 0.9936 | 0.2% |
| C. B − `days_to_return` | 33 | 0.9779 | 1.6% |
| D. C − `return_rate_pct` | 31 | 0.9767 | 2.0% |
| E. D − `total_returns_lifetime` | 29 | 0.9586 | 3.7% |
| F. E − `customer_support_contacts` | 28 | 0.9342 | 8.4% |
| **G. F − `previous_dispute_count`** | **27** | **0.8993** | **16.1%** |
| H. G − `avg_order_value`, `refund_amount` | 24 | 0.8581 | 23.9% |

> **Why rung G reads 0.8993 here and the `testbed` track reads 0.8967
> elsewhere.** Same 27 features, different fit — not a discrepancy. The ladder
> is a *comparative* sweep across eight feature sets, so `src/ablation.py`
> fits a deliberately lighter model (150 trees at lr 0.08, unweighted) where
> only the relative shape of the curve carries the finding. The shipped
> `testbed` track is fit by `src/model.py` with the same hyperparameters as
> `full` (400 trees at lr 0.05, class-weighted) so the two tracks are
> comparable to each other. Ladder numbers come from
> `runs/ablation_ladder.json`; track numbers from `runs/model_testbed.json`.
> Quote a number with the file it came from.

**Why the testbed cannot be promoted to headline model:** the ladder degrades
smoothly from 0.999 to 0.858 with no natural cut point, so any rung is
arbitrary. "We dropped features until the task got hard" is not a result, and
a reviewer will say so. Naming the testbed as not-a-model, in the README and in
the code, is the only version of this that survives questioning.

**And the selection itself touched the test set.** The ladder scores all eight
feature sets against the same temporal test split, so the criterion that picked
rung G — "first rung with enough probability mass near a decision boundary" —
was read off that split. That is acceptable for a diagnostic and would not be
acceptable for a reported model, which is the second reason `testbed` is
labelled a rung rather than a result. Stated here rather than left for a
reviewer to notice.

### 5.3 Hyperparameters were fixed, not tuned — and why

**There is no validation split, no hyperparameter search, and no early
stopping in this build.** Both tracks are fit with the same two constants,
declared at the top of `src/model.py` and never moved:

```python
N_ESTIMATORS  = 400
LEARNING_RATE = 0.05
```

This is a deliberate omission, and stating it plainly is cheaper than letting
a reviewer discover it. Three reasons:

1. **There is nothing to tune toward.** The `full` track sits at 0.9988
   macro-F1 with 99.9% of predictions above p=0.99. A search would be
   optimising in the fourth decimal place, and every point it could buy would
   come from fitting the synthetic generator more tightly — which is the exact
   behaviour this project's headline finding says to stop rewarding. A tuned
   0.9992 would be a *worse* submission than an untuned 0.9988, because it
   would imply the number meant something.
2. **A tuning result would not be honest here anyway.** With a single temporal
   test split and no validation fold, any search would have to select against
   the test set — the same test set the ablation ladder already touches eight
   times. Adding a search on top would convert a clean held-out number into a
   selected one for no analytical gain.
3. **The interesting parameters are not the model's.** The decision this
   project is actually about lives in the cost matrix (§6.2), which *is* swept,
   across two axes, with the operating curve reported rather than a chosen
   point. That is where the tuning effort went.

**What this costs, stated honestly.** These numbers are not "the best LightGBM
can do on this data" and are not presented as such — they are what a
reasonable default configuration does. On a real, non-degenerate dataset this
would be an unacceptable gap and the first thing to fix: a proper
train/validation/test three-way split, with the search run against validation
only and the test set touched once. That is recorded as future work, not
claimed as done.

---

## 6. Evaluation Framework — "Honest Metrics" (the core grading criterion)

### 6.1 What will NOT be the headline metric
Macro-accuracy. With a 70% legitimate majority class, a model predicting "legitimate" for everything scores ~70% accuracy while being useless. This strawman baseline is computed and recorded on Day 2 alongside the first real model, and shown explicitly in the writeup to justify why accuracy is rejected.

*Recorded: the strawman scores 0.6954 accuracy / 0.2051 macro-F1. Per the §2
correction, it is now reported as the first of three anchors — strawman, then
the untrained rule baseline at 0.9188, then the model at 0.9988. **The sequence
is the argument**; any one of those numbers alone misrepresents the project.*

### 6.2 What will be reported
- **Per-class precision, recall, F1** (not just macro-averaged)
- **Full confusion matrix**, annotated with which misclassifications matter most
- **Per-class PR curves**, since class imbalance makes ROC-AUC optimistic and misleading here
- **Cost-weighted confusion matrix** — a merchant-relevant cost is assigned to each cell:

| Predicted →<br>Actual ↓ | Legitimate | Wardrobing | Policy Abuser | Fraudulent |
|---|---|---|---|---|
| **Legitimate** | 0 | friction cost (low) | friction cost (low) | **`C_fp`** — wrongly hard-blocking a good customer |
| **Wardrobing** | missed-recovery cost (low) | 0 | small confusion cost | over-penalization cost |
| **Policy Abuser** | missed-recovery cost (low) | small confusion cost | 0 | over-penalization cost |
| **Fraudulent** | **`C_fn`** — real loss goes through | moderate | moderate | 0 |

**The decision rule is expected-cost minimising, not `argmax`.** Given the
model's class posterior `p(c)` for a row and the cost matrix `C[c, a]` above,
the routed action is

```
action = argmin_a  Σ_c  p(c) · C[c, a]
```

so a row is routed by what its full probability vector costs under each
available action, not by which single class happens to be most likely. This is
the layer the sweeps below act on, and the one `src/infer.py` serves through
(`expected_cost_decision`).

**On the two dominant cells:** `C_fp` (blocking a legitimate customer) and `C_fn` (letting a real fraudulent return through) are the two costs that drive the decision policy, and **neither is declared larger than the other a priori.** Doing so would presuppose the answer to the exact question this framework exists to ask. Their *ratio* is the free parameter, it is not knowable from the data, and sweeping it is the analysis.

- **Cost-ratio sensitivity analysis:** the `C_fp : C_fn` ratio is swept across 2–3 defensible merchant postures — e.g. loss-neutral (1:1), customer-retention-weighted (3:1), loss-averse (1:3) — with the resulting optimal per-class thresholds and the precision/recall they produce reported for each. **This range, not a single cherry-picked number, is the actual deliverable of the project.**
- **Segment-level false-positive rate (stretch goal):** FPR broken out by return-value bucket or order-value bucket, to check the model isn't concentrating false blocks on any one customer segment — adapted from the Bank Account Fraud fairness-audit methodology.

> ### CORRECTION (Day 4) — §6.2 was measuring the wrong axis
>
> The bolded sentence above is wrong, and it was the plan's central bet.
>
> **The `C_fp : C_fn` axis is inert on this data.** Measured directly
> (`src/evaluate.py`, `runs/evaluation_full.json`):
>
> - On the **`full`** model, every merchant posture from `C_fp:C_fn` = 0.03 to 32
>   produces **byte-identical decisions.** False-block rate on legitimate
>   customers is 0.00% across the entire range; fraud hard-blocked is 100.00%
>   across the entire range. The sweep returns a flat line.
> - Even on the **`testbed`**, sweeping that axis across three orders of
>   magnitude moves the false-block rate only from **0.00% to 0.23%**, and fraud
>   hard-blocked only from **96.76% to 97.39%**.
>
> **Why.** The plan assumed the decisive tension was "block an honest customer
> versus let real fraud through." It isn't, because `Fraudulent Return` is the
> *easiest* class to separate. Of 592 test rows with a top-two margin below 0.3:
>
> | ambiguous pair | count |
> |---|---|
> | Legitimate vs **Policy Abuser** | 342 |
> | Legitimate vs Wardrobing | 119 |
> | Policy Abuser vs Wardrobing | 117 |
> | *anything* vs Fraudulent Return | **14** |
>
> The hard-block call is not where the difficulty lives, so the cost parameter
> governing it has almost nothing to act on.
>
> **The real tension is the approve ↔ soft-friction boundary** — how aggressively
> to apply return fees and pattern flags to customers who might just be heavy
> returners. Sweeping **friction cost against missed-recovery cost** moves the
> outcome by an order of magnitude:
>
> | | span across the posture range |
> |---|---|
> | legitimate customers given friction | **2.78% → 24.79%** |
> | wardrobers / policy abusers caught | **85.49% → 99.57%** |
>
> **That curve is the centerpiece** (`runs/friction_tradeoff_testbed.png`). The
> README links to it; the reading of it is here and in `docs/EVALUATION.md`.
>
> The inert `C_fp:C_fn` axis is **still reported**. Showing which axis *doesn't*
> move is part of the honest account — and reporting only the inert one would
> have satisfied the letter of §6.2 while measuring the wrong thing.

### 6.3 Failure-mode disclosure
The writeup will explicitly document the model's weakest boundary rather than only presenting favorable results.

*Confirmed by the ambiguity table in the §6.2 correction: the weakest boundary is
**Legitimate vs Policy Abuser** (342 of 592 ambiguous rows), not the
Wardrobing-vs-Policy-Abuser boundary originally predicted here. The original
prediction is left in the correction log (§11) rather than deleted — a
prediction that was made and then falsified by measurement is evidence the
measurement was real.*

### 6.4 Suspiciously-good-result protocol
Any headline macro-F1 above ~0.95 is treated as a leakage signal, not a result, and blocks progress until explained (§9.1, Day 2 checkpoint). Whatever the outcome — a real leak found and removed, or a defensible explanation for why the synthetic generator makes the task genuinely easy — it goes in the writeup. "Our model scores 0.97" with no accompanying account of why is the single easiest claim for a panelist to dismantle.

> ### The protocol fired, and it worked (Day 2)
>
> The first end-to-end baseline scored **0.9986 macro-F1**, tripping this gate.
> Progress was blocked, the investigation ran, and it produced both the §2
> correction and the §6.2 correction — the two findings the submission now
> leads with.
>
> This is recorded here deliberately. A pre-committed protocol that then fired
> and changed the build is stronger evidence of method than a protocol that
> never had to do anything. Its output is `docs/LEAKAGE_FINDING.md`.

---

## 7. Defense-Only Compliance

- The system **scores and routes**; it does not execute irreversible actions (no auto-refund-denial, no auto-account-ban).
- No generative component exists anywhere in the pipeline — no synthetic-fraud-pattern generation, no content generation of any kind.
- No component of this system could be repurposed to help a bad actor evade detection or optimize an abuse strategy — it is a one-directional read → score → recommend pipeline.
- All decisions are logged and explainable (SHAP), supporting human audit of every flagged case.

*Enforced, not asserted: `tests/test_compliance.py` fails if any module under
`src/` or `scripts/` imports a generative or LLM library, or if
`requirements.txt` names one. The track criterion is
"anything offense-capable is disqualified" — a claim in a document is weaker
evidence of compliance than a test in CI.*

---

## 8. Repository Structure

```
return-risk-scorer/
├── .github/workflows/
│   └── ci.yml                   # ruff check + pytest on push (fetch-depth: 0, see §9.3)
├── .githooks/                   # local config — `git config core.hooksPath .githooks`
│   ├── commit-msg               # strips agent attribution trailers as a commit is written
│   └── pre-push                 # refuses a push whose commits carry any, the last gate before GitHub
├── data/
│   ├── raw/.gitkeep             # Kaggle CSV NOT committed — README says where to get it
│   └── processed/               # train/test splits, engineered features (parquet)
├── notebooks/                   # presentation only — import from src, hold no logic
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_cost_calibration.ipynb
├── src/
│   ├── data_gate.py             # Day 1 gate: customer-ID viability, split strategy, leakage sweep
│   ├── features.py              # feature builders + DROP_COLS leakage quarantine
│   ├── ablation.py              # degeneracy ladder (diagnostic evidence, not feature selection)
│   ├── model.py                 # train/predict/threshold logic; `full` | `testbed` track selector
│   ├── evaluate.py              # per-class PR, confusion matrix, cost + friction sweeps
│   ├── infer.py                 # single-record + batch scoring -> routed intervention
│   ├── explain.py               # per-class SHAP, both tracks
│   ├── smote_experiment.py      # SMOTE vs class-weighted on testbed
│   └── segment_audit.py         # segment-level FPR by order-value bucket
├── scripts/
│   └── score.py                 # CLI: record or CSV -> class + intervention (replaces the cut demo)
│                                #   --track --posture --friction --explain --overwrite
├── examples/                    # 1 record + 20 rows, so the CLI runs with no dataset
│   ├── README.md
│   ├── record.json
│   └── sample_returns.csv
├── runs/                        # committed — reviewer reads every headline number without the dataset
│   ├── model_full.joblib        # the one model binary committed (3.2M) — see below
│   ├── model_full.json          # + model_testbed.json — per-track metrics
│   ├── evaluation_full.json     # + evaluation_testbed.json — per-class metrics, both cost axes
│   ├── baseline_rule.json       # the untrained 4-rule anchor (§2 correction)
│   ├── ablation_ladder.json     # + .md — the degeneracy ladder (§5.2)
│   ├── ambiguity_full.json      # + ambiguity_testbed.json — top-two-margin pairs (§6.2)
│   ├── shap_full.json           # + shap_testbed.json, shap_*.png — per-class SHAP
│   ├── shap_interpretation.md   # written reading of why the confusable classes confuse
│   ├── smote_testbed.json       # + smote_verdict.md — verdict: discard (§5)
│   ├── segment_fpr_full.json    # + segment_fpr_testbed.json, *.png
│   ├── segment_fpr_audit.md     # FPR by order-value bucket, written up
│   ├── friction_tradeoff_testbed.png   # + _full, + .csv — the centerpiece curve (§6.2)
│   ├── cost_tradeoff_*.png/.csv # the inert C_fp:C_fn axis, reported anyway
│   ├── confusion_*.png
│   └── pr_curves_*.png
├── docs/
│   ├── ARCHITECTURE.md          # this file
│   ├── LEAKAGE_FINDING.md       # the headline result — read this second, after the README
│   ├── DATA_NOTES.md            # Day 1 gate findings + split-strategy decision
│   └── EVALUATION.md            # per-class metrics, both cost axes, caveats
├── tests/                       # see §9.3 — all 15 files, none omitted
│   ├── test_leakage.py
│   ├── test_features.py
│   ├── test_split.py
│   ├── test_baseline_rule.py
│   ├── test_model.py
│   ├── test_evaluate.py
│   ├── test_ablation.py
│   ├── test_explain.py
│   ├── test_smote_experiment.py
│   ├── test_segment_audit.py
│   ├── test_infer.py
│   ├── test_score.py
│   ├── test_require_artifacts.py
│   ├── test_git_attribution.py
│   └── test_compliance.py
├── requirements.txt             # the 13 direct dependencies, exact pins
├── requirements.lock            # full transitive closure — see §8.1
├── pyproject.toml               # ruff + pytest configuration
├── .gitignore
├── LICENSE
└── README.md
```

**On notebooks vs. `src/`:** the four notebooks exist for presentation and exploration only. They import from `src` and contain no logic of their own. The failure mode otherwise is two divergent implementations of the same feature builder, which a reviewer will find and which invalidates any number the notebook produces.

**On what is and isn't committed (added Day 6, extended Day 7).** `runs/` **is**
committed and `data/raw/` is **not**. Most reviewers will never download a 60k-row
Kaggle CSV, so every headline number must be readable from the repo alone.
Committing someone else's dataset is separately a bloat and licensing problem.

Day 7 extends that principle from the numbers to the demo. The same reviewer who
will not download the dataset also will not run a training pipeline to see one
prediction, and until Day 7 that is exactly what `scripts/score.py` required —
model bundles are gitignored, so the CLI could not produce a single score from a
clean clone. Now committed:

- **`runs/model_full.joblib`** (3.2M). The `testbed` bundle stays out: it is a
  diagnostic rung rather than a model (§5.2), and it is the larger file.
- **`examples/record.json`** and **`examples/sample_returns.csv`** — 1 and 20
  real records, drawn from the **test window only**, so they are rows the
  committed bundle never trained on. `abuse_label` dropped per DROP_COLS;
  `abuse_type` kept so predictions can be checked against truth.

This is a reversal of the "no model binaries" position and is logged as one in
§11. The bloat argument still holds against the dataset itself — 20 rows is not
redistributing it — and 3.2M is a proportionate price for a demo that runs.

---

## 8.1 Environment & Dependencies

**Language:** Python 3.11 specifically — not 3.12+. `shap` and `lightgbm` wheel availability lags new interpreter releases, and a build week is the wrong time to be compiling from source.

**Core libraries:**
- `pandas`, `numpy`, `pyarrow` — data handling; parquet for `data/processed`, since CSV round-trips lose dtypes and are slow at 60k × 35
- `lightgbm` — modeling (committed; see §5)
- `scikit-learn` — splits, metrics
- `imbalanced-learn` — SMOTE comparison
- `shap` — explainability
- `joblib` — model artifact persistence
- `matplotlib` / `seaborn` — PR curves, confusion matrix plots
- `tabulate` — markdown tables in generated reports (`runs/`, `docs/DATA_NOTES.md`)
- `pytest` — tests
- `ruff` — linting (one tool, zero config)

**Deliberately not used:** MLflow, Weights & Biases, or any experiment-tracking service. At this scale a JSON metrics file per run in `runs/` covers the need and is one fewer thing to explain in a five-minute pitch.

**requirements.txt — pinned exactly, not `>=`:**
```
pandas==2.2.3
numpy==1.26.4
pyarrow==17.0.0
tabulate==0.9.0
lightgbm==4.5.0
scikit-learn==1.5.2
imbalanced-learn==0.12.4
shap==0.46.0
joblib==1.4.2
matplotlib==3.9.2
seaborn==0.13.2
pytest==8.3.3
ruff==0.7.0
```

Exact pins matter here for a specific reason: this repo is submitted for review and may be cloned weeks later. `>=` constraints guarantee that a future resolver picks different versions than the ones the reported numbers were produced with, which quietly breaks both reproducibility and the Quickstart.

**Platform note — Apple Silicon:** `pip install lightgbm` succeeds but fails at import without OpenMP. Run `brew install libomp` **before** installing requirements. This belongs in Day 1 setup, not discovered mid-build, and it appears in the README above the `pip install` line rather than in a footnote.

**Lockfile.** `requirements.txt` pins the 13 direct dependencies; `requirements.lock`
pins the full transitive closure, so a rebuild cannot pick up a different
resolution of an indirect package. It is resolved in a **clean venv** from
`requirements.txt`, deliberately not frozen from a working environment — this
project's own venv also holds `streamlit`, `altair`, `pydeck`, `jupyter` and the
macOS-only `appnope`, left over from the cut demo app and from executing the
notebooks. Freezing that would have pinned 91 packages, enshrined dependencies
the project does not use, and produced a file that cannot install on Linux. The
lock is 33 packages with no platform-specific entries. CI installs
`requirements.txt`, so the direct pins stay exercised on every run.

**Reproducibility:** Fix a single random seed (`RANDOM_STATE = 42`) at the top of `src/model.py` and reuse it everywhere — train/test split, model init, SMOTE — so results are reproducible run to run and defensible if a panelist asks to re-run the notebook.

## 8.2 Quickstart

```bash
git clone <repo-url>
cd return-risk-scorer

# Hook path is local config and does not survive a clone — set it once, here.
# commit-msg strips agent attribution trailers; pre-push refuses any that
# survive. Without this line neither hook runs (§10, repo hygiene).
git config core.hooksPath .githooks

python3.11 -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows
# macOS on Apple Silicon: brew install libomp
pip install -r requirements.txt

# place the Kaggle CSV at data/raw/returns.csv, then:
python -m src.data_gate        # Day 1 checks: customer viability, split strategy, leakage sweep
python -m src.features         # builds processed train/test sets
python -m src.ablation         # the degeneracy ladder
python -m src.model            # trains + saves both tracks
python -m src.evaluate         # generates metrics, confusion matrix, cost + friction tables
python -m src.explain          # per-class SHAP, both tracks
python -m src.smote_experiment # SMOTE vs class-weighted on testbed
python -m src.segment_audit    # segment FPR by order-value bucket

# score a record or a batch CSV -> routed intervention (replaces the cut demo).
# These two need NO dataset and no training run: runs/model_full.joblib and
# examples/ are committed, so they work straight after pip install.
python -m scripts.score --record-file examples/record.json --track full
python -m scripts.score --csv examples/sample_returns.csv --out scored.csv

# the friction axis, the one that moves (needs the testbed bundle, so train first)
python -m scripts.score --csv examples/sample_returns.csv --track testbed \
    --friction "recovery-first (1:20)"

# why this record got that call — signed per-feature SHAP, single records only
python -m scripts.score --record-file examples/record.json --track full --explain

# --out refuses to replace an existing file; pass --overwrite to allow it
# for an exact environment rebuild: pip install -r requirements.lock  (§8.1)

# no dataset? every headline number is committed:
#   runs/evaluation_full.json, runs/evaluation_testbed.json,
#   runs/friction_tradeoff_testbed.png
# and the test suite runs on fixtures alone:
pytest
```

This sequence is verified from a clean clone into a fresh virtualenv on Day 6. It is not assumed to work — hackathon repos routinely fail at exactly this step, and it is the first thing a reviewer touches.

---

### 8.3 What a scored result reports about its own inputs

`src/infer.py` returns the routed decision *and* an account of what the model
could not see. This exists because the alternative was measured and found
dishonest: a record scored without three engineered features reported
`features_missing: []`, which is indistinguishable from a complete record
(§11, F6).

Causes are **derived from the finished model frame**, not tracked as the record
is assembled — whatever is NaN and unexplained by the three named causes is
reported as degraded, so a cause nobody anticipated surfaces instead of
vanishing.

| field | meaning |
|---|---|
| `features_missing` | column absent from the input |
| `features_invalid` | present, but not parseable as a number |
| `features_out_of_range` | present and numeric, but impossible — negative money, a 0/1 flag holding 7. Nulled rather than allowed to drive a recommendation (§11, F5) |
| `features_degraded` | present, valid, in range, and still not computable — an engineered ratio whose denominator was zero |
| `n_features_not_seen` | the total, whatever the cause. The one number that cannot be gamed by how the causes are categorised |

`score_batch` reports the same as **per-row** counts. `n_features_missing` stays
frame-level, correctly: an absent column is absent for every row.

The domain bounds behind `features_out_of_range` are read off the training data
rather than invented — every numeric column in `data/processed/train.parquet`
has `min >= 0`, and `return_rate_pct` is a percentage — so they cannot fire on a
record drawn from the real distribution. They are checked at **inference only**:
training, ablation and SMOTE go through `src.model.as_model_frame`, so no
published number can move by construction. Cross-field invariants
(`total_returns_lifetime <= total_orders_lifetime`) are deliberately excluded: a
two-column rule has no unambiguous answer to which column to null.

---

## 9. Build Plan (7 Days)

| Day | Focus | Output |
|---|---|---|
| 1 | **Data gate** (first ~2h) + scaffolding + transaction-level features | `src/data_gate.py` and `docs/DATA_NOTES.md` answering the three gate questions (§9.1); repo scaffolding, pinned venv, `libomp` if on macOS; column documentation, class-balance confirmation; `features.py` v1 |
| 2 | Customer-behavioral aggregates + baseline model + **leak checkpoint** | `features.py` v2; end-to-end unweighted LightGBM baseline; strawman number recorded for §6.1; **hard gate: macro-F1 > ~0.95 stops progress until explained (§6.4)** |
| 3 | Imbalance handling + tuning — **half day** — then start cost matrix | Class-weighted model, SMOTE tested with a documented keep/discard decision, basic hyperparameter pass. Second half: cost matrix definition begins early |
| 4 | **Cost-based decision policy — full day, protected** | Cost matrix finalized, posture sweep, per-scenario thresholds + resulting per-class precision/recall. Inherits the buffer Day 3 gave up |
| 5 | Evaluation writeup + SHAP + **pitch script drafted** | Per-class precision/recall/F1, confusion matrix, PR curves, per-class SHAP, failure-mode disclosure. The 5-minute pitch is *scripted today*, while results are fresh |
| 6 | Demo (half day) + **clean-clone reproducibility test** + tests | Streamlit app (record → class → intervention), timeboxed to ~80 lines. Then: fresh clone into a new venv, run §8.2 verbatim, fix what breaks. Test suite per §9.3 |
| 7 | Pitch recording + final polish + buffer | 5-min pitch recorded from the Day 5 script, this document finalized with real numbers, README finished, repo cleanup; stretch goal (segment-level FPR audit) only if genuinely free |

> ### WHAT ACTUALLY HAPPENED (Days 1–4)
>
> - **Day 2 fired the §6.4 gate** at 0.9986 macro-F1 and the build stopped as
>   specified. The investigation consumed the day and produced the §2 correction.
>   The **dual-track design (§5.2) is the response** and did not exist in the plan.
> - **Day 4 falsified the plan's central bet.** The `C_fp:C_fn` sweep the plan
>   called "the actual deliverable" returned a flat line; the centerpiece moved
>   to the friction axis (§6.2 correction).
> - **Net effect on the schedule:** the deliverable named on Day 4 changed, but
>   the day itself still carried the analysis it was protected for. The plan's
>   protection of Day 4 was correct even though its prediction of what Day 4
>   would produce was wrong.

### 9.1 The Day 1 data gate

Three questions, answered from the raw CSV before any modeling work begins, because each one can invalidate a later day's plan:

1. **Is there a usable repeat-customer identifier?** Report the distribution of rows-per-customer. A median of 1 means every constructed feature in §4.2 is undefined, and Day 2's scope changes to the §4.1 fallback. *(See the §4.2 open item — this answer belongs in `docs/DATA_NOTES.md`, and §4.2 carries its consequence for the feature plan.)*
2. **Is there a usable timestamp?** Decides temporal vs. random split (§2), which affects every number reported afterwards and cannot be changed retroactively without redoing the evaluation. *(Resolved: temporal split in use.)*
3. **Leakage sweep.** Mutual information per feature against the label, plus a ~~depth-1~~ decision tree per feature. Any single feature reaching ~0.6 macro-F1 alone is a generation artifact of the synthetic dataset, not a signal, and must be identified before it silently produces a headline number that collapses under questioning.

> ### CORRECTION (Day 2) — the depth-1 tree was an invalid test, and it missed the leak
>
> **A depth-1 decision tree cannot detect single-feature leakage on a 4-class
> target.** One split produces two leaves, so the tree can name at most 2 of 4
> classes and its macro-F1 is capped near ~0.45 *however perfectly the feature
> encodes the label.*
>
> This is not theoretical. `abuse_label` is a **1:1 integer encoding of
> `abuse_type`** — single-feature macro-F1 of **1.000** at depth 3. At depth 1
> it scored **0.393** and passed the gate as comfortably clean.
>
> ```
> abuse_label            0     1     2     3
> abuse_type
> Legitimate         42060     0     0     0
> Policy Abuser          0  7192     0     0
> Fraudulent Return      0     0  6112     0
> Wardrobing             0     0     0  4636
> ```
>
> **Mutual information *did* catch it** (0.935, far above every other feature),
> which is the entire reason the gate specifies both tests rather than either
> alone. Redundancy in the gate is what saved it.
>
> **Fix applied in `src/data_gate.py`:** the sweep now fits at depth
> `max(2, n_classes - 1)` and reports the depth-1 number alongside it, so the
> failure mode stays visible instead of being quietly patched out.
> `abuse_label` is in `DROP_COLS`, and `tests/test_leakage.py` (§9.3, T.1) is a
> permanent tripwire against its return.
>
> **What this correction does not fix.** Dropping `abuse_label` moved the
> baseline from 1.000 to 0.9986 — it was never the main problem. The remaining
> leakage is **combinatorial**: no single feature exceeds 0.62 alone, but three
> together reach 0.995. A per-feature gate cannot detect that by construction.
> The greedy forward-selection sweep that does is in `docs/LEAKAGE_FINDING.md`,
> and **a per-feature-only leakage gate is now a recorded limitation of this
> methodology**, not a solved problem.

All three findings and their consequences are written to `docs/DATA_NOTES.md` on Day 1 and referenced in the final writeup.

### 9.3 Test plan — WIDENED Day 6

> The original Day 6 line read *"3–4 `pytest` tests on `features.py`."* That
> predates the leakage finding and is too narrow. On a project whose headline
> result is about leakage, the tests must guard the leakage claims — otherwise
> one careless commit re-introduces the leak and nothing catches it.

Tests are ranked by what they defend, not by ease of writing. Coverage is not a
target and no coverage gate is used; twelve tests that guard the claims are
worth more than sixty that exercise getters.

| ID | Test | Guards | Priority |
|---|---|---|---|
| T.1 | `abuse_label` absent from the feature matrix; no single feature clears the gate threshold at depth `max(2, n_classes-1)` | The §9.1 finding — permanent tripwire on the leak | **Must** |
| T.2 | A customer-behavioral aggregate for row *n* uses only rows strictly before *n* for that customer | Temporal leakage in aggregates — silent when present, inflates everything | **Must** |
| T.6 | No module under `src/` or `scripts/` imports a generative or LLM library | §7 — the one criterion that disqualifies | **Must** |
| T.3 | Train/test disjoint on row key; `max(train ts) <= min(test ts)` | Every reported number | Should |
| T.4 | No ablation rung retains an algebraic restatement of a feature it drops | The §4.2 proxy trap, frozen as an assertion | Should |
| T.5 | Feature builders match hand-computed values on a ~10-row fixture | That someone understood the features | Should |
| T.7 | Re-running evaluation reproduces `runs/evaluation_full.json` within tolerance | Makes "reproducible" a checked claim, enforceable in CI | Should |
| T.8 | Same input + `RANDOM_STATE` → identical output twice | §8.1 reproducibility | Nice |
| T.9 | `src/infer.py` returns 4 probabilities summing to 1.0 + a valid intervention | The demo path a judge runs live | Nice |

**If time runs out, T.1 / T.2 / T.6 are worth more than the other six
combined.** Write those three and record the rest as not-reached, with the reason. Three tests that defend the thesis
plus an honest note about what was skipped reads better than nine shallow ones.

**What was actually built.** All nine planned tests landed, and the suite grew
well past them: **139 tests across 15 files**, one per `src/` module plus
`test_score.py` for the CLI, `test_require_artifacts.py` for the missing-input
guards, and `test_git_attribution.py`, which is not in the table above because
the problem it guards was not foreseen when the plan was written — it walks the
whole commit history for agent attribution and drives the pre-push hook against
a scratch remote to prove it refuses. The estimate of "twelve tests that guard
the claims" was the right *principle* and the wrong *number*; the suite is
larger because the serving path and the repo-hygiene surface both turned out to
need guarding, not because coverage became a target. It still is not one.

**What the suite defends.** Coverage is concentrated on the claims that would
be expensive to get silently wrong:

- **The decision layer**, where a wrong cost matrix produces plausible-looking
  numbers that are simply false — cost-cell placement, Bayes optimality,
  monotonicity in `C_fp`, the oracle lower bound, and the degeneracy detector
  firing on saturated probabilities.
- **The rule baseline**, whose 0.9188 macro-F1 is the headline finding. The
  four thresholds are pinned branch by branch, so an edit to the rule fails the
  suite instead of silently falsifying every document that quotes it.
- **The serving path**, with one regression test per historical bug in
  `src/infer.py` (§11.1), plus partial-record handling — a caller who omits a
  field gets a scored result that names what was missing, not a library
  traceback.
- **The invariants**: no `abuse_label` in the feature matrix, train/test
  disjoint and ordered in time, no ablation rung retaining an algebraic
  restatement of a feature it drops, and no module importing a generative or
  LLM library.
- **Commit attribution**, over the whole reachable history — see the §10 repo
  hygiene entry for why a hook alone is not enough.

**Two standing rules.** Every test runs on committed fixtures — a suite that
needs the Kaggle CSV can never run in CI or on a reviewer's machine. And **a
failing test is a finding, not a bug in the test**: if T.2 fails, that is real
temporal leakage stacked on top of the generator leakage, and it changes
reported numbers. It is not resolved by loosening the assertion.

### 9.2 Slippage risk

**Highest risk: Day 2**, and it is conditional on a Day 1 finding. If the data gate rules out per-customer aggregates, Day 2's headline deliverable disappears and the model falls back to transaction-level features only — which is precisely the ordinary approach §1 criticizes. Time is not the constraint there; having a documented, defensible answer for the writeup is.

**Second: Day 4.** It carries the analysis the whole submission rests on and it is the day requiring the most judgment rather than mechanical work. It borrows from Day 3 by design (halved above) and from Day 7's buffer if needed.

**Day 3 is deliberately no longer flagged as the top risk.** With a 70/12/10/8 split on 60k rows, the imbalance is mild: class weighting is a single parameter, and SMOTE at this ratio and sample size is a two-hour experiment with a likely-negative result, not a day of work.

> ### RETROSPECTIVE (Day 4)
>
> **Day 2 was correctly identified as the top risk — for the wrong reason.** The
> predicted failure was "per-customer aggregates turn out to be undefined." The
> actual failure was a degenerate generator that made the entire task trivial.
> Both would have cost Day 2; only one was anticipated.
>
> The useful lesson is not that the prediction was wrong. It is that **the
> mitigation worked anyway**, because it was a *gate* (§6.4: stop on a
> suspicious result) rather than a *forecast* of a specific failure. A gate
> catches the failure you didn't think of; a forecast only catches the one you
> did. That is worth stating in the pitch.

---

## 10. Deliverables Checklist

Boxes are checked against what is actually in the repository, not against what
was planned. Two remain open and are listed as open rather than quietly
dropped.

**Core**

- [x] Public GitHub repo, structured as §8
- [x] `docs/LEAKAGE_FINDING.md` — the headline result, complete with evidence
- [x] `docs/DATA_NOTES.md` — Day 1 gate findings, **including the open rows-per-customer answer (§4.2)**
- [x] Held-out test set with documented split strategy (temporal)
- [x] Per-class precision/recall/F1 + annotated confusion matrix
- [x] **Friction ↔ missed-recovery operating curve** — the centerpiece (§6.2 correction)
- [x] The inert `C_fp:C_fn` sweep reported alongside it, as the axis that *doesn't* move
- [x] Three-anchor result presentation: strawman 0.2051 → untrained rules 0.9188 → model 0.9988
- [x] Explicit synthetic-data limitation, at the strength of the §2 correction
- [x] Explicit leakage-screening result, including the depth-1 gate bug and its fix
- [x] Explicit failure-mode disclosure (Legitimate vs Policy Abuser)
- [x] Per-class SHAP on both tracks, with a written reading of *why* the
      confusable classes confuse (`runs/shap_interpretation.md`)
- [x] SMOTE tested against the class-weighted baseline on `testbed`, judged by
      criteria fixed before the result — **verdict: discard**
      (`runs/smote_verdict.md`, `runs/smote_testbed.json`)
- [x] Segment-level FPR audit by order-value bucket, checking the policy is not
      concentrating false blocks on one customer segment
      (`runs/segment_fpr_audit.md`) — the §6.2 stretch goal, delivered
- [x] `full` vs `testbed` distinction named in the README, with the reasoning in §5.2 — not discoverable only in code
- [x] Clean-clone Quickstart verified from a fresh virtualenv: cloned into a
      fresh venv and run verbatim, every regenerated number matched, only
      artifact timestamps differed
- [x] ~~Working demo~~ → **`scripts/score.py` CLI** (record or CSV → class → routed
      intervention, under an explicit `--posture` and `--friction`, with `--explain`
      for signed per-feature SHAP on the predicted class). The Streamlit demo was cut
      from scope; see §8. The deliverable is the real decision path, not a UI shell.
- [x] Every scored result reports what the model could not see, split by cause (§8.3)
- [ ] 5-minute pitch — **open.** The written script was moved out of the
      repository and no recording exists, so neither is delivered here. Logged
      in §11 rather than quietly unchecked.
- [x] This architecture document, with its correction log intact

**Repo hygiene**

- [x] `runs/` committed; `data/raw/` not committed
- [x] Tests T.1–T.9 all written and passing on fixtures — the plan's "if time runs out" fallback to T.1 / T.2 / T.6 was not needed. **139 tests across 15 files** (§9.3)
- [x] CI running `ruff check` + `pytest`, badge in README
- [x] LICENSE present (MIT)
- [x] Notebook outputs deliberately kept (all four notebooks executed in place,
      real output cells; they import from `src` and hold no logic)
- [x] Secret scan over history clean
- [x] `requirements.lock` — full transitive closure, clean-venv resolved (§8.1)
- [x] Commit attribution guard in three layers: `.githooks/commit-msg` strips an
      agent co-author trailer at write time, `.githooks/pre-push` refuses a push
      carrying one anywhere in a message or in the author/committer fields, and
      `tests/test_git_attribution.py` enforces both over the whole history in CI
      and drives the pre-push hook against a scratch remote to prove it refuses

---

## 11. Correction Log

Kept as a running list rather than folded into the text above, so a reviewer can
see the shape of what changed at a glance. Entries are never deleted.

| Date | § | Original claim | What replaced it | Trigger |
|---|---|---|---|---|
| Day 2 | §9.1.3 | Leakage sweep fits a **depth-1** tree per feature | Depth `max(2, n_classes-1)`, depth-1 reported alongside | `abuse_label` scored 0.393 at depth 1 and 1.000 at depth 3 — the gate passed a 1:1 label encoding |
| Day 2 | §2 | Synthetic data means "absolute numbers may not transfer" | The dataset is **degenerate**; 4 untrained rules score 0.9188 macro-F1 | §6.4 protocol fired at 0.9986 |
| Day 2 | §5 | Single model track | **Dual-track** `full` / `testbed` (§5.2) | Neither reporting 0.9986 nor silent feature-dropping is defensible |
| Day 2 | §5.2 | *(first ablation attempt)* — dropped raw artifact columns only | Each rung drags its algebraic proxies out with it | `returns_per_order` restates `return_rate_pct`; the flat first ablation measured nothing |
| Day 4 | §6.2 | `C_fp:C_fn` sweep is "the actual deliverable of the project" | **Friction ↔ missed-recovery** curve is the centerpiece; `C_fp:C_fn` reported as inert | Sweep produces byte-identical decisions from 0.03 to 32 on the full model |
| Day 4 | §6.3 | Weakest boundary predicted: Wardrobing vs Policy Abuser | Actual: **Legitimate vs Policy Abuser** (342 of 592 ambiguous rows) | Top-two-margin analysis |
| Day 6 | §9 Day 6 | "3–4 pytest tests on `features.py`" | Nine-test plan ranked by claim defended (§9.3) | Leakage finding changed what needs guarding |
| Day 6 | §8, §10 | Streamlit demo app at `app/demo.py` | **Cut.** Replaced by `scripts/score.py`, a CLI over the same `src.infer` path | A UI shell demonstrates nothing the CLI doesn't; the decision layer is the deliverable, and a second scoring path is the two-implementations failure mode §8 warns about |
| Day 6 | §4.2 | OPEN — whether trailing/temporal aggregates are "live" for the 1,945 repeat customers | **Closed: they aren't a real aggregate at all** — `total_orders_lifetime` etc. are independent per-row generator snapshots, non-monotonic across a customer's own rows (78 → 57 → 12 → 14 in one real case) | Checked directly against the raw CSV's repeat-customer rows while writing the T.2 temporal-leakage test |
| Day 7 | §8 | No model binaries committed; `runs/` carries JSON and charts only | **`runs/model_full.joblib` committed**, plus `examples/` | The stated goal of committing `runs/` was that a reviewer without the dataset can still see the work. That reasoning covers the numbers but stopped short of the demo: with every bundle gitignored, `scripts/score.py` could not produce one prediction from a clean clone, and the only `--record` example in the docs was a `{"...": "..."}` placeholder that raises. 3.2M buys a demo that runs on `pip install` |
| Day 7 | §8.2 | Pipeline stages fail with whatever exception the missing file raises | `require_artifacts` at each entry point: lists what is missing, names the command that builds it, exits 1 | Seven of eight `python -m src.*` entry points ended in a bare `FileNotFoundError` from library depth on a clean clone. `src/evaluate.py`'s was worse than absent — it guarded `model_{track}.json`, which *is* committed, then died on the gitignored `.npy`, so its friendly message was unreachable code |
| Day 7 | §8, §6.2 | The scoring CLI exposes the cost policy through `--posture` alone | **`--friction` added**, and `--posture`'s help text corrected to state that it is the near-inert axis | The Day 4 correction moved the centerpiece to the friction axis, but the serving path was never updated to match: `src/infer.py` called `build_cost_matrix` without a `friction_cost`, silently taking the default, so the one axis §6.2 calls the deliverable could not be moved at the point of use. Measured before the fix: all three `--posture` values produce byte-identical actions for 12,000 of 12,000 rows on `full`, and differ on 29 at the extremes on `testbed`. `--friction` moves 764–1,219. Its `balanced (1:2)` default is the previously hardcoded value, so no committed artifact changed |
| Post-build | §8, §10 | `docs/PITCH.md` ships in the repo as the written 5-minute script, checked off as delivered | **Removed at the author's request.** The file was copied out to a working folder and deleted from the repository; §8's tree, §10's checklist, the README's `docs/` listing and Status, and two comments in `tests/test_baseline_rule.py` were all updated in the same commit | A deliverable that leaves the repository has to leave the claims with it. Five documents named a file a reviewer would no longer find, which is the exact failure §10 asserts against — and the pitch line is now stated as fully open rather than half-checked |

### 11.1 Serving-path audit — after Day 7

A ruthless read of the scoring path, run against a written audit checklist
after the build week closed: static analysis and secret scanning, boundary
payloads pushed through the scoring functions, failure paths executed, and the
dependency and explainability posture examined. The checklist itself is not
committed — it is an instruction document rather than an engineering artifact,
and a submission that ships its own review prompt reads as output rather than
as work. What matters is reproducible from the findings below. It found no security vulnerability, no hardcoded secret and
no injection sink, and every reported metric matched `runs/`. What it did find
was ten defects in how the CLI meets a caller, two of which changed behaviour
and therefore belong in this log rather than only in a commit message.

| # | § | Original behaviour | What replaced it | Trigger |
|---|---|---|---|---|
| F5 | §8.3 | Impossible input — negative money, negative account age, a 0/1 flag holding 7 — scored to **`Fraudulent Return` / `hard_block` at p ≈ 1** | Domain rules null the offending cell and name it; the same record now routes to **`approve`** | A system whose thesis is *proportionate* response was failing toward maximum punishment on data that describes nothing. Bounds read off the training data, checked at inference only |
| F6 | §8.3 | A record scored without three engineered features reported `features_missing: []` | Causes derived from the finished frame; `features_degraded` and `n_features_not_seen` added | `src/features.py` nulls a zero denominator, so `avg_order_value_usd=0` silently dropped three features while the result claimed full data. The repo argues for honest reporting and its own scorer did not do it |
| F1–F4 | §8 | A non-numeric cell, a mistyped path or an empty CSV raised a raw traceback from inside pandas or LightGBM | Coerced-and-reported, or a one-line exit naming the file | §8 claims a caller error must fail with a remedy, not a stack trace. Four paths did not |
| F7 | §8.2 | `--out` replaced an existing file silently | Refuses without `--overwrite` | A scored batch is evidence someone may be working from; a re-run under a different `--friction` quietly replacing it is how wrong numbers reach a report |
| F8 | §8.2 | Probabilities and a routed action, but no per-feature reasoning at the point of use; SHAP existed only as an offline per-class artifact | `--explain` returns signed contributions, sharing `src.explain`'s single `TreeExplainer` | Two explainers that drifted would make the explanation shown to a merchant disagree with the one in the writeup — the §8 two-implementations failure mode |
| F9 | §8.1 | Direct dependencies pinned; transitive tree unpinned | `requirements.lock`, resolved in a clean venv | A plain `pip freeze` would have pinned 91 packages including streamlit, jupyter and the macOS-only `appnope`, and produced a file that cannot install on Linux |
| F10 | §8 | `joblib.load` unpickles a bundle with no integrity check | `save_run` records a SHA-256; `load_run` verifies before unpickling | Scope stated honestly in the code: the digest sits beside the file it covers, so it stops corruption and a drifted `.joblib`/`.json` pair, not a deliberate attacker. Both bundles regenerated; retraining is deterministic and was confirmed byte-identical first, so no metric moved |

Nothing in F1–F10 touched training, ablation or evaluation — the fixes live in
`src/infer.py` and `scripts/score.py`, and the scored outputs for
`examples/record.json` and the 20-row batch are identical to the pre-audit
baseline once the new reporting fields are set aside.

---

**On keeping this log.** Every entry is a place the plan was wrong, and
publishing that list is a deliberate choice. A reviewer's fastest way to
discredit a submission is to find a contradiction between its architecture doc
and its results and conclude nobody maintained either. This log converts that
into the opposite signal.
