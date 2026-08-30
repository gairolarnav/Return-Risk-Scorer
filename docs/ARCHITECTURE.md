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
> | Always-predict-Legitimate strawman | 0.6950 | 0.2050 |
> | **Four hand-written rules, no training** | **0.9425** | **0.9188** |
> | LightGBM, full feature set | 0.9994 | 0.9986 |
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

**Why the testbed cannot be promoted to headline model:** the ladder degrades
smoothly from 0.999 to 0.858 with no natural cut point, so any rung is
arbitrary. "We dropped features until the task got hard" is not a result, and
a reviewer will say so. Naming the testbed as not-a-model, in the README and in
the code, is the only version of this that survives questioning.

---

## 6. Evaluation Framework — "Honest Metrics" (the core grading criterion)

### 6.1 What will NOT be the headline metric
Macro-accuracy. With a 70% legitimate majority class, a model predicting "legitimate" for everything scores ~70% accuracy while being useless. This strawman baseline is computed and recorded on Day 2 alongside the first real model, and shown explicitly in the writeup to justify why accuracy is rejected.

*Recorded: the strawman scores 0.6950 accuracy / 0.2050 macro-F1. Per the §2
correction, it is now reported as the first of three anchors — strawman, then
the untrained rule baseline at 0.9188, then the model at 0.9986. **The sequence
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
> **That curve is the centerpiece** (`runs/friction_tradeoff_testbed.png`), and
> it is what the README leads to after the leakage finding.
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
`src/` or `app/` imports a generative or LLM library. The track criterion is
"anything offense-capable is disqualified" — a claim in a document is weaker
evidence of compliance than a test in CI.*

---

## 8. Repository Structure

```
return-risk-scorer/
├── .github/workflows/
│   └── ci.yml                   # ruff check + pytest on push
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
│   ├── model.py                 # train/predict/threshold logic; `full` | `testbed` track selector
│   ├── evaluate.py              # per-class PR, confusion matrix, cost + friction sweeps
│   └── infer.py                 # single-record scoring function
├── app/
│   └── demo.py                  # Streamlit demo
├── runs/                        # committed — reviewer reads every headline number without the dataset
│   ├── evaluation_full.json
│   ├── evaluation_testbed.json
│   └── friction_tradeoff_testbed.png
├── docs/
│   ├── ARCHITECTURE.md          # this file
│   ├── LEAKAGE_FINDING.md       # the headline result — read this second, after the README
│   └── DATA_NOTES.md            # Day 1 gate findings + split-strategy decision
├── tests/                       # see §9.3
│   ├── test_leakage.py
│   ├── test_features.py
│   ├── test_split.py
│   └── test_compliance.py
├── requirements.txt
├── LICENSE
└── README.md
```

**On notebooks vs. `src/`:** the four notebooks exist for presentation and exploration only. They import from `src` and contain no logic of their own. The failure mode otherwise is two divergent implementations of the same feature builder, which a reviewer will find and which invalidates any number the notebook produces.

**On what is and isn't committed (added Day 6).** `runs/` **is** committed and
`data/raw/` is **not**. Most reviewers will never download a 60k-row Kaggle CSV,
so every headline number must be readable from the repo alone. Committing
someone else's dataset is separately a bloat and licensing problem.

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
- `streamlit` — demo app
- `matplotlib` / `seaborn` — PR curves, confusion matrix plots
- `pytest` — tests
- `ruff` — linting (one tool, zero config)

**Deliberately not used:** MLflow, Weights & Biases, or any experiment-tracking service. At this scale a JSON metrics file per run in `runs/` covers the need and is one fewer thing to explain in a five-minute pitch.

**requirements.txt — pinned exactly, not `>=`:**
```
pandas==2.2.3
numpy==1.26.4
pyarrow==17.0.0
lightgbm==4.5.0
scikit-learn==1.5.2
imbalanced-learn==0.12.4
shap==0.46.0
joblib==1.4.2
streamlit==1.39.0
matplotlib==3.9.2
seaborn==0.13.2
pytest==8.3.3
ruff==0.7.0
```

Exact pins matter here for a specific reason: this repo is submitted for review and may be cloned weeks later. `>=` constraints guarantee that a future resolver picks different versions than the ones the reported numbers were produced with, which quietly breaks both reproducibility and the Quickstart.

**Platform note — Apple Silicon:** `pip install lightgbm` succeeds but fails at import without OpenMP. Run `brew install libomp` **before** installing requirements. This belongs in Day 1 setup, not discovered mid-build, and it appears in the README above the `pip install` line rather than in a footnote.

**Reproducibility:** Fix a single random seed (`RANDOM_STATE = 42`) at the top of `src/model.py` and reuse it everywhere — train/test split, model init, SMOTE — so results are reproducible run to run and defensible if a panelist asks to re-run the notebook.

## 8.2 Quickstart

```bash
git clone <repo-url>
cd return-risk-scorer
python3.11 -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows
# macOS on Apple Silicon: brew install libomp
pip install -r requirements.txt

# place the Kaggle CSV at data/raw/returns.csv, then:
python -m src.data_gate        # Day 1 checks: customer viability, split strategy, leakage sweep
python -m src.features         # builds processed train/test sets
python -m src.model            # trains + saves the model
python -m src.evaluate         # generates metrics, confusion matrix, cost + friction tables
streamlit run app/demo.py      # launches the interactive demo

# no dataset? every headline number is committed:
#   runs/evaluation_full.json, runs/evaluation_testbed.json,
#   runs/friction_tradeoff_testbed.png
# and the test suite runs on fixtures alone:
pytest
```

This sequence is verified from a clean clone into a fresh virtualenv on Day 6. It is not assumed to work — hackathon repos routinely fail at exactly this step, and it is the first thing a reviewer touches.

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

1. **Is there a usable repeat-customer identifier?** Report the distribution of rows-per-customer. A median of 1 means every constructed feature in §4.2 is undefined, and Day 2's scope changes to the §4.1 fallback. *(See the §4.2 open item — this answer belongs in `docs/DATA_NOTES.md` and must be stated in the README before submission.)*
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
| T.6 | No module under `src/` or `app/` imports a generative or LLM library | §7 — the one criterion that disqualifies | **Must** |
| T.3 | Train/test disjoint on row key; `max(train ts) <= min(test ts)` | Every reported number | Should |
| T.4 | No ablation rung retains an algebraic restatement of a feature it drops | The §4.2 proxy trap, frozen as an assertion | Should |
| T.5 | Feature builders match hand-computed values on a ~10-row fixture | That someone understood the features | Should |
| T.7 | Re-running evaluation reproduces `runs/evaluation_full.json` within tolerance | Makes "reproducible" a checked claim, enforceable in CI | Should |
| T.8 | Same input + `RANDOM_STATE` → identical output twice | §8.1 reproducibility | Nice |
| T.9 | `src/infer.py` returns 4 probabilities summing to 1.0 + a valid intervention | The demo path a judge runs live | Nice |

**If time runs out, T.1 / T.2 / T.6 are worth more than the other six
combined.** Write those three and record the rest as not-reached, with the
reason. Three tests that defend the thesis
plus an honest note about what was skipped reads better than nine shallow ones.

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

**Core**

- [ ] Public GitHub repo, structured as §8
- [ ] `docs/LEAKAGE_FINDING.md` — the headline result, complete with evidence
- [ ] `docs/DATA_NOTES.md` — Day 1 gate findings, **including the open rows-per-customer answer (§4.2)**
- [ ] Held-out test set with documented split strategy (temporal)
- [ ] Per-class precision/recall/F1 + annotated confusion matrix
- [ ] **Friction ↔ missed-recovery operating curve** — the centerpiece (§6.2 correction)
- [ ] The inert `C_fp:C_fn` sweep reported alongside it, as the axis that *doesn't* move
- [ ] Three-anchor result presentation: strawman 0.2050 → untrained rules 0.9188 → model 0.9986
- [ ] Explicit synthetic-data limitation, at the strength of the §2 correction
- [ ] Explicit leakage-screening result, including the depth-1 gate bug and its fix
- [ ] Explicit failure-mode disclosure (Legitimate vs Policy Abuser)
- [ ] `full` vs `testbed` distinction stated in the README, not discoverable only in code
- [ ] Clean-clone Quickstart verified from a fresh virtualenv
- [ ] Working demo (record → class → intervention)
- [ ] 5-minute pitch video
- [ ] This architecture document, with its correction log intact

**Repo hygiene**

- [ ] `runs/` committed; `data/raw/` not committed
- [ ] Tests T.1 / T.2 / T.6 written and passing on fixtures
- [ ] CI running `ruff check` + `pytest`, badge in README
- [ ] LICENSE present
- [ ] Notebook outputs cleared or deliberately kept
- [ ] Secret scan over history clean

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
| Day 6 | §4.2 | OPEN — whether trailing/temporal aggregates are "live" for the 1,945 repeat customers | **Closed: they aren't a real aggregate at all** — `total_orders_lifetime` etc. are independent per-row generator snapshots, non-monotonic across a customer's own rows (78 → 57 → 12 → 14 in one real case) | Checked directly against the raw CSV's repeat-customer rows while writing the T.2 temporal-leakage test |

**On keeping this log.** Every entry is a place the plan was wrong, and
publishing that list is a deliberate choice. A reviewer's fastest way to
discredit a submission is to find a contradiction between its architecture doc
and its results and conclude nobody maintained either. This log converts that
into the opposite signal.
