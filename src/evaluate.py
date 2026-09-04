"""
Evaluation framework (ARCHITECTURE.md §6) — "honest metrics".

Reports per-class precision/recall/F1, the confusion matrix, per-class PR
curves, and the cost-calibrated decision policy (§6.2) — the centerpiece.

The decision rule
-----------------
The model emits a probability vector p over the four classes. Rather than
taking argmax, the decision layer picks the action minimising expected cost:

    action* = argmin_a  sum_c  p(c) * C[c, a]

where C[c, a] is the merchant's cost of taking action `a` when the true class
is `c`. This is the standard Bayes-optimal decision rule, and it is what makes
the cost matrix an actual policy instrument rather than a table in a slide:
changing C changes which returns get blocked, and by how much.

Note this replaces argmax rather than layering a threshold on top of it. A
per-class threshold is a special case of the above for two classes; with four
asymmetric actions the expected-cost form is the honest generalisation, and
sweeping `C_fp : C_fn` moves the whole decision boundary rather than one cut
point.

On the two dominant costs
-------------------------
Per §6.2, `C_fp` (hard-blocking a legitimate customer) and `C_fn` (letting a
real fraudulent return through) are NOT declared with one larger than the
other a priori — that would presuppose the answer. Their ratio is swept
across defensible merchant postures and the range is the deliverable.

What the sweep shows on this dataset
------------------------------------
On the `full` track the sweep is FLAT: ~0% of test rows sit near a decision
boundary, so every posture yields identical decisions. That flatness is
reported as a positive finding — it is direct evidence that the dataset is
degenerate (docs/LEAKAGE_FINDING.md), not a failed experiment. The method is
then demonstrated on the `testbed` track, where ~17% of rows are
threshold-sensitive and the postures genuinely diverge.

Run as:
    python -m src.evaluate
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — this runs in CI and from a plain shell
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize

from src.data_gate import NEEDS_MODEL, require_artifacts

RUNS_DIR = Path("runs")

# Actions the decision layer can route to (ARCHITECTURE.md §1/§3), ordered
# from least to most punitive.
ACTIONS = ["approve", "soft_friction", "hard_block"]

# Which action each true class *should* receive, absent any uncertainty.
CLASS_TARGET_ACTION = {
    "Legitimate": "approve",
    "Wardrobing": "soft_friction",
    "Policy Abuser": "soft_friction",
    "Fraudulent Return": "hard_block",
}


# Merchant postures, denominated in USD so the numbers mean something rather
# than being abstract 1:3 ratios. All three are defensible; which one a given
# merchant holds is a business posture, not a fact recoverable from the data,
# which is exactly why §6.2 sweeps rather than picks.
#
#   C_fn  cost of approving a fraudulent return: the item, the shipping,
#         the restocking that never happens. Roughly the order value.
#   C_fp  cost of hard-blocking an honest customer: not the order — the
#         customer. Anchored on lost lifetime value, which is where the
#         postures genuinely disagree.
DEFAULT_POSTURES: dict[str, tuple[float, float]] = {
    # Treats a lost customer and a lost item as comparable.
    "loss-neutral (1:1)": (120.0, 120.0),
    # High-margin/high-CLV merchant: churning a good customer dwarfs eating
    # the occasional fraudulent refund.
    "retention-weighted (8:1)": (960.0, 120.0),
    # Thin-margin/high-fraud merchant: refund losses are existential, and a
    # blocked customer can be won back through support.
    "loss-averse (1:8)": (120.0, 960.0),
}

# Named postures on the OTHER axis — the one that actually moves (Day 4
# correction, see sweep_friction_curve's docstring). DEFAULT_POSTURES sweeps
# C_fp : C_fn, which this dataset renders nearly inert: on `full` every posture
# above produces byte-identical decisions, and on `testbed` the extremes differ
# on 29 of 12,000 rows. The live tension is approve vs. soft-friction — how
# aggressively to fee and flag customers who might just be heavy returners.
#
# Values are `friction_cost`, against build_cost_matrix's fixed
# missed_recovery_cost of 2.0; the ratio in each name is friction : recovery.
# Holding the denominator at the serving default rather than adopting
# sweep_friction_curve's 20.0 is deliberate — 1.0 is the value the decision
# layer has always used, so "balanced" reproduces every previously committed
# artifact (runs/segment_fpr_*.json especially) exactly.
#
# Measured on the testbed track's real test probabilities, these three span
# essentially the whole published operating curve
# (runs/friction_tradeoff_testbed.csv: 2.78%-24.79% / 85.49%-99.57%):
#
#   posture                legitimate frictioned   wardrobers/abusers caught
#   recovery-first (1:20)          22.35%                    99.37%
#   balanced (1:2)                  9.06%                    95.44%
#   approve-first (4:1)             2.91%                    85.65%
FRICTION_POSTURES: dict[str, float] = {
    # Friction is cheap next to an abuse pattern going unrecovered: fee and
    # flag liberally, and accept that roughly a fifth of honest customers
    # feel it.
    "recovery-first (1:20)": 0.1,
    # The historical serving default.
    "balanced (1:2)": 1.0,
    # Friction is expensive — a return fee on an honest customer is a
    # retention event. Intervene only on strong evidence.
    "approve-first (4:1)": 8.0,
}

# A fraudulent return that gets soft-friction (a return fee, inspection)
# instead of a hard block isn't a full miss — the fee recovers part of the
# loss and the inspection may still catch it. Costed as a fraction of the
# full C_fn (approving it outright), not swept, for the same reason the
# other second-order cells in build_cost_matrix aren't: only C_fp and C_fn
# are the sweep's free parameters.
FRAUD_SOFT_FRICTION_COST_FACTOR = 0.4


def build_cost_matrix(
    class_names: list[str],
    c_fp: float,
    c_fn: float,
    friction_cost: float = 1.0,
    missed_recovery_cost: float = 2.0,
    over_penalisation_cost: float = 3.0,
) -> pd.DataFrame:
    """Cost of taking each action given the true class (§6.2).

    Rows are true classes, columns are actions. Only `c_fp` and `c_fn` are
    swept; the remaining cells are fixed at modest values because they are
    genuinely second-order (a wrongly-applied return fee is an annoyance,
    not a lost customer or a lost refund) and letting every cell float would
    make the sweep uninterpretable.

    c_fp  cost of hard-blocking a genuinely Legitimate return
    c_fn  cost of approving a genuinely Fraudulent Return
    """
    cost = pd.DataFrame(0.0, index=class_names, columns=ACTIONS)

    for cls in class_names:
        target = CLASS_TARGET_ACTION.get(cls)
        for action in ACTIONS:
            if action == target:
                cost.loc[cls, action] = 0.0
            elif cls == "Legitimate":
                # Over-reacting to an honest customer.
                cost.loc[cls, action] = friction_cost if action == "soft_friction" else c_fp
            elif cls == "Fraudulent Return":
                # Under-reacting to real fraud.
                cost.loc[cls, action] = (
                    c_fn if action == "approve" else c_fn * FRAUD_SOFT_FRICTION_COST_FACTOR
                )
            else:
                # Wardrobing / Policy Abuser: mild either way.
                cost.loc[cls, action] = (
                    missed_recovery_cost if action == "approve" else over_penalisation_cost
                )
    return cost


def expected_cost_decision(proba: np.ndarray, cost_matrix: pd.DataFrame) -> np.ndarray:
    """Bayes-optimal action per row: argmin_a sum_c p(c) C[c,a].

    proba columns must be ordered to match cost_matrix.index.
    """
    expected = proba @ cost_matrix.to_numpy()  # (n_rows, n_actions)
    return expected.argmin(axis=1)


def realised_cost(y_true_idx: np.ndarray, action_idx: np.ndarray, cost_matrix: pd.DataFrame) -> float:
    """Total cost actually incurred: sum of cost_matrix[true_class, taken_action]
    over all rows. Not comparable across postures on its own — see oracle_cost."""
    cost = cost_matrix.to_numpy()
    return float(cost[y_true_idx, action_idx].sum())


def oracle_cost(y_true_idx: np.ndarray, cost_matrix: pd.DataFrame) -> float:
    """Cost a perfect classifier would incur under this matrix.

    Needed because realised cost is denominated in each posture's own units —
    a loss-averse posture reports a bigger number simply because its numbers
    are bigger, so comparing raw cost across postures is meaningless. Regret
    against this oracle is the comparable quantity.
    """
    cost = cost_matrix.to_numpy()
    return float(cost[y_true_idx, :].min(axis=1).sum())


def sweep_cost_ratios(
    y_true_idx: np.ndarray,
    proba: np.ndarray,
    class_names: list[str],
    postures: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Sweep C_fp : C_fn across merchant postures (§6.2).

    Reports, per posture: the action mix, the realised total cost, the
    false-block rate on legitimate customers, and fraud recall. Those last
    two are the tradeoff the whole framework exists to expose — a posture
    that improves one necessarily worsens the other, and the point is to
    show by how much rather than to pick a winner.
    """
    postures = postures or DEFAULT_POSTURES

    legit_idx = class_names.index("Legitimate") if "Legitimate" in class_names else None
    fraud_idx = class_names.index("Fraudulent Return") if "Fraudulent Return" in class_names else None
    block_action = ACTIONS.index("hard_block")
    approve_action = ACTIONS.index("approve")

    rows = []
    for name, (c_fp, c_fn) in postures.items():
        cost_matrix = build_cost_matrix(class_names, c_fp=c_fp, c_fn=c_fn)
        actions = expected_cost_decision(proba, cost_matrix)

        realised = realised_cost(y_true_idx, actions, cost_matrix)
        oracle = oracle_cost(y_true_idx, cost_matrix)
        n = len(y_true_idx)
        row = {
            "posture": name,
            "c_fp": c_fp,
            "c_fn": c_fn,
            "ratio_fp_to_fn": c_fp / c_fn,
            # Regret per 1,000 returns against a perfect classifier. Unlike raw
            # cost this IS comparable across postures.
            "regret_per_1k": (realised - oracle) / n * 1000,
        }
        for i, action in enumerate(ACTIONS):
            row[f"pct_{action}"] = float((actions == i).mean())

        if legit_idx is not None:
            legit_mask = y_true_idx == legit_idx
            row["false_block_rate_on_legit"] = float((actions[legit_mask] == block_action).mean())
        if fraud_idx is not None:
            fraud_mask = y_true_idx == fraud_idx
            row["fraud_caught_rate"] = float((actions[fraud_mask] != approve_action).mean())
            row["fraud_hard_blocked_rate"] = float((actions[fraud_mask] == block_action).mean())

        rows.append(row)

    return pd.DataFrame(rows)


def sweep_tradeoff_curve(
    y_true_idx: np.ndarray,
    proba: np.ndarray,
    class_names: list[str],
    c_fn: float = 120.0,
    ratios: np.ndarray | None = None,
) -> pd.DataFrame:
    """Continuous sweep of C_fp/C_fn, holding C_fn fixed.

    Three named postures are three points; the operating curve they sit on is
    the actual deliverable, because it shows a merchant the whole menu of
    achievable (false-block, fraud-caught) pairs and lets them locate their
    own posture on it rather than accepting ours.
    """
    ratios = ratios if ratios is not None else np.logspace(-1.5, 1.5, 31)
    legit_idx = class_names.index("Legitimate")
    fraud_idx = class_names.index("Fraudulent Return")
    block_action = ACTIONS.index("hard_block")
    approve_action = ACTIONS.index("approve")

    legit_mask = y_true_idx == legit_idx
    fraud_mask = y_true_idx == fraud_idx

    rows = []
    for ratio in ratios:
        c_fp = c_fn * ratio
        cost_matrix = build_cost_matrix(class_names, c_fp=c_fp, c_fn=c_fn)
        actions = expected_cost_decision(proba, cost_matrix)
        rows.append(
            {
                "ratio_fp_to_fn": float(ratio),
                "c_fp": float(c_fp),
                "c_fn": float(c_fn),
                "false_block_rate_on_legit": float((actions[legit_mask] == block_action).mean()),
                "fraud_caught_rate": float((actions[fraud_mask] != approve_action).mean()),
                "fraud_hard_blocked_rate": float((actions[fraud_mask] == block_action).mean()),
                "pct_hard_block": float((actions == block_action).mean()),
            }
        )
    return pd.DataFrame(rows)


def sweep_friction_curve(
    y_true_idx: np.ndarray,
    proba: np.ndarray,
    class_names: list[str],
    missed_recovery_cost: float = 20.0,
    friction_costs: np.ndarray | None = None,
) -> pd.DataFrame:
    """Sweep the *friction* axis: approve vs. soft-friction.

    Why this axis exists at all — a Day 4 finding, not part of the original
    plan. ARCHITECTURE.md §6.2 assumed the decisive tension was
    `C_fp : C_fn`, i.e. blocking an honest customer vs. letting real fraud
    through. On this data that axis is nearly inert: Fraudulent Return is the
    *easiest* class to separate, and only 14 of 592 genuinely ambiguous test
    rows involve it at all. Sweeping it moves the false-block rate from 0.00%
    to 0.23%.

    The ambiguity is concentrated somewhere else entirely — 342 of those 592
    rows are Legitimate vs. Policy Abuser. So the decision a merchant actually
    agonises over is not "block or not", it is "how aggressively do I apply
    return fees and pattern flags to customers who might just be heavy
    returners". That is the approve/soft-friction boundary, governed by
    friction cost against missed-recovery cost, and it moves the outcome by
    an order of magnitude.

    Reporting the inert axis and stopping would have satisfied the letter of
    §6.2 while hiding the fact that it measured the wrong thing.
    """
    friction_costs = (
        friction_costs if friction_costs is not None else np.logspace(np.log10(0.5), np.log10(80), 25)
    )
    legit_idx = class_names.index("Legitimate")
    abuser_idx = [class_names.index(c) for c in ("Policy Abuser", "Wardrobing") if c in class_names]
    approve_action = ACTIONS.index("approve")

    legit_mask = y_true_idx == legit_idx
    abuser_mask = np.isin(y_true_idx, abuser_idx)

    rows = []
    for fc in friction_costs:
        cost_matrix = build_cost_matrix(
            class_names,
            c_fp=120.0,
            c_fn=120.0,
            friction_cost=float(fc),
            missed_recovery_cost=missed_recovery_cost,
        )
        actions = expected_cost_decision(proba, cost_matrix)
        rows.append(
            {
                "friction_cost": float(fc),
                "missed_recovery_cost": missed_recovery_cost,
                "ratio_friction_to_recovery": float(fc) / missed_recovery_cost,
                # Share of honest customers who get *any* intervention.
                "legit_frictioned_rate": float((actions[legit_mask] != approve_action).mean()),
                # Share of wardrobers/policy abusers who get caught by one.
                "abuser_caught_rate": float((actions[abuser_mask] != approve_action).mean()),
                "pct_approve": float((actions == approve_action).mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_friction_curve(curve: pd.DataFrame, track: str, out_path: Path) -> None:
    """Render the friction-axis operating curve (the §6.2 centerpiece) as
    two panels: rate vs. posture, and the achievable-tradeoff scatter.

    Args: `curve` from sweep_friction_curve; `track` for the title;
    `out_path` to save the PNG to. Returns None — writes the file and
    closes the figure as a side effect (not closing it would leak memory
    across the two tracks' calls in evaluate_track's loop)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(curve["ratio_friction_to_recovery"], curve["legit_frictioned_rate"] * 100,
             marker="o", ms=3, label="legitimate customers given friction")
    ax1.plot(curve["ratio_friction_to_recovery"], curve["abuser_caught_rate"] * 100,
             marker="s", ms=3, label="wardrobers / policy abusers caught")
    ax1.set_xscale("log")
    ax1.set_xlabel("friction cost / missed-recovery cost\n(higher = more reluctant to add friction)")
    ax1.set_ylabel("% of class")
    ax1.set_title(f"Friction policy vs. cost posture — {track}")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(curve["legit_frictioned_rate"] * 100, curve["abuser_caught_rate"] * 100,
             marker="o", ms=3)
    ax2.set_xlabel("legitimate customers given friction (%)")
    ax2.set_ylabel("wardrobers / policy abusers caught (%)")
    ax2.set_title("Achievable operating points — friction axis")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_tradeoff_curve(curve: pd.DataFrame, track: str, out_path: Path) -> None:
    """Render the C_fp:C_fn operating curve as two panels: rate vs.
    posture, and the achievable-tradeoff scatter. On `full` this curve is
    the flat-sweep finding itself (see sweep_is_degenerate) — the plot
    still renders, it just shows a horizontal line, which is the point.

    Args: `curve` from sweep_tradeoff_curve; `track` for the title;
    `out_path` to save the PNG to. Returns None, closes the figure
    (see plot_friction_curve for why that matters)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(curve["ratio_fp_to_fn"], curve["false_block_rate_on_legit"] * 100,
             marker="o", ms=3, label="false hard-block rate on legitimate")
    ax1.plot(curve["ratio_fp_to_fn"], curve["fraud_hard_blocked_rate"] * 100,
             marker="s", ms=3, label="fraudulent returns hard-blocked")
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$C_{fp} / C_{fn}$   (higher = more customer-protective)")
    ax1.set_ylabel("% of class")
    ax1.set_title(f"Decision policy vs. cost posture — {track}")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(curve["false_block_rate_on_legit"] * 100,
             curve["fraud_hard_blocked_rate"] * 100, marker="o", ms=3)
    ax2.set_xlabel("false hard-block rate on legitimate customers (%)")
    ax2.set_ylabel("fraudulent returns hard-blocked (%)")
    ax2.set_title("Achievable operating points")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def sweep_is_degenerate(sweep: pd.DataFrame, tol: float = 1e-9) -> bool:
    """True when every posture produced the same decisions.

    This is the check that turns a flat sweep into a reportable finding
    instead of an unnoticed non-result.
    """
    cols = [c for c in sweep.columns if c.startswith("pct_") or c.endswith("_rate")]
    return bool(all(sweep[c].max() - sweep[c].min() < tol for c in cols))


def per_class_report(y_true_idx, pred_idx, class_names: list[str]) -> pd.DataFrame:
    """Per-class precision/recall/F1/support as a DataFrame (rows = classes
    plus accuracy/macro avg/weighted avg). The non-obvious part:
    `zero_division=0` rather than the sklearn default warning-and-NaN — a
    class with zero predicted samples reports 0.0, not a silent NaN that
    would otherwise break the macro average downstream."""
    report = classification_report(
        y_true_idx, pred_idx, target_names=class_names, output_dict=True, zero_division=0
    )
    return pd.DataFrame(report).transpose()


def plot_confusion_matrix(y_true_idx, pred_idx, class_names: list[str], out_path: Path) -> None:
    """Render a 4x4 annotated confusion-matrix heatmap to out_path.
    `labels=range(len(class_names))` pins the row/column order to
    class_names explicitly, rather than trusting sklearn's default (the
    sorted set of labels actually present) to match — those can silently
    diverge if a class has zero predictions in a given run."""
    cm = confusion_matrix(y_true_idx, pred_idx, labels=range(len(class_names)))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax, cbar=False,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_class_pr_curves(y_true_idx, proba, class_names: list[str], out_path: Path) -> None:
    """PR rather than ROC — under a 70% majority class ROC-AUC is optimistic
    and hides exactly the minority-class behaviour that matters here (§6.2)."""
    y_bin = label_binarize(y_true_idx, classes=range(len(class_names)))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for i, name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], proba[:, i])
        ax.plot(recall, precision, label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Per-class precision-recall curves")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


AMBIGUITY_MARGIN = 0.3


def ambiguous_class_pairs(
    proba: np.ndarray, class_names: list[str], margin: float = AMBIGUITY_MARGIN
) -> dict:
    """Rows where the model's own top-two predicted probabilities are within
    `margin` of each other -- i.e. rows a cost posture could actually move.

    This is the runnable source for the "592 ambiguous rows, 342 of them
    Legitimate vs Policy Abuser, only 14 involving Fraudulent Return" figures
    quoted throughout docs/, which previously existed only as
    prose with no committed code behind them (Day 6 correction: every quoted
    figure must be regenerable). Grouped by the model's *predicted*
    top-two classes, not the true label, since the question is which classes
    the model itself struggles to tell apart.
    """
    sorted_p = np.sort(proba, axis=1)
    is_ambiguous = (sorted_p[:, -1] - sorted_p[:, -2]) < margin
    top2_idx = np.argsort(proba, axis=1)[:, -2:]

    pair_counts: dict[str, int] = {}
    for row in top2_idx[is_ambiguous]:
        pair = " vs ".join(sorted(class_names[i] for i in row))
        pair_counts[pair] = pair_counts.get(pair, 0) + 1

    fraud_involved = sum(n for pair, n in pair_counts.items() if "Fraudulent Return" in pair)
    return {
        "margin_threshold": margin,
        "n_total": int(len(proba)),
        "n_ambiguous": int(is_ambiguous.sum()),
        "pair_counts": dict(sorted(pair_counts.items(), key=lambda kv: -kv[1])),
        "fraudulent_return_involved": int(fraud_involved),
    }


def weakest_boundary(y_true_idx, pred_idx, class_names: list[str]) -> dict:
    """Largest off-diagonal confusion pair — the failure-mode disclosure (§6.3),
    computed rather than asserted so the writeup reports what actually happened
    instead of what the architecture doc predicted would happen."""
    cm = confusion_matrix(y_true_idx, pred_idx, labels=range(len(class_names)))
    np.fill_diagonal(cm, 0)
    i, j = np.unravel_index(cm.argmax(), cm.shape)
    return {
        "actual": class_names[i],
        "predicted_as": class_names[j],
        "n_cases": int(cm[i, j]),
    }


def evaluate_track(track: str) -> dict:
    """Full evaluation for one track ("full" or "testbed"): per-class
    metrics, confusion/PR-curve plots, both cost-sweep axes, and the
    ambiguity analysis, all written to runs/.

    Args: track name — loads runs/model_{track}.{json,joblib,_proba.npy,
        _ytest.npy}, which src.model.run() must have already produced.
    Returns: the same dict written to runs/evaluation_{track}.json (not a
    superset — ambiguous_class_pairs' output is written separately to
    runs/ambiguity_{track}.json rather than folded in here, so that
    already-committed file's content never changes when this runs again).
    Exits with a "run src.model first" message rather than a bare traceback if
    the model hasn't been trained yet.
    """
    run_json = RUNS_DIR / f"model_{track}.json"
    # All four artifacts, not just the JSON. The JSON is committed (it holds the
    # headline numbers, so a reviewer can read them without the dataset) while
    # the arrays and the testbed bundle are gitignored — so guarding on the JSON
    # alone checked the one file that is always present and then died on
    # np.load. That made this the only unreachable error path in the repo, and
    # a clean clone got a bare FileNotFoundError from library depth instead.
    require_artifacts(
        [
            run_json,
            RUNS_DIR / f"model_{track}_proba.npy",
            RUNS_DIR / f"model_{track}_ytest.npy",
            RUNS_DIR / f"model_{track}.joblib",
        ],
        NEEDS_MODEL,
    )

    meta = json.loads(run_json.read_text())
    proba = np.load(RUNS_DIR / f"model_{track}_proba.npy")
    y_true_idx = np.load(RUNS_DIR / f"model_{track}_ytest.npy")

    import joblib

    bundle = joblib.load(RUNS_DIR / f"model_{track}.joblib")
    class_names = list(bundle["label_encoder"].classes_)

    pred_idx = proba.argmax(axis=1)

    report = per_class_report(y_true_idx, pred_idx, class_names)
    plot_confusion_matrix(y_true_idx, pred_idx, class_names, RUNS_DIR / f"confusion_{track}.png")
    plot_per_class_pr_curves(y_true_idx, proba, class_names, RUNS_DIR / f"pr_curves_{track}.png")

    sweep = sweep_cost_ratios(y_true_idx, proba, class_names)
    degenerate = sweep_is_degenerate(sweep)

    curve = sweep_tradeoff_curve(y_true_idx, proba, class_names)
    plot_tradeoff_curve(curve, track, RUNS_DIR / f"cost_tradeoff_{track}.png")
    curve.to_csv(RUNS_DIR / f"cost_tradeoff_{track}.csv", index=False)

    friction = sweep_friction_curve(y_true_idx, proba, class_names)
    plot_friction_curve(friction, track, RUNS_DIR / f"friction_tradeoff_{track}.png")
    friction.to_csv(RUNS_DIR / f"friction_tradeoff_{track}.csv", index=False)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_idx, pred_idx, labels=range(len(class_names)), zero_division=0
    )

    out = {
        "track": track,
        "macro_f1": meta["macro_f1"],
        "accuracy": meta["accuracy"],
        "strawman": meta["strawman"],
        "threshold_sensitive_frac": meta["threshold_sensitive_frac"],
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(class_names)
        },
        "weakest_boundary": weakest_boundary(y_true_idx, pred_idx, class_names),
        "cost_sweep": sweep.to_dict(orient="records"),
        "cost_sweep_is_degenerate": degenerate,
        "tradeoff_curve_span": {
            "false_block_rate_min": float(curve["false_block_rate_on_legit"].min()),
            "false_block_rate_max": float(curve["false_block_rate_on_legit"].max()),
            "fraud_hard_blocked_min": float(curve["fraud_hard_blocked_rate"].min()),
            "fraud_hard_blocked_max": float(curve["fraud_hard_blocked_rate"].max()),
        },
        "friction_curve_span": {
            "legit_frictioned_min": float(friction["legit_frictioned_rate"].min()),
            "legit_frictioned_max": float(friction["legit_frictioned_rate"].max()),
            "abuser_caught_min": float(friction["abuser_caught_rate"].min()),
            "abuser_caught_max": float(friction["abuser_caught_rate"].max()),
        },
    }

    (RUNS_DIR / f"evaluation_{track}.json").write_text(json.dumps(out, indent=2))

    # Written to its own file rather than folded into evaluation_{track}.json,
    # which is frozen output other docs already cite by exact hash.
    ambiguity = ambiguous_class_pairs(proba, class_names)
    (RUNS_DIR / f"ambiguity_{track}.json").write_text(json.dumps(ambiguity, indent=2))

    print(f"\n{'=' * 78}\nTRACK: {track}\n{'=' * 78}")
    print(f"macro-F1 {meta['macro_f1']:.4f}   accuracy {meta['accuracy']:.4f}   "
          f"(strawman: macro-F1 {meta['strawman']['macro_f1']:.4f}, "
          f"acc {meta['strawman']['accuracy']:.4f})")
    print(f"threshold-sensitive rows: {meta['threshold_sensitive_frac']:.1%}\n")
    print(report.round(4).to_string())
    wb = out["weakest_boundary"]
    print(f"\nWeakest boundary: {wb['actual']} misread as {wb['predicted_as']} "
          f"({wb['n_cases']} cases)")
    print("\nCost-ratio sweep (C_fp : C_fn):")
    print(sweep.round(4).to_string(index=False))
    span = out["tradeoff_curve_span"]
    print(
        f"\nContinuous sweep span (ratio 0.03x - 32x):"
        f"\n  false hard-block on legitimate : "
        f"{span['false_block_rate_min']:.2%} -> {span['false_block_rate_max']:.2%}"
        f"\n  fraudulent returns hard-blocked: "
        f"{span['fraud_hard_blocked_min']:.2%} -> {span['fraud_hard_blocked_max']:.2%}"
    )
    fspan = out["friction_curve_span"]
    print(
        f"\nFriction-axis sweep (the axis that actually moves; see docstring):"
        f"\n  legitimate customers given friction: "
        f"{fspan['legit_frictioned_min']:.2%} -> {fspan['legit_frictioned_max']:.2%}"
        f"\n  wardrobers / policy abusers caught  : "
        f"{fspan['abuser_caught_min']:.2%} -> {fspan['abuser_caught_max']:.2%}"
    )
    if degenerate:
        print(
            "\n  *** SWEEP IS DEGENERATE — every posture produced identical decisions. ***\n"
            "  This is a reportable finding, not a bug: with almost no probability mass\n"
            "  near a decision boundary, the cost matrix cannot influence any decision.\n"
            "  See docs/LEAKAGE_FINDING.md."
        )
    return out


def run() -> None:
    """CLI entrypoint: evaluate_track for both tracks, in a fixed order,
    so console output is always full-then-testbed regardless of dict
    ordering elsewhere."""
    for track in ("full", "testbed"):
        evaluate_track(track)


if __name__ == "__main__":
    run()
