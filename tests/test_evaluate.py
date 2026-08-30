"""
Tests for the cost-calibrated decision layer (src/evaluate.py).

These matter more than the usual unit test: the decision rule is the
centerpiece deliverable, and a silently wrong cost matrix would produce
plausible-looking numbers that are simply false.
"""

from itertools import pairwise

import numpy as np

from src.evaluate import (
    ACTIONS,
    build_cost_matrix,
    expected_cost_decision,
    oracle_cost,
    sweep_cost_ratios,
    sweep_friction_curve,
    sweep_is_degenerate,
    sweep_tradeoff_curve,
)

CLASSES = ["Fraudulent Return", "Legitimate", "Policy Abuser", "Wardrobing"]


def test_correct_action_costs_nothing():
    """Each class's target action must be free, or the decision rule is
    minimising against a shifted baseline."""
    cost = build_cost_matrix(CLASSES, c_fp=100.0, c_fn=100.0)
    assert cost.loc["Legitimate", "approve"] == 0.0
    assert cost.loc["Fraudulent Return", "hard_block"] == 0.0
    assert cost.loc["Wardrobing", "soft_friction"] == 0.0
    assert cost.loc["Policy Abuser", "soft_friction"] == 0.0


def test_cfp_and_cfn_land_in_the_intended_cells():
    cost = build_cost_matrix(CLASSES, c_fp=999.0, c_fn=777.0)
    # Hard-blocking an honest customer.
    assert cost.loc["Legitimate", "hard_block"] == 999.0
    # Approving real fraud.
    assert cost.loc["Fraudulent Return", "approve"] == 777.0


def test_confident_prediction_gets_its_target_action():
    """A one-hot probability vector must route to that class's target action
    under any posture — if this fails the rule is not Bayes-optimal."""
    cost = build_cost_matrix(CLASSES, c_fp=100.0, c_fn=100.0)
    proba = np.eye(len(CLASSES))
    actions = expected_cost_decision(proba, cost)
    for i, cls in enumerate(CLASSES):
        expected = {
            "Fraudulent Return": "hard_block",
            "Legitimate": "approve",
            "Policy Abuser": "soft_friction",
            "Wardrobing": "soft_friction",
        }[cls]
        assert ACTIONS[actions[i]] == expected


def test_raising_cfp_never_increases_blocking():
    """Monotonicity: making false blocks more expensive must not make the
    policy block more people. A violation means the cost matrix and the
    decision rule disagree about which cell is which."""
    proba = np.random.default_rng(0).dirichlet(np.ones(4), size=2000)
    block = ACTIONS.index("hard_block")
    rates = []
    for c_fp in (10.0, 100.0, 1000.0, 10000.0):
        cost = build_cost_matrix(CLASSES, c_fp=c_fp, c_fn=100.0)
        actions = expected_cost_decision(proba, cost)
        rates.append((actions == block).mean())
    assert all(a >= b - 1e-12 for a, b in pairwise(rates))


def test_oracle_cost_is_a_lower_bound():
    rng = np.random.default_rng(1)
    proba = rng.dirichlet(np.ones(4), size=500)
    y = rng.integers(0, 4, size=500)
    cost = build_cost_matrix(CLASSES, c_fp=100.0, c_fn=100.0)
    actions = expected_cost_decision(proba, cost)
    realised = cost.to_numpy()[y, actions].sum()
    assert oracle_cost(y, cost) <= realised + 1e-9


def test_saturated_probabilities_produce_a_degenerate_sweep():
    """The degeneracy detector must fire on exactly the situation it was
    written for — a model so confident that no cost posture can move it."""
    rng = np.random.default_rng(2)
    y = rng.integers(0, 4, size=1000)
    proba = np.full((1000, 4), 1e-6)
    proba[np.arange(1000), y] = 1.0
    proba /= proba.sum(axis=1, keepdims=True)

    sweep = sweep_cost_ratios(y, proba, CLASSES)
    assert sweep_is_degenerate(sweep)


def test_uncertain_probabilities_produce_a_live_sweep():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 4, size=4000)
    proba = rng.dirichlet(np.ones(4) * 3.0, size=4000)
    sweep = sweep_cost_ratios(y, proba, CLASSES)
    assert not sweep_is_degenerate(sweep)


def test_tradeoff_curve_is_monotone_in_the_ratio():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 4, size=3000)
    proba = rng.dirichlet(np.ones(4) * 2.0, size=3000)
    curve = sweep_tradeoff_curve(y, proba, CLASSES)
    fb = curve.sort_values("ratio_fp_to_fn")["false_block_rate_on_legit"].to_numpy()
    assert np.all(np.diff(fb) <= 1e-12)


def test_friction_curve_trades_the_two_populations_against_each_other():
    """Higher friction cost must mean fewer honest customers frictioned AND
    fewer abusers caught. If both move the same way the axis is mis-wired."""
    rng = np.random.default_rng(5)
    y = rng.integers(0, 4, size=3000)
    proba = rng.dirichlet(np.ones(4) * 2.0, size=3000)
    curve = sweep_friction_curve(y, proba, CLASSES).sort_values("friction_cost")
    legit = curve["legit_frictioned_rate"].to_numpy()
    abuser = curve["abuser_caught_rate"].to_numpy()
    assert np.all(np.diff(legit) <= 1e-12)
    assert np.all(np.diff(abuser) <= 1e-12)


def test_unknown_class_names_do_not_silently_produce_zero_costs():
    cost = build_cost_matrix(["Legitimate", "Fraudulent Return"], c_fp=50.0, c_fn=60.0)
    assert set(cost.columns) == set(ACTIONS)
    assert cost.loc["Legitimate", "hard_block"] == 50.0
