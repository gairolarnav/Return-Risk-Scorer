"""
Tests for the ablation proxy-restatement rule (src/ablation.py::_PROXIES).

The Day 2 correction this project made: an ablation rung that drops a raw
column but keeps a derived feature that algebraically restates it (e.g.
returns_per_order == return_rate_pct / 100) measures nothing -- the first
version of this ladder did exactly that and came out nearly flat. This test
freezes that lesson as an assertion instead of leaving it as prose someone
could regress past.

Runs against the actual frozen ABLATION_LADDER / _PROXIES constants; no CSV
or model fitting needed.
"""

from src.ablation import _PROXIES, _RUNGS, ABLATION_LADDER, _with_proxies


def test_every_rung_drags_its_proxies_out_with_it():
    """For every rung, if a raw column is dropped, every derived proxy of
    that column (per _PROXIES) must be dropped too -- otherwise the rung
    keeps an algebraic restatement of a feature it claims to have removed."""
    for name, dropped in ABLATION_LADDER:
        dropped_set = set(dropped)
        for raw_col, proxies in _PROXIES.items():
            if raw_col in dropped_set:
                for proxy in proxies:
                    assert proxy in dropped_set, (
                        f"{name!r} drops {raw_col!r} but keeps its proxy {proxy!r}"
                    )


def test_returns_per_order_does_not_survive_the_rung_that_drops_return_rate_pct():
    """The exact proxy trap described in src/ablation.py's `_PROXIES`."""
    checked_at_least_one = False
    for name, dropped in ABLATION_LADDER:
        if "return_rate_pct" in dropped:
            checked_at_least_one = True
            assert "returns_per_order" in dropped, name
    assert checked_at_least_one, "no rung in ABLATION_LADDER drops return_rate_pct"


def test_with_proxies_never_drops_less_than_it_was_given_and_is_idempotent():
    """_with_proxies must never shrink its input, and applying it twice must
    not change the result -- a stability guarantee if a future rung ever
    chains it."""
    for _, raw_cols in _RUNGS:
        once = _with_proxies(raw_cols)
        twice = _with_proxies(once)
        assert set(raw_cols) <= set(once)
        assert once == twice
