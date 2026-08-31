"""
CLI entry point for scoring return records.

Replaces the cut `app/demo.py` Streamlit UI (docs/ARCHITECTURE.md §11,
correction log) with something a merchant's pipeline could actually shell out
to: a single JSON record or a batch CSV in, class probabilities and a routed
intervention out, under explicit `--posture` and `--friction` flags.

This module contains no scoring logic of its own. It parses arguments, loads
a run bundle, and calls `src.infer.score_record` / `src.infer.score_batch` —
the same functions `tests/test_infer.py` exercises directly. Reimplementing
feature engineering or the decision rule here would be exactly the
two-implementations failure mode docs/ARCHITECTURE.md §8 warns about.

Run as (from the repo root, after `python -m src.model` has produced
runs/model_<track>.joblib):

    python -m scripts.score --record-file examples/record.json
    python -m scripts.score --record-file examples/record.json --track testbed
    python -m scripts.score --csv examples/sample_returns.csv --out scored.csv \
        --friction "recovery-first (1:20)"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.evaluate import DEFAULT_POSTURES, FRICTION_POSTURES
from src.features import FEATURE_SETS
from src.infer import (
    DEFAULT_FRICTION_POSTURE,
    DEFAULT_POSTURE,
    load_run,
    score_batch,
    score_record,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score return record(s) and route to a recommended intervention.",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--record",
        metavar="JSON",
        help="A single return record as a JSON object string.",
    )
    input_group.add_argument(
        "--record-file",
        metavar="PATH",
        help="Path to a JSON file containing a single return record object.",
    )
    input_group.add_argument(
        "--csv",
        metavar="PATH",
        help="Path to a CSV of raw return records to score in a batch.",
    )
    parser.add_argument(
        "--track",
        choices=sorted(FEATURE_SETS),
        default="full",
        help="Which trained track to score with (default: full — the honest model; "
        "'testbed' is a diagnostic ablation rung, not a model, and its scores are "
        "never a result — see docs/ARCHITECTURE.md §5.2).",
    )
    parser.add_argument(
        "--posture",
        choices=sorted(DEFAULT_POSTURES),
        default=DEFAULT_POSTURE,
        help=f"C_fp : C_fn posture from src.evaluate.DEFAULT_POSTURES (default: {DEFAULT_POSTURE!r}) "
        "— blocking an honest customer vs. approving real fraud. Reported honestly: "
        "on this dataset this axis is near-inert. It changes 0 of 12,000 decisions on "
        "--track full and 29 at the extremes on testbed, because Fraudulent Return is "
        "the easiest class to separate (docs/LEAKAGE_FINDING.md). Use --friction for "
        "the axis that moves.",
    )
    parser.add_argument(
        "--friction",
        choices=sorted(FRICTION_POSTURES),
        default=DEFAULT_FRICTION_POSTURE,
        help="Approve vs. soft-friction posture from src.evaluate.FRICTION_POSTURES "
        f"(default: {DEFAULT_FRICTION_POSTURE!r}) — how aggressively to fee and flag "
        "customers who might just be heavy returners. This is the axis that moves, "
        "but only where anything can: on --track testbed it spans 2.9%-22.4% of "
        "legitimate customers given friction (764-1219 of 12,000 decisions change), "
        "while on --track full it changes nothing, because that model is degenerate "
        "and no cost posture can move a saturated probability. The default "
        "reproduces the decision layer's historical operating point exactly.",
    )
    parser.add_argument(
        "--run-dir",
        default="runs",
        help="Directory containing model_<track>.joblib (default: runs).",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        help="Where to write output. Single record: JSON. Batch: CSV. "
        "Omit to print to stdout.",
    )
    return parser


def _load_bundle(track: str, run_dir: str) -> dict:
    run_path = Path(run_dir) / f"model_{track}"
    try:
        return load_run(run_path)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"No trained bundle at {run_path.with_suffix('.joblib')}. "
            f"Run `python -m src.model {track}` (or `python -m src.model` for both "
            "tracks) first."
        ) from exc


def _read_record(args: argparse.Namespace) -> dict:
    if args.record is not None:
        raw = args.record
    else:
        raw = Path(args.record_file).read_text()
    try:
        record = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--record/--record-file must be a JSON object: {exc}") from exc
    if not isinstance(record, dict):
        raise SystemExit("--record/--record-file must decode to a single JSON object, not a list.")
    return record


def _warn_if_partial(missing: list[str]) -> None:
    """Print a stderr note when the input did not carry every trained feature.

    Scoring proceeds (LightGBM handles NaN, and a real integration will not
    always carry all 35 columns), but a partial record is a caveat on the
    recommendation and must not be invisible to whoever runs the command.
    Stderr, not stdout, so it never contaminates piped JSON/CSV output.
    """
    if not missing:
        return
    shown = ", ".join(missing[:8])
    more = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
    print(
        f"WARNING: {len(missing)} trained feature(s) absent from the input and "
        f"scored as missing: {shown}{more}",
        file=sys.stderr,
    )


def _note_if_postures_are_inert(track: str) -> None:
    """Say out loud, at the point of use, that the posture flags cannot move a
    decision on the `full` track.

    src/evaluate.py already prints a banner when a cost sweep comes out flat
    (`sweep_is_degenerate`), on the principle that a non-result has to be
    reported rather than pass unnoticed. The same principle applies here: a
    reviewer who runs this command twice under different postures and sees
    identical output deserves to be told that is the finding, not a bug in the
    flag. Measured: on `full`, all three --posture values and all three
    --friction values produce byte-identical actions for 12,000 of 12,000 test
    rows. Stderr, so it never contaminates piped JSON/CSV.
    """
    if track != "full":
        return
    print(
        "NOTE: on --track full, neither --posture nor --friction can change a "
        "recommendation. 99.9% of this model's predictions sit above p=0.99, so "
        "no row is near an action boundary for a cost matrix to move. That is "
        "the project's headline finding, not a broken flag — see "
        "docs/LEAKAGE_FINDING.md. Use --track testbed to see the decision layer "
        "actually respond.",
        file=sys.stderr,
    )


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bundle = _load_bundle(args.track, args.run_dir)
    _note_if_postures_are_inert(args.track)

    if args.csv is not None:
        records = pd.read_csv(args.csv)
        result = score_batch(
            records, bundle, posture=args.posture, friction_posture=args.friction
        )
        _warn_if_partial(result.attrs.get("missing_features", []))
        if args.out:
            result.to_csv(args.out, index=False)
            print(f"Scored {len(result)} records -> {args.out}", file=sys.stderr)
        else:
            print(result.to_csv(index=False), end="")
        return 0

    record = _read_record(args)
    result = score_record(
        record, bundle, posture=args.posture, friction_posture=args.friction
    )
    _warn_if_partial(result["features_missing"])
    payload = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(payload + "\n")
        print(f"Scored 1 record -> {args.out}", file=sys.stderr)
    else:
        print(payload)
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
