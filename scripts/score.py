"""
CLI entry point for scoring return records.

Replaces the cut `app/demo.py` Streamlit UI (docs/ARCHITECTURE.md §11,
correction log) with something a merchant's pipeline could actually shell out
to: a single JSON record or a batch CSV in, class probabilities and a routed
intervention out, under an explicit `--posture`.

This module contains no scoring logic of its own. It parses arguments, loads
a run bundle, and calls `src.infer.score_record` / `src.infer.score_batch` —
the same functions `tests/test_infer.py` exercises directly. Reimplementing
feature engineering or the decision rule here would be exactly the
two-implementations failure mode docs/ARCHITECTURE.md §8 warns about.

Run as (from the repo root, after `python -m src.model` has produced
runs/model_<track>.joblib):

    python -m scripts.score --record '{"avg_order_value_usd": 120, ...}'
    python -m scripts.score --record-file record.json --track testbed
    python -m scripts.score --csv returns.csv --out scored.csv --posture "loss-averse (1:8)"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.evaluate import DEFAULT_POSTURES
from src.features import FEATURE_SETS
from src.infer import load_run, score_batch, score_record

DEFAULT_POSTURE = "loss-neutral (1:1)"


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
        "'testbed' is a diagnostic rung, never present it as a result — see docs/ARCHITECTURE.md §5.2).",
    )
    parser.add_argument(
        "--posture",
        choices=sorted(DEFAULT_POSTURES),
        default=DEFAULT_POSTURE,
        help=f"Merchant cost posture from src.evaluate.DEFAULT_POSTURES (default: {DEFAULT_POSTURE!r}). "
        "Explicit on purpose: the posture changes the recommendation.",
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


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bundle = _load_bundle(args.track, args.run_dir)

    if args.csv is not None:
        records = pd.read_csv(args.csv)
        result = score_batch(records, bundle, posture=args.posture)
        if args.out:
            result.to_csv(args.out, index=False)
            print(f"Scored {len(result)} records -> {args.out}", file=sys.stderr)
        else:
            print(result.to_csv(index=False), end="")
        return 0

    record = _read_record(args)
    result = score_record(record, bundle, posture=args.posture)
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
