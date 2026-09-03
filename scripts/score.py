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
        "Omit to print to stdout. Refuses to overwrite an existing file unless "
        "--overwrite is given.",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Add per-feature SHAP contributions for the predicted class to a "
        "single-record result: which evidence argued for the call and which "
        "against, signed. Off by default because it loads shap, which is slow. "
        "Single records only — a batch would recompute the explainer per row. "
        "Worth reading on --track full with the leakage finding in mind: the "
        "contributions show the model resting on the features the generator "
        "made near-separable (docs/LEAKAGE_FINDING.md).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow --out to replace an existing file. Off by default: a scored "
        "batch is evidence someone may be working from, and silently replacing "
        "it on a re-run with different --track or --friction is how the wrong "
        "numbers end up in a report.",
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


def _refuse_to_clobber(out: str | None, overwrite: bool) -> None:
    """Stop --out replacing a file that is already there.

    Checked before scoring rather than after, so a rejected run costs nothing
    and leaves the existing file untouched.
    """
    if not out or overwrite:
        return
    if Path(out).exists():
        raise SystemExit(
            f"{out} already exists. Pass --overwrite to replace it, or choose "
            "another path."
        )


def _read_record(args: argparse.Namespace) -> dict:
    if args.record is not None:
        raw = args.record
    else:
        try:
            raw = Path(args.record_file).read_text()
        except FileNotFoundError as exc:
            raise SystemExit(
                f"No such file: {args.record_file}. --record-file takes a path to a "
                "JSON file; pass the JSON itself with --record."
            ) from exc
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


def _warn_if_values_were_unusable(invalid: list[str]) -> None:
    """Print a stderr note when a field was present but could not be read as a
    number and was scored as missing instead.

    The record still scores -- a spreadsheet export full of "N/A" should not
    take a whole batch down -- but substituting NaN changes what the model saw,
    so it is stated rather than absorbed. Stderr, like the partial-record
    warning above, so piped JSON/CSV stays clean.
    """
    if not invalid:
        return
    shown = ", ".join(invalid[:8])
    more = f" (+{len(invalid) - 8} more)" if len(invalid) > 8 else ""
    print(
        f"WARNING: {len(invalid)} field(s) held a value that is not a number and "
        f"were scored as missing: {shown}{more}",
        file=sys.stderr,
    )


def _warn_if_values_are_impossible(out_of_range: list[str]) -> None:
    """Print a stderr note when a field held a value it cannot legitimately
    hold and was discarded.

    This is the loudest of the three warnings for a reason. A negative refund
    amount used to route straight to `hard_block` with full confidence: the most
    punitive action this system can recommend, computed from a number that
    cannot exist. Discarding the value is the right response, but doing it
    quietly would just move the problem.
    """
    if not out_of_range:
        return
    shown = ", ".join(out_of_range[:8])
    more = f" (+{len(out_of_range) - 8} more)" if len(out_of_range) > 8 else ""
    print(
        f"WARNING: {len(out_of_range)} field(s) held a value outside their "
        f"possible range and were discarded: {shown}{more}",
        file=sys.stderr,
    )


def _warn_if_features_degraded(degraded: list[str]) -> None:
    """Print a stderr note for features that arrived intact and still could not
    be computed -- an engineered ratio whose denominator was zero.

    Nothing the caller sent was wrong, so this is a caveat rather than a
    correction, but a recommendation resting on fewer features than it appears
    to has to say so.
    """
    if not degraded:
        return
    shown = ", ".join(degraded[:8])
    more = f" (+{len(degraded) - 8} more)" if len(degraded) > 8 else ""
    print(
        f"NOTE: {len(degraded)} feature(s) could not be computed from otherwise "
        f"valid input (usually a zero denominator): {shown}{more}",
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


def _score_or_exit(scorer, *args, **kwargs):
    """Run a src.infer scorer, turning its schema ValueError into a clean exit.

    `prepare_frame` raises ValueError when a payload supplies none of the
    trained features, and its message already names the track, the columns
    received and the ones expected — it is a good error. But letting it escape
    as a traceback makes a *caller* mistake look like a crash inside the tool,
    which is the one impression this repo cannot afford to give a reviewer.
    Every other bad-input path here exits with a single line; this makes the
    schema path match.

    The wrap is deliberately narrow — one call, not the body of `run` — so a
    genuine ValueError from anywhere else still surfaces as a traceback
    instead of being swallowed into a tidy-looking exit. The message itself
    stays owned by src/infer.py, where tests/test_infer.py asserts on it.
    """
    try:
        return scorer(*args, **kwargs)
    except (ValueError, TypeError) as exc:
        raise SystemExit(str(exc)) from exc


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bundle = _load_bundle(args.track, args.run_dir)
    _refuse_to_clobber(args.out, args.overwrite)

    if args.csv is not None:
        if args.explain:
            raise SystemExit(
                "--explain works on a single record, not --csv: a batch would "
                "rebuild the SHAP explainer for every row. Use --record or "
                "--record-file, or run `python -m src.explain` for the "
                "per-class study across the whole test set."
            )
        try:
            records = pd.read_csv(args.csv)
        except FileNotFoundError as exc:
            raise SystemExit(f"No such file: {args.csv}") from exc
        except pd.errors.EmptyDataError as exc:
            raise SystemExit(
                f"{args.csv} is empty -- no header row to read. A batch file needs "
                "a header naming the input columns; see examples/sample_returns.csv."
            ) from exc
        result = _score_or_exit(
            score_batch,
            records,
            bundle,
            posture=args.posture,
            friction_posture=args.friction,
        )
        _note_if_postures_are_inert(args.track)
        _warn_if_partial(result.attrs.get("missing_features", []))
        _warn_if_values_were_unusable(result.attrs.get("invalid_features", []))
        _warn_if_values_are_impossible(result.attrs.get("out_of_range_features", []))
        _warn_if_features_degraded(result.attrs.get("degraded_features", []))
        if args.out:
            result.to_csv(args.out, index=False)
            print(f"Scored {len(result)} records -> {args.out}", file=sys.stderr)
        else:
            print(result.to_csv(index=False), end="")
        return 0

    record = _read_record(args)
    result = _score_or_exit(
        score_record,
        record,
        bundle,
        posture=args.posture,
        friction_posture=args.friction,
        explain=args.explain,
    )
    _note_if_postures_are_inert(args.track)
    _warn_if_partial(result["features_missing"])
    _warn_if_values_were_unusable(result["features_invalid"])
    _warn_if_values_are_impossible(result["features_out_of_range"])
    _warn_if_features_degraded(result["features_degraded"])
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
