"""
eval/run_evals.py

CLI entry point for the eval framework.

Usage:
    python run_evals.py                          # run all cases
    python run_evals.py --case tamera_preventive_v1
    python run_evals.py --parallel               # async batch
    python run_evals.py --output results.json    # save results to file
    python run_evals.py --fail-fast              # stop on first failure

Exit codes:
    0 — all cases passed
    1 — one or more cases failed
    2 — fatal error (agent unreachable, no cases found, etc.)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from golden_cases import load_all_cases, load_case_by_id
from runner import run_all_cases
from scorer import EvalReport, EvalResult, score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run claims coding evals against the live A2A agent."
    )
    parser.add_argument(
        "--case",
        metavar="CASE_ID",
        help="Run a single case by ID instead of all cases.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run all cases concurrently using asyncio (faster, less readable logs).",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        help="Save full results to a JSON file.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        dest="fail_fast",
        help="Stop after the first failed case.",
    )
    parser.add_argument(
        "--f1-threshold",
        type=float,
        default=0.80,
        dest="f1_threshold",
        metavar="FLOAT",
        help="Minimum F1 score to pass a case (default: 0.80).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed per-code breakdown.",
    )
    parser.add_argument("--debug", action="store_true", help="Dump raw agent responses to eval/debug_responses/")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> int:
    # Load cases
    try:
        if args.case:
            cases = [load_case_by_id(args.case)]
        else:
            cases = load_all_cases()
    except FileNotFoundError as e:
        logger.error("Could not load cases: %s", e)
        return 2

    print(f"\n{'='*60}")
    print(f"CLAIMS CODING EVAL — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Cases: {len(cases)} | Parallel: {args.parallel} | F1 threshold: {args.f1_threshold}")
    print(f"{'='*60}\n")

    if args.debug:
        import runner as _runner_mod
        _runner_mod.DEBUG_MODE = True

    # Run cases against live agent
    try:
        run_results = await run_all_cases(cases, parallel=args.parallel)
    except Exception as e:
        logger.error("Fatal error during run: %s", e)
        return 2

    # Score each result
    eval_results: list[EvalResult] = []
    for case, audit_log, latency in run_results:
        if audit_log is None:
            result = EvalResult(
                case_id=case.case_id,
                patient_id=case.patient_id,
                payer=case.payer,
                latency_seconds=latency,
                error="Agent did not return a parseable ClaimAuditLog",
            )
        else:
            result = score(case, audit_log, latency_seconds=latency)

        eval_results.append(result)

        if args.verbose:
            _print_verbose(result)

        if args.fail_fast and not result.passed:
            logger.warning("Fail-fast triggered on case: %s", case.case_id)
            break

    # Build and print report
    report = EvalReport(results=eval_results)
    report.print_summary()

    # Save results if requested
    if args.output:
        output_data = {
            "run_at": datetime.now(timezone.utc).isoformat(),
            "macro_f1": report.macro_f1,
            "macro_precision": report.macro_precision,
            "macro_recall": report.macro_recall,
            "modifier_accuracy": report.modifier_accuracy,
            "hallucination_rate": report.hallucination_rate,
            "passed": report.passed,
            "failed": report.failed,
            "cases": [
                {
                    "case_id": r.case_id,
                    "passed": r.passed,
                    "f1": r.f1,
                    "precision": r.precision,
                    "recall": r.recall,
                    "overall_status": r.overall_status,
                    "overall_confidence": r.overall_confidence,
                    "latency_seconds": r.latency_seconds,
                    "true_positives": [
                        {"cpt": c.cpt, "modifier": c.modifier, "confidence": c.confidence}
                        for c in r.true_positives
                    ],
                    "false_positives": [
                        {"cpt": c.cpt, "modifier": c.modifier}
                        for c in r.false_positives
                    ],
                    "false_negatives": [
                        {"cpt": c.cpt, "modifier": c.modifier}
                        for c in r.false_negatives
                    ],
                    "missing_modifiers": r.missing_modifiers,
                    "hallucinations": [
                        {"cpt": v.cpt, "modifier": v.modifier}
                        for v in r.must_not_include_violations
                    ],
                    "error": r.error,
                }
                for r in eval_results
            ],
        }
        output_path = Path(args.output)
        output_path.write_text(json.dumps(output_data, indent=2))
        print(f"\nResults saved to: {output_path.resolve()}")

    return 0 if report.failed == 0 else 1


def _print_verbose(result: EvalResult) -> None:
    print(f"\n  [{result.case_id}]")
    print(f"  Latency: {result.latency_seconds:.1f}s | "
          f"Agent status: {result.overall_status} | "
          f"Confidence: {result.overall_confidence}")
    if result.true_positives:
        print(f"  ✓ Correct: {', '.join(str(c) for c in result.true_positives)}")
    if result.false_negatives:
        print(f"  ✗ Missed:  {', '.join(c.cpt + (f'-{c.modifier}' if c.modifier else '') for c in result.false_negatives)}")
    if result.false_positives:
        print(f"  + Extra:   {', '.join(str(c) for c in result.false_positives)}")
    if result.missing_modifiers:
        print(f"  ~ No mod:  {result.missing_modifiers}")
    if result.must_not_include_violations:
        print(f"  ! Halluc:  {[v.cpt for v in result.must_not_include_violations]}")


if __name__ == "__main__":
    args = parse_args()
    sys.exit(asyncio.run(main(args)))