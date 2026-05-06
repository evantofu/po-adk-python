"""
eval/chaos_run.py — Stability stress test.

Runs all golden cases N times sequentially and reports:
  - Pass rate per case
  - Overall flake rate
  - Infra retry distribution
  - Any non-deterministic failures (passed sometimes, failed others)

Usage:
    python chaos_run.py                  # 20 iterations (default)
    python chaos_run.py --iterations 50
    python chaos_run.py --iterations 10 --output chaos_results.json

Exit codes:
    0 — flake rate == 0%
    1 — any flakes detected
    2 — fatal error
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from golden_cases import load_all_cases
from runner import run_all_cases, _get_fhir_token, _invalidate_token
import runner as _runner_mod

logging.basicConfig(
    level=logging.WARNING,   # suppress per-case INFO noise during chaos run
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
# Keep our own logger at INFO so progress prints
logger = logging.getLogger("chaos_run")
logger.setLevel(logging.INFO)

from scorer import score


def _check_token_freshness(n_iterations: int, n_cases: int, avg_seconds_per_case: float = 9.0) -> None:
    """
    Check token has enough time for the full chaos run.
    Estimates run time and aborts if the token will expire mid-run.
    """
    import time
    remaining = _runner_mod._token_expires_at - time.time()
    if remaining <= 0:
        print("\nERROR: FHIR token is EXPIRED.")
        print("  1. Open any patient in PO to trigger a consult")
        print("  2. Run: ./refresh_token.sh")
        print("  3. Re-run chaos_run.py")
        sys.exit(2)

    remaining_min = remaining / 60
    estimated_run_min = (n_iterations * n_cases * avg_seconds_per_case) / 60
    safe_iterations = int((remaining - 60) / (n_cases * avg_seconds_per_case))  # 60s buffer

    print(f"  Token expires in   : {remaining_min:.1f} min")
    print(f"  Estimated run time : {estimated_run_min:.1f} min ({n_iterations} iter × {n_cases} cases × ~{avg_seconds_per_case:.0f}s)")

    if estimated_run_min > remaining_min - 1:
        print(f"\n  ⚠️  TOKEN WILL EXPIRE before iteration {safe_iterations + 1}.")
        print(f"  Safe maximum with current token: {safe_iterations} iterations.")
        print(f"\n  Options:")
        print(f"    1. Refresh token then rerun:  ./refresh_token.sh && python chaos_run.py -n {n_iterations}")
        print(f"    2. Run fewer iterations now:  python chaos_run.py -n {safe_iterations}")
        response = input("\nContinue anyway? [y/N] ").strip().lower()
        if response != 'y':
            print("Aborted.")
            sys.exit(2)
    else:
        print(f"  Token OK — {remaining_min - estimated_run_min:.1f} min to spare")


def _maybe_refresh_token(buffer_minutes: float = 3.0) -> None:
    """
    Silently refresh the FHIR token if it expires within buffer_minutes.
    Called before each chaos iteration to prevent mid-run expiry.
    Requires refresh_token.sh to be in the parent directory (po-adk-python/).
    """
    import subprocess, time, os
    remaining = _runner_mod._token_expires_at - time.time()
    if remaining > buffer_minutes * 60:
        return
    logger.warning(
        "Token expires in %.1f min — refreshing before next iteration …",
        remaining / 60,
    )
    refresh_script = Path(__file__).parent.parent / "refresh_token.sh"
    if not refresh_script.exists():
        logger.error("refresh_token.sh not found at %s — cannot auto-refresh", refresh_script)
        return
    result = subprocess.run(
        ["bash", str(refresh_script)],
        capture_output=True, text=True, timeout=15,
    )
    if result.returncode == 0:
        # Invalidate the runner's token cache so it re-reads .env
        _invalidate_token()
        logger.info("Token refreshed successfully.")
    else:
        logger.error("Token refresh failed:\n%s", result.stderr[:200])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chaos/stability stress test for the eval suite.")
    parser.add_argument("--iterations", "-n", type=int, default=20, metavar="N",
                        help="Number of full suite runs (default: 20)")
    parser.add_argument("--output", metavar="FILE",
                        help="Save full results to a JSON file")
    parser.add_argument("--sleep", type=float, default=2.0, metavar="SECONDS",
                        help="Seconds to sleep between iterations (default: 2.0)")
    return parser.parse_args()


async def main(args: argparse.Namespace) -> int:
    try:
        cases = load_all_cases()
    except FileNotFoundError as e:
        logger.error("Could not load cases: %s", e)
        return 2

    n = args.iterations
    total_runs = n * len(cases)

    print(f"\n{'='*60}")
    print(f"CHAOS RUN — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Cases: {len(cases)} | Iterations: {n} | Total agent calls: {total_runs} | Sleep: {args.sleep}s")
    print(f"{'='*60}\n")

    # Ensure token is fresh enough to survive the full run
    try:
        _get_fhir_token()   # seeds the cache if empty
        _check_token_freshness(n_iterations=n, n_cases=len(cases))
    except SystemExit:
        raise
    except Exception as e:
        logger.warning("Could not verify token freshness: %s", e)

    # Per-case accumulators
    results_by_case: dict[str, list[bool]] = defaultdict(list)   # case_id → [passed, ...]
    retries_by_case: dict[str, list[int]]  = defaultdict(list)   # case_id → [retry_count, ...]
    infra_failures:  list[tuple[int, str]] = []                  # (iteration, case_id)
    logic_failures:  list[tuple[int, str]] = []

    for i in range(1, n + 1):
        if i > 1 and args.sleep > 0:
            await asyncio.sleep(args.sleep)
        # Refresh token if it expires within 3 minutes
        _maybe_refresh_token()
        print(f"  Iteration {i:>3}/{n} ...", end=" ", flush=True)
        run_results = await run_all_cases(cases, parallel=False)

        iter_passed = 0
        for case, audit_log, latency, infra_retries in run_results:
            retries_by_case[case.case_id].append(infra_retries)

            if audit_log is None:
                passed = False
                if infra_retries > 0:
                    infra_failures.append((i, case.case_id))
                else:
                    logic_failures.append((i, case.case_id))
            else:
                result = score(case, audit_log, latency_seconds=latency)
                passed = result.passed

            results_by_case[case.case_id].append(passed)
            if passed:
                iter_passed += 1

        status = "✓" if iter_passed == len(cases) else f"✗ {iter_passed}/{len(cases)}"
        print(status)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"CHAOS RUN RESULTS — {n} iterations × {len(cases)} cases = {total_runs} calls")
    print(f"{'='*60}\n")

    total_passed  = sum(sum(v) for v in results_by_case.values())
    total_failed  = total_runs - total_passed
    flake_rate    = total_failed / total_runs * 100
    total_retries = sum(sum(v) for v in retries_by_case.values())

    print(f"  Overall pass rate   : {total_passed}/{total_runs} ({100-flake_rate:.1f}%)")
    print(f"  Flake rate          : {total_failed}/{total_runs} ({flake_rate:.1f}%)  ← target: 0.0%")
    print(f"  Infra failures      : {len(infra_failures)}  (unrecoverable after retries)")
    print(f"  Logic/parse failures: {len(logic_failures)}")
    print(f"  Total infra retries : {total_retries}  (recovered transients)")
    print()

    # Per-case breakdown
    print(f"  {'Case':<30} {'Pass':>6}  {'Flakes':>6}  {'Retries':>8}  {'Status'}")
    print(f"  {'-'*30}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*10}")
    all_clean = True
    for case in cases:
        cid = case.case_id
        passes   = sum(results_by_case[cid])
        flakes   = n - passes
        retries  = sum(retries_by_case[cid])
        status   = "✅ CLEAN" if flakes == 0 else f"⚠️  {flakes} FLAKE{'S' if flakes > 1 else ''}"
        if flakes:
            all_clean = False
        print(f"  {cid:<30}  {passes:>4}/{n}  {flakes:>6}  {retries:>8}  {status}")

    print()
    if all_clean:
        print("  ✅ ALL CASES CLEAN — zero flakes across all iterations.")
    else:
        print("  ⚠️  FLAKES DETECTED — see per-case breakdown above.")
        if logic_failures:
            print(f"\n  Logic failure details ({len(logic_failures)}):")
            for it, cid in logic_failures:
                print(f"    iteration {it:>3}: {cid}")
        if infra_failures:
            print(f"\n  Infra failure details ({len(infra_failures)}):")
            for it, cid in infra_failures:
                print(f"    iteration {it:>3}: {cid}")

    print(f"\n{'='*60}\n")

    # ── Optional JSON output ──────────────────────────────────────────────────
    if args.output:
        output_data = {
            "run_at":          datetime.now(timezone.utc).isoformat(),
            "iterations":      n,
            "total_calls":     total_runs,
            "flake_rate_pct":  round(flake_rate, 2),
            "infra_failures":  len(infra_failures),
            "logic_failures":  len(logic_failures),
            "total_retries":   total_retries,
            "cases": {
                cid: {
                    "passes":  sum(results_by_case[cid]),
                    "flakes":  n - sum(results_by_case[cid]),
                    "retries": sum(retries_by_case[cid]),
                }
                for cid in results_by_case
            },
            "infra_failure_detail": [{"iteration": it, "case_id": cid} for it, cid in infra_failures],
            "logic_failure_detail": [{"iteration": it, "case_id": cid} for it, cid in logic_failures],
        }
        Path(args.output).write_text(json.dumps(output_data, indent=2))
        print(f"Results saved to: {args.output}\n")

    return 0 if all_clean else 1


if __name__ == "__main__":
    args = parse_args()
    sys.exit(asyncio.run(main(args)))