"""
eval/runner.py

Sends golden cases to the live A2A agent and collects ClaimAuditLog responses.

The runner calls the agent via its A2A endpoint — the same path the Prompt
Opinion platform uses. This tests the full stack: FHIR tools, MCP tools,
Chroma retrieval, and LLM reasoning.

Usage:
    python run_evals.py                        # run all cases sequentially
    python run_evals.py --case tamera_preventive_v1
    python run_evals.py --parallel             # async batch (coming soon)
    python run_evals.py --debug                # dump raw HTTP response to disk

Environment variables (from .env):
    HEALTHCARE_AGENT_URL  — base URL of the A2A agent
    EVAL_API_KEY          — X-API-Key for the agent (same as middleware key)

Token auto-refresh (Priority 1 fix):
    Minimum required (static):
        EVAL_FHIR_TOKEN        — static bearer token (warns when used, still works)

    For real auto-refresh, add ONE of:
        Option A — refresh-token grant:
            EVAL_FHIR_TOKEN_URL    + EVAL_FHIR_REFRESH_TOKEN
            (+ optionally EVAL_FHIR_CLIENT_ID / EVAL_FHIR_CLIENT_SECRET)
        Option B — client-credentials grant:
            EVAL_FHIR_TOKEN_URL    + EVAL_FHIR_CLIENT_ID + EVAL_FHIR_CLIENT_SECRET

    EVAL_FHIR_REFRESH_BUFFER_SEC  — how many seconds before expiry to pre-refresh
                                    (default: 300 = 5 minutes)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import threading
from pathlib import Path
from typing import Optional

import httpx

from golden_cases import GoldenCase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

import os
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

AGENT_BASE_URL = os.getenv(
    "HEALTHCARE_AGENT_URL", "http://localhost:8001"
).rstrip("/")

EVAL_API_KEY    = os.getenv("EVAL_API_KEY", os.getenv("MIDDLEWARE_API_KEY", ""))
EVAL_FHIR_URL   = os.getenv("EVAL_FHIR_URL", "")

# ── Token config — read once; _get_fhir_token() handles the rest ──────────
_TOKEN_URL      = os.getenv("EVAL_FHIR_TOKEN_URL", "")
_STATIC_TOKEN   = os.getenv("EVAL_FHIR_TOKEN", "")
_REFRESH_TOKEN  = os.getenv("EVAL_FHIR_REFRESH_TOKEN", "")
_CLIENT_ID      = os.getenv("EVAL_FHIR_CLIENT_ID", "")
_CLIENT_SECRET  = os.getenv("EVAL_FHIR_CLIENT_SECRET", "")
_SCOPE          = os.getenv("EVAL_FHIR_SCOPE", "system/*.read")
_REFRESH_BUFFER = int(os.getenv("EVAL_FHIR_REFRESH_BUFFER_SEC", "300"))

# How long to wait for the agent to respond (seconds)
# Live A2A calls can take 30-60s due to multiple tool calls
AGENT_TIMEOUT = 120

# Set to True via --debug flag in run_evals.py to dump raw responses
DEBUG_MODE = False
DEBUG_DIR = Path(__file__).parent / "debug_responses"


# ---------------------------------------------------------------------------
# Token cache (module-level singleton, thread-safe)
# ---------------------------------------------------------------------------

_token_lock       = threading.Lock()
_cached_token     = ""
_token_expires_at = 0.0   # epoch seconds; 0 means "not set"


def _token_is_valid() -> bool:
    return bool(_cached_token) and time.time() < _token_expires_at - _REFRESH_BUFFER


def _store_token(token: str, expires_in: int) -> None:
    global _cached_token, _token_expires_at
    _cached_token     = token
    _token_expires_at = time.time() + expires_in
    logger.info(
        "FHIR token cached — expires in %ds (at %s)",
        expires_in,
        time.strftime("%H:%M:%S", time.localtime(_token_expires_at)),
    )


def _try_refresh() -> Optional[str]:
    """
    Attempt a token refresh via OAuth2.
    Tries refresh-token grant first, then client-credentials.
    Returns the new access token string, or None on failure.
    """
    import requests as _requests  # stdlib-free import — only pulled in when needed

    if not _TOKEN_URL:
        return None

    strategies = []

    if _REFRESH_TOKEN:
        strategies.append(("refresh_token", {
            "grant_type":    "refresh_token",
            "refresh_token": _REFRESH_TOKEN,
            "client_id":     _CLIENT_ID,
            "client_secret": _CLIENT_SECRET,
        }))

    if _CLIENT_ID and _CLIENT_SECRET:
        strategies.append(("client_credentials", {
            "grant_type":    "client_credentials",
            "client_id":     _CLIENT_ID,
            "client_secret": _CLIENT_SECRET,
            "scope":         _SCOPE,
        }))

    for strategy_name, data in strategies:
        try:
            logger.info("FHIR token: attempting refresh via %s …", strategy_name)
            resp = _requests.post(_TOKEN_URL, data=data, timeout=10)
            resp.raise_for_status()
            payload    = resp.json()
            new_token  = payload["access_token"]
            expires_in = int(payload.get("expires_in", 3600))
            _store_token(new_token, expires_in)
            return new_token
        except Exception as exc:
            logger.warning("FHIR token: refresh via %s failed — %s", strategy_name, exc)

    return None


def _get_fhir_token() -> str:
    """
    Return a valid FHIR bearer token, refreshing automatically when needed.

    Resolution order:
      1. Cached token (still valid with buffer)
      2. OAuth2 refresh (refresh-token or client-credentials)
      3. Static EVAL_FHIR_TOKEN env var (fallback; warns)

    Raises RuntimeError if nothing works.
    """
    global _cached_token, _token_expires_at

    with _token_lock:
        if _token_is_valid():
            return _cached_token

    # Needs refresh — try OAuth2 outside the lock so we don't block parallel cases
    new_token = _try_refresh()
    if new_token:
        return new_token

    # Fall back to static env token — seed the cache so we only warn once per hour
    if _STATIC_TOKEN:
        logger.warning(
            "FHIR token: using static EVAL_FHIR_TOKEN — "
            "set EVAL_FHIR_TOKEN_URL for auto-refresh. "
            "Token will expire in ~1h and eval runs will break."
        )
        with _token_lock:
            _store_token(_STATIC_TOKEN, 3600)
        return _STATIC_TOKEN

    raise RuntimeError(
        "No valid FHIR token available.\n"
        "Set EVAL_FHIR_TOKEN (static) or\n"
        "EVAL_FHIR_TOKEN_URL + one of:\n"
        "  - EVAL_FHIR_REFRESH_TOKEN\n"
        "  - EVAL_FHIR_CLIENT_ID + EVAL_FHIR_CLIENT_SECRET"
    )


def _invalidate_token() -> None:
    """Force the next _get_fhir_token() call to refresh. Called on 401."""
    global _token_expires_at
    with _token_lock:
        _token_expires_at = 0.0
    logger.info("FHIR token: invalidated — will refresh on next request")


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def _dump_debug(case_id: str, raw_json: dict, extracted_text: str) -> None:
    """Write the full raw A2A response and extracted text to disk."""
    DEBUG_DIR.mkdir(exist_ok=True)
    out_path = DEBUG_DIR / f"{case_id}_raw.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "case_id": case_id,
                "raw_a2a_response": raw_json,
                "extracted_text_preview": extracted_text[:2000],
                "extracted_text_length": len(extracted_text),
            },
            f,
            indent=2,
        )
    logger.info("DEBUG: Raw response written to %s", out_path)

    # Also print a compact structural summary to stdout so you can see
    # immediately what shape the response has — without flooding the terminal.
    print("\n" + "=" * 60)
    print(f"DEBUG — raw A2A response structure for {case_id}")
    print("=" * 60)
    _print_structure(raw_json, indent=0, max_depth=4)
    print("-" * 60)
    print(f"Extracted text ({len(extracted_text)} chars):")
    print(extracted_text[:800] if extracted_text else "<empty>")
    print("=" * 60 + "\n")


def _print_structure(obj, indent: int, max_depth: int) -> None:
    """Recursively print the keys/types of a JSON object — no values."""
    prefix = "  " * indent
    if indent > max_depth:
        print(f"{prefix}...")
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                print(f"{prefix}{k}: {type(v).__name__}({len(v)})")
                _print_structure(v, indent + 1, max_depth)
            else:
                # Show value only for small scalars so we can see status codes etc.
                short = str(v)[:80] if v is not None else "null"
                print(f"{prefix}{k}: {type(v).__name__} = {short!r}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj[:5]):  # cap at 5 items
            print(f"{prefix}[{i}]:")
            _print_structure(item, indent + 1, max_depth)
        if len(obj) > 5:
            print(f"{prefix}... ({len(obj) - 5} more items)")


# ---------------------------------------------------------------------------
# A2A message builder
# ---------------------------------------------------------------------------

def build_a2a_request(case: GoldenCase) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": f"eval-{case.case_id}",
        "method": "message/send",  # A2A protocol v0.3.0
        "params": {
            "message": {
                "messageId": f"eval-{case.case_id}",
                "role": "user",
                "parts": [
                    {
                        "type": "text",
                        "text": (
                            f"Please code the encounter for patient ID: "
                            f"{case.patient_id}. "
                            f"Produce a complete ClaimAuditLog with confidence "
                            f"scores and payer-specific evidence for all codes."
                        ),
                    }
                ],
                # A2A extension: FHIR context required by claims_coding_agent.
                # Key must contain "fhir-context" (matched by fhir_hook.py).
                # Field names are camelCase (fhirUrl, fhirToken, patientId).
                # Token is fetched here — not at module load — so it auto-refreshes.
                "metadata": {
                    "https://app.promptopinion.ai/schemas/a2a/v1/fhir-context": {
                        "fhirUrl":   EVAL_FHIR_URL,
                        "fhirToken": _get_fhir_token(),   # ← refreshes if needed
                        "patientId": case.patient_id,
                        "payer":     case.payer,           # ← injected from golden case
                    }
                },
            },
        },
    }


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def extract_audit_log(response_text: str) -> Optional[dict]:
    """
    Extract the ClaimAuditLog JSON from the agent's A2A response text.

    The agent returns a mixed response — markdown claims summary + JSON
    audit log. We extract the JSON block.
    """
    if not response_text or not response_text.strip():
        logger.warning("extract_audit_log: response_text is empty")
        return None

    # Try to find a JSON block in the response
    # The agent wraps it in ```json ... ``` or outputs it directly
    json_block_pattern = r"```json\s*(\{.*?\})\s*```"
    match = re.search(json_block_pattern, response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError as e:
            logger.warning("extract_audit_log: ```json block parse failed — %s", e)

    # Try to find raw JSON object starting with "claim_id"
    raw_json_pattern = r'(\{"claim_id".*\})'
    match = re.search(raw_json_pattern, response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError as e:
            logger.warning("extract_audit_log: claim_id pattern parse failed — %s", e)

    # Try parsing the entire response as JSON
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass

    # -----------------------------------------------------------------------
    # MARKDOWN TABLE PARSER
    # Agent returns a markdown table with columns:
    # | CPT | Description | Modifier | Status | Evidence |
    # -----------------------------------------------------------------------
    codes = _extract_codes_from_markdown(response_text)
    if codes:
        logger.info("extract_audit_log: parsed %d codes from markdown table", len(codes))
        return {"codes": codes, "source": "markdown"}

    logger.warning(
        "extract_audit_log: all patterns failed.\n"
        "  Response length : %d chars\n"
        "  First 300 chars : %r\n"
        "  Contains 'claim_id': %s\n"
        "  Contains '```json' : %s",
        len(response_text),
        response_text[:300],
        "YES" if "claim_id" in response_text else "NO",
        "YES" if "```json" in response_text else "NO",
    )
    return None


def _extract_codes_from_markdown(text: str) -> list[dict]:
    """
    Parse CPT codes from the agent's markdown table response.

    Detects column positions from the header row so the parser is robust
    to varying column counts and Evidence cells that contain pipe characters.

    Expected header: | CPT | Description | Modifier | Status | Evidence |
    """
    codes = []
    in_table = False
    col_cpt = col_modifier = col_status = -1

    for line in text.splitlines():
        line = line.strip()

        # ── Detect the billable codes table header ─────────────────────────
        if line.startswith("|") and "CPT" in line and "Modifier" in line:
            headers = [h.strip().lower() for h in line.split("|")]
            for i, h in enumerate(headers):
                if h == "cpt":
                    col_cpt = i
                elif h == "modifier":
                    col_modifier = i
                elif h == "status":
                    col_status = i
            in_table = True
            continue

        # ── Skip separator row ─────────────────────────────────────────────
        if in_table and re.match(r"^\|[-| ]+\|$", line):
            continue

        # ── Stop at next section header ────────────────────────────────────
        if in_table and line.startswith("#"):
            break

        # ── Parse data rows ────────────────────────────────────────────────
        if in_table and line.startswith("|") and col_cpt >= 0:
            # The Evidence column often contains pipe characters inside quoted
            # clinical text (e.g. 'assessment of health | social care needs').
            # Fix: split only far enough to reach the last structural column
            # (CPT / Modifier / Status), then ignore everything beyond it.
            # This makes parsing immune to any pipes in the Evidence cell.
            last_structural_col = max(col_cpt, col_modifier, col_status)
            parts = line.split("|", last_structural_col + 2)
            parts = [p.strip() for p in parts]

            def _get(idx):
                return parts[idx].strip() if 0 <= idx < len(parts) else ""

            cpt      = _get(col_cpt)
            modifier = _get(col_modifier)
            status   = _get(col_status)

            # Must be a real CPT code
            if not re.match(r"^[A-Z]?\d{4,5}[A-Z]?$", cpt):
                continue

            codes.append({
                "cpt":      cpt,
                "modifier": modifier if modifier else None,
                "status":   status,
            })

    return codes


def extract_text_from_a2a_response(a2a_response: dict) -> str:
    """
    Extract the text content from an A2A JSON-RPC response.

    Logs each path attempted so --debug output shows exactly where
    the traversal succeeded or fell through.
    """
    try:
        result = a2a_response.get("result", {})

        # ── Path 1: result.artifacts[*].parts[type=text] ──────────────────
        artifacts = result.get("artifacts", [])
        if artifacts:
            for artifact in artifacts:
                for part in artifact.get("parts", []):
                    if part.get("kind") == "text" or part.get("type") == "text":
                        logger.debug(
                            "extract_text: found text via artifacts path (%d chars)",
                            len(part.get("text", "")),
                        )
                        return part.get("text", "")
        else:
            logger.debug("extract_text: no artifacts in result")

        # ── Path 2: result.status.message.parts[type=text] ────────────────
        status = result.get("status", {})
        message = status.get("message", {})
        for part in message.get("parts", []):
            if part.get("kind") == "text" or part.get("type") == "text":
                logger.debug(
                    "extract_text: found text via status.message path (%d chars)",
                    len(part.get("text", "")),
                )
                return part.get("text", "")

        # ── Path 3: top-level result is a plain string ─────────────────────
        if isinstance(result, str):
            logger.debug("extract_text: result is a plain string")
            return result

        # ── Path 4: result has a direct "text" key ─────────────────────────
        if "text" in result:
            logger.debug("extract_text: found result.text directly")
            return result["text"]

        # ── Fallback: dump the whole response so the parser can try ────────
        logger.warning(
            "extract_text: no text path matched — falling back to full JSON dump.\n"
            "  result keys: %s",
            list(result.keys()) if isinstance(result, dict) else type(result).__name__,
        )
        return json.dumps(a2a_response)

    except Exception as e:
        logger.error("extract_text_from_a2a_response: unexpected error — %s", e)
        return str(a2a_response)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_case(
    case: GoldenCase,
    client: httpx.AsyncClient,
) -> tuple[Optional[dict], float]:
    """
    Send a single golden case to the agent and return the parsed audit log.

    Returns:
        (audit_log_dict, latency_seconds)
        audit_log_dict is None if the agent call failed or parsing failed.
    """
    request_body = build_a2a_request(case)   # token freshness checked here
    headers = {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true",
    }

    if not EVAL_API_KEY:
        raise ValueError("EVAL_API_KEY is not set")

    headers["X-API-Key"] = EVAL_API_KEY

    start = time.monotonic()
    try:
        logger.info("Running case: %s → %s", case.case_id, AGENT_BASE_URL)
        response = await client.post(
            f"{AGENT_BASE_URL}/",
            json=request_body,
            headers=headers,
            timeout=AGENT_TIMEOUT,
        )
        latency = time.monotonic() - start

        # ── 401 handling: invalidate token and retry once ─────────────────
        if response.status_code == 401:
            logger.warning(
                "Case %s: got 401 — token may have expired mid-run. "
                "Invalidating and retrying once …",
                case.case_id,
            )
            _invalidate_token()
            # Rebuild the request so it picks up the fresh token
            request_body = build_a2a_request(case)
            start = time.monotonic()
            response = await client.post(
                f"{AGENT_BASE_URL}/",
                json=request_body,
                headers=headers,
                timeout=AGENT_TIMEOUT,
            )
            latency = time.monotonic() - start

        # ── Latency sanity check ──────────────────────────────────────────
        if latency < 1.0:
            logger.warning(
                "Case %s returned in %.2fs — suspiciously fast. "
                "The agent likely didn't do real work. Run with --debug to inspect.",
                case.case_id,
                latency,
            )

        response.raise_for_status()

        a2a_response = response.json()
        text = extract_text_from_a2a_response(a2a_response)
        audit_log = extract_audit_log(text)

        # ── Debug dump ────────────────────────────────────────────────────
        if DEBUG_MODE:
            _dump_debug(case.case_id, a2a_response, text)

        if audit_log is None:
            logger.warning(
                "Case %s: agent responded (%.1fs) but audit log could not be parsed. "
                "Re-run with --debug to see the full response.",
                case.case_id,
                latency,
            )

        return audit_log, latency

    except httpx.TimeoutException:
        latency = time.monotonic() - start
        logger.error("Case %s: agent timed out after %.1fs", case.case_id, latency)
        return None, latency
    except httpx.HTTPStatusError as e:
        latency = time.monotonic() - start
        logger.error(
            "Case %s: HTTP %d — %s",
            case.case_id, e.response.status_code, e.response.text[:200]
        )
        return None, latency
    except Exception as e:
        latency = time.monotonic() - start
        logger.error("Case %s: unexpected error — %s", case.case_id, e)
        return None, latency


async def run_all_cases(
    cases: list[GoldenCase],
    parallel: bool = False,
) -> list[tuple[GoldenCase, Optional[dict], float]]:
    """
    Run all golden cases against the live agent.

    Args:
        cases    — list of GoldenCase objects
        parallel — if True, run all cases concurrently (faster but harder to debug)

    Returns:
        List of (case, audit_log, latency) tuples
    """
    async with httpx.AsyncClient() as client:
        if parallel:
            logger.info("Running %d cases in parallel (async)", len(cases))
            tasks = [run_case(case, client) for case in cases]
            results_raw = await asyncio.gather(*tasks)
            return [
                (case, audit_log, latency)
                for case, (audit_log, latency) in zip(cases, results_raw)
            ]
        else:
            logger.info("Running %d cases sequentially", len(cases))
            results = []
            for case in cases:
                audit_log, latency = await run_case(case, client)
                results.append((case, audit_log, latency))
                logger.info(
                    "  %s: %.1fs — %s",
                    case.case_id,
                    latency,
                    "OK" if audit_log else "FAILED",
                )
            return results