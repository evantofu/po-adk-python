"""
token_manager.py  —  po-adk-python/eval/token_manager.py

Auto-refreshing FHIR JWT token for eval runs.

Supports two refresh strategies:
  1. ENV-based   — reads EVAL_FHIR_TOKEN + EVAL_FHIR_REFRESH_TOKEN from env
                   and hits EVAL_FHIR_TOKEN_URL to get a new access token
  2. Client-creds — reads EVAL_FHIR_CLIENT_ID + EVAL_FHIR_CLIENT_SECRET
                    and uses OAuth2 client-credentials grant

Usage in runner.py:
    from token_manager import get_fhir_token
    token = get_fhir_token()          # cached; refreshes only when < 5 min left
"""

from __future__ import annotations

import os
import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Config (all from environment — no secrets in source)                        #
# --------------------------------------------------------------------------- #

_TOKEN_URL        = os.getenv("EVAL_FHIR_TOKEN_URL")          # required for refresh
_STATIC_TOKEN     = os.getenv("EVAL_FHIR_TOKEN")              # fallback / seed
_REFRESH_TOKEN    = os.getenv("EVAL_FHIR_REFRESH_TOKEN")      # for refresh-token grant
_CLIENT_ID        = os.getenv("EVAL_FHIR_CLIENT_ID")
_CLIENT_SECRET    = os.getenv("EVAL_FHIR_CLIENT_SECRET")
_SCOPE            = os.getenv("EVAL_FHIR_SCOPE", "system/*.read")
_REFRESH_BUFFER   = int(os.getenv("EVAL_FHIR_REFRESH_BUFFER_SEC", "300"))  # 5 min


# --------------------------------------------------------------------------- #
#  Token cache (module-level singleton, thread-safe)                           #
# --------------------------------------------------------------------------- #

@dataclass
class _TokenCache:
    access_token: str = ""
    expires_at: float = 0.0          # epoch seconds
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def is_valid(self) -> bool:
        return bool(self.access_token) and time.time() < self.expires_at - _REFRESH_BUFFER

    def store(self, token: str, expires_in: int) -> None:
        with self._lock:
            self.access_token = token
            self.expires_at   = time.time() + expires_in
        logger.info(
            "Token cached — expires in %ds (at %s)",
            expires_in,
            time.strftime("%H:%M:%S", time.localtime(self.expires_at)),
        )


_cache = _TokenCache()


# --------------------------------------------------------------------------- #
#  Refresh strategies                                                          #
# --------------------------------------------------------------------------- #

def _refresh_via_refresh_token() -> dict:
    """OAuth2 refresh-token grant."""
    resp = requests.post(
        _TOKEN_URL,
        data={
            "grant_type":    "refresh_token",
            "refresh_token": _REFRESH_TOKEN,
            "client_id":     _CLIENT_ID or "",
            "client_secret": _CLIENT_SECRET or "",
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def _refresh_via_client_credentials() -> dict:
    """OAuth2 client-credentials grant."""
    resp = requests.post(
        _TOKEN_URL,
        data={
            "grant_type":    "client_credentials",
            "client_id":     _CLIENT_ID,
            "client_secret": _CLIENT_SECRET,
            "scope":         _SCOPE,
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def _try_refresh() -> Optional[str]:
    """
    Attempt a token refresh.  Returns new access token string, or None on failure.
    Tries refresh-token grant first; falls back to client-credentials.
    """
    if not _TOKEN_URL:
        logger.debug("EVAL_FHIR_TOKEN_URL not set — cannot refresh")
        return None

    for strategy_name, strategy_fn, precondition in [
        ("refresh_token",       _refresh_via_refresh_token,       bool(_REFRESH_TOKEN)),
        ("client_credentials",  _refresh_via_client_credentials,  bool(_CLIENT_ID and _CLIENT_SECRET)),
    ]:
        if not precondition:
            continue
        try:
            logger.info("Attempting token refresh via %s …", strategy_name)
            payload = strategy_fn()
            token      = payload["access_token"]
            expires_in = int(payload.get("expires_in", 3600))
            _cache.store(token, expires_in)
            return token
        except Exception as exc:                           # noqa: BLE001
            logger.warning("Refresh via %s failed: %s", strategy_name, exc)

    return None


# --------------------------------------------------------------------------- #
#  Public API                                                                  #
# --------------------------------------------------------------------------- #

def get_fhir_token() -> str:
    """
    Return a valid FHIR bearer token, refreshing automatically when needed.

    Resolution order:
      1. Cached token (if still valid)
      2. Refresh via OAuth2
      3. EVAL_FHIR_TOKEN env var (static fallback — warns if close to expiry)

    Raises RuntimeError if no valid token can be obtained.
    """
    with _cache._lock:
        if _cache.is_valid():
            return _cache.access_token

    # Not in cache — try refresh
    token = _try_refresh()
    if token:
        return token

    # Fall back to static env token
    if _STATIC_TOKEN:
        logger.warning(
            "Using static EVAL_FHIR_TOKEN — set EVAL_FHIR_TOKEN_URL for auto-refresh"
        )
        # Seed the cache with a 1-hour TTL so we don't spam this warning
        _cache.store(_STATIC_TOKEN, 3600)
        return _STATIC_TOKEN

    raise RuntimeError(
        "No valid FHIR token available.\n"
        "Set EVAL_FHIR_TOKEN (static) or\n"
        "EVAL_FHIR_TOKEN_URL + (EVAL_FHIR_REFRESH_TOKEN or EVAL_FHIR_CLIENT_ID/SECRET)."
    )


def invalidate_token() -> None:
    """Force the next call to get_fhir_token() to refresh. Useful after a 401."""
    with _cache._lock:
        _cache.expires_at = 0.0
    logger.info("Token invalidated — will refresh on next call")


def token_status() -> dict:
    """Return a debug dict — safe to log (no token value)."""
    remaining = max(0.0, _cache.expires_at - time.time())
    return {
        "has_token":        bool(_cache.access_token),
        "seconds_remaining": int(remaining),
        "valid":            _cache.is_valid(),
        "refresh_buffer_sec": _REFRESH_BUFFER,
        "token_url_set":    bool(_TOKEN_URL),
        "refresh_token_set": bool(_REFRESH_TOKEN),
        "client_creds_set": bool(_CLIENT_ID and _CLIENT_SECRET),
    }