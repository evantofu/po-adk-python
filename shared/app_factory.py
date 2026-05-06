"""
A2A application factory — shared by all agents in this repo.

Each agent's app.py calls create_a2a_app() with its own name, description,
URL, and optional FHIR extension URI.  The factory handles the AgentCard
boilerplate, wires up the A2A transport, and optionally attaches API key
middleware.

Security modes
──────────────
  require_api_key=True  (default)
      Agent card advertises X-API-Key as required.
      All requests except /.well-known/agent-card.json are blocked without a
      valid key.  Use this for agents that handle sensitive data (e.g. FHIR).

  require_api_key=False
      Agent card declares no security scheme — any caller can send requests
      without a key.  The agent card itself makes this discoverable so Prompt
      Opinion and other callers know no key is needed.  Use this for public or
      read-only utility agents (e.g. ICD-10 lookups, date/time queries).

Usage:
    from shared.app_factory import create_a2a_app
    from .agent import root_agent

    # Authenticated agent (requires X-API-Key) with FHIR + SMART scopes
    a2a_app = create_a2a_app(
        agent=root_agent,
        name="healthcare_fhir_agent",
        description="Queries patient FHIR data.",
        url="http://localhost:8001",
        port=8001,
        fhir_extension_uri="https://your-workspace/schemas/a2a/v1/fhir-context",
        fhir_scopes=[
            {"name": "patient/Patient.rs",           "required": True},
            {"name": "patient/MedicationRequest.rs", "required": True},
            {"name": "patient/Condition.rs",         "required": True},
            {"name": "patient/Observation.rs",       "required": True},
        ],
        require_api_key=True,   # default — can be omitted
    )

    # Anonymous agent (no key needed)
    a2a_app = create_a2a_app(
        agent=root_agent,
        name="general_agent",
        description="Public utility agent.",
        url="http://localhost:8002",
        port=8002,
        require_api_key=False,
    )
"""
import json as _json
import logging as _logging
from typing import Any
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentExtension,
    AgentSkill,
)
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from pydantic import Field
from starlette.middleware.base import BaseHTTPMiddleware as _Base

_shim_logger = _logging.getLogger("method_shim")


class AgentExtensionV1(AgentExtension):
    """AgentExtension subclass that adds the `params` field for SMART scope declarations.

    The installed a2a-sdk does not define `params` on AgentExtension, so the
    scopes payload would be silently dropped during JSON serialisation.  Declaring
    the field explicitly here ensures it is included in the agent card output.

    Once the a2a-sdk ships native v1 support this subclass can be removed.
    """

    params: dict[str, Any] | None = Field(default=None)


class AgentCardV1(AgentCard):
    """AgentCard subclass that patches two fields missing/changed in A2A v1.

    1. supportedInterfaces — not defined in installed a2a-sdk; added here so
       it is included in the serialised JSON.
    2. securitySchemes — the parent types this as dict[str, SecurityScheme],
       which forces Pydantic to serialise using the OLD flat format
       (type/name/in).  Po v1 expects the NEW nested-key format
       (apiKeySecurityScheme/...).  Overriding to dict[str, Any] lets the
       v1-format dict pass through unmodified.

    Both overrides can be removed once the a2a-sdk ships native v1 support.
    """

    supportedInterfaces: list[dict[str, Any]] = Field(default_factory=list)
    securitySchemes: dict[str, Any] | None = None  # override parent's typed field


from shared.middleware import ApiKeyMiddleware

# Protobuf-style role enum -> A2A spec role
_ROLE_MAP = {
    "ROLE_USER":  "user",
    "ROLE_AGENT": "agent",
}

# Non-standard PO method name -> A2A spec method name
_METHOD_MAP = {
    "SendStreamingMessage": "message/stream",
    "SendMessage":          "message/send",
}


def _normalize_roles(obj: Any) -> None:
    """Recursively rewrite ROLE_USER/ROLE_AGENT -> user/agent in-place."""
    if isinstance(obj, dict):
        if "role" in obj and obj["role"] in _ROLE_MAP:
            obj["role"] = _ROLE_MAP[obj["role"]]
        for v in obj.values():
            _normalize_roles(v)
    elif isinstance(obj, list):
        for item in obj:
            _normalize_roles(item)


class _MethodShim(_Base):
    """Temporary shim: normalises non-standard PO requests to A2A spec format.

    Prompt Opinion currently sends:
      - 'SendMessage' / 'SendStreamingMessage' instead of 'message/send' / 'message/stream'
      - role: 'ROLE_USER' / 'ROLE_AGENT' instead of 'user' / 'agent'

    This middleware rewrites both before the request reaches the A2A handler.
    Remove once PO fixes their client.
    """

    async def dispatch(self, request, call_next):
        if request.method == "POST":
            body = await request.body()
            try:
                data = _json.loads(body)
                original_method = data.get("method")

                # Rewrite method name
                rewritten_method = _METHOD_MAP.get(original_method, original_method)
                if rewritten_method != original_method:
                    data["method"] = rewritten_method

                # Rewrite protobuf-style role enums
                _normalize_roles(data)

                body = _json.dumps(data).encode()
                _shim_logger.warning(
                    "SHIM: method %s -> %s", original_method, rewritten_method
                )
            except Exception as e:
                _shim_logger.warning("SHIM parse failed: %s", e)
            request._body = body
        return await call_next(request)


def create_a2a_app(
    agent,
    name: str,
    description: str,
    url: str,
    port: int = 8001,
    version: str = "1.0.0",
    fhir_extension_uri: str | None = None,
    fhir_scopes: list[dict[str, Any]] | None = None,
    require_api_key: bool = True,
    skills: list[AgentSkill] | None = None,
):
    """
    Build and return an A2A ASGI application for the given ADK agent.

    Args:
        agent:               The ADK Agent instance (root_agent from agent.py).
        name:                Agent name — shown in the agent card and Prompt Opinion UI.
        description:         Short description of what this agent does.
        url:                 Public base URL where this agent is reachable.
        port:                Port the agent listens on (used by to_a2a).
        version:             Semver string, e.g. "1.0.0".
        fhir_extension_uri:  If provided, advertises FHIR context support in the
                             agent card.  Callers use this URI as the metadata key
                             when sending FHIR credentials.  Omit for non-FHIR agents.
        fhir_scopes:         SMART-on-FHIR scopes the agent requires, expressed as a
                             list of dicts: [{"name": "patient/Patient.rs", "required": True}, ...]
                             Scopes with "required": True cannot be unchecked by the user
                             in the Prompt Opinion UI.  Only used when fhir_extension_uri
                             is also provided.  Omit for non-FHIR agents.
        require_api_key:     If True (default), the agent card declares X-API-Key as
                             required and ApiKeyMiddleware is attached — all requests
                             without a valid key are rejected with 401/403.
                             If False, no security scheme is declared and no middleware
                             is attached — the agent is publicly accessible.

    Returns:
        A Starlette ASGI application ready to be served with uvicorn.
    """
    extensions = []
    if fhir_extension_uri:
        extension_params = None
        if fhir_scopes:
            extension_params = {"scopes": fhir_scopes}
        extensions = [
            AgentExtensionV1(
                uri=fhir_extension_uri,
                description="FHIR context allowing the agent to query a FHIR server securely",
                required=False,
                params=extension_params,
            )
        ]

    if require_api_key:
        security_schemes = {
            "apiKey": {
                "apiKeySecurityScheme": {
                    "name": "X-API-Key",
                    "location": "header",
                    "description": "API key required to access this agent.",
                }
            }
        }
        security = [{"apiKey": []}]
    else:
        security_schemes = None
        security = None

    agent_card = AgentCardV1(
        name=name,
        description=description,
        url=url,
        version=version,
        defaultInputModes=["text/plain"],
        defaultOutputModes=["text/plain"],
        capabilities=AgentCapabilities(
            streaming=False,
            pushNotifications=False,
            stateTransitionHistory=False,
            extensions=extensions,
        ),
        supportedInterfaces=[
            {"url": url, "protocolBinding": "JSONRPC", "protocolVersion": "1.0"},
        ],
        skills=skills or [],
        securitySchemes=security_schemes,
        security=security,
    )

    runner = Runner(
        app_name=name,
        agent=agent,
        session_service=InMemorySessionService(),
    )

    app = to_a2a(agent, port=port, agent_card=agent_card, runner=runner)

    # Method shim must be added before ApiKeyMiddleware so it fires first.
    app.add_middleware(_MethodShim)

    if require_api_key:
        app.add_middleware(ApiKeyMiddleware)

    return app