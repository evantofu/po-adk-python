"""
A2A application factory — shared by all agents in this repo.

Each agent's app.py calls create_a2a_app() with its own name, description,
URL, and optional FHIR extension URI.  The factory handles the AgentCard
boilerplate, wires up the A2A transport, and attaches the API key middleware.

Usage:
    from shared.app_factory import create_a2a_app
    from .agent import root_agent

    a2a_app = create_a2a_app(
        agent=root_agent,
        name="my_agent",
        description="Does useful things.",
        url="http://localhost:8001",
        port=8001,
        fhir_extension_uri="https://your-workspace/schemas/a2a/v1/fhir-context",
    )
"""
import os

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentExtension,
    APIKeySecurityScheme,
    In,
    SecurityScheme,
)
from google.adk.a2a.utils.agent_to_a2a import to_a2a

from shared.middleware import ApiKeyMiddleware


def create_a2a_app(
    agent,
    name: str,
    description: str,
    url: str,
    port: int = 8001,
    version: str = "1.0.0",
    fhir_extension_uri: str | None = None,
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

    Returns:
        A Starlette ASGI application ready to be served with uvicorn.
    """
    # Optional FHIR extension — only included when the agent supports it.
    extensions = []
    if fhir_extension_uri:
        extensions = [
            AgentExtension(
                uri=fhir_extension_uri,
                description="FHIR R4 context — allows the agent to query the patient's FHIR server.",
                required=False,
            )
        ]

    agent_card = AgentCard(
        name=name,
        description=description,
        url=url,
        version=version,
        defaultInputModes=["text/plain"],
        defaultOutputModes=["text/plain"],
        capabilities=AgentCapabilities(
            streaming=True,
            pushNotifications=False,
            stateTransitionHistory=True,
            extensions=extensions,
        ),
        skills=[],
        securitySchemes={
            "apiKey": SecurityScheme(
                root=APIKeySecurityScheme(
                    type="apiKey",
                    name="X-API-Key",
                    in_=In.header,
                    description="API key required to access this agent.",
                )
            )
        },
        security=[{"apiKey": []}],
    )

    app = to_a2a(agent, port=port, agent_card=agent_card)
    app.add_middleware(ApiKeyMiddleware)
    return app
