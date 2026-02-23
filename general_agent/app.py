"""
general_agent — A2A application entry point.

Start the server with:
    uvicorn general_agent.app:a2a_app --host 0.0.0.0 --port 8002

The agent card is served publicly at:
    GET http://localhost:8002/.well-known/agent-card.json

All other endpoints require an X-API-Key header (see shared/middleware.py).
"""
import os

from shared.app_factory import create_a2a_app

from .agent import root_agent

a2a_app = create_a2a_app(
    agent=root_agent,
    name="general_agent",
    description=(
        "A general-purpose clinical assistant for date/time queries and ICD-10-CM "
        "code lookups. Does not require patient context or FHIR credentials."
    ),
    url=os.getenv("GENERAL_AGENT_URL", "http://localhost:8002"),
    port=8002,
    # No fhir_extension_uri — this agent does not use FHIR context.
)
