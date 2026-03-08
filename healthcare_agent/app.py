"""
healthcare_agent/app.py — Claims Agent A2A application entry point.

Start the server with:
    uvicorn healthcare_agent.app:a2a_app --host 0.0.0.0 --port 8001

The agent card is served publicly at:
    GET http://localhost:8001/.well-known/agent-card.json

All other endpoints require an X-API-Key header (see shared/middleware.py).

Environment variables (set in .env):
    HEALTHCARE_AGENT_URL  — public ngrok URL for this agent, e.g. https://xxxx.ngrok-free.app
    CLAIMS_MCP_URL        — URL of your po-community-mcp server, e.g. http://localhost:5000/mcp
    PO_PLATFORM_BASE_URL  — Prompt Opinion workspace base URL (for FHIR extension URI)
"""
import os

from a2a.types import AgentSkill
from shared.app_factory import create_a2a_app

from .agent import root_agent

a2a_app = create_a2a_app(
    agent=root_agent,
    name="claims_coding_agent",
    description=(
        "A medical billing specialist that reads a patient's FHIR clinical record, "
        "assigns CPT procedure codes, runs a pre-flight denial check against "
        "payer-specific rules, and produces an auditable claims summary with "
        "cited evidence for every coding decision."
    ),
    url=os.getenv("HEALTHCARE_AGENT_URL", os.getenv("BASE_URL", "http://localhost:8001")),
    port=8001,
    fhir_extension_uri=f"{os.getenv('PO_PLATFORM_BASE_URL', 'http://localhost:5139')}/schemas/a2a/v1/fhir-context",
    skills=[
        AgentSkill(
            id="auto-coding",
            name="auto-coding",
            description=(
                "Reads a patient's clinical notes and assigns the correct CPT procedure "
                "codes for the encounter. Cites the specific clinical note sentence that "
                "justifies each code."
            ),
            tags=["claims", "cpt", "coding", "billing", "fhir"],
        ),
        AgentSkill(
            id="denial-prediction",
            name="denial-prediction",
            description=(
                "Runs a pre-flight denial check for each CPT code against CMS baseline "
                "rules and payer-specific policy PDFs. Flags hard stops, auto-applies "
                "required modifiers, and cites the payer PDF section that requires them."
            ),
            tags=["claims", "denial", "modifiers", "payer-rules", "billing"],
        ),
        AgentSkill(
            id="claims-summary",
            name="claims-summary",
            description=(
                "Produces a structured, auditable claims summary including billable codes, "
                "modifiers, denial risk flags, documentation gaps, and payer-specific "
                "evidence citations. Ready for human reviewer sign-off before submission."
            ),
            tags=["claims", "summary", "audit", "billing"],
        ),
    ],
)