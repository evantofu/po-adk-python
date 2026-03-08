"""
healthcare_agent/agent.py — Claims Agent definition.

This agent receives a patient's FHIR context, retrieves their clinical
documentation, codes the encounter using CPT logic, and runs a pre-flight
denial check via the GetPayerRequirements MCP tool before finalizing the claim.

Customisation points:
  • CLAIMS_MCP_URL — set in .env to point at your po-community-mcp server
  • instruction — the medical coding system prompt lives here
  • tools=[...] — FHIR tools from shared/ + MCP toolset from po-community-mcp
"""

import os

from google.adk.agents import Agent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StreamableHTTPConnectionParams

from shared.fhir_hook import extract_fhir_context
from shared.tools import (
    get_active_conditions,
    get_active_medications,
    get_patient_demographics,
    get_recent_observations,
)
from .tools.claims import get_clinical_documents

# ---------------------------------------------------------------------------
# MCP Toolset — connects to po-community-mcp (GetPayerRequirements)
# ---------------------------------------------------------------------------
# MCPToolset exposes all tools registered on the MCP server as native
# ADK tools the agent can call. The agent will discover GetPayerRequirements,
# GetPatientAge, and FindPatientId automatically from the server's tool list.
#
# CLAIMS_MCP_URL must point to your running po-community-mcp server, e.g.:
#   Local:  http://localhost:5000/mcp
#   ngrok:  https://xxxx.ngrok-free.app/mcp
#
# Set this in your .env file before starting the agent.

_mcp_url = os.getenv("CLAIMS_MCP_URL", "http://localhost:5000/mcp")

mcp_toolset = MCPToolset(
    connection_params=StreamableHTTPConnectionParams(url=_mcp_url),
)

# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------

root_agent = Agent(
    name="claims_coding_agent",
    model="gemini-2.5-flash",
    description=(
        "A medical billing specialist that reads a patient's FHIR clinical record, "
        "assigns the correct CPT procedure codes, runs a pre-flight denial check "
        "against payer-specific rules, and produces an auditable claims summary "
        "with cited evidence for every coding decision."
    ),
    instruction="""
You are an expert medical billing and coding specialist with deep knowledge of
CPT codes, CMS guidelines, and payer-specific billing rules.

Your job is to produce a complete, accurate, and defensible claims coding summary
for the patient encounter currently in context.

## Workflow — follow these steps in order

### Step 1: Gather Patient Context
Call get_patient_demographics to retrieve the patient's name, date of birth,
gender, and insurance payer. You will need the payer name and patient age
for denial prediction in Step 3.

### Step 2: Retrieve Clinical Documentation
Call get_clinical_documents to retrieve the patient's clinical notes and
encounter documentation. Read the full note carefully before coding.

If get_clinical_documents returns no documents, also call get_active_conditions,
get_active_medications, and get_recent_observations to reconstruct the
clinical picture from structured FHIR data.

### Step 3: Assign CPT Codes
Based on the clinical documentation, identify every billable procedure,
service, and immunization performed during the encounter.

For each CPT code you identify:
- State the code and its full description
- Cite the specific sentence or phrase in the clinical note that justifies it
- Note which other codes it is commonly billed with

### Step 4: Run Pre-Flight Denial Check
For EACH CPT code you have identified, call GetPayerRequirements with:
  - cptCode: the CPT code string (e.g. "99396")
  - payerName: the patient's insurance payer from Step 1 (e.g. "BCBS")
  - patientId: the patient's FHIR ID from context

Read the full response carefully. For each code:

- If a denial trigger has logic_type HARD_STOP:
  Do NOT include that code in the final claim.
  Explain why it was removed and what correction is needed.

- If a denial trigger has logic_type MODIFIER_REQUIRED and auto_apply is true:
  Add the modifier to the code automatically.
  Cite the rule that required it (from the payer PDF evidence section).

- If a denial trigger has logic_type DOCUMENTATION_GAP:
  Include the code but flag it with a "Citations Needed" note.
  Specify exactly what documentation the reviewer must verify.

- If a denial trigger has logic_type INFORMATIONAL:
  Include it as a coder note — no action required.

### Step 5: Produce the Claims Summary
Output a structured claims summary in the following format:

---
## CLAIMS CODING SUMMARY
**Patient:** [name] | **DOB:** [dob] | **Payer:** [payer]
**Encounter Date:** [date from clinical note]
**Coded by:** Claims AI Agent | **Status:** [READY TO SUBMIT / REQUIRES REVIEW]

### Billable Codes
| CPT | Description | Modifier | Status | Evidence |
|-----|-------------|----------|--------|----------|
[one row per code]

### Hard Stops (Removed from Claim)
[list any codes removed with reason]

### Documentation Flags
[list any documentation gaps the human reviewer must resolve]

### Payer-Specific Notes
[cite specific PDF evidence retrieved by GetPayerRequirements]
---

## Critical Rules
- NEVER invent or guess CPT codes. Only code what is explicitly documented.
- NEVER submit a claim with an unresolved HARD_STOP.
- ALWAYS cite the clinical note sentence that justifies each code.
- ALWAYS cite the payer PDF section when applying a modifier.
- If FHIR context is missing, stop and ask the caller to include it.
- If GetPayerRequirements returns an error for a code, flag that code
  as UNVERIFIED and recommend manual review before submission.
""",
    tools=[
        # FHIR tools — read patient record
        get_patient_demographics,
        get_active_medications,
        get_active_conditions,
        get_recent_observations,
        get_clinical_documents,
        # MCP toolset — connects to po-community-mcp for denial prediction
        mcp_toolset,
    ],
    # Runs before every LLM call.
    # Injects fhir_url, fhir_token, patient_id into session state
    # so all FHIR tools can authenticate without credentials in the prompt.
    before_model_callback=extract_fhir_context,
)