# Prompt Opinion Agent Examples
### Built with Google ADK · A2A Protocol · Python

Runnable examples showing how to build external agents that connect to **[Prompt Opinion](https://promptopinion.ai)** — the multi-agent platform for healthcare and enterprise workflows.

This is not a single-file template. It is a **monorepo with three working agents** that share a common infrastructure library. Clone it, run `adk web .` to see all three agents in a browser UI, then copy whichever example matches your use case and customise from there.

---

## Contents

- [What's in this repo](#whats-in-this-repo)
- [Architecture](#architecture)
- [Quick start](#quick-start)
- [The three agents](#the-three-agents)
  - [healthcare\_agent](#healthcare_agent--fhir-connected-clinical-assistant)
  - [general\_agent](#general_agent--general-purpose-assistant-no-fhir)
  - [orchestrator](#orchestrator--multi-agent-orchestrator)
- [The shared library](#the-shared-library)
- [Adding tools](#adding-tools)
- [FHIR context (optional)](#fhir-context-optional)
- [Configuration reference](#configuration-reference)
- [API security](#api-security)
- [Testing locally](#testing-locally)
- [Connecting to Prompt Opinion](#connecting-to-prompt-opinion)

---

## What's in this repo

| Agent | Description | FHIR? | Port |
|---|---|---|---|
| `healthcare_agent` | Queries a patient's FHIR R4 record — demographics, meds, conditions, observations | ✅ Yes | 8001 |
| `general_agent` | Date/time queries and ICD-10-CM code lookups — no patient data needed | ❌ No | 8002 |
| `orchestrator` | Delegates to the other two agents using ADK's built-in sub-agent routing | ✅ Optional | 8003 |

All three share a `shared/` library that provides middleware, logging, the FHIR context hook, FHIR R4 tools, and an app factory — so each agent's own files stay small and focused.

---

## Architecture

```
Prompt Opinion
     │  POST /  X-API-Key  A2A JSON-RPC
     │
     ▼
┌──────────────────────────────────────────────────┐
│  shared/middleware.py  (ApiKeyMiddleware)         │
│  · validates X-API-Key                           │
│  · bridges FHIR metadata to params.metadata      │
└──────────────┬───────────────────────────────────┘
               │
   ┌───────────┼───────────┐
   ▼           ▼           ▼
healthcare_  general_   orchestrator
agent        agent           │
   │           │          delegates
   │           │          via AgentTool
   ▼           ▼              │
shared/      local            ├──► healthcare_agent
fhir_hook    tools/           └──► general_agent
   │          general.py
   ▼
session state
(fhir_url, fhir_token, patient_id)
   │
   ▼
shared/tools/fhir.py  ──►  FHIR R4 server
```

**Key design principle:** FHIR credentials travel in the A2A message metadata — they never appear in the LLM prompt. The `extract_fhir_context` callback intercepts them before the model is called and stores them in session state, where tools read them at call time.

---

## Quick start

### Prerequisites

- Python 3.11 or later
- A [Google AI Studio](https://aistudio.google.com/app/apikey) API key (free)
- Git

### 1 — Clone the repository

```bash
git clone https://github.com/your-org/my-adk-project.git
cd my-adk-project
```

### 2 — Create a virtual environment and install dependencies

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (Command Prompt)
.venv\Scripts\activate

pip install -r requirements.txt
```

### 3 — Configure environment variables

```bash
# macOS / Linux
cp .env.example .env

# Windows (Command Prompt)
copy .env.example .env
```

Open `.env` and set your Google API key:

```env
GOOGLE_API_KEY=your-google-api-key-here
```

### 4 — Run the agents

**Option A — `adk web` (recommended for local development)**

Opens a visual chat UI in your browser. All three agents appear in the dropdown. No API key header required.

```bash
adk web .
```

Then open **http://localhost:8000** and select which agent to chat with.

> **Note:** `adk web` bypasses the A2A middleware, so FHIR tools will report missing credentials (no metadata is sent). Everything else — tool calls, model responses, instructions — works normally. Use this for developing and testing agent logic before wiring to Prompt Opinion.

---

**Option B — A2A servers (required to connect to Prompt Opinion)**

**All three at once with `honcho` (recommended):**

```bash
pip install -r requirements-dev.txt   # one-time
honcho start
```

All three agents start in a single terminal with colour-coded logs — `healthcare` in one colour, `general` in another, `orchestrator` in a third. Ports 8001, 8002, and 8003 are all live simultaneously.

**Or start them individually in separate terminals:**

```bash
# Terminal 1 — FHIR healthcare agent
uvicorn healthcare_agent.app:a2a_app --host 0.0.0.0 --port 8001

# Terminal 2 — General-purpose agent
uvicorn general_agent.app:a2a_app --host 0.0.0.0 --port 8002

# Terminal 3 — Orchestrator (delegates to agents 1 & 2)
uvicorn orchestrator.app:a2a_app --host 0.0.0.0 --port 8003
```

### 5 — Verify an agent is running

```bash
curl http://localhost:8001/.well-known/agent-card.json
```

You should see the agent card JSON describing the agent's capabilities and security requirements.

---

## The three agents

### `healthcare_agent` — FHIR-connected clinical assistant

The most complete example. Receives FHIR credentials from the caller via A2A metadata, extracts them into session state, and uses them to query a FHIR R4 server.

**Files to change when building your own:**

| File | What to change |
|---|---|
| `healthcare_agent/agent.py` | Model, instruction, tools list |
| `healthcare_agent/app.py` | Agent name, description, URL, FHIR extension URI |
| `shared/tools/fhir.py` | Add or modify FHIR query tools |
| `shared/middleware.py` | Update `VALID_API_KEYS` |

**Use this as your starting point if** your agent needs to query patient data from a FHIR server.

---

### `general_agent` — General-purpose assistant (no FHIR)

The minimal example. No `before_model_callback`, no FHIR tools. Demonstrates that the FHIR layer is completely optional.

Includes two tools that work offline with no external APIs:
- `get_current_datetime(timezone)` — current date/time in any IANA timezone
- `look_up_icd10(term)` — ICD-10-CM code lookup from a built-in reference table (15 common conditions)

**Files to change when building your own:**

| File | What to change |
|---|---|
| `general_agent/agent.py` | Model, instruction, tools list |
| `general_agent/app.py` | Agent name, description, URL |
| `general_agent/tools/general.py` | Replace with your own tools |

**Use this as your starting point if** your agent does not need patient data (knowledge lookup, scheduling, notifications, etc.).

---

### `orchestrator` — Multi-agent orchestrator

Shows ADK's built-in sub-agent routing (`AgentTool`). Gemini decides which specialist to call based on the question. Both `healthcare_agent` and `general_agent` run in-process as sub-agents — no separate HTTP calls needed.

Session state is shared, so FHIR credentials extracted by the orchestrator's `before_model_callback` are immediately available to the `healthcare_agent`'s tools.

**Use this as your starting point if** you want a single endpoint that coordinates multiple specialties.

To add a third sub-agent:
1. Create a new agent package (copy `general_agent` as a template)
2. Import its `root_agent` in `orchestrator/agent.py`
3. Add `AgentTool(agent=your_new_agent)` to the tools list
4. Update the instruction

---

## The shared library

```
shared/
├── logging_utils.py    ANSI-colour logger, configure_logging(package_name)
├── middleware.py        API key enforcement + FHIR metadata bridging
├── fhir_hook.py        before_model_callback — extracts FHIR credentials into state
├── app_factory.py      create_a2a_app() — builds the A2A ASGI app for any agent
└── tools/
    ├── __init__.py     Re-exports all shared tools
    └── fhir.py         FHIR R4 query tools (demographics, meds, conditions, observations)
```

Think of `shared/` as a class library. Any agent can import from it:

```python
from shared.fhir_hook import extract_fhir_context
from shared.tools import get_patient_demographics
from shared.app_factory import create_a2a_app
```

`shared/` is never run directly — it has no `agent.py` or `app.py`.

---

## Adding tools

### To an existing agent

**Step 1** — Write the tool function (last param must be `tool_context: ToolContext`):

```python
# general_agent/tools/general.py
from google.adk.tools import ToolContext
import logging

logger = logging.getLogger(__name__)

def get_care_team(tool_context: ToolContext) -> dict:
    """Returns the patient's care team members."""
    patient_id = tool_context.state.get("patient_id", "unknown")
    logger.info("tool_get_care_team patient_id=%s", patient_id)
    # your implementation here
    return {"status": "success", "care_team": [...]}
```

**Step 2** — Export it from the tools `__init__.py`:

```python
from .general import get_current_datetime, look_up_icd10, get_care_team
__all__ = [..., "get_care_team"]
```

**Step 3** — Register it in `agent.py`:

```python
from .tools import get_current_datetime, look_up_icd10, get_care_team

root_agent = Agent(..., tools=[..., get_care_team])
```

### As a shared FHIR tool

Add it to `shared/tools/fhir.py`, export from `shared/tools/__init__.py`, then import it in any agent that needs it.

---

## FHIR context (optional)

FHIR context is **completely optional**. Agents that don't need it simply omit `before_model_callback` — `general_agent` is the example.

### How credentials flow

```
A2A request
  └── params.message.metadata
        └── "http://.../fhir-context": { fhirUrl, fhirToken, patientId }
              │
              ▼  shared/middleware.py bridges to params.metadata
              │
              ▼  extract_fhir_context() runs before every LLM call
              │
              ▼
        session state
              ├── fhir_url   → tool_context.state["fhir_url"]
              ├── fhir_token → tool_context.state["fhir_token"]
              └── patient_id → tool_context.state["patient_id"]
```

### What Prompt Opinion sends

```json
{
  "jsonrpc": "2.0",
  "method": "message/stream",
  "params": {
    "message": {
      "metadata": {
        "https://your-workspace.promptopinion.ai/schemas/a2a/v1/fhir-context": {
          "fhirUrl":   "https://your-fhir-server.example.org/r4",
          "fhirToken": "<short-lived-bearer-token>",
          "patientId": "patient-uuid"
        }
      },
      "parts": [{ "kind": "text", "text": "What medications is this patient on?" }],
      "role": "user"
    }
  }
}
```

### What if FHIR context is not sent?

`extract_fhir_context` writes nothing to session state. FHIR tools return a clear error message explaining that credentials were not provided. The agent passes that back to the caller rather than hallucinating data.

### Log markers to watch

| Log marker | Meaning |
|---|---|
| `FHIR_URL_FOUND` | FHIR server URL received |
| `FHIR_TOKEN_FOUND fingerprint=len=N sha256=X` | Token received (value never logged) |
| `FHIR_PATIENT_FOUND` | Patient ID received |
| `hook_called_fhir_found` | All three credentials stored in state |
| `hook_called_no_metadata` | Request had no metadata |
| `hook_called_fhir_not_found` | Metadata present but FHIR key not found |
| `hook_called_fhir_malformed` | FHIR key found but value was not a JSON object |

---

## Configuration reference

Copy `.env.example` to `.env` and set values before starting any server.

| Variable | Required | Default | Description |
|---|---|---|---|
| `GOOGLE_API_KEY` | **Yes** | — | Google AI Studio key for Gemini |
| `LOG_FULL_PAYLOAD` | No | `true` | Log full JSON-RPC request body on each request |
| `LOG_HOOK_RAW_OBJECTS` | No | `false` | Dump raw ADK callback objects — debug only |
| `HEALTHCARE_AGENT_URL` | No | `http://localhost:8001` | Public URL for the healthcare agent |
| `GENERAL_AGENT_URL` | No | `http://localhost:8002` | Public URL for the general agent |
| `ORCHESTRATOR_URL` | No | `http://localhost:8003` | Public URL for the orchestrator |

---

## API security

All A2A endpoints except the agent card require an `X-API-Key` header.

### Updating the allowed keys

Open `shared/middleware.py` and update `VALID_API_KEYS`:

```python
VALID_API_KEYS: set = {
    "my-secret-key-123",   # ← replace with your real keys
    "another-valid-key",
}
```

In production, load from environment variables or a secrets manager:

```python
import os

VALID_API_KEYS: set = {
    k for k in [
        os.getenv("API_KEY_PRIMARY"),
        os.getenv("API_KEY_SECONDARY"),
    ]
    if k
}
```

### Endpoints (per agent)

| Endpoint | Auth required | Description |
|---|---|---|
| `GET /.well-known/agent-card.json` | No | Agent discovery |
| `POST /` | Yes (`X-API-Key`) | A2A JSON-RPC — all agent interactions |

---

## Testing locally

A shell script exercises the full `healthcare_agent` pipeline with `curl`:

```bash
# Start the healthcare agent first (separate terminal)
uvicorn healthcare_agent.app:a2a_app --host 127.0.0.1 --port 8001 --log-level info

# Run all test cases
bash scripts/test_fhir_hook.sh
```

| Case | Description | Expected log marker |
|---|---|---|
| A | Missing API key | `security_rejected_missing_api_key` |
| B | Valid key, no metadata | `hook_called_no_metadata` |
| C | Valid key, wrong metadata key | `hook_called_fhir_not_found` |
| D | Valid key + FHIR context — clinical summary | `hook_called_fhir_found` |
| D2 | Valid key + FHIR context — vital signs | `tool_get_recent_observations` |
| E | Valid key + malformed FHIR value | `hook_called_fhir_malformed` |

---

## Connecting to Prompt Opinion

[Prompt Opinion](https://promptopinion.ai) is a multi-agent platform that orchestrates agents like these — routing conversations, injecting patient context, and composing results across multiple specialised agents.

### Registration steps

1. **Deploy your agent** to a publicly reachable URL (e.g. `https://my-agent.example.com`).

2. **Set the public URL** via environment variable or directly in `app.py`:
   ```bash
   HEALTHCARE_AGENT_URL=https://my-agent.example.com
   ```

3. **Update the FHIR extension URI** in `app.py` to match your Prompt Opinion workspace:
   ```python
   fhir_extension_uri="https://your-workspace.promptopinion.ai/schemas/a2a/v1/fhir-context"
   ```

4. **Register the agent in Prompt Opinion** by providing:
   - Agent card URL: `https://my-agent.example.com/.well-known/agent-card.json`
   - Your `X-API-Key` value (Prompt Opinion sends this on every request)

5. **Prompt Opinion discovers your agent** by fetching the agent card, learns that an API key is required, and begins routing requests to it.

### What Prompt Opinion provides

When your agent is called from Prompt Opinion, the platform automatically injects into the A2A message metadata:

- The patient's **FHIR server URL** for your workspace
- A **short-lived bearer token** scoped to the current user session
- The **patient ID** selected in the active encounter

Your tools receive these transparently from `tool_context.state` — you never handle FHIR authentication yourself.

---

## License

MIT

---

*Built on [Google ADK](https://google.github.io/adk-docs/) and the [A2A protocol](https://google.github.io/A2A/). Designed for the [Prompt Opinion](https://promptopinion.ai) multi-agent platform.*
