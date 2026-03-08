"""
healthcare_agent/tools/claims.py

FHIR tools specific to the claims coding agent.
Follows the exact pattern from shared/tools/fhir.py.

Tools defined here:
  - get_clinical_documents: fetches DocumentReference / clinical notes for the patient

Export from healthcare_agent/tools/__init__.py and add to agent.py tools=[...].
"""

import logging
import base64

import httpx
from google.adk.tools import ToolContext

from shared.tools.fhir import _get_fhir_context, _fhir_get, _http_error_result, _connection_error_result

logger = logging.getLogger(__name__)


def get_clinical_documents(tool_context: ToolContext) -> dict:
    """
    Retrieves clinical notes and encounter documentation for the current patient
    from the FHIR server.

    Queries DocumentReference resources and decodes any base64-encoded note
    content so the agent can read the full clinical text directly.

    No arguments required — patient identity comes from session context.

    Returns a list of documents with title, date, content type, and decoded text.
    This is the primary input for CPT code assignment in Step 2 of the workflow.
    """
    ctx = _get_fhir_context(tool_context)
    if isinstance(ctx, dict):
        return ctx
    fhir_url, fhir_token, patient_id = ctx

    logger.info("tool_get_clinical_documents patient_id=%s", patient_id)

    try:
        bundle = _fhir_get(
            fhir_url, fhir_token, "DocumentReference",
            params={
                "patient": patient_id,
                "_sort": "-date",
                "_count": "10",
            },
        )
    except httpx.HTTPStatusError as e:
        return _http_error_result(e)
    except Exception as e:
        return _connection_error_result(e)

    documents = []

    for entry in bundle.get("entry", []):
        res = entry.get("resource", {})

        # Document title — try description, then category text, then type text
        title = (
            res.get("description")
            or (res.get("category") or [{}])[0].get("text")
            or (res.get("type") or {}).get("text")
            or "Clinical Document"
        )

        # Document date
        doc_date = (
            res.get("date")
            or res.get("context", {}).get("period", {}).get("start")
        )

        # Extract content from attachment(s)
        contents = []
        for content_item in res.get("content", []):
            attachment = content_item.get("attachment", {})
            content_type = attachment.get("contentType", "unknown")
            raw_data = attachment.get("data")  # base64-encoded string
            url = attachment.get("url")
            title_attr = attachment.get("title", "")

            decoded_text = None

            if raw_data:
                try:
                    decoded_text = base64.b64decode(raw_data).decode("utf-8", errors="replace")
                except Exception:
                    decoded_text = "[Could not decode document content]"
            elif url:
                # Some FHIR servers store the note content at a URL reference
                # Attempt to fetch it using the same bearer token
                try:
                    response = httpx.get(
                        url if url.startswith("http") else f"{fhir_url}/{url}",
                        headers={
                            "Authorization": f"Bearer {fhir_token}",
                            "Accept": "text/plain, text/html, application/fhir+json",
                        },
                        timeout=15,
                    )
                    response.raise_for_status()
                    decoded_text = response.text
                except Exception:
                    decoded_text = f"[Content available at URL but could not be retrieved: {url}]"

            contents.append({
                "content_type": content_type,
                "title": title_attr,
                "text": decoded_text,
            })

        # Document author
        authors = [
            (a.get("display") or a.get("reference", "Unknown"))
            for a in res.get("author", [])
        ]

        documents.append({
            "document_id": res.get("id"),
            "title": title,
            "date": doc_date,
            "status": res.get("status"),
            "authors": authors,
            "contents": contents,
        })

    return {
        "status": "success",
        "patient_id": patient_id,
        "count": len(documents),
        "documents": documents,
    }