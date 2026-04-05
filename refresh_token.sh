#!/bin/bash
# refresh_token.sh — grab latest FHIR token from uvicorn log and update .env
#
# Usage:
#   1. Trigger a consult on the PO platform (opens a patient chart)
#   2. Run: ./refresh_token.sh
#   3. Token is written to .env automatically
#   4. No uvicorn restart needed — runner.py reads .env fresh each run
#
# Place this file in po-adk-python/ (same directory as .env)

LOG_FILE="/tmp/uvicorn.log"
ENV_FILE="$(cd "$(dirname "$0")" && pwd)/.env"

# Extract fhirToken value from the most recent EVAL_CAPTURE line
TOKEN=$(grep "EVAL_CAPTURE" "$LOG_FILE" | tail -1 | grep -o 'fhirToken=[^ ]*' | cut -d= -f2)

if [ -z "$TOKEN" ]; then
    echo "ERROR: No EVAL_CAPTURE line found in $LOG_FILE"
    echo ""
    echo "Steps to fix:"
    echo "  1. Make sure uvicorn is running and logging to /tmp/uvicorn.log"
    echo "  2. Open any patient in the PO platform to trigger a consult"
    echo "  3. Re-run this script"
    exit 1
fi

# Validate it looks like a JWT (starts with eyJ)
if [[ "$TOKEN" != eyJ* ]]; then
    echo "ERROR: Extracted token doesn't look like a JWT: ${TOKEN:0:20}..."
    echo "Check the EVAL_CAPTURE line format in $LOG_FILE"
    exit 1
fi

# Update or insert EVAL_FHIR_TOKEN in .env
if grep -q "^EVAL_FHIR_TOKEN=" "$ENV_FILE"; then
    sed -i '' "s|^EVAL_FHIR_TOKEN=.*|EVAL_FHIR_TOKEN=$TOKEN|" "$ENV_FILE"
    echo "Updated EVAL_FHIR_TOKEN in $ENV_FILE"
else
    echo "EVAL_FHIR_TOKEN=$TOKEN" >> "$ENV_FILE"
    echo "Added EVAL_FHIR_TOKEN to $ENV_FILE"
fi

echo "Token preview: ${TOKEN:0:40}..."
echo ""
echo "Done. Run your evals:"
echo "  cd eval && python run_evals.py --case tamera_preventive_v1 --verbose"