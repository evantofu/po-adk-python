# eval/refresh_token.sh
echo "Go to the PO platform and trigger any consult."
echo "Waiting for EVAL_CAPTURE line in uvicorn log..."
tail -f /tmp/uvicorn.log | grep --line-buffered "EVAL_CAPTURE" | head -1 | while read line; do
    TOKEN=$(echo "$line" | sed 's/.*fhirToken=\([^ ]*\).*/\1/')
    URL=$(echo "$line" | sed 's/.*fhirUrl=\([^ ]*\).*/\1/')
    # Update .env in place
    sed -i '' "s|EVAL_FHIR_TOKEN=.*|EVAL_FHIR_TOKEN=$TOKEN|" ../.env
    sed -i '' "s|EVAL_FHIR_URL=.*|EVAL_FHIR_URL=$URL|" ../.env
    echo "✓ Token refreshed in .env (expires in ~1h)"
    exit 0
done