#!/usr/bin/env bash
# Run locally after: gcloud auth login
# Grants the local dev service account (a2a-next-speech) permission to query billing export.
set -euo pipefail
PROJECT="${1:-gls-training-486405}"
SA="a2a-next-speech@${PROJECT}.iam.gserviceaccount.com"

for ROLE in roles/bigquery.jobUser roles/bigquery.dataViewer; do
  echo ">>> Adding $ROLE for $SA"
  gcloud projects add-iam-policy-binding "$PROJECT" \
    --member="serviceAccount:$SA" \
    --role="$ROLE" \
    --condition=None \
    --quiet
done
echo "Done."
