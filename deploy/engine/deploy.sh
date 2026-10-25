#!/bin/sh
# The website's engine on Cloud Run (project frt-motion-lab, Blaze plan), behind frt-motion-lab.web.app/api/**.
# Once: install the Google Cloud CLI and sign in yourself (gcloud auth login). Then: sh deploy/engine/deploy.sh
#
#   one instance at most (one solve at a time, a queue of 8: a burst of visitors waits, it never multiplies cost),
#   none while idle (it starts on the first request, ~20 s), 4 vCPU / 4 GB while it runs.
#   CPU stays on between requests (--no-cpu-throttling): the solve runs after the POST has answered.
set -e
cd "$(dirname "$0")/../.."
PROJECT=frt-motion-lab REGION=us-central1
python deploy/engine/assemble.py
gcloud config set project $PROJECT
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
gcloud run deploy frt-engine --source out/engine_image --region $REGION \
  --cpu 4 --memory 4Gi --max-instances 1 --min-instances 0 --no-cpu-throttling \
  --concurrency 20 --timeout 300 --allow-unauthenticated
gcloud run services describe frt-engine --region $REGION --format 'value(status.url)'
