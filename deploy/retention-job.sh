#!/usr/bin/env sh
# The 90-day purge, scheduled. A Cloud Run job on the agent image running the
# `nutrigraph-purge` console script, and a Cloud Scheduler job that runs it at
# 03:00 Asia/Manila every day. Running the purge is therefore not a manual step.
#
# Both commands are create-or-update, so this is safe to re-run, and the merge
# pipeline runs it after the agent image is built. It reads the same
# `neon-connection-string` secret the agent service reads; the job needs no
# other secret, because the purge makes no provider call.
set -eu

PROJECT="${PROJECT:-nutrigraph-2026ldm}"
REGION="${REGION:-asia-southeast1}"
# `current` is the rolling tag the merge pipeline pushes beside the commit tag.
IMAGE="${IMAGE:-asia-southeast1-docker.pkg.dev/${PROJECT}/nutrigraph/agent:current}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-nutrigraph-agent@${PROJECT}.iam.gserviceaccount.com}"
JOB=nutrigraph-purge
SCHEDULE="${SCHEDULE:-0 3 * * *}"

gcloud run jobs deploy "$JOB" \
  --project "$PROJECT" \
  --region "$REGION" \
  --image "$IMAGE" \
  --command nutrigraph-purge \
  --service-account "$SERVICE_ACCOUNT" \
  --set-secrets "DATABASE_URL=neon-connection-string:latest" \
  --max-retries 1 \
  --task-timeout 5m

# The scheduler calls the Cloud Run Admin API as the same service account, which
# holds roles/run.invoker on the job. No key, and nothing public.
RUN_URL="https://run.googleapis.com/v2/projects/${PROJECT}/locations/${REGION}/jobs/${JOB}:run"
VERB=create
gcloud scheduler jobs describe "$JOB" --project "$PROJECT" --location "$REGION" \
  >/dev/null 2>&1 && VERB=update

gcloud scheduler jobs "$VERB" http "$JOB" \
  --project "$PROJECT" \
  --location "$REGION" \
  --schedule "$SCHEDULE" \
  --time-zone Asia/Manila \
  --uri "$RUN_URL" \
  --http-method POST \
  --oauth-service-account-email "$SERVICE_ACCOUNT" \
  --description "Nulls message.raw_text after 90 days (ADR 0002)"
