# Everything below was first made by hand with gcloud (docs/deploy.md,
# "Bootstrapping this from nothing"). These blocks adopt the live resources into
# state instead of recreating them. The goal of the first run is
# `terraform plan` with nothing to add, change or destroy. After the first
# successful apply, these blocks can be deleted; they are no-ops once the
# resources are in state.

locals {
  p = var.project
  r = var.region
}

import {
  for_each = toset(local.apis)
  to       = google_project_service.api[each.key]
  id       = "${local.p}/${each.key}.googleapis.com"
}

import {
  to = google_compute_subnetwork.default
  id = "projects/${local.p}/regions/${local.r}/subnetworks/default"
}

import {
  to = google_artifact_registry_repository.images
  id = "projects/${local.p}/locations/${local.r}/repositories/nutrigraph"
}

import {
  for_each = toset(["agent", "gateway", "build"])
  to       = google_service_account.sa[each.key]
  id       = "projects/${local.p}/serviceAccounts/nutrigraph-${each.key}@${local.p}.iam.gserviceaccount.com"
}

import {
  for_each = toset(["roles/artifactregistry.writer", "roles/run.admin", "roles/iam.serviceAccountUser", "roles/logging.logWriter"])
  to       = google_project_iam_member.build[each.key]
  id       = "${local.p} ${each.key} serviceAccount:nutrigraph-build@${local.p}.iam.gserviceaccount.com"
}

import {
  for_each = local.secret_readers
  to       = google_secret_manager_secret.s[each.key]
  id       = "projects/${local.p}/secrets/${each.key}"
}

import {
  for_each = local.secret_grants
  to       = google_secret_manager_secret_iam_member.reader[each.key]
  id       = "projects/${local.p}/secrets/${each.value.secret} roles/secretmanager.secretAccessor serviceAccount:nutrigraph-${each.value.reader}@${local.p}.iam.gserviceaccount.com"
}

import {
  to = module.agent.google_cloud_run_v2_service.this
  id = "projects/${local.p}/locations/${local.r}/services/nutrigraph-agent"
}

import {
  to = module.gateway.google_cloud_run_v2_service.this
  id = "projects/${local.p}/locations/${local.r}/services/nutrigraph-gateway"
}

import {
  to = google_cloud_run_v2_service_iam_member.agent_invoker
  id = "projects/${local.p}/locations/${local.r}/services/nutrigraph-agent roles/run.invoker serviceAccount:nutrigraph-gateway@${local.p}.iam.gserviceaccount.com"
}

import {
  to = google_cloud_run_v2_service_iam_member.gateway_public
  id = "projects/${local.p}/locations/${local.r}/services/nutrigraph-gateway roles/run.invoker allUsers"
}

import {
  to = google_cloud_run_v2_job.purge
  id = "projects/${local.p}/locations/${local.r}/jobs/nutrigraph-purge"
}

import {
  to = google_cloud_run_v2_job_iam_member.purge_invoker
  id = "projects/${local.p}/locations/${local.r}/jobs/nutrigraph-purge roles/run.invoker serviceAccount:nutrigraph-agent@${local.p}.iam.gserviceaccount.com"
}

import {
  to = google_cloud_scheduler_job.purge
  id = "projects/${local.p}/locations/${local.r}/jobs/nutrigraph-purge"
}

# Triggers import by id, not by name.
import {
  to = google_cloudbuild_trigger.main
  id = "projects/${local.p}/locations/${local.r}/triggers/8a1330c6-5731-4a19-bac3-2b312c9f7411"
}

import {
  to = google_cloudbuild_trigger.pr
  id = "projects/${local.p}/locations/${local.r}/triggers/42788db3-2433-4fa3-a54f-60a8b7377265"
}
