locals {
  registry = "${var.region}-docker.pkg.dev/${var.project}/${google_artifact_registry_repository.images.repository_id}"

  # The first image a new environment starts from. After that, Cloud Build owns
  # the image tag (cloudbuild.yaml) and Terraform ignores it.
  agent_image   = "${local.registry}/agent:current"
  gateway_image = "${local.registry}/gateway:current"

  apis = ["compute", "run", "artifactregistry", "secretmanager", "cloudbuild", "cloudscheduler", "iam"]
}

resource "google_project_service" "api" {
  for_each           = toset(local.apis)
  service            = "${each.key}.googleapis.com"
  disable_on_destroy = false
}

# --- Network ----------------------------------------------------------------
# The agent takes internal ingress only, so the gateway's call must leave
# through this VPC, and the gateway must reach run.app over Private Google
# Access. Without it every Turn ends in agent_unreachable (docs/deploy.md).
resource "google_compute_subnetwork" "default" {
  name                     = "default"
  region                   = var.region
  network                  = "default"
  ip_cidr_range            = "10.148.0.0/20" # the auto-mode range for asia-southeast1
  private_ip_google_access = true

  # Not read from the live project (billing was off). If the plan disagrees,
  # fix the range here; never let Terraform replace the default subnet.
  lifecycle { prevent_destroy = true }
}

# --- Images -----------------------------------------------------------------
resource "google_artifact_registry_repository" "images" {
  repository_id = "nutrigraph"
  location      = var.region
  format        = "DOCKER"
  description   = "NutriGraph service images"

  lifecycle { prevent_destroy = true }
}
