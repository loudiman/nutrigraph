# The GitHub connection and its repository link are not here. The connection
# holds a GitHub OAuth token and is authorised in a browser once, so it stays
# outside Terraform and is referenced by name.
locals {
  github_repo = "projects/${var.project}/locations/${var.region}/connections/nutrigraph-github/repositories/nutrigraph"
}

resource "google_cloudbuild_trigger" "main" {
  name            = "nutrigraph-main"
  location        = var.region
  filename        = "cloudbuild.yaml"
  service_account = google_service_account.sa["build"].id

  repository_event_config {
    repository = local.github_repo
    push { branch = "^main$" }
  }
}

resource "google_cloudbuild_trigger" "pr" {
  name            = "nutrigraph-pr"
  location        = var.region
  filename        = "cloudbuild.pr.yaml"
  service_account = google_service_account.sa["build"].id

  repository_event_config {
    repository = local.github_repo
    pull_request { branch = "^main$" }
  }
}
