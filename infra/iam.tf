# One runtime identity per service, and one for the pipeline. Each reads only
# its own secrets (docs/deploy.md, "The secrets").
resource "google_service_account" "sa" {
  for_each     = toset(["agent", "gateway", "build"])
  account_id   = "nutrigraph-${each.key}"
  display_name = "nutrigraph-${each.key}"
}

# What the pipeline may do on the project: push images, deploy Cloud Run,
# act as the two runtime identities, write its logs.
resource "google_project_iam_member" "build" {
  for_each = toset([
    "roles/artifactregistry.writer",
    "roles/run.admin",
    "roles/iam.serviceAccountUser",
    "roles/logging.logWriter",
  ])
  project = var.project
  role    = each.key
  member  = google_service_account.sa["build"].member
}
