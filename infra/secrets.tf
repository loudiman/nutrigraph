# Terraform owns each secret's name, replication and readers. It never owns a
# value: no google_secret_manager_secret_version here, so no key ever lands in
# the state file. Values are added with `gcloud secrets versions add`.
locals {
  secret_readers = {
    "neon-connection-string" = ["agent", "build"] # build: the migration step
    "gemini-api-key"         = ["agent", "build"] # build: the eval gate
    "langsmith-api-key"      = ["agent", "build"]
    "fdc-api-key"            = ["agent"]
    "cookie-signing-secret"  = ["gateway"]
  }

  secret_grants = merge([
    for secret, readers in local.secret_readers : {
      for r in readers : "${secret}/${r}" => { secret = secret, reader = r }
    }
  ]...)
}

resource "google_secret_manager_secret" "s" {
  for_each  = local.secret_readers
  secret_id = each.key

  replication {
    auto {}
  }

  # Replacing a secret deletes its values. A plan that wants to must stop here.
  lifecycle { prevent_destroy = true }
}

resource "google_secret_manager_secret_iam_member" "reader" {
  for_each  = local.secret_grants
  secret_id = google_secret_manager_secret.s[each.value.secret].id
  role      = "roles/secretmanager.secretAccessor"
  member    = google_service_account.sa[each.value.reader].member
}
