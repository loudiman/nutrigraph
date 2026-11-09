module "agent" {
  source          = "./modules/cloud_run_service"
  name            = "nutrigraph-agent"
  region          = var.region
  image           = local.agent_image
  service_account = google_service_account.sa["agent"].email

  # No public ingress: a request that did not leave through this project's VPC
  # is refused by the platform and never reaches the application.
  ingress = "INGRESS_TRAFFIC_INTERNAL_ONLY"

  env = {
    LANGSMITH_TRACING  = "true"
    LANGSMITH_ENDPOINT = "https://apac.api.smith.langchain.com" # APAC data plane; the US default answers 403
  }
  secrets = {
    DATABASE_URL      = google_secret_manager_secret.s["neon-connection-string"].secret_id
    GOOGLE_API_KEY    = google_secret_manager_secret.s["gemini-api-key"].secret_id
    FDC_API_KEY       = google_secret_manager_secret.s["fdc-api-key"].secret_id
    LANGSMITH_API_KEY = google_secret_manager_secret.s["langsmith-api-key"].secret_id
  }

  depends_on = [google_secret_manager_secret_iam_member.reader]
}

module "gateway" {
  source          = "./modules/cloud_run_service"
  name            = "nutrigraph-gateway"
  region          = var.region
  image           = local.gateway_image
  service_account = google_service_account.sa["gateway"].email
  ingress         = "INGRESS_TRAFFIC_ALL"
  vpc_egress      = true # the call to the internal agent must leave through the VPC

  env = {
    AGENT_URL = module.agent.uri
  }
  secrets = {
    SESSION_SECRET = google_secret_manager_secret.s["cookie-signing-secret"].secret_id
  }

  depends_on = [google_secret_manager_secret_iam_member.reader, google_compute_subnetwork.default]
}

# Only the gateway may call the agent; anyone may call the gateway.
resource "google_cloud_run_v2_service_iam_member" "agent_invoker" {
  name     = module.agent.name
  location = var.region
  role     = "roles/run.invoker"
  member   = google_service_account.sa["gateway"].member
}

resource "google_cloud_run_v2_service_iam_member" "gateway_public" {
  name     = module.gateway.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# --- The 90-day purge (ADR 0002) ---------------------------------------------
resource "google_cloud_run_v2_job" "purge" {
  name     = "nutrigraph-purge"
  location = var.region

  template {
    template {
      service_account       = google_service_account.sa["agent"].email
      max_retries           = 1
      timeout               = "300s"
      execution_environment = "EXECUTION_ENVIRONMENT_GEN2"

      containers {
        image   = local.agent_image
        command = ["nutrigraph-purge"]

        env {
          name = "DATABASE_URL"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.s["neon-connection-string"].secret_id
              version = "latest"
            }
          }
        }

        resources {
          limits = { cpu = "1000m", memory = "512Mi" }
        }
      }
    }
  }

  lifecycle {
    ignore_changes = [template[0].template[0].containers[0].image, client, client_version]
  }
}

# The scheduler runs the job as the agent's identity, which may invoke it.
resource "google_cloud_run_v2_job_iam_member" "purge_invoker" {
  name     = google_cloud_run_v2_job.purge.name
  location = var.region
  role     = "roles/run.invoker"
  member   = google_service_account.sa["agent"].member
}

resource "google_cloud_scheduler_job" "purge" {
  name        = "nutrigraph-purge"
  region      = var.region
  schedule    = "0 3 * * *"
  time_zone   = "Asia/Manila"
  description = "Nulls message.raw_text after 90 days (ADR 0002)"

  # No retries: a missed purge runs again the next night, and the job itself
  # already retries once (max_retries above).
  retry_config {
    retry_count          = 0
    max_retry_duration   = "0s"
    min_backoff_duration = "5s"
    max_backoff_duration = "3600s"
    max_doublings        = 5
  }

  http_target {
    http_method = "POST"
    uri         = "https://run.googleapis.com/v2/projects/${var.project}/locations/${var.region}/jobs/${google_cloud_run_v2_job.purge.name}:run"

    oauth_token {
      service_account_email = google_service_account.sa["agent"].email
    }
  }
}
