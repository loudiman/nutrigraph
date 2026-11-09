# One Cloud Run service as NutriGraph runs it: scale to zero, at most three
# instances, 1 CPU / 512 MiB, plain env plus Secret Manager references.
variable "name" { type = string }
variable "region" { type = string }
variable "image" { type = string }
variable "service_account" { type = string }
variable "ingress" { type = string }

variable "env" {
  type    = map(string)
  default = {}
}

variable "secrets" {
  description = "Env var name => Secret Manager secret id, read at its latest version."
  type        = map(string)
  default     = {}
}

variable "vpc_egress" {
  description = "Send all egress through the default VPC (direct VPC egress)."
  type        = bool
  default     = false
}

resource "google_cloud_run_v2_service" "this" {
  name     = var.name
  location = var.region
  ingress  = var.ingress

  template {
    service_account = var.service_account

    scaling {
      min_instance_count = 0
      max_instance_count = 3
    }

    dynamic "vpc_access" {
      for_each = var.vpc_egress ? [1] : []
      content {
        egress = "ALL_TRAFFIC"
        network_interfaces {
          network    = "default"
          subnetwork = "default"
        }
      }
    }

    containers {
      image = var.image

      resources {
        limits            = { cpu = "1000m", memory = "512Mi" }
        cpu_idle          = true # CPU only while a request is in flight: the scale-to-zero price
        startup_cpu_boost = true
      }

      dynamic "env" {
        for_each = var.env
        content {
          name  = env.key
          value = env.value
        }
      }

      dynamic "env" {
        for_each = var.secrets
        content {
          name = env.key
          value_source {
            secret_key_ref {
              secret  = env.value
              version = "latest"
            }
          }
        }
      }
    }
  }

  lifecycle {
    # Cloud Build deploys a new image tag on every merge (cloudbuild.yaml).
    # Without this, every plan after a deploy would try to roll the image back.
    ignore_changes = [template[0].containers[0].image, client, client_version]
  }
}

output "name" {
  value = google_cloud_run_v2_service.this.name
}

output "uri" {
  value = google_cloud_run_v2_service.this.uri
}
