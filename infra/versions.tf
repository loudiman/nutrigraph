terraform {
  required_version = ">= 1.7" # import blocks with for_each

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 7.0"
    }
  }

  # The state bucket is the one resource Terraform cannot create for itself.
  # It is made once by hand: see README.md, "Bootstrap". GCS locks the state
  # on its own, so two runs cannot write at once.
  backend "gcs" {
    bucket = "nutrigraph-2026ldm-tfstate"
    prefix = "nutrigraph"
  }
}

provider "google" {
  project = var.project
  region  = var.region
}
