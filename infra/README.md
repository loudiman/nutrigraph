# infra

NutriGraph's GCP stack as Terraform. Why, and what it leaves out: [ADR 0006](../docs/adr/0006-terraform-adopts-the-live-gcp-stack.md).

| File | Holds |
|---|---|
| `main.tf` | APIs, the `default` subnet (Private Google Access), the image registry |
| `iam.tf` | the three service accounts and the pipeline's project roles |
| `secrets.tf` | the five secrets and who may read each one — never a value |
| `run.tf` | the agent and gateway (one module, used twice), who may call each, the purge job and its schedule |
| `build.tf` | the merge and pull-request triggers |
| `imports.tf` | adopts the live resources; delete after the first apply |
| `modules/cloud_run_service` | one Cloud Run service as NutriGraph runs it |

## Bootstrap (once)

Billing must be on for the project: Secret Manager, Artifact Registry, Cloud Scheduler and Compute refuse every call without it.

```sh
gcloud storage buckets create gs://nutrigraph-2026ldm-tfstate \
  --project nutrigraph-2026ldm --location asia-southeast1 --uniform-bucket-level-access
gcloud storage buckets update gs://nutrigraph-2026ldm-tfstate --versioning
gcloud services enable compute.googleapis.com --project nutrigraph-2026ldm   # the subnet import reads it
gcloud auth application-default login
```

## Run

```sh
cd infra
terraform init
terraform plan          # first run: imports, and the goal is "0 to add, 0 to change, 0 to destroy"
terraform apply         # records the imports in state
```

If the first plan shows a change, the code and the live project disagree. Read the diff, then fix the code to match what is live. Do not apply your way out of it. A plan that wants to **replace** a secret or the registry fails on `prevent_destroy`, as it should.

## What Terraform does not own

- **The image tag.** Cloud Build deploys a new tag on every merge, so the module ignores `image`.
- **Secret values.** Add them with `gcloud secrets versions add <name> --data-file=-`.
- **The GitHub connection** `nutrigraph-github`. It is authorised once in a browser.
- **The state bucket.** See Bootstrap.
