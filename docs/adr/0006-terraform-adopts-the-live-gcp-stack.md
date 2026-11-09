# Terraform adopts the live GCP stack

The GCP side of NutriGraph was made by hand with `gcloud`: the two Cloud Run services, the purge job and its schedule, the service accounts and their grants, the secrets, the image registry and the build triggers. The steps were written down in `docs/deploy.md`, but a list of commands is not a description of the system: nothing showed when the live project drifted from it, a second environment meant replaying the list by hand, and an IAM change was not reviewable in a pull request. We therefore describe the stack in Terraform under `infra/`, adopt the live resources with `import` blocks rather than recreating them, and keep the state in a versioned GCS bucket.

## Considered options

- **Keep the gcloud scripts.** No new tool. Rejected because a script says how a resource was made once, not what it should be now, and cannot report drift.
- **Recreate the stack from Terraform and delete the old one.** Clean state, but it changes the service URLs, re-creates secrets (losing their values) and costs a deploy for nothing. Import keeps every live resource and proves the code matches reality: the first plan must show no changes.
- **Pulumi or Config Connector.** Both work on GCP. Terraform is the one most teams already read, and it needs nothing running in the project.

## Consequences

- **Cloud Build still owns the image tag.** Every merge deploys a new tag with `gcloud run deploy`, so Terraform ignores the container image (and the `client` fields `gcloud` writes). Terraform owns the shape of a service; the pipeline owns which build it runs.
- **Terraform never holds a secret value.** It owns each secret's name, replication and readers, never a version, so no key reaches the state file.
- **Secrets and the image registry carry `prevent_destroy`.** A plan that would replace either stops instead of deleting values or images.
- **The state bucket and the GitHub connection stay outside Terraform.** The bucket is the one thing Terraform cannot make for itself; the connection holds a GitHub OAuth token and is authorised in a browser.
- **Pull requests check the code, not the cloud.** `cloudbuild.pr.yaml` runs `fmt`, `init -backend=false` and `validate`, which need no credentials. A `plan` in CI would need the pipeline's identity to read IAM and secrets metadata across the project; that grant is a decision for later, and until then a plan is run from a laptop before an apply.
