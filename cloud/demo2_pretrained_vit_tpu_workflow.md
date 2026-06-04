# Demo 2 TPU Workflow Reference

This is the reference document for Demo 2 Cloud TPU execution. For the
recommended executable path from local baseline to TPU cleanup, start with
[demo2_tpu_quickstart.md](demo2_tpu_quickstart.md).

This document keeps details that are useful after the quickstart: resource
creation variants, queued-resource state interpretation, spot and TRC spot
notes, monitoring and evidence guidance, cleanup discipline, troubleshooting
notes, optional classifier-head fine-tuning smoke evidence, and the
course-specific TRC spot TPU evidence appendices.

TRC was the funding/quota path used for the course project. TRC is not required
by the code itself. Any valid Google Cloud TPU quota and funding path may be
used if the selected accelerator type, runtime, zone, and cleanup plan are
appropriate.

## Document Roles

- `cloud/demo2_tpu_quickstart.md`: main user-facing executable quickstart.
- `cloud/demo2_pretrained_vit_tpu_workflow.md`: reference material and workflow
  variants.
- `report/google_cloud_trc_setup.md`: course-specific Google Cloud / TRC setup
  and evidence record.
- `report/results/README.md`: index of curated report result artifacts.

## Command Environments

- **Local Ubuntu/WSL repo root**: local CPU baseline, local validation, and local
  comparison commands.
- **Google Cloud Shell or local terminal with `gcloud`**: cloud auth, resource
  creation, SSH, artifact retrieval, and cleanup.
- **TPU VM shell**: repository checkout, dependency setup, JAX TPU verification,
  and Demo 2 execution, including the optional classifier-head fine-tuning
  smoke workflow.

## Placeholders And Privacy

Use placeholders until values are safe and intentional.

Cloud resource placeholders are used from Google Cloud Shell or a local
`gcloud` terminal:

- `<PROJECT_ID>`: Google Cloud project ID.
- `<PROJECT_NUMBER>`: Google Cloud project number for TRC submission.
- `<ZONE>`: TPU zone.
- `<TPU_NAME>`: TPU VM resource name.
- `<ACCELERATOR_TYPE>`: TPU accelerator type.
- `<RUNTIME_VERSION>`: TPU VM runtime version.
- `<QUEUED_RESOURCE_ID>`: queued-resource identifier.
- `<NETWORK_NAME>`: VPC network name.
- `<SUBNET_NAME>`: regional subnet name.

Repository checkout placeholders are used inside the TPU VM shell:

- `<REPO_URL>`: Git URL for the repository checkout.
- `<BRANCH>`: branch to test.
- `<COMMIT_SHA>`: optional exact commit hash for reproducible benchmark mode.
  Latest-branch smoke-test mode may omit checkout and record `git rev-parse
  HEAD` instead.

`<COMMIT_SHA>` is the run-input pin used before execution. New Demo 2 raw JSON
results also record observed code provenance as `git_commit`, `git_branch`, and
`git_dirty` when Git metadata is available inside the TPU VM shell. Use the JSON
`git_commit` field to audit the code version that actually produced a new
artifact; do not invent these fields for legacy artifacts.

Do not document or commit project numbers, billing account IDs, private
hostnames, private IP addresses, credential paths, SSH key fingerprints, raw
terminal logs, private screenshots, service account keys, or cloud credential
files, subnet CIDR ranges, or detailed network topology unless intentionally
needed and safe.

## Resource And Funding Paths

| Situation | Resource path | Notes |
| --- | --- | --- |
| Normal on-demand TPU quota | Direct TPU VM or queued resource | May incur regular cost |
| Spot quota | Queued resource with `--spot` | Availability and interruption behavior differ from on-demand |
| TRC spot quota | Queued resource with `--spot` | Course project used this path |
| No TPU quota or no budget | Local CPU only | Demo 2 still runs locally |
| Long `WAITING_FOR_RESOURCES` | Delete queued resource and try another zone/type | Do not leave unused resources queued |

The quickstart uses the queued-resource path because it matches the successful
course run and works naturally with spot or TRC spot quota.

## Direct TPU VM Creation Path

Use direct TPU VM creation only when it matches your quota/funding path. Keep the
delete command ready before creating the VM.

```bash
gcloud compute tpus tpu-vm create <TPU_NAME> \
  --project=<PROJECT_ID> \
  --zone=<ZONE> \
  --accelerator-type=<ACCELERATOR_TYPE> \
  --version=<RUNTIME_VERSION> \
  --network=<NETWORK_NAME> \
  --subnetwork=<SUBNET_NAME>

gcloud compute tpus tpu-vm describe <TPU_NAME> \
  --project=<PROJECT_ID> \
  --zone=<ZONE>

gcloud compute tpus tpu-vm ssh <TPU_NAME> \
  --project=<PROJECT_ID> \
  --zone=<ZONE>

gcloud compute tpus tpu-vm delete <TPU_NAME> \
  --project=<PROJECT_ID> \
  --zone=<ZONE>
```

## Queued-Resource Path

Queued resources are useful for spot quota, TRC spot quota, and cases where the
resource may wait for capacity before provisioning.

Generic queued resource:

```bash
gcloud compute tpus queued-resources create <QUEUED_RESOURCE_ID> \
  --node-id=<TPU_NAME> \
  --project=<PROJECT_ID> \
  --zone=<ZONE> \
  --accelerator-type=<ACCELERATOR_TYPE> \
  --runtime-version=<RUNTIME_VERSION> \
  --network=<NETWORK_NAME> \
  --subnetwork=<SUBNET_NAME>
```

Spot or TRC spot queued resource:

```bash
gcloud compute tpus queued-resources create <QUEUED_RESOURCE_ID> \
  --node-id=<TPU_NAME> \
  --project=<PROJECT_ID> \
  --zone=<ZONE> \
  --accelerator-type=<ACCELERATOR_TYPE> \
  --runtime-version=<RUNTIME_VERSION> \
  --network=<NETWORK_NAME> \
  --subnetwork=<SUBNET_NAME> \
  --spot
```

The subnet must exist in the region corresponding to the selected TPU zone. The
course smoke run used the default VPC network and default subnet in the selected
region, but other valid VPC/subnet setups may be used.

Optional queue expiration guard: if the installed `gcloud` version and selected
queued-resource API support it, `--valid-until-duration` can limit how long a
queued resource should remain valid. Check support with
`gcloud compute tpus queued-resources create --help` before using it. This was
not recorded as used in the completed course smoke run, and it does not replace
manual cleanup.

Delete a queued resource after artifact retrieval, failed provisioning, or an
abandoned wait:

```bash
gcloud compute tpus queued-resources delete <QUEUED_RESOURCE_ID> \
  --project=<PROJECT_ID> \
  --zone=<ZONE> \
  --force
```

## Queued-Resource State Interpretation

Inspect state:

```bash
gcloud compute tpus queued-resources describe <QUEUED_RESOURCE_ID> \
  --project=<PROJECT_ID> \
  --zone=<ZONE>
```

Interpret state conservatively:

- `WAITING_FOR_RESOURCES`: request accepted but waiting for capacity. Keep
  waiting only if the run window and cost plan still make sense.
- `PROVISIONING`: allocation has started; wait until the TPU VM is available.
- `ACTIVE`: TPU VM should exist; verify with `tpu-vm describe`, then SSH and
  run backend checks before benchmarking.
- Error or unexpected state: do not benchmark. Preserve only privacy-safe notes,
  then clean up if a resource exists.

If `WAITING_FOR_RESOURCES` blocks the available run window, delete the queued
resource before trying another zone or accelerator type.

## TPU VM Setup And Demo 2 Run

Use [demo2_tpu_quickstart.md](demo2_tpu_quickstart.md) for the complete
repository checkout, `uv` setup, TPU-compatible JAX install, backend/device
verification, Demo 2 command, artifact retrieval, and local comparison sequence.
That quickstart also contains Imagenette 320 TPU preparation, retrieval, and
curated table generation commands.

Reference command for the public `b4` TPU smoke run:

```bash
uv run --group pretrained python examples/pretrained_vit_inference.py \
  --jax-platform tpu \
  --image-manifest examples/assets/manifest.txt \
  --batch-size 4 \
  --warmup-steps 1 \
  --benchmark-steps 5 \
  --output runs/vit-inference/demo2_cloud_public_examples_tpu_b4.json
```

Do not claim TPU execution succeeded unless backend/device verification reports
TPU and the Demo 2 command completes on the TPU VM.

Optional classifier-head fine-tuning is also part of Demo 2. The script
`examples/demo2_pretrained_vit_finetune.py` freezes the ViT backbone, updates
only the classifier head, and writes generated artifacts under
`runs/vit-finetune/`. Use the quickstart for the executable TPU command and
resume sequence. Do not claim TPU fine-tuning evidence until the command has
actually completed on a TPU VM and artifacts have been retrieved.

## Imagenette 320 Cloud TPU Inference Evidence

The repository now has retrieved TPU JSON artifacts for Imagenette 320
validation-manifest inference runs:

- `val64`: batch sizes `b1`, `b4`, and `b8`.
- `val256`: batch sizes `b1`, `b4`, and `b8`.
- `val_full`: batch sizes `b1`, `b4`, and `b8`.

The repository does not download Imagenette automatically. Download Imagenette
320 from its official source and preserve the same path on the local machine and
TPU VM. Concrete `curl` and `tar` commands using the official fastai Imagenette
320 archive are documented in
[../docs/demo2_pretrained_vit.md](../docs/demo2_pretrained_vit.md).

```text
data/local/imagenette2-320/val/manifest_val_64.txt
data/local/imagenette2-320/val/manifest_val_256.txt
data/local/imagenette2-320/val/manifest_val_full.txt
```

Use `scripts/build_image_manifest.py` to generate the `manifest_val_64.txt`,
`manifest_val_256.txt`, and `manifest_val_full.txt` files from an existing
extracted validation directory. Do not commit `data/local/`, generated
manifests, dataset files, model caches, raw JSON artifacts, or raw cloud logs.

The retrieved cloud TPU output names are:

```text
runs/vit-inference/demo2_cloud_imagenette320_val64_tpu_b1.json
runs/vit-inference/demo2_cloud_imagenette320_val64_tpu_b4.json
runs/vit-inference/demo2_cloud_imagenette320_val64_tpu_b8.json
runs/vit-inference/demo2_cloud_imagenette320_val256_tpu_b1.json
runs/vit-inference/demo2_cloud_imagenette320_val256_tpu_b4.json
runs/vit-inference/demo2_cloud_imagenette320_val256_tpu_b8.json
runs/vit-inference/demo2_cloud_imagenette320_valfull_tpu_b1.json
runs/vit-inference/demo2_cloud_imagenette320_valfull_tpu_b4.json
runs/vit-inference/demo2_cloud_imagenette320_valfull_tpu_b8.json
```

Retrieve a complete TPU VM result folder from **Google Cloud Shell or a local
terminal with `gcloud`**, preferably at the local repository root:

```bash
mkdir -p runs

gcloud compute tpus tpu-vm scp --recurse \
  "$TPU_NAME":~/numerical-jax-project/runs/vit-inference \
  runs/ \
  --project "$PROJECT_ID" \
  --zone "$ZONE"
```

The curated TPU Markdown table paths are:

```text
report/results/demo2_cloud_imagenette320_val64_tpu.md
report/results/demo2_cloud_imagenette320_val256_tpu.md
report/results/demo2_cloud_imagenette320_valfull_tpu.md
```

These are ViT inference timing tables only. They are not training results, not
Imagenette accuracy evaluation, not a full controlled benchmark study, and not a
universal TPU speedup claim.

## Artifact Policy

- Raw TPU JSON belongs under ignored `runs/vit-inference/`.
- Generated comparison JSON also belongs under ignored `runs/vit-inference/`.
- Raw fine-tuning summaries, metrics, logs, predictions, checkpoints, and
  resume artifacts belong under ignored `runs/vit-finetune/`.
- Curated Markdown tables may be committed under `report/results/` when they are
  intentionally report-ready.
- Do not commit model caches, Hugging Face cache files, private images, raw cloud
  logs, credential files, or unredacted screenshots.

## Monitoring Evidence Guidance

Useful monitoring and observability evidence can include:

- Application metrics: loss, step time, examples per second, runtime,
  checkpoint step, resume start step, and final step.
- Checkpoint evidence: checkpoint directory, latest checkpoint step, restored
  start step, final step, and whether the run resumed.
- Google Cloud Console resource lifecycle notes.
- Cloud Monitoring charts for TPU utilization, host CPU, memory, or idle time,
  if available.
- Cloud Logging notes for run timeline or troubleshooting.
- Terminal command notes for backend/device verification, benchmark execution,
  artifact retrieval, cleanup, and deletion verification.

Screenshots should be included only when they add report value and do not expose
project identifiers, private images, secrets, hostnames, IP addresses, or
credentials.

For the first completed smoke run, preserved evidence is application-level JSON
metrics, artifact retrieval, comparison generation, and cleanup verification.
Broader monitoring analysis remains future work.

## Evidence Checklist

Capture these items for each real TPU attempt:

- Branch and exact commit hash used on the TPU VM.
- TPU VM name, zone, accelerator type, runtime version, and quota/funding path.
- Backend/device output, including JAX version, default backend, device count,
  local device count, and devices.
- Demo command and output artifact path.
- TPU JSON artifact path.
- For fine-tuning attempts: `summary.json`, `metrics.csv`, checkpoint directory,
  latest checkpoint step, resume command, resume start step, final step, and
  whether controlled SIGTERM or real spot interruption was used.
- Artifact retrieval command.
- Cleanup command and deletion verification output.
- Comparison command and curated Markdown table path.
- Limitations and failed attempts, if any.

For spot or preemptible TPU risk, controlled SIGTERM plus resume is the primary
checkpoint evidence path. Real interruption is non-deterministic and may not
send graceful SIGTERM. Durable resume after TPU VM deletion requires copying
checkpoints to Google Cloud Storage or another durable location before deleting
the VM.

## Cleanup Discipline

Every resource-creation command should have a visible cleanup command before the
resource is created. After deleting resources, verify both queued resources and
TPU VMs:

```bash
gcloud compute tpus queued-resources list \
  --project=<PROJECT_ID> \
  --zone=<ZONE>

gcloud compute tpus tpu-vm list \
  --project=<PROJECT_ID> \
  --zone=<ZONE>
```

For queued resources, use `--force` when deleting:

```bash
gcloud compute tpus queued-resources delete <QUEUED_RESOURCE_ID> \
  --project=<PROJECT_ID> \
  --zone=<ZONE> \
  --force
```

Record cleanup evidence in report materials only after cleanup actually occurs.

## Troubleshooting Notes

- If `WAITING_FOR_RESOURCES` lasts longer than the run window, delete the queued
  resource and consider another zone, accelerator type, or quota path.
- If resource creation fails before provisioning, verify the selected network
  and regional subnet. A missing default subnet in the selected region can block
  TPU resource creation; either create/select a valid regional subnet or use a
  different zone/network/subnet combination.
- If backend/device verification does not show TPU devices, do not run or claim
  TPU benchmark evidence. Re-check runtime version, JAX TPU install, and the VM
  environment.
- If first-run model download fails, check network access and Hugging Face cache
  state. Do not commit model caches.
- If artifact retrieval fails, verify the output file exists on the TPU VM and
  that the local destination directory exists.
- If cleanup verification still shows resources, inspect and delete remaining
  resources before ending the run.

## Appendix: Course TRC Spot Smoke-Run Evidence

The course project's first successful Demo 2 TPU run used the queued-resource
path with TRC spot quota.

Non-private resource details:

- Zone: `us-east1-d`.
- Google Cloud accelerator type: `v6e-1`.
- TPU runtime version: `v2-alpha-tpuv6e`.
- Quota/funding path: TRC spot quota.
- JSON-visible device kind: `TPU v6 lite`.
- Branch used on the TPU VM: `feat/demo2-tpu-evidence`.
- Exact TPU checkout commit: not preserved in the available report notes. Do
  not substitute a later local commit SHA; treat this as a reproducibility
  limitation.

Operational note:

- An initial v4 queued resource in `us-central2-b` remained in
  `WAITING_FOR_RESOURCES` for several days and was abandoned.
- The successful smoke run used the smaller v6e TRC spot queued resource in
  `us-east1-d`.
- This is an availability and queue behavior note, not a performance claim.

Artifact and comparison evidence:

- TPU JSON artifact was generated and retrieved:
  `runs/vit-inference/demo2_cloud_public_examples_tpu_b4.json`.
- Curated comparison table was generated:
  `report/results/demo2_local_cpu_vs_cloud_tpu_public_examples_b4.md`.
- Cleanup was verified after artifact retrieval: queued-resource deletion
  succeeded, and queued-resource and TPU-VM list commands returned zero items in
  `us-east1-d`.

Recorded TPU JSON fields for this smoke run:

```text
backend=tpu
device_kind=TPU v6 lite
mode=image_manifest
manifest_kind=public_example
num_images=5
batch_size=4
num_batches=2
num_padded_images=3
timed_batch_runs=10
total_timed_inference_sec=about 0.01098252
throughput_images_per_sec=about 2276.3446
```

The generated comparison table reports about 1931.76x throughput speedup versus
the local CPU `b4` public-examples artifact for this specific five-image smoke
run. This should not be generalized to TPU performance overall.

Smoke-run limitations:

- Five public example images.
- Batch size 4.
- Final-batch padding with `num_padded_images = 3`.
- Short benchmark loop.
- No dataset-level accuracy evaluation.
- Not a full controlled hardware benchmark.

## Appendix: Course Imagenette 320 TPU Inference Evidence

After the public-example smoke run, Demo 2 TPU JSON artifacts were retrieved for
Imagenette 320 validation-manifest inference:

- `val64`: `b1`, `b4`, and `b8`.
- `val256`: `b1`, `b4`, and `b8`.
- `val_full`: `b1`, `b4`, and `b8`.

The curated Markdown tables are:

```text
report/results/demo2_cloud_imagenette320_val64_tpu.md
report/results/demo2_cloud_imagenette320_val256_tpu.md
report/results/demo2_cloud_imagenette320_valfull_tpu.md
```

Recorded table scope:

- Backend: `tpu`.
- JSON-visible device kind: `TPU v6 lite`.
- Benchmark shape: one warmup step and five benchmark steps.
- Full validation manifest size: 3925 images.
- Full validation `b4` and `b8` runs include 3 padded final-batch entries.

Interpretation limits:

- ViT inference timing only.
- No model training or fine-tuning.
- No Imagenette dataset-level accuracy evaluation.
- No full controlled hardware benchmark.
- No universal TPU speedup claim.

## References

- Google Cloud TPU VM management:
  https://cloud.google.com/tpu/docs/managing-tpus-tpu-vm
- `gcloud compute tpus tpu-vm create` reference:
  https://cloud.google.com/sdk/gcloud/reference/compute/tpus/tpu-vm/create
- JAX installation guide:
  https://docs.jax.dev/en/latest/installation.html
- Google Cloud JAX on TPU guide:
  https://cloud.google.com/tpu/docs/run-calculation-jax
