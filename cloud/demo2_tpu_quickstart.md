# Demo 2 TPU Quickstart

## Purpose

This is the reusable Cloud TPU quickstart for Demo 2 pretrained ViT inference
and the optional classifier-head fine-tuning smoke extension.
It separates shared resource setup from three executable paths: public-example
TPU inference smoke, Imagenette 320 TPU inference artifacts, and optional
classifier-head fine-tuning with GCS checkpoint/resume. It also documents
artifact retrieval, local comparison commands, and cleanup. Fine-tuning outputs
use `runs/vit-finetune/` and are not part of the existing inference result
tables.

Local CPU remains the stable default path for this repository. TPU execution is
optional and requires a Google Cloud project, billing or another funding path,
Cloud TPU API access, suitable TPU quota, and strict cleanup discipline.

The course project used TRC spot quota for its first successful run. TRC is not
required by the code itself: a reader may use normal on-demand TPU quota, spot
quota, TRC quota, institutional quota, or another valid Google Cloud TPU funding
setup.

## Where Commands Run

- **Local Ubuntu/WSL repo root**: run local baseline commands and local
  comparison commands.
- **Google Cloud Shell or local terminal with `gcloud`**: configure Google
  Cloud, create resources, SSH, retrieve artifacts, and clean up.
- **TPU VM shell**: clone the repository, install dependencies, verify JAX TPU
  devices, and run Demo 2.

## Local CPU Baseline Preparation

Run from the **local Ubuntu/WSL repo root**.

Install dependencies if needed:

```bash
uv sync --frozen --group pretrained
```

If the local CPU `b4` public-example JSON does not already exist, create it:

```bash
uv run --group pretrained python examples/pretrained_vit_inference.py \
  --jax-platform cpu \
  --image-manifest examples/assets/manifest.txt \
  --batch-size 4 \
  --warmup-steps 1 \
  --benchmark-steps 5 \
  --output runs/vit-inference/demo2_local_public_examples_cpu_b4.json
```

Raw JSON artifacts stay under ignored `runs/vit-inference/`. Do not commit raw
JSON unless a later artifact policy explicitly changes.

## Environment Variables

Use this section as a variable hierarchy. The common cloud resource variables
apply to every TPU path. Scenario-specific blocks then set the zone, TPU shape,
runtime, and resource names for a particular run. GCS variables are needed only
for the optional fine-tuning checkpoint/resume workflow. Repository checkout and
run-directory variables are used only after SSH inside the TPU VM shell.

### Common Cloud Resource Variables

These variables are used from **Google Cloud Shell or a local terminal with
`gcloud`**:

```bash
export PROJECT_ID="<PROJECT_ID>"
export REGION="<REGION>"
export ZONE="<ZONE>"
export TPU_NAME="<TPU_NAME>"
export QUEUED_RESOURCE_ID="<QUEUED_RESOURCE_ID>"
export ACCELERATOR_TYPE="<ACCELERATOR_TYPE>"
export RUNTIME_VERSION="<RUNTIME_VERSION>"
export NETWORK_NAME="<NETWORK_NAME>"
export SUBNET_NAME="<SUBNET_NAME>"
```

The selected subnet must exist in the region corresponding to the selected TPU
zone. Network and subnet names are not secrets, but project-specific network
topology details should still be documented conservatively.

### Scenario-Specific TPU Blocks

**A. Public-example inference smoke run**

```bash
export PROJECT_ID="<PROJECT_ID>"
export REGION="us-east1"
export ZONE="us-east1-d"
export TPU_NAME="demo2-vit-v6e1-use1-spot"
export QUEUED_RESOURCE_ID="demo2-vit-v6e1-use1-spot-qr"
export ACCELERATOR_TYPE="v6e-1"
export RUNTIME_VERSION="v2-alpha-tpuv6e"
export NETWORK_NAME="default"
export SUBNET_NAME="default"
```

This reproduces the successful public-example inference smoke-run resource
shape, subject to matching quota and funding availability.

**B. Fine-tuning first-run shape**

```bash
export PROJECT_ID="<PROJECT_ID>"
export REGION="us-east1"
export ZONE="us-east1-d"
export TPU_NAME="<TPU_NAME>"
export QUEUED_RESOURCE_ID="<QUEUED_RESOURCE_ID>"
export ACCELERATOR_TYPE="v6e-1"
export RUNTIME_VERSION="v2-alpha-tpuv6e"
export NETWORK_NAME="default"
export SUBNET_NAME="default"
# Use --spot when creating the queued resource.
```

This shape produced a successful first classifier-head fine-tuning smoke run,
but `us-east1-d` `v6e-1` spot later repeatedly showed maintenance or
preemption risk. Do not treat the zone as guaranteed stable.

**C. Fine-tuning GCS resume shape**

```bash
export PROJECT_ID="<PROJECT_ID>"
export REGION="europe-west4"
export ZONE="europe-west4-a"
export TPU_NAME="demo2-vit-ft-v6e1-ew4a-spot"
export QUEUED_RESOURCE_ID="demo2-vit-ft-v6e1-ew4a-spot-qr"
export ACCELERATOR_TYPE="v6e-1"
export RUNTIME_VERSION="v2-alpha-tpuv6e"
export NETWORK_NAME="default"
export SUBNET_NAME="default"
# Use --spot when creating the queued resource.
```

This was the successful GCS checkpoint restore/resume region in the observed
workflow. Do not overgeneralize it as guaranteed stable.

**D. On-demand fallback candidate**

```bash
export PROJECT_ID="<PROJECT_ID>"
export REGION="us-central2"
export ZONE="us-central2-b"
export TPU_NAME="demo2-vit-ft-v4-32-usc2-ondemand"
export QUEUED_RESOURCE_ID="demo2-vit-ft-v4-32-usc2-ondemand-qr"
export ACCELERATOR_TYPE="v4-32"
export RUNTIME_VERSION="<check with gcloud compute tpus versions list>"
export NETWORK_NAME="default"
export SUBNET_NAME="default"
```

Do not claim the on-demand path was run unless there is artifact evidence.
On-demand should not use `--spot`; it may cost more, but is less exposed to
spot preemption.

### Fine-Tuning GCS Variables

These are used only for Path C. The prefix is arbitrary, but use one value
consistently across first-run checkpoint copy, restore, artifact copy, and
cleanup.

```bash
export BUCKET_NAME="<BUCKET_NAME>"
export GCS_RUN_ROOT="gs://$BUCKET_NAME/numerical-jax-project/demo2-vit-finetune"
```

Use placeholders in reusable docs. Do not publish real personal bucket names or
project identifiers.

### TPU VM Checkout Variables

Repository checkout variables such as `REPO_URL`, `BRANCH`, and optional
`COMMIT_SHA` are used later inside the **TPU VM shell**. Shell variables exported
before SSH do not automatically exist inside the TPU VM shell.

```bash
export REPO_URL="<REPO_URL>"
export BRANCH="<BRANCH>"
# export COMMIT_SHA="<real commit SHA>"
```

### TPU VM Run Variables

`RUN_NAME`, `RUN_DIR`, and `CKPT_DIR` are scoped to one command run inside the
TPU VM shell. Fine-tuning uses absolute paths because Orbax checkpoint paths
must be absolute.

```bash
export RUN_NAME="<RUN_NAME>"
export RUN_DIR="$(pwd)/runs/vit-finetune/$RUN_NAME"
export CKPT_DIR="$RUN_DIR/checkpoints"
```

## Cloud Preflight

Run from **Google Cloud Shell or a local terminal with `gcloud`** after setting
the environment variables above.

Discover or confirm the active Google Cloud project without committing real
project IDs or project numbers:

```bash
gcloud projects list --format="table(projectId,name,projectNumber)"
gcloud config get-value project
gcloud projects describe <PROJECT_ID> --format="table(projectId,name,projectNumber)"
```

```bash
gcloud auth login
gcloud config set project "$PROJECT_ID"
gcloud config set compute/zone "$ZONE"
gcloud services enable tpu.googleapis.com

gcloud compute networks list \
  --project "$PROJECT_ID"

gcloud compute networks subnets list \
  --project "$PROJECT_ID" \
  --regions "$REGION"

gcloud compute networks subnets describe "$SUBNET_NAME" \
  --project "$PROJECT_ID" \
  --region "$REGION"

gcloud compute tpus accelerator-types list \
  --project "$PROJECT_ID" \
  --zone "$ZONE"

gcloud compute tpus versions list \
  --project "$PROJECT_ID" \
  --zone "$ZONE"

gcloud compute tpus queued-resources list \
  --project "$PROJECT_ID" \
  --zone "$ZONE"

gcloud compute tpus tpu-vm list \
  --project "$PROJECT_ID" \
  --zone "$ZONE"
```

## Resource And Funding Decision

| Situation | Suggested path | Notes |
| --- | --- | --- |
| Normal on-demand TPU quota | Direct TPU VM or queued resource | May incur regular cost |
| Spot quota | Queued resource with `--spot` | Cheaper/available differently, can be interrupted |
| TRC spot quota | Queued resource with `--spot` | Course project used this path |
| No TPU quota or no budget | Local CPU only | Still supports Demo 2 workflow |
| Long `WAITING_FOR_RESOURCES` | Delete queued resource and try another zone/type | Do not leave unused resources queued |

## Queued-Resource Creation Path

The queued-resource path is the canonical quickstart path because it matches the
successful course run and works with spot or TRC spot quota.
For direct TPU VM creation, see
[demo2_pretrained_vit_tpu_workflow.md](demo2_pretrained_vit_tpu_workflow.md).

Generic queued resource:

```bash
gcloud compute tpus queued-resources create "$QUEUED_RESOURCE_ID" \
  --node-id "$TPU_NAME" \
  --project "$PROJECT_ID" \
  --zone "$ZONE" \
  --accelerator-type "$ACCELERATOR_TYPE" \
  --runtime-version "$RUNTIME_VERSION" \
  --network "$NETWORK_NAME" \
  --subnetwork "$SUBNET_NAME"
```

Add `--spot` when using spot or TRC spot quota. The course project's successful
TRC spot run used `--spot`:

```bash
gcloud compute tpus queued-resources create "$QUEUED_RESOURCE_ID" \
  --node-id "$TPU_NAME" \
  --project "$PROJECT_ID" \
  --zone "$ZONE" \
  --accelerator-type "$ACCELERATOR_TYPE" \
  --runtime-version "$RUNTIME_VERSION" \
  --network "$NETWORK_NAME" \
  --subnetwork "$SUBNET_NAME" \
  --spot
```

Optional queue expiration guard:

If your installed `gcloud` version and selected queued-resource API support it,
you can add `--valid-until-duration` to limit how long the queued resource
should remain valid. Check support first:

```bash
gcloud compute tpus queued-resources create --help
```

Example:

```bash
gcloud compute tpus queued-resources create "$QUEUED_RESOURCE_ID" \
  --node-id "$TPU_NAME" \
  --project "$PROJECT_ID" \
  --zone "$ZONE" \
  --accelerator-type "$ACCELERATOR_TYPE" \
  --runtime-version "$RUNTIME_VERSION" \
  --network "$NETWORK_NAME" \
  --subnetwork "$SUBNET_NAME" \
  --spot \
  --valid-until-duration=45m
```

This is a queue-expiration guard, not a substitute for cleanup. If a wait is
abandoned, still verify and delete queued resources explicitly.

## Wait For Resource

Run from **Google Cloud Shell or a local terminal with `gcloud`**.

```bash
while true; do
  date
  gcloud compute tpus queued-resources describe "$QUEUED_RESOURCE_ID" \
    --project "$PROJECT_ID" \
    --zone "$ZONE" \
    --format="value(state)"
  sleep 20
done
```

Interpret the state conservatively:

- `WAITING_FOR_RESOURCES`: accepted but waiting for capacity.
- `PROVISIONING`: allocation has started.
- `ACTIVE`: TPU VM should exist.
- If the run window is blocked, delete the queued resource before trying another
  zone or accelerator type.
- A queued resource can become `ACTIVE` but still be unusable if the TPU VM
  immediately shows maintenance or `PREEMPTED` behavior. Inspect the queued
  resource and TPU VM list, delete the unusable queued resource, then try
  another zone/type or an on-demand fallback if the run window requires more
  stability.

Observed pattern, not a guarantee:

- `us-east1-d` `v6e-1` spot completed a first fine-tuning run, but later
  repeatedly showed maintenance/preemption risk.
- `europe-west4-a` `v6e-1` spot completed the GCS checkpoint restore/resume
  run.

## SSH

When the queued resource is `ACTIVE`, run from **Google Cloud Shell or a local
terminal with `gcloud`**:

```bash
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --project "$PROJECT_ID" \
  --zone "$ZONE"
```

## Shared TPU VM Checkout And Setup

Run inside the **TPU VM shell**.

Set repository checkout variables inside the TPU VM shell if they were not
already set:

```bash
export REPO_URL="<REPO_URL>"
export BRANCH="<BRANCH>"
# export COMMIT_SHA="<real commit SHA>"

# Example for this repository and current evidence branch:
# export REPO_URL="https://github.com/anthroplankton/numerical-jax-project.git"
# export BRANCH="feat/demo2-tpu-evidence"
```

`COMMIT_SHA` is optional. Choose one checkout mode:

- Latest-branch smoke-test mode: leave `COMMIT_SHA` unset or as a placeholder,
  do not run `git checkout "$COMMIT_SHA"`, and record `git rev-parse HEAD`.
- Reproducible benchmark mode: replace `COMMIT_SHA` with a real commit SHA and
  run `git checkout "$COMMIT_SHA"`.

New Demo 2 JSON records `git_commit` from the observed checkout when Git
metadata is available. `COMMIT_SHA` remains an optional checkout pin; it is not
used as the source of truth for result provenance.

```bash
git clone "$REPO_URL" numerical-jax-project
cd numerical-jax-project
git fetch --all --prune
git switch "$BRANCH"

# Reproducible mode only, after replacing <COMMIT_SHA> with a real commit:
# git checkout "$COMMIT_SHA"

git status --short --branch
git rev-parse HEAD

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version

# Path A/B inference setup:
uv sync --frozen --group pretrained

# Path C fine-tuning setup:
# uv sync --frozen --group pretrained --group training

uv pip install -U "jax[tpu]" \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

uv run python -c "import jax; print('jax_version=', jax.__version__); print('default_backend=', jax.default_backend()); print('device_count=', jax.device_count()); print('local_device_count=', jax.local_device_count()); print('devices=', jax.devices()); raise SystemExit(0 if jax.default_backend() == 'tpu' else 1)"
```

The backend check must report TPU backend/devices before interpreting the Demo 2
run as TPU evidence.

## Path A: Public-Example TPU Inference Smoke Run

Run inside the **TPU VM shell** after the shared checkout and TPU setup:

```bash
uv run --group pretrained python examples/pretrained_vit_inference.py \
  --jax-platform tpu \
  --image-manifest examples/assets/manifest.txt \
  --batch-size 4 \
  --warmup-steps 1 \
  --benchmark-steps 5 \
  --output runs/vit-inference/demo2_cloud_public_examples_tpu_b4.json

ls -lh runs/vit-inference/demo2_cloud_public_examples_tpu_b4.json
head -40 runs/vit-inference/demo2_cloud_public_examples_tpu_b4.json
```

## Inference Artifact Retrieval

Run from **Google Cloud Shell or a local `gcloud` terminal**.

For the simplest workflow, run the `scp` command from the same local repository
checkout where `scripts/compare_vit_results.py` will be run. If using Google
Cloud Shell for `scp`, either run the comparison in a Cloud Shell checkout or
copy/download the retrieved JSON back to the local repository before local
comparison.

To retrieve the entire Demo 2 result folder from the TPU VM, run this from the
**local repository root in Google Cloud Shell or a local terminal with
`gcloud`**:

```bash
mkdir -p runs

gcloud compute tpus tpu-vm scp --recurse \
  "$TPU_NAME":~/numerical-jax-project/runs/vit-inference \
  runs/ \
  --project "$PROJECT_ID" \
  --zone "$ZONE"
```

This copies the remote inference folder to local `runs/vit-inference/`, which
is ignored by Git. Keep raw TPU JSON and generated comparison JSON there;
commit only intentionally curated Markdown tables under `report/results/`.

To retrieve only the public-example smoke-run JSON, use the single-file form:

```bash
mkdir -p runs/vit-inference

gcloud compute tpus tpu-vm scp \
  "$TPU_NAME":~/numerical-jax-project/runs/vit-inference/demo2_cloud_public_examples_tpu_b4.json \
  runs/vit-inference/demo2_cloud_public_examples_tpu_b4.json \
  --project "$PROJECT_ID" \
  --zone "$ZONE"
```

## Local Comparison

Run from the **local Ubuntu/WSL repo root** after the TPU JSON has been copied
back.

```bash
uv run python scripts/compare_vit_results.py \
  runs/vit-inference/demo2_local_public_examples_cpu_b4.json \
  runs/vit-inference/demo2_cloud_public_examples_tpu_b4.json \
  --output runs/vit-inference/demo2_local_cpu_vs_cloud_tpu_public_examples_b4_compare.json \
  --markdown-output report/results/demo2_local_cpu_vs_cloud_tpu_public_examples_b4.md
```

Keep raw JSON and comparison JSON under ignored `runs/vit-inference/`. Curated
Markdown comparison tables belong under `report/results/`.

## Optional Run-When-Active Helper

The manual quickstart above remains the primary workflow because it keeps cloud
resource creation, artifact retrieval, comparison, and cleanup visible. For an
existing queued resource, this optional run-when-active helper waits for the
resource to become `ACTIVE`, runs the same public-example TPU `b4` smoke command
on the TPU VM, retrieves the JSON artifact, runs the existing local comparison
command, and prints cleanup instructions:

This helper only runs and retrieves the public-example `b4` smoke artifact. It
does not run, retrieve, or compare Imagenette 320 TPU artifacts.

```bash
export PROJECT_ID="<PROJECT_ID>"
export ZONE="<ZONE>"
export TPU_NAME="<TPU_NAME>"
export QUEUED_RESOURCE_ID="<QUEUED_RESOURCE_ID>"
export REPO_URL="<REPO_URL>"
export BRANCH="<BRANCH>"

bash scripts/demo2_tpu_run_when_active.sh
```

The helper does not create queued resources. It writes a small sanitized command
log under `runs/vit-inference/` and does not record expanded project IDs, repo
URLs, IPs, or private local paths. By default it does not delete resources; it
prints cleanup commands for review. To explicitly delete the queued resource
after artifact retrieval and comparison, opt in:

```bash
bash scripts/demo2_tpu_run_when_active.sh --delete-after
```

## Path B: Imagenette 320 TPU Inference Artifacts

The repository now has retrieved TPU JSON artifacts for Imagenette 320
inference runs on `val64`, `val256`, and the full validation manifest, each with
batch sizes `b1`, `b4`, and `b8`. The raw JSON artifacts stay under ignored
`runs/vit-inference/`; the curated Markdown tables are:

```text
report/results/demo2_cloud_imagenette320_val64_tpu.md
report/results/demo2_cloud_imagenette320_val256_tpu.md
report/results/demo2_cloud_imagenette320_valfull_tpu.md
```

These tables are TPU inference timing evidence. They are not training evidence,
not dataset-level accuracy evaluation, not a full controlled benchmark study,
and not a universal TPU speedup claim.

### Regenerate Curated Markdown Tables

After retrieving Imagenette 320 TPU JSON artifacts, generate the curated TPU
Markdown tables from the **local Ubuntu/WSL repo root**:

```bash
uv run python scripts/compare_vit_results.py \
  runs/vit-inference/demo2_cloud_imagenette320_val64_tpu_b1.json \
  runs/vit-inference/demo2_cloud_imagenette320_val64_tpu_b4.json \
  runs/vit-inference/demo2_cloud_imagenette320_val64_tpu_b8.json \
  --markdown-output report/results/demo2_cloud_imagenette320_val64_tpu.md

uv run python scripts/compare_vit_results.py \
  runs/vit-inference/demo2_cloud_imagenette320_val256_tpu_b1.json \
  runs/vit-inference/demo2_cloud_imagenette320_val256_tpu_b4.json \
  runs/vit-inference/demo2_cloud_imagenette320_val256_tpu_b8.json \
  --markdown-output report/results/demo2_cloud_imagenette320_val256_tpu.md

uv run python scripts/compare_vit_results.py \
  runs/vit-inference/demo2_cloud_imagenette320_valfull_tpu_b1.json \
  runs/vit-inference/demo2_cloud_imagenette320_valfull_tpu_b4.json \
  runs/vit-inference/demo2_cloud_imagenette320_valfull_tpu_b8.json \
  --markdown-output report/results/demo2_cloud_imagenette320_valfull_tpu.md
```

Prepare Imagenette 320 before running the TPU commands. Use the official
Imagenette source, then extract files so this path exists:

```text
data/local/imagenette2-320/val
```

Concrete local download and extraction commands using the official fastai
Imagenette 320 archive are documented in
[../docs/demo2_pretrained_vit.md](../docs/demo2_pretrained_vit.md). For cloud
Imagenette benchmark runs, the TPU VM must have the same
`data/local/imagenette2-320/val` path before benchmark commands run.

Build the validation manifests used by the retrieved artifact families:

```bash
uv run python scripts/build_image_manifest.py \
  data/local/imagenette2-320/val \
  --output data/local/imagenette2-320/val/manifest_val_64.txt \
  --limit 64

uv run python scripts/build_image_manifest.py \
  data/local/imagenette2-320/val \
  --output data/local/imagenette2-320/val/manifest_val_256.txt \
  --limit 256

uv run python scripts/build_image_manifest.py \
  data/local/imagenette2-320/val \
  --output data/local/imagenette2-320/val/manifest_val_full.txt
```

Do not commit `data/local/`, generated manifests, dataset files, model caches,
or raw JSON benchmark outputs.

On the **TPU VM shell**, either download and extract Imagenette 320 on the TPU
VM or copy a prepared local `data/local/imagenette2-320/` directory to the TPU
VM. Preserve the same manifest path expected by the benchmark commands:

```text
data/local/imagenette2-320/val/manifest_val_64.txt
data/local/imagenette2-320/val/manifest_val_256.txt
data/local/imagenette2-320/val/manifest_val_full.txt
```

Cloud TPU Imagenette val64 command pattern:

```bash
uv run --group pretrained python examples/pretrained_vit_inference.py \
  --jax-platform tpu \
  --image-manifest data/local/imagenette2-320/val/manifest_val_64.txt \
  --batch-size 1 \
  --warmup-steps 1 \
  --benchmark-steps 5 \
  --output runs/vit-inference/demo2_cloud_imagenette320_val64_tpu_b1.json

uv run --group pretrained python examples/pretrained_vit_inference.py \
  --jax-platform tpu \
  --image-manifest data/local/imagenette2-320/val/manifest_val_64.txt \
  --batch-size 4 \
  --warmup-steps 1 \
  --benchmark-steps 5 \
  --output runs/vit-inference/demo2_cloud_imagenette320_val64_tpu_b4.json

uv run --group pretrained python examples/pretrained_vit_inference.py \
  --jax-platform tpu \
  --image-manifest data/local/imagenette2-320/val/manifest_val_64.txt \
  --batch-size 8 \
  --warmup-steps 1 \
  --benchmark-steps 5 \
  --output runs/vit-inference/demo2_cloud_imagenette320_val64_tpu_b8.json
```

Use the same pattern with `manifest_val_256.txt` and `manifest_val_full.txt` for
the retrieved `val256` and `valfull` TPU artifact families.

If a local CPU-vs-cloud TPU Imagenette comparison is added later, generate it
only after choosing a clear comparison baseline and preserving the limitations.
The current generated cross-device Imagenette summary is produced from the
result directory:

```bash
uv run python scripts/generate_vit_summary_tables.py --input-dir runs/vit-inference --output-dir report/results
```

The existing generated summary file for this cross-device view is
`report/results/demo2_imagenette320_cpu_vs_tpu.md`. Do not treat that table as a
universal speedup claim; it is still inference-only timing evidence from
specific artifacts.

## Path C: Optional Classifier-Head Fine-Tuning With GCS Checkpoint/Resume

This optional path is still Demo 2. It freezes the pretrained ViT backbone,
trains only the classifier head, and uses Orbax checkpoints to demonstrate
checkpoint/resume behavior under spot or preemptible TPU risk. It is not full
ViT fine-tuning and not an accuracy benchmark.

### GCS Durable Checkpoint Setup

Run this section from **Google Cloud Shell or a local terminal with `gcloud`**.
GCS is used as a durable copy of checkpoints and artifacts. Orbax still writes
local checkpoint files first under the TPU VM run directory. Use a temporary
class-demo bucket rather than a shared bucket for commands that later show
destructive cleanup examples.

```bash
export PROJECT_ID="<PROJECT_ID>"
export REGION="<REGION>"
export BUCKET_NAME="<BUCKET_NAME>"
export GCS_RUN_ROOT="gs://$BUCKET_NAME/numerical-jax-project/demo2-vit-finetune"

gcloud config set project "$PROJECT_ID"
gcloud storage buckets create "gs://$BUCKET_NAME" \
  --project "$PROJECT_ID" \
  --location "$REGION" \
  --uniform-bucket-level-access \
  --public-access-prevention

gcloud storage buckets update "gs://$BUCKET_NAME" --clear-soft-delete
gcloud storage buckets describe "gs://$BUCKET_NAME"
```

For short-lived class demo buckets, clear soft delete immediately after bucket
creation. For a class reset, use a new bucket name and clear soft delete at
creation time. The `GCS_RUN_ROOT` prefix is arbitrary, but it must stay
consistent across first-run checkpoint copy, restore, artifact copy, and
cleanup.

Local GCS write/read preflight:

```bash
export LOCAL_PREFLIGHT_DIR="/tmp/demo2-vit-ft-gcs-preflight"
export LOCAL_PREFLIGHT_READBACK="/tmp/demo2-vit-ft-gcs-readback"

mkdir -p "$LOCAL_PREFLIGHT_DIR"
printf "local preflight %s\n" "$(date -Is)" > "$LOCAL_PREFLIGHT_DIR/local.txt"
gcloud storage rsync --recursive "$LOCAL_PREFLIGHT_DIR" "$GCS_RUN_ROOT/preflight/local"

rm -rf "$LOCAL_PREFLIGHT_READBACK"
mkdir -p "$LOCAL_PREFLIGHT_READBACK"
gcloud storage rsync --recursive "$GCS_RUN_ROOT/preflight/local" "$LOCAL_PREFLIGHT_READBACK"
cat "$LOCAL_PREFLIGHT_READBACK/local.txt"
```

After SSH to the TPU VM, run a TPU VM GCS write preflight:

```bash
export GCS_RUN_ROOT="gs://<BUCKET_NAME>/numerical-jax-project/demo2-vit-finetune"
export TPU_PREFLIGHT_DIR="/tmp/demo2-vit-ft-tpu-preflight"

mkdir -p "$TPU_PREFLIGHT_DIR"
printf "tpu vm preflight %s\n" "$(date -Is)" > "$TPU_PREFLIGHT_DIR/tpu-vm.txt"
gcloud storage rsync --recursive "$TPU_PREFLIGHT_DIR" "$GCS_RUN_ROOT/preflight/tpu-vm"
gcloud storage ls "$GCS_RUN_ROOT/preflight/tpu-vm/"
```

### First Fine-Tuning Smoke Run

Run inside the **TPU VM shell** after the shared checkout and TPU setup. If the
shared setup was done for inference only, sync the training dependency group
before running fine-tuning:

```bash
uv sync --frozen --group pretrained --group training

uv run python -c "import jax; print('jax_version=', jax.__version__); print('default_backend=', jax.default_backend()); print('device_count=', jax.device_count()); print('local_device_count=', jax.local_device_count()); print('devices=', jax.devices()); raise SystemExit(0 if jax.default_backend() == 'tpu' else 1)"
```

Download and extract Imagenette 320 on the TPU VM:

```bash
mkdir -p data/local
curl -L --fail --show-error \
  -o data/local/imagenette2-320.tgz \
  https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar -xzf data/local/imagenette2-320.tgz -C data/local
test -d data/local/imagenette2-320/train
test -d data/local/imagenette2-320/val
```

Build small train/eval manifests from the existing Imagenette files:

```bash
uv run python scripts/build_image_manifest.py \
  data/local/imagenette2-320/train \
  --output data/local/imagenette2-320/train/manifest_train_64.txt \
  --limit 64

uv run python scripts/build_image_manifest.py \
  data/local/imagenette2-320/val \
  --output data/local/imagenette2-320/val/manifest_val_64.txt \
  --limit 64
```

Set absolute run paths. Do not use relative checkpoint paths for TPU fine
tuning; Orbax can fail with `ValueError: Checkpoint path should be absolute`.

```bash
export RUN_NAME="demo2_cloud_vit_head_finetune_tpu_first"
export RUN_DIR="$(pwd)/runs/vit-finetune/$RUN_NAME"
export CKPT_DIR="$RUN_DIR/checkpoints"
mkdir -p "$RUN_DIR" "$CKPT_DIR"
```

Run the time-controlled TPU smoke command:

```bash
uv run --group pretrained --group training python examples/demo2_pretrained_vit_finetune.py \
  --jax-platform tpu \
  --train-manifest data/local/imagenette2-320/train/manifest_train_64.txt \
  --eval-manifest data/local/imagenette2-320/val/manifest_val_64.txt \
  --batch-size 8 \
  --learning-rate 0.001 \
  --max-steps 100000 \
  --min-train-seconds 120 \
  --checkpoint-every-steps 0 \
  --checkpoint-every-seconds 30 \
  --checkpoint-dir "$CKPT_DIR" \
  --output-dir "$RUN_DIR" \
  --save-predictions
```

`--max-steps 100000` is only an upper bound so the run can be controlled by
`--min-train-seconds`; it is not a large benchmark target. Prefer
`--checkpoint-every-steps 0` and `--checkpoint-every-seconds 30` on TPU.
Step-based checkpointing such as `--checkpoint-every-steps 20` created too many
checkpoints on TPU and made live GCS sync fragile.

After training completes, copy the final run directory and checkpoint directory
to GCS:

```bash
export GCS_RUN_ROOT="gs://<BUCKET_NAME>/numerical-jax-project/demo2-vit-finetune"
export GCS_RUN_URI="$GCS_RUN_ROOT/artifacts/$RUN_NAME"
export GCS_CKPT_URI="$GCS_RUN_ROOT/checkpoints/$RUN_NAME"

gcloud storage rsync --recursive "$RUN_DIR" "$GCS_RUN_URI"
gcloud storage rsync --recursive "$CKPT_DIR" "$GCS_CKPT_URI"
gcloud storage ls "$GCS_CKPT_URI/"
```

The observed first run produced durable GCS checkpoint copies at steps `15100`,
`15120`, and `15140`.

### Restore And Resume From GCS Checkpoint

Run this section on the replacement or resumed **TPU VM shell** after repository
checkout, dependency setup, JAX TPU verification, Imagenette preparation, and
manifest generation are complete.

```bash
export GCS_RUN_ROOT="gs://<BUCKET_NAME>/numerical-jax-project/demo2-vit-finetune"
export BASE_RUN_NAME="demo2_cloud_vit_head_finetune_tpu_first"
export RESUME_RUN_NAME="demo2_cloud_vit_head_finetune_tpu_resume"
export RESUME_RUN_DIR="$(pwd)/runs/vit-finetune/$RESUME_RUN_NAME"
export RESUME_CKPT_DIR="$RESUME_RUN_DIR/checkpoints"
export GCS_CKPT_URI="$GCS_RUN_ROOT/checkpoints/$BASE_RUN_NAME"

mkdir -p "$RESUME_RUN_DIR" "$RESUME_CKPT_DIR"
gcloud storage rsync --recursive "$GCS_CKPT_URI" "$RESUME_CKPT_DIR"
find "$RESUME_CKPT_DIR" -name "*.orbax-checkpoint-tmp" -type d -prune -exec rm -rf {} +
```

Resume from the latest restored Orbax checkpoint:

```bash
uv run --group pretrained --group training python examples/demo2_pretrained_vit_finetune.py \
  --jax-platform tpu \
  --train-manifest data/local/imagenette2-320/train/manifest_train_64.txt \
  --eval-manifest data/local/imagenette2-320/val/manifest_val_64.txt \
  --batch-size 8 \
  --learning-rate 0.001 \
  --max-steps 100000 \
  --min-train-seconds 120 \
  --checkpoint-every-steps 0 \
  --checkpoint-every-seconds 30 \
  --checkpoint-dir "$RESUME_CKPT_DIR" \
  --output-dir "$RESUME_RUN_DIR" \
  --resume \
  --save-predictions
```

Verify the resume summary:

```bash
uv run python - <<'PY'
import json
import os
from pathlib import Path

summary = json.loads((Path(os.environ["RESUME_RUN_DIR"]) / "summary.json").read_text())
checks = {
    "backend": summary["backend"] == "tpu",
    "resumed_from_checkpoint": summary["resumed_from_checkpoint"] is True,
    "start_step": summary["start_step"] > 0,
    "final_step": summary["final_step"] > summary["start_step"],
    "trainable_scope": summary["trainable_scope"] == "classifier_head_only",
    "frozen_scope": summary["frozen_scope"] == "vit_backbone",
}
print(json.dumps({key: summary.get(key) for key in [
    "backend",
    "resumed_from_checkpoint",
    "start_step",
    "final_step",
    "trainable_scope",
    "frozen_scope",
]}, indent=2))
failed = [name for name, ok in checks.items() if not ok]
raise SystemExit(f"failed checks: {failed}" if failed else 0)
PY
```

In the observed workflow, the resume restored checkpoint step `15140` and
completed successfully with `backend=tpu`, `resumed_from_checkpoint=true`,
`start_step=15140`, `final_step=51538`, `trainable_scope=classifier_head_only`,
and `frozen_scope=vit_backbone`.

### Monitoring While Training Runs

From **Google Cloud Shell or a local terminal with `gcloud`**:

```bash
gcloud compute tpus queued-resources describe "$QUEUED_RESOURCE_ID" \
  --project "$PROJECT_ID" \
  --zone "$ZONE"

gcloud compute tpus tpu-vm list \
  --project "$PROJECT_ID" \
  --zone "$ZONE"
```

From the **TPU VM shell**:

```bash
tail -f "$RUN_DIR/train.log"

while true; do
  date
  tail -n 5 "$RUN_DIR/metrics.csv"
  sleep 10
done

ls -lh "$CKPT_DIR"
find "$CKPT_DIR" -maxdepth 2 -type f | head
```

For a resume run, replace `RUN_DIR` and `CKPT_DIR` with `RESUME_RUN_DIR` and
`RESUME_CKPT_DIR`.

From **GCS**:

```bash
gcloud storage ls "$GCS_CKPT_URI/"
gcloud storage du --summarize "$GCS_CKPT_URI"
gcloud storage ls "$GCS_RUN_ROOT/artifacts/$RUN_NAME/"
```

Avoid live-rsyncing the entire Orbax checkpoint root while Orbax is actively
writing checkpoints. In the observed run, `rsync` of the whole checkpoint root
could race with Orbax checkpoint creation or cleanup and report
`FileNotFoundError` for `_CHECKPOINT_METADATA`. Prefer final `rsync` after
training completes. If live sync is needed, sync only a completed/stable
checkpoint step directory and document it as experimental.

### Metrics And Report Review After Training

Inspect these generated artifacts under the run directory:

```text
summary.json
metrics.csv
train.log
predictions_before.json
predictions_after.json
checkpoints/
```

Useful inspection commands:

```bash
ls -lh "$RUN_DIR"
uv run python -m json.tool "$RUN_DIR/summary.json" | head -80
head -5 "$RUN_DIR/metrics.csv"
tail -5 "$RUN_DIR/metrics.csv"
tail -40 "$RUN_DIR/train.log"
```

Expected `summary.json` fields include:

- `backend`
- `devices`
- `selected_jax_platform`
- `model_name`
- `trainable_scope`
- `frozen_scope`
- `start_step`
- `final_step`
- `resumed_from_checkpoint`
- `latest_checkpoint_step`
- `initial_loss`
- `final_loss`
- `mean_step_time_sec`
- `examples_per_second`
- `total_runtime_sec`
- `git_commit`
- `git_branch`
- `git_dirty`

`mean_step_time_sec` and `examples_per_second` measure training-step time and
exclude checkpoint write time. `total_runtime_sec` includes setup, evaluation,
checkpointing, prediction writing, and summary writing overhead.

Near-zero loss can be expected in the tiny `train64`/`val64` smoke setup and
does not imply dataset-level accuracy. The manifest limit may select a small
and class-skewed subset. Treat this path as workflow evidence,
checkpoint/resume evidence, and TPU execution evidence, not an accuracy study.

### Fine-Tuning Artifact Retrieval

Retrieve fine-tuning artifacts after the run using one of these scoped paths.
If the TPU VM is still available, copy `runs/vit-finetune/` from the VM:

```bash
mkdir -p runs

gcloud compute tpus tpu-vm scp --recurse \
  "$TPU_NAME":~/numerical-jax-project/runs/vit-finetune \
  runs/ \
  --project "$PROJECT_ID" \
  --zone "$ZONE"
```

If the final artifact copy was written to GCS, retrieve from the GCS artifact
prefix instead:

```bash
export RUN_NAME="<RUN_NAME>"
export GCS_RUN_ROOT="gs://<BUCKET_NAME>/numerical-jax-project/demo2-vit-finetune"
export GCS_RUN_URI="$GCS_RUN_ROOT/artifacts/$RUN_NAME"
export GCS_CKPT_URI="$GCS_RUN_ROOT/checkpoints/$RUN_NAME"

mkdir -p "runs/vit-finetune/$RUN_NAME"
gcloud storage rsync --recursive "$GCS_RUN_URI" "runs/vit-finetune/$RUN_NAME"
gcloud storage rsync --recursive \
  "$GCS_CKPT_URI" \
  "runs/vit-finetune/$RUN_NAME/checkpoints"
```

Both retrieval paths copy raw summaries, metrics, logs, predictions, and
checkpoints under ignored `runs/vit-finetune/`. Commit only intentionally
curated, small report summaries under `report/results/`.

### Cleanup

Use this queued-resource cleanup for Path A, Path B, or Path C after artifact
retrieval, or after abandoning a blocked allocation:

```bash
gcloud compute tpus queued-resources delete "$QUEUED_RESOURCE_ID" \
  --project "$PROJECT_ID" \
  --zone "$ZONE" \
  --force

gcloud compute tpus queued-resources list \
  --project "$PROJECT_ID" \
  --zone "$ZONE"

gcloud compute tpus tpu-vm list \
  --project "$PROJECT_ID" \
  --zone "$ZONE"
```

Optionally delete GCS checkpoint and artifact objects after retrieving evidence.
Do this only for a temporary class-demo bucket that is not shared with other
work. Before deleting a demo bucket, save a small local manual note describing
what was retrieved and why the bucket can be removed.

```bash
gcloud storage rm --recursive "gs://$BUCKET_NAME/**"
gcloud storage buckets delete "gs://$BUCKET_NAME"
```

For future class demos, use a new bucket name and clear soft delete at creation
time:

```bash
gcloud storage buckets update "gs://$BUCKET_NAME" --clear-soft-delete
```

Real spot or preemptible interruption is non-deterministic and is not guaranteed
to deliver graceful SIGTERM. Controlled SIGTERM plus resume remains the primary
interruption evidence path when graceful behavior must be demonstrated. Durable
resume after TPU VM deletion requires copying checkpoints to GCS or another
durable location before the VM is deleted.

## Limitations

- Demo 2 TPU inference tables remain ViT inference timing evidence only.
- Optional fine-tuning TPU evidence is classifier-head-only smoke workflow and
  GCS checkpoint/resume evidence, not full ViT fine-tuning and not an accuracy
  benchmark.
- The public TPU smoke run uses five public images, batch size 4, final-batch
  padding with `num_padded_images = 3`, and a short benchmark loop.
- The Imagenette 320 TPU tables use validation manifests for inference timing
  only. They do not compute Imagenette labels, top-k accuracy, or dataset-level
  evaluation metrics.
- The tables are not a full controlled hardware benchmark study.
- Do not generalize any reported speedup beyond the specific artifacts and
  command settings used to produce it.
