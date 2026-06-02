# Demo 2 TPU Quickstart

## Purpose

This is the reusable Cloud TPU quickstart for Demo 2 pretrained ViT inference.
It runs the same small public-example TPU smoke test documented in the course
project, then retrieves the JSON artifact, deletes the TPU resource, and
generates a local CPU-vs-TPU comparison table.

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

Use this generic block first, replacing placeholders with values available to
your project and quota path:

```bash
export PROJECT_ID="<PROJECT_ID>"
export ZONE="<ZONE>"
export REGION="<REGION>"
export TPU_NAME="<TPU_NAME>"
export QUEUED_RESOURCE_ID="<QUEUED_RESOURCE_ID>"
export ACCELERATOR_TYPE="<ACCELERATOR_TYPE>"
export RUNTIME_VERSION="<RUNTIME_VERSION>"
export REPO_URL="https://github.com/anthroplankton/numerical-jax-project.git"
export BRANCH="feat/demo2-tpu-evidence"
export COMMIT_SHA="<COMMIT_SHA>"
```

Optional concrete block for reproducing the successful course smoke-run shape:

```bash
export PROJECT_ID="<PROJECT_ID>"
export ZONE="us-east1-d"
export REGION="us-east1"
export TPU_NAME="demo2-vit-v6e1-use1-spot"
export QUEUED_RESOURCE_ID="demo2-vit-v6e1-use1-spot-qr"
export ACCELERATOR_TYPE="v6e-1"
export RUNTIME_VERSION="v2-alpha-tpuv6e"
export REPO_URL="https://github.com/anthroplankton/numerical-jax-project.git"
export BRANCH="feat/demo2-tpu-evidence"
export COMMIT_SHA="<COMMIT_SHA>"
```

The second block reproduces the successful course smoke-run resource shape, but
it requires matching quota and funding availability. `<PROJECT_ID>` and
`<COMMIT_SHA>` remain placeholders; do not use a later local commit SHA as a
substitute for the TPU-run commit.

## Cloud Preflight

Run from **Google Cloud Shell or a local terminal with `gcloud`** after setting
the environment variables above.

```bash
gcloud auth login
gcloud config set project "$PROJECT_ID"
gcloud config set compute/zone "$ZONE"
gcloud services enable tpu.googleapis.com

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
  --runtime-version "$RUNTIME_VERSION"
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
  --spot
```

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

## SSH

When the queued resource is `ACTIVE`, run from **Google Cloud Shell or a local
terminal with `gcloud`**:

```bash
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --project "$PROJECT_ID" \
  --zone "$ZONE"
```

## TPU VM Setup And Run

Run inside the **TPU VM shell**.

Choose one checkout mode:

- Reproducible mode: set `COMMIT_SHA` to a real commit SHA, then run
  `git checkout "$COMMIT_SHA"`.
- Latest-branch smoke-test mode: leave `COMMIT_SHA` as a placeholder, omit the
  checkout command, and record `git rev-parse HEAD`.

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
uv sync --group pretrained

uv pip install -U "jax[tpu]" \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

uv run python -c "import jax; print('jax_version=', jax.__version__); print('default_backend=', jax.default_backend()); print('device_count=', jax.device_count()); print('local_device_count=', jax.local_device_count()); print('devices=', jax.devices())"

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

The backend check must report TPU backend/devices before interpreting the Demo 2
run as TPU evidence.

## Artifact Retrieval And Cleanup

Run from **Google Cloud Shell or a local `gcloud` terminal**.

For the simplest workflow, run the `scp` command from the same local repository
checkout where `scripts/compare_vit_results.py` will be run. If using Google
Cloud Shell for `scp`, either run the comparison in a Cloud Shell checkout or
copy/download the retrieved JSON back to the local repository before local
comparison.

```bash
mkdir -p runs/vit-inference

gcloud compute tpus tpu-vm scp \
  "$TPU_NAME":~/numerical-jax-project/runs/vit-inference/demo2_cloud_public_examples_tpu_b4.json \
  runs/vit-inference/demo2_cloud_public_examples_tpu_b4.json \
  --project "$PROJECT_ID" \
  --zone "$ZONE"

gcloud compute tpus queued-resources delete "$QUEUED_RESOURCE_ID" \
  --project "$PROJECT_ID" \
  --zone "$ZONE" \
  --force \
  --quiet

gcloud compute tpus queued-resources list \
  --project "$PROJECT_ID" \
  --zone "$ZONE"

gcloud compute tpus tpu-vm list \
  --project "$PROJECT_ID" \
  --zone "$ZONE"
```

Both list commands should show no remaining resource for the completed smoke-run
resource.

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

## Limitations

- This smoke run uses five public images.
- Batch size is 4.
- The final batch is padded with `num_padded_images = 3`.
- The benchmark loop is short.
- This is not dataset-level accuracy evaluation.
- This is not a full controlled hardware benchmark.
- Do not generalize the speedup beyond the specific five-image public smoke-run
  comparison.
