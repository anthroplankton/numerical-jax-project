# Demo 2 TPU Quickstart

## Purpose

This is the reusable Cloud TPU quickstart for Demo 2 pretrained ViT inference
and the optional classifier-head fine-tuning smoke extension.
It runs the same small public-example TPU smoke test documented in the course
project, retrieves the JSON artifact, runs local comparison commands, and keeps
cleanup visible. It also documents how to retrieve the full
`runs/vit-inference/` folder and regenerate curated Imagenette 320 TPU Markdown
tables after those JSON artifacts exist. Fine-tuning outputs use
`runs/vit-finetune/` and are not part of the existing inference result tables.

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

These cloud resource variables are used from **Google Cloud Shell or a local
terminal with `gcloud`**. Use this generic block first, replacing placeholders
with values available to your project and quota path:

```bash
export PROJECT_ID="<PROJECT_ID>"
export ZONE="<ZONE>"
export REGION="<REGION>"
export TPU_NAME="<TPU_NAME>"
export QUEUED_RESOURCE_ID="<QUEUED_RESOURCE_ID>"
export ACCELERATOR_TYPE="<ACCELERATOR_TYPE>"
export RUNTIME_VERSION="<RUNTIME_VERSION>"
export NETWORK_NAME="<NETWORK_NAME>"
export SUBNET_NAME="<SUBNET_NAME>"
```

Optional concrete cloud-resource block for reproducing the successful course
smoke-run shape:

```bash
export PROJECT_ID="<PROJECT_ID>"
export ZONE="us-east1-d"
export REGION="us-east1"
export TPU_NAME="demo2-vit-v6e1-use1-spot"
export QUEUED_RESOURCE_ID="demo2-vit-v6e1-use1-spot-qr"
export ACCELERATOR_TYPE="v6e-1"
export RUNTIME_VERSION="v2-alpha-tpuv6e"
export NETWORK_NAME="default"
export SUBNET_NAME="default"
```

The second block reproduces the successful course smoke-run resource shape, but
it requires matching quota and funding availability. `<PROJECT_ID>` remains a
placeholder.

The successful course smoke run used the default VPC network and default subnet
in the selected region. Other users may use another valid VPC/subnet. The subnet
must exist in the region corresponding to the selected TPU zone. Network and
subnet names are not secrets, but project-specific network topology details
should still be documented conservatively.

Repository checkout variables such as `REPO_URL`, `BRANCH`, and optional
`COMMIT_SHA` are used later inside the **TPU VM shell**. Shell variables exported
before SSH do not automatically exist inside the TPU VM shell.

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

Set repository checkout variables inside the TPU VM shell:

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

This copies the remote folder to local `runs/vit-inference/`, which is ignored
by Git. Keep raw TPU JSON and generated comparison JSON there; commit only
intentionally curated Markdown tables under `report/results/`.

To retrieve the full optional fine-tuning output directory, use the same
recursive pattern:

```bash
mkdir -p runs

gcloud compute tpus tpu-vm scp --recurse \
  "$TPU_NAME":~/numerical-jax-project/runs/vit-finetune \
  runs/ \
  --project "$PROJECT_ID" \
  --zone "$ZONE"
```

This copies raw fine-tuning summaries, metrics, logs, predictions, and
checkpoints under ignored `runs/vit-finetune/`. Commit only intentionally
curated, small report summaries after a real run.

To retrieve only the public-example smoke-run JSON, use the single-file form:

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

## Imagenette 320 TPU Inference Benchmark

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

## Optional Demo 2 ViT Head Fine-Tuning TPU Smoke Run

This optional path is still Demo 2. It freezes the pretrained ViT backbone,
trains only the classifier head, and uses Orbax checkpoints to demonstrate
checkpoint/resume behavior under spot or preemptible TPU risk. It is not full
ViT fine-tuning and not an accuracy benchmark.

Run inside the **TPU VM shell** after repository checkout, Imagenette
preparation, and TPU JAX installation. Prefer a frozen dependency sync for this
extension:

```bash
uv sync --frozen --group pretrained --group training
```

Build small train/eval manifests from existing Imagenette 320 files:

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
  --checkpoint-every-steps 20 \
  --checkpoint-every-seconds 30 \
  --checkpoint-dir runs/vit-finetune/demo2_cloud_imagenette320_train64_tpu/checkpoints \
  --output-dir runs/vit-finetune/demo2_cloud_imagenette320_train64_tpu \
  --save-predictions
```

`--max-steps 100000` is only an upper bound so the run can be controlled by
`--min-train-seconds`; it is not a large benchmark target.

For primary interruption evidence, use a controlled SIGTERM from another TPU VM
shell and then resume from the latest checkpoint:

```bash
pkill -TERM -f "examples/demo2_pretrained_vit_finetune.py"

uv run --group pretrained --group training python examples/demo2_pretrained_vit_finetune.py \
  --jax-platform tpu \
  --train-manifest data/local/imagenette2-320/train/manifest_train_64.txt \
  --eval-manifest data/local/imagenette2-320/val/manifest_val_64.txt \
  --batch-size 8 \
  --learning-rate 0.001 \
  --max-steps 100000 \
  --min-train-seconds 120 \
  --checkpoint-every-steps 20 \
  --checkpoint-every-seconds 30 \
  --checkpoint-dir runs/vit-finetune/demo2_cloud_imagenette320_train64_tpu/checkpoints \
  --output-dir runs/vit-finetune/demo2_cloud_imagenette320_train64_tpu_resume \
  --resume \
  --save-predictions
```

Real spot or preemptible interruption is non-deterministic and is not guaranteed
to deliver graceful SIGTERM. Treat it as optional evidence, separate from the
controlled SIGTERM smoke test. Durable resume after TPU VM deletion requires
copying checkpoints to Google Cloud Storage or another durable location before
the VM is deleted.

## Limitations

- Completed Demo 2 TPU evidence is ViT inference only. The optional fine-tuning
  extension should be reported only after a real run produces artifacts.
- The public TPU smoke run uses five public images, batch size 4, final-batch
  padding with `num_padded_images = 3`, and a short benchmark loop.
- The Imagenette 320 TPU tables use validation manifests for inference timing
  only. They do not compute Imagenette labels, top-k accuracy, or dataset-level
  evaluation metrics.
- The tables are not a full controlled hardware benchmark study.
- Do not generalize any reported speedup beyond the specific artifacts and
  command settings used to produce it.
