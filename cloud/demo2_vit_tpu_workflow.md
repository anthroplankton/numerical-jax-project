# Demo 2: Pretrained ViT TPU VM Workflow

This document is a conservative, documentation-only workflow for running Demo 2
on a Google Cloud TPU VM. It uses placeholders and example commands only. Do not
run these commands without confirming project, zone, quota, cost, and cleanup
requirements.

Current status: TPU execution for Demo 2 is planned, not completed.

## Placeholders

Use placeholders until safe project-specific values are known:

- `<PROJECT_ID>`: Google Cloud project ID
- `<ZONE>`: TPU zone, such as a zone where the selected TPU type is available
- `<TPU_NAME>`: TPU VM resource name
- `<ACCELERATOR_TYPE>`: TPU accelerator type, such as `v4-8` or a course-approved
  alternative
- `<REPO_URL>`: Git URL for this repository
- `<BRANCH>`: branch to test, such as `feat/vit-inference-benchmark`

## Where Commands Run

- **Local machine / Ubuntu WSL**: local repository checks, Git operations, and
  artifact review.
- **Google Cloud Shell or local gcloud terminal**: `gcloud` configuration and
  TPU VM create, connect, describe, and delete commands.
- **TPU VM shell**: repository checkout, Python environment setup, JAX backend
  verification, and Demo 2 benchmark execution.

## Prerequisites

Before creating resources, verify:

- Google Cloud account and billing are available.
- Google Cloud CLI is installed or Cloud Shell is available.
- The target project is selected and authenticated.
- Cloud TPU API is enabled.
- TPU quota and accelerator availability exist in `<ZONE>`.
- The selected TPU VM may incur cost while it exists.
- No service account keys, `.env` files, or credentials are committed to Git.

Example project setup, run from Google Cloud Shell or local `gcloud` terminal:

```bash
gcloud auth login
gcloud config set project <PROJECT_ID>
gcloud config set compute/zone <ZONE>
gcloud services enable tpu.googleapis.com
```

## Create A TPU VM

Example only, run from Google Cloud Shell or local `gcloud` terminal:

```bash
gcloud compute tpus tpu-vm create <TPU_NAME> \
  --zone=<ZONE> \
  --accelerator-type=<ACCELERATOR_TYPE> \
  --version=tpu-ubuntu2204-base
```

Inspect the resource:

```bash
gcloud compute tpus tpu-vm describe <TPU_NAME> --zone=<ZONE>
```

Connect to the TPU VM:

```bash
gcloud compute tpus tpu-vm ssh <TPU_NAME> --zone=<ZONE>
```

## Set Up The Repository On The TPU VM

Run inside the TPU VM shell:

```bash
git clone <REPO_URL> numerical-jax-project
cd numerical-jax-project
git switch <BRANCH>
git status --short --branch
```

Install `uv` if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

Create or sync the Python environment with the optional pretrained group:

```bash
uv sync --group pretrained
```

JAX on TPU needs `jaxlib` and `libtpu` support. If backend verification below
does not show TPU devices, install TPU-compatible JAX in the active environment
and recheck:

```bash
uv pip install -U "jax[tpu]" \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

Note: the current project default dependency is configured for local JAX usage.
The TPU package installation should be verified on the TPU VM before treating
any run as evidence.

## Verify JAX Backend And Devices

Run inside the TPU VM shell:

```bash
uv run python -m jax_tpu_project.cli devices
```

Expected evidence to capture:

- `default_backend` should report a TPU backend when configured correctly.
- The `devices` list should show TPU devices.
- Save terminal output or copy it into a report note.

Do not claim TPU execution succeeded unless this command and the benchmark run
actually complete on a TPU VM.

## Run Demo 2 On TPU

Run inside the TPU VM shell after backend verification:

```bash
uv run --group pretrained python examples/pretrained_vit_inference.py \
  --jax-platform tpu \
  --image examples/assets/chihuahua_pet_licorice.jpg \
  --batch-size 1 \
  --warmup-steps 1 \
  --benchmark-steps 5 \
  --output report/results/demo2_vit_tpu_b1.json
```

The tracked public example manifest is also available after a normal Git
checkout:

```bash
uv run --group pretrained python examples/pretrained_vit_inference.py \
  --jax-platform tpu \
  --image-manifest examples/assets/manifest.txt \
  --batch-size 4 \
  --warmup-steps 1 \
  --benchmark-steps 5 \
  --output report/results/demo2_vit_tpu_public_b4.json
```

Private live-demo images under `data/local/demo2_vit_images/` are ignored by
Git. They will not appear on a TPU VM from repository checkout alone. If a
private manifest TPU run is needed, copy that directory to the TPU VM manually,
verify the files are present, and avoid committing private images or cloud
transfer artifacts.

For an initial exploratory run, it is also acceptable to write to an ignored
temporary path first:

```bash
uv run --group pretrained python examples/pretrained_vit_inference.py \
  --jax-platform tpu \
  --image examples/assets/chihuahua_pet_licorice.jpg \
  --batch-size 1 \
  --warmup-steps 1 \
  --benchmark-steps 5 \
  --output runs/vit-inference/demo2_tpu_b1.json
```

Record:

- exact command
- branch and commit SHA
- model name
- selected JAX platform
- actual backend and devices
- input shape and batch size
- warmup and benchmark steps
- mean step time and throughput
- predicted label
- output artifact path

## Retrieve Result Artifacts

From Google Cloud Shell or local `gcloud` terminal:

```bash
gcloud compute tpus tpu-vm scp \
  <TPU_NAME>:~/numerical-jax-project/report/results/demo2_vit_tpu_b1.json \
  ./report/results/demo2_vit_tpu_b1.json \
  --zone=<ZONE>
```

If results were first written under `runs/`, copy them locally, inspect them,
and only move curated small artifacts into `report/results/` with clear notes.

## Monitoring Suggestions

Use whichever options are available in the course environment:

- Google Cloud Console TPU resource page for status and lifecycle.
- Cloud Monitoring dashboards for TPU utilization, host CPU, and memory signals
  if metrics are available.
- Terminal logs showing backend/device detection and benchmark JSON output.
- Screenshots only when they add report-ready evidence and do not expose secrets
  or private project details.

## Cleanup

Delete TPU resources as soon as the run and artifact retrieval are complete.
Run from Google Cloud Shell or local `gcloud` terminal:

```bash
gcloud compute tpus tpu-vm delete <TPU_NAME> --zone=<ZONE>
```

Verify cleanup:

```bash
gcloud compute tpus tpu-vm list --zone=<ZONE>
```

Keep cleanup notes in the progress log or report materials if TPU execution is
attempted.

## Warnings And Current Limitations

- TPU VM resources can incur cost until deleted.
- TPU availability and quota vary by project, zone, and accelerator type.
- First-run model download from Hugging Face can take time and requires network
  access unless files are already cached on the TPU VM.
- Do not commit model caches, Hugging Face cache files, service account keys,
  `.env` files, or cloud credentials.
- This document does not create resources automatically.
- TPU execution for Demo 2 has not yet been attempted or completed in this
  repository.

## References

- Google Cloud TPU VM management:
  https://cloud.google.com/tpu/docs/managing-tpus-tpu-vm
- `gcloud compute tpus tpu-vm create` reference:
  https://cloud.google.com/sdk/gcloud/reference/compute/tpus/tpu-vm/create
- JAX installation guide:
  https://docs.jax.dev/en/latest/installation.html
- Google Cloud JAX on TPU guide:
  https://cloud.google.com/tpu/docs/run-calculation-jax
