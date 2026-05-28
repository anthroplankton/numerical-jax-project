# Demo 2: Pretrained ViT TPU VM Workflow

This is a safe, documentation-first workflow for moving Demo 2 from local CPU
execution to Google Cloud TPU VM execution. It is a pre-execution workflow with
placeholders and example commands only. Do not create, modify, or delete cloud
resources until project, zone, quota, cost, funding, and cleanup plans are
confirmed.

Current status: local CPU Demo 2 is completed, the Google Cloud TPU workflow is
documented, Google Cloud / TRC setup is recorded in
`report/google_cloud_trc_setup.md`, TRC confirmation has been received, and TPU
execution is not completed. CPU-vs-TPU comparison, TPU monitoring evidence,
cleanup evidence, and TPU performance evidence remain pending until a real TPU
run occurs and its JSON artifact is retrieved.

## Placeholders

Use placeholders until safe project-specific values are known:

- `<PROJECT_ID>`: Google Cloud project ID.
- `<PROJECT_NUMBER>`: Google Cloud project number for TRC submission.
- `<ZONE>`: TPU zone.
- `<TPU_NAME>`: TPU VM resource name.
- `<ACCELERATOR_TYPE>`: TPU accelerator type.
- `<RUNTIME_VERSION>`: TPU VM runtime version.
- `<QUEUED_RESOURCE_ID>`: optional queued-resource identifier, if that creation
  path is used.
- `<REPO_URL>`: Git URL for this repository.
- `<BRANCH>`: branch to test.
- `<COMMIT_SHA>`: exact commit hash to test on the TPU VM.

## Google Cloud / TRC Setup Checklist

Complete these steps before attempting TPU execution:

- Create a dedicated Google Cloud project for this course project.
- Record the project ID locally as `<PROJECT_ID>`.
- Record the project number locally as `<PROJECT_NUMBER>`.
- Submit the project number to the Google TPU Research Cloud / TRC form.
- Confirm TRC confirmation has been received before creating TPU resources.
- Avoid committing real project IDs, project numbers, billing details, service
  account keys, `.env` files, credential files, or local cloud config files.
- Keep repository documentation on placeholders such as `<PROJECT_ID>`,
  `<PROJECT_NUMBER>`, `<ZONE>`, `<TPU_NAME>`, `<ACCELERATOR_TYPE>`, and
  `<RUNTIME_VERSION>`.
- Avoid running TPU VM creation commands until the actual cloud experiment is
  ready to start.
- Keep cleanup commands visible next to any resource-creation commands.

## Cost-Control Notes

- Prefer TRC-provided resources before spending Google Cloud free-trial credits.
- Do not create TPU resources until the run plan, artifact retrieval command,
  and cleanup command are ready.
- Use a short first run to verify backend, devices, model loading, and JSON
  output before attempting longer measurements.
- Delete the TPU VM immediately after the experiment and artifact retrieval.
- Use the Google Cloud Web Console billing, budget, quota, and resource pages
  for human verification.
- Do not claim budget alerts, billing setup, quota allocation, TRC approval, or
  resource cleanup were configured unless those steps actually happened.

## Where Commands Run

- **Local machine / Ubuntu WSL**: run local validation, review Git state, inspect
  copied JSON artifacts, and compare local CPU versus TPU result files.
- **Google Cloud Shell or local terminal with `gcloud` installed**: authenticate,
  select project/zone, create/connect/describe/delete TPU VM resources, and copy
  result artifacts.
- **TPU VM shell**: clone or update the repository, install dependencies, verify
  JAX TPU backend/devices, and run Demo 2.
- **Optional container shell**: only for a later Docker-based experiment. Direct
  TPU VM execution is the first path.

## Local Preflight

Run on the local machine / Ubuntu WSL before any cloud work:

```bash
git status --short --branch
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Confirm the local CPU raw JSON artifact and curated Markdown table exist for the
comparison you plan to run:

```text
runs/vit-inference/demo2_local_public_examples_cpu_b4.json
report/results/demo2_local_public_examples_cpu.md
```

Optional local comparison dry run:

```bash
uv run python scripts/compare_vit_results.py \
  runs/vit-inference/demo2_local_public_examples_cpu_b1.json \
  runs/vit-inference/demo2_local_public_examples_cpu_b4.json \
  --output runs/vit-inference/demo2_local_public_examples_compare.json
```

The helper reads existing JSON files only. It does not need TPU access, model
weights, image files, or network access.

## Cloud Prerequisites

Run from Google Cloud Shell or a local terminal with `gcloud` installed only
after TRC confirmation has been received and the cloud experiment is ready to
start, or after explicitly deciding to use non-TRC Google Cloud credits:

```bash
gcloud auth login
gcloud config set project <PROJECT_ID>
gcloud config set compute/zone <ZONE>
gcloud services enable tpu.googleapis.com
```

Before creating resources, verify:

- Billing and TPU quota are available for `<PROJECT_ID>`.
- TRC confirmation has been received, or fallback funding is explicitly selected
  for a non-TRC run.
- `<ACCELERATOR_TYPE>` is available in `<ZONE>`.
- `<RUNTIME_VERSION>` is appropriate for the selected TPU VM.
- `<TPU_NAME>` and optional `<QUEUED_RESOURCE_ID>` are unique and safe to use.
- The cleanup command for the chosen creation path is visible and ready.
- The TPU VM may incur cost until it is deleted.
- No service account keys, `.env` files, model caches, or credentials are
  committed to Git.

## First TPU Smoke-Run Sequence

Use this as the first Demo 2 TPU runbook after TRC confirmation has been
received and zone, accelerator type, runtime version, funding, and cleanup
readiness are confirmed.
Choose either direct TPU VM creation or queued-resource creation; do not run both
unless there is a deliberate reason.

Record the placeholders before creating resources:

```text
PROJECT_ID=<PROJECT_ID>
ZONE=<ZONE>
TPU_NAME=<TPU_NAME>
ACCELERATOR_TYPE=<ACCELERATOR_TYPE>
RUNTIME_VERSION=<RUNTIME_VERSION>
QUEUED_RESOURCE_ID=<QUEUED_RESOURCE_ID>  # optional
REPO_URL=<REPO_URL>
BRANCH=<BRANCH>
COMMIT_SHA=<COMMIT_SHA>
```

Direct TPU VM creation path, example only:

```bash
gcloud compute tpus tpu-vm create <TPU_NAME> \
  --project=<PROJECT_ID> \
  --zone=<ZONE> \
  --accelerator-type=<ACCELERATOR_TYPE> \
  --version=<RUNTIME_VERSION>

# Cleanup command to keep ready before creation:
gcloud compute tpus tpu-vm delete <TPU_NAME> --project=<PROJECT_ID> --zone=<ZONE>
```

Queued-resource creation path, example only:

```bash
gcloud compute tpus queued-resources create <QUEUED_RESOURCE_ID> \
  --project=<PROJECT_ID> \
  --zone=<ZONE> \
  --node-id=<TPU_NAME> \
  --accelerator-type=<ACCELERATOR_TYPE> \
  --runtime-version=<RUNTIME_VERSION>

# Cleanup command to keep ready before creation:
gcloud compute tpus queued-resources delete <QUEUED_RESOURCE_ID> \
  --project=<PROJECT_ID> \
  --zone=<ZONE>
```

Inspect the created VM and connect to it:

```bash
gcloud compute tpus tpu-vm describe <TPU_NAME> \
  --project=<PROJECT_ID> \
  --zone=<ZONE>

gcloud compute tpus tpu-vm ssh <TPU_NAME> \
  --project=<PROJECT_ID> \
  --zone=<ZONE>
```

Inside the TPU VM, clone the repository and pin the exact branch and commit:

```bash
git clone <REPO_URL> numerical-jax-project
cd numerical-jax-project
git fetch --all --prune
git switch <BRANCH>
git checkout <COMMIT_SHA>
git status --short --branch
git rev-parse HEAD
```

Install project dependencies and TPU-compatible JAX support:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
uv sync --group pretrained

uv pip install -U "jax[tpu]" \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

The exact TPU JAX installation may need adjustment based on the TPU runtime.
Do not proceed to the benchmark until the sanity command below reports a TPU
backend and TPU devices.

Run a JAX TPU sanity command inside the TPU VM:

```bash
uv run python -c "import jax; print('jax_version=', jax.__version__); print('default_backend=', jax.default_backend()); print('device_count=', jax.device_count()); print('local_device_count=', jax.local_device_count()); print('devices=', jax.devices())"
```

Run the first Demo 2 public examples TPU smoke benchmark inside the TPU VM:

```bash
uv run --group pretrained python examples/pretrained_vit_inference.py \
  --jax-platform tpu \
  --image-manifest examples/assets/manifest.txt \
  --batch-size 4 \
  --warmup-steps 1 \
  --benchmark-steps 5 \
  --output runs/vit-inference/demo2_cloud_public_examples_tpu_b4.json
```

Copy the TPU JSON artifact back to the local machine or Cloud Shell:

```bash
gcloud compute tpus tpu-vm scp \
  <TPU_NAME>:~/numerical-jax-project/runs/vit-inference/demo2_cloud_public_examples_tpu_b4.json \
  ./runs/vit-inference/demo2_cloud_public_examples_tpu_b4.json \
  --project=<PROJECT_ID> \
  --zone=<ZONE>
```

Only after the real TPU JSON exists locally, generate the future local CPU vs
cloud TPU comparison artifacts:

```bash
uv run python scripts/compare_vit_results.py \
  runs/vit-inference/demo2_local_public_examples_cpu_b4.json \
  runs/vit-inference/demo2_cloud_public_examples_tpu_b4.json \
  --output runs/vit-inference/demo2_local_cpu_vs_cloud_tpu_public_examples_b4_compare.json \
  --markdown-output report/results/demo2_local_cpu_vs_cloud_tpu_public_examples_b4.md
```

Do not create placeholder comparison JSON or Markdown files before a real TPU
JSON artifact exists.

Collect monitoring and logging notes only after the real run:

- Cloud Monitoring note or screenshot placeholder for TPU utilization, host CPU,
  memory, or idle time if available.
- Cloud Logging note if useful for the run timeline or errors.
- Terminal transcript or command log covering setup, sanity check, benchmark,
  artifact copy, comparison generation, and cleanup.

Delete resources after artifact retrieval and note the transcript:

```bash
gcloud compute tpus tpu-vm delete <TPU_NAME> --project=<PROJECT_ID> --zone=<ZONE>

# If a queued resource was used, delete it too:
gcloud compute tpus queued-resources delete <QUEUED_RESOURCE_ID> \
  --project=<PROJECT_ID> \
  --zone=<ZONE>
```

Verify deletion:

```bash
gcloud compute tpus tpu-vm list --project=<PROJECT_ID> --zone=<ZONE>
gcloud compute tpus queued-resources list --project=<PROJECT_ID> --zone=<ZONE>
```

## Create And Inspect A TPU VM

Example only, run from Google Cloud Shell or local `gcloud` terminal:

```bash
gcloud compute tpus tpu-vm create <TPU_NAME> \
  --zone=<ZONE> \
  --accelerator-type=<ACCELERATOR_TYPE> \
  --version=<RUNTIME_VERSION>
```

Keep the cleanup command ready before creating the VM:

```bash
gcloud compute tpus tpu-vm delete <TPU_NAME> --zone=<ZONE>
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

If the repository already exists on the TPU VM, update it through Git:

```bash
cd numerical-jax-project
git fetch --all --prune
git switch <BRANCH>
git pull --ff-only
git status --short --branch
```

Install `uv` if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

Install project dependencies, including the optional pretrained group:

```bash
uv sync --group pretrained
```

Install or verify TPU-compatible JAX support in the active environment:

```bash
uv pip install -U "jax[tpu]" \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

The exact TPU JAX installation may need adjustment based on the TPU runtime.
Do not treat the environment as ready until backend/device verification below
shows TPU devices.

## Verify JAX Backend And Devices

Run inside the TPU VM shell:

```bash
uv run python -m jax_tpu_project.cli devices
```

Expected evidence to capture:

- `default_backend` reports a TPU backend.
- `devices` lists TPU devices.
- Terminal output is saved or copied into report notes.

Do not claim TPU execution succeeded unless both this check and a Demo 2
benchmark command actually complete on the TPU VM.

## Run Demo 2 On TPU

Run inside the TPU VM shell after backend verification.

Public five-image manifest, `b1`:

```bash
uv run --group pretrained python examples/pretrained_vit_inference.py \
  --jax-platform tpu \
  --image-manifest examples/assets/manifest.txt \
  --batch-size 1 \
  --warmup-steps 1 \
  --benchmark-steps 5 \
  --output runs/vit-inference/demo2_cloud_public_examples_tpu_b1.json
```

Public five-image manifest, `b4`:

```bash
uv run --group pretrained python examples/pretrained_vit_inference.py \
  --jax-platform tpu \
  --image-manifest examples/assets/manifest.txt \
  --batch-size 4 \
  --warmup-steps 1 \
  --benchmark-steps 5 \
  --output runs/vit-inference/demo2_cloud_public_examples_tpu_b4.json
```

Private live-demo images under `data/local/demo2_vit_images/` are ignored by
Git and will not be available from repository checkout alone. If a private
manifest TPU run is needed, copy that directory manually to the TPU VM, verify
the files are present, and keep private images out of Git.

The Demo 2 JSON output records:

- `command_used`
- `output_path`
- `image_path` or `manifest_path`
- `mode` and `processing_mode`
- `backend`
- `devices`
- `batch_size`
- `num_images`, `num_batches`, `timed_batch_runs`, and `num_padded_images`
- `total_timed_inference_sec`
- `mean_step_time_sec`
- `throughput_counted_images`
- `throughput_images_per_sec`
- per-image predictions under `image_results` for manifest mode

Also save terminal logs showing the command, backend/device output, and any
first-run model download messages.

## Retrieve Result Artifacts

Run from Google Cloud Shell or local `gcloud` terminal:

```bash
gcloud compute tpus tpu-vm scp \
  <TPU_NAME>:~/numerical-jax-project/runs/vit-inference/demo2_cloud_public_examples_tpu_b1.json \
  ./runs/vit-inference/demo2_cloud_public_examples_tpu_b1.json \
  --zone=<ZONE>
```

For the public manifest run:

```bash
gcloud compute tpus tpu-vm scp \
  <TPU_NAME>:~/numerical-jax-project/runs/vit-inference/demo2_cloud_public_examples_tpu_b4.json \
  ./runs/vit-inference/demo2_cloud_public_examples_tpu_b4.json \
  --zone=<ZONE>
```

Inspect copied artifacts locally before curating any small result files under
`report/results/`.

## Compare Local CPU And TPU Results

Run on the local machine / Ubuntu WSL after TPU JSON files are retrieved:

```bash
uv run python scripts/compare_vit_results.py \
  runs/vit-inference/demo2_local_public_examples_cpu_b4.json \
  runs/vit-inference/demo2_cloud_public_examples_tpu_b4.json \
  --output runs/vit-inference/demo2_local_cpu_vs_cloud_tpu_public_examples_b4_compare.json \
  --markdown-output report/results/demo2_local_cpu_vs_cloud_tpu_public_examples_b4.md
```

If a matching local CPU manifest artifact does not already exist, create it with
the same image set and batch size before comparing:

```bash
uv run --group pretrained python examples/pretrained_vit_inference.py \
  --jax-platform cpu \
  --image-manifest examples/assets/manifest.txt \
  --batch-size 4 \
  --warmup-steps 1 \
  --benchmark-steps 5 \
  --output runs/vit-inference/demo2_local_public_examples_cpu_b4.json
```

Then compare:

```bash
uv run python scripts/compare_vit_results.py \
  runs/vit-inference/demo2_local_public_examples_cpu_b4.json \
  runs/vit-inference/demo2_cloud_public_examples_tpu_b4.json \
  --output runs/vit-inference/demo2_local_cpu_vs_cloud_tpu_public_examples_b4_compare.json \
  --markdown-output report/results/demo2_local_cpu_vs_cloud_tpu_public_examples_b4.md
```

The comparison helper summarizes command, input path or manifest, backend,
devices, batch size, image count, total runtime, throughput, per-image time
derived from throughput, and output path when those fields are present. This
comparison remains pending until a real TPU JSON artifact exists.

## Monitoring Evidence

Collect monitoring evidence only after a real TPU run is attempted:

- Google Cloud Console TPU resource status and lifecycle.
- Cloud Monitoring charts for TPU utilization, host CPU, memory, or idle time if
  available.
- Terminal logs for backend/device verification and Demo 2 benchmark output.
- Screenshots only when they add report-ready evidence and do not expose
  project identifiers, secrets, private images, or credentials.

## Evidence Checklist

Capture these items only after a real TPU attempt. Until then, they remain
planned evidence placeholders:

- Exact branch and commit hash used on the TPU VM: `<BRANCH>` and
  `<COMMIT_SHA>`.
- TPU VM name, zone, accelerator type, and runtime version:
  `<TPU_NAME>`, `<ZONE>`, `<ACCELERATOR_TYPE>`, and `<RUNTIME_VERSION>`.
- Terminal transcript or command log for setup, sanity checks, benchmark,
  artifact retrieval, comparison generation, cleanup, and deletion verification.
- JAX backend/device output from the sanity command, including `jax.__version__`,
  `jax.default_backend()`, `jax.device_count()`, `jax.local_device_count()`, and
  `jax.devices()`.
- Demo 2 TPU JSON artifact:
  `runs/vit-inference/demo2_cloud_public_examples_tpu_b4.json`.
- Cloud Monitoring note or screenshot placeholder, if available and safe to
  include.
- Cloud Logging note, if useful for run timeline or troubleshooting.
- Cleanup command transcript.
- Deletion verification output from TPU VM and queued-resource list commands.
- Generated report-ready CPU-vs-TPU Markdown table path, after the TPU artifact
  exists:
  `report/results/demo2_local_cpu_vs_cloud_tpu_public_examples_b4.md`.

## Cleanup

Run from Google Cloud Shell or local `gcloud` terminal when the run and artifact
retrieval are complete:

```bash
gcloud compute tpus tpu-vm delete <TPU_NAME> --project=<PROJECT_ID> --zone=<ZONE>

# If a queued resource was used:
gcloud compute tpus queued-resources delete <QUEUED_RESOURCE_ID> \
  --project=<PROJECT_ID> \
  --zone=<ZONE>
```

Verify cleanup:

```bash
gcloud compute tpus tpu-vm list --project=<PROJECT_ID> --zone=<ZONE>
gcloud compute tpus queued-resources list --project=<PROJECT_ID> --zone=<ZONE>
```

Record cleanup notes in `report/progress_log.md` or final report material only
after cleanup actually occurs.

## Optional Container Follow-Up

Direct TPU VM execution is the default path. A Docker or TPU container workflow
can be considered later only if direct VM execution works and containerization
would improve reproducibility.

If explored later, keep it documentation-first:

- Use placeholders for image names, registries, and project IDs.
- Do not add CI cloud jobs or automated deployment.
- Verify TPU device visibility inside the container before benchmarking.
- Keep result retrieval, comparison, monitoring, and cleanup steps explicit.

## Warnings And Current Limitations

- TPU VM resources can incur cost until deleted.
- TPU availability and quota vary by project, zone, and accelerator type.
- First-run model download from Hugging Face can take time and requires network
  access unless files are already cached on the TPU VM.
- Do not commit model caches, Hugging Face cache files, service account keys,
  `.env` files, cloud credentials, private images, raw cloud logs, or uncurated
  generated artifacts.
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
