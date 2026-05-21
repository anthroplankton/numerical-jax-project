# Numerical JAX Project

Course project repository for Numerical Computation with JAX.

The current course presentation focus is **Demo 2: pretrained ViT inference
with JAX/Flax** using `google/vit-base-patch16-224`. Local CPU inference has
been run successfully. Raw JSON benchmark outputs live under ignored
`runs/vit-inference/`, and curated report-ready Markdown tables live under
`report/results/`. Google Cloud TPU execution is the next planned workflow and
has not been completed yet.

Demo 1 remains in the repository as preserved background work: a raw-JAX
hand-written CNN benchmark foundation. Demo 3 remains optional future work for a
larger pretrained or Gemma-like cloud workflow.

## Local Sanity Check

Run the JAX device summary script first:

```bash
bash scripts/check_jax_device.sh
```

The script defaults `JAX_PLATFORMS` to `cpu` unless the caller already set it,
keeping the local sanity path explicit on machines that may have unrelated GPU
drivers or plugins installed.

You can also run the package CLI directly:

```bash
uv run python -m jax_tpu_project.cli devices
```

The command prints the active JAX backend and visible devices as JSON.

Continuous integration is intentionally limited to lightweight local CPU checks:
Ruff linting, Ruff formatting, pytest, and a small JAX backend/device sanity
command. It does not run pretrained inference, formal benchmarks, Docker builds,
Google Cloud commands, or TPU workflows.

## Fresh Demo 2 Benchmark Machine Setup

Run from an Ubuntu or WSL terminal at the repository root:

```bash
uv sync --frozen --group dev --group pretrained
bash scripts/check_jax_device.sh
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Run a public five-image manifest smoke benchmark before using local-only
datasets:

```bash
uv run --group pretrained python examples/pretrained_vit_inference.py \
  --jax-platform cpu \
  --image-manifest examples/assets/manifest.txt \
  --batch-size 4 \
  --warmup-steps 1 \
  --benchmark-steps 1 \
  --output runs/vit-inference/demo2_public_examples_smoke_cpu_b4.json
```

Imagenette 320 is not downloaded by tests or project scripts. Before running
Imagenette benchmarks, manually download and extract `imagenette2-320` so this
local ignored path exists:

```text
data/local/imagenette2-320/val
```

## Demo 2: Pretrained ViT Inference

Demo 2 benchmarks inference for `google/vit-base-patch16-224` with
Hugging Face Transformers, Flax, and JAX. Pretrained dependencies are optional:

```bash
uv sync --frozen --group pretrained
```

Run the stable local CPU public examples benchmark. This is the formal Demo 2
local public `b1` baseline command:

```bash
uv run --group pretrained python examples/pretrained_vit_inference.py \
  --jax-platform cpu \
  --image-manifest examples/assets/manifest.txt \
  --batch-size 1 \
  --warmup-steps 1 \
  --benchmark-steps 5 \
  --output runs/vit-inference/demo2_local_public_examples_cpu_b1.json
```

The script writes JSON metrics with the selected JAX platform, actual backend,
devices, input shape, timing, throughput, and predicted class. The tracked
public example set contains five Wikimedia Commons images under
`examples/assets/`, including the single-image smoke-test input.

Run the public five-image manifest. This is the formal Demo 2 public manifest
`b4` command:

```bash
uv run --group pretrained python examples/pretrained_vit_inference.py \
  --jax-platform cpu \
  --image-manifest examples/assets/manifest.txt \
  --batch-size 4 \
  --warmup-steps 1 \
  --benchmark-steps 5 \
  --output runs/vit-inference/demo2_local_public_examples_cpu_b4.json
```

For a private live demo, keep local photos and the optional manifest under
`data/local/demo2_vit_images/`. That path is ignored by Git; see
[docs/pretrained_vit_demo.md](docs/pretrained_vit_demo.md) for the manifest
format and command. Manifest mode uses true mixed-image batches; the final
partial batch is padded by repeating its last real image, padded entries are
ignored for predictions and throughput, and `num_padded_images` records that
padding. The current local live-demo manifest is expected to contain 15 images.
The private manifest workflow is for qualitative live predictions only; it is
not a public benchmark dataset or an accuracy benchmark.

For optional local benchmark work, use Imagenette 320 (`imagenette2-320`) as a
local-only dataset. Current curated tables exist for local and supplementary
external CPU Imagenette runs, but the extracted dataset and generated manifests
remain under ignored `data/local/imagenette2-320/`. Build a manifest before
running a benchmark. `scripts/build_image_manifest.py` scans existing local
images only; it does not download Imagenette. The primary documented path is the
64-image validation manifest:

```bash
uv run python scripts/build_image_manifest.py \
  data/local/imagenette2-320/val \
  --output data/local/imagenette2-320/val/manifest_val_64.txt \
  --limit 64

wc -l data/local/imagenette2-320/val/manifest_val_64.txt
head data/local/imagenette2-320/val/manifest_val_64.txt
```

Curated Demo 2 Markdown tables:

```text
report/results/demo2_local_public_examples_cpu.md
report/results/demo2_external_ryzen7735hs_wsl_public_examples_cpu.md
report/results/demo2_local_imagenette320_val64_cpu.md
report/results/demo2_external_ryzen7735hs_wsl_imagenette320_val64_cpu.md
report/results/demo2_local_imagenette320_val256_cpu.md
report/results/demo2_external_ryzen7735hs_wsl_imagenette320_val256_cpu.md
report/results/demo2_local_private_examples_cpu.md
```

Local CPU tables are the primary current-machine evidence. External Ryzen
7735HS WSL CPU tables are supplementary and are kept separate.

For full local instructions, expected output, model notes, and limitations, see
[docs/pretrained_vit_demo.md](docs/pretrained_vit_demo.md).

For the planned TPU workflow, see
[cloud/demo2_vit_tpu_workflow.md](cloud/demo2_vit_tpu_workflow.md). The
Google Cloud / TRC setup record is tracked in
[report/google_cloud_trc_setup.md](report/google_cloud_trc_setup.md). This is
documentation-only at the current stage; TRC confirmation is pending and no TPU
run is claimed.

After TPU JSON artifacts are copied back locally, compare existing result files
without TPU access:

```bash
uv run python scripts/compare_vit_results.py \
  runs/vit-inference/demo2_local_public_examples_cpu_b4.json \
  runs/vit-inference/demo2_cloud_public_examples_tpu_b4.json \
  --output runs/vit-inference/demo2_cpu_vs_tpu_public_examples_b4_compare.json \
  --markdown-output runs/vit-inference/demo2_cpu_vs_tpu_public_examples_b4_table.md
```

## Demo 1: Preserved Raw-JAX CNN Foundation

Demo 1 trains a small CNN on MNIST-shaped image data using raw JAX. The model is
implemented manually with explicit parameter initialization, `jax.jit`,
`jax.value_and_grad`, `jax.vmap`, `jax.lax.conv_general_dilated`, and explicit
reshape-based average pooling. It does not use Flax, Optax, PyTorch, or
TensorFlow.

The current implemented dataset path is deterministic synthetic data shaped like
MNIST: `[N, 28, 28, 1]` images and integer labels from 0 to 9.

Run a quick local smoke benchmark:

```bash
uv run python examples/cnn_mnist_benchmark.py \
  --dataset synthetic \
  --steps 3 \
  --batch-size 16 \
  --seed 0 \
  --output-dir runs/smoke \
  --platform-label local
```

Demo 1 is preserved as background/foundation work and is not the primary path
for the current course presentation. Real MNIST/Fashion-MNIST loading and TPU
comparison for Demo 1 remain future work.

## Demo 3: Optional Future Work

Demo 3 is reserved for a possible larger pretrained-model or Gemma-like cloud
workflow if scope, access, quota, hardware, and time permit. It is not part of
the current presentation scope.
