# Numerical JAX Project

Course project repository for Numerical Computation with JAX.

The current course presentation focus is **Demo 2: pretrained ViT inference
with JAX/Flax** using `google/vit-base-patch16-224`. Local CPU inference has
been run successfully and curated JSON baseline artifacts are stored under
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

You can also run the package CLI directly:

```bash
uv run python -m jax_tpu_project.cli devices
```

The command prints the active JAX backend and visible devices as JSON.

## Demo 2: Pretrained ViT Inference

Demo 2 benchmarks inference for `google/vit-base-patch16-224` with
Hugging Face Transformers, Flax, and JAX. Pretrained dependencies are optional:

```bash
uv sync --group pretrained
```

Run the stable local CPU classroom benchmark:

```bash
uv run --group pretrained python examples/pretrained_vit_inference.py \
  --jax-platform cpu \
  --image examples/assets/chihuahua_pet_licorice.jpg \
  --batch-size 1 \
  --warmup-steps 1 \
  --benchmark-steps 5 \
  --output runs/vit-inference/demo2_cpu_b1.json
```

The script writes JSON metrics with the selected JAX platform, actual backend,
devices, input shape, timing, throughput, and predicted class. The included
sample image is a small public-domain Wikimedia Commons image used only for
reproducible classroom demonstration.

Curated local CPU baseline artifacts:

```text
report/results/demo2_vit_local_cpu_b1.json
report/results/demo2_vit_local_cpu_b4.json
report/results/demo2_vit_local_cpu_b8.json
```

For full local instructions, expected output, model notes, and limitations, see
[docs/pretrained_vit_demo.md](docs/pretrained_vit_demo.md).

For the planned TPU workflow, see
[cloud/demo2_vit_tpu_workflow.md](cloud/demo2_vit_tpu_workflow.md). This is
documentation-only at the current stage; no TPU run is claimed.

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
