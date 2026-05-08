# Numerical JAX Project

Course project repository for Numerical Computation with JAX. The current demo
focus is a hand-written CNN benchmark foundation for local training now and
later Google Cloud TPU comparison.

## Local Sanity Check

Run the JAX device summary script first:

```bash
bash scripts/check_jax_device.sh
```

You can also run the package CLI directly:

```bash
uv run python -m jax_tpu_project.cli devices
```

The command prints the active JAX backend and visible devices as JSON. It should
work on CPU-only machines and GPU-enabled machines.

## Demo 1: Hand-Written CNN Benchmark

Demo 1 trains a small CNN on MNIST-shaped image data using raw JAX. The model is
implemented manually with explicit parameter initialization, `jax.jit`,
`jax.value_and_grad`, `jax.vmap`, `jax.lax.conv_general_dilated`, and explicit
reshape-based average pooling. It does not use Flax, Optax, PyTorch, or
TensorFlow.

The default dataset is deterministic synthetic data shaped like MNIST:
`[N, 28, 28, 1]` images and integer labels from 0 to 9. This keeps local smoke
tests and default runs independent of network access while leaving the CLI ready
for later local-file MNIST or Fashion-MNIST support under `data/raw/`.

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

The example script defaults to CPU when `JAX_PLATFORMS` is unset so smoke runs
stay safe on ordinary local machines. Set `JAX_PLATFORMS` before launching the
script when testing a specific JAX backend later.

Run a slightly longer local benchmark:

```bash
uv run python examples/cnn_mnist_benchmark.py \
  --dataset synthetic \
  --steps 50 \
  --batch-size 64 \
  --learning-rate 0.05 \
  --warmup-steps 5 \
  --seed 0 \
  --output-dir runs/local-cnn-mnist \
  --platform-label local
```

Expected terminal output includes per-step loss, synthetic accuracy, step time,
the active JAX backend, visible devices, average step time, examples per second,
and the metrics file path.

Each run writes JSON metrics to:

```text
<output-dir>/cnn_mnist_metrics.json
```

The metrics include backend, devices, platform label, dataset name, seed, batch
size, steps, warmup steps, learning rate, total training time, average step time,
examples per second, initial and final loss, synthetic accuracy, and output
artifact path.

The CNN code is written to be portable across JAX backends and is intended for a
future local-versus-Google Cloud TPU benchmark. TPU execution is planned but has
not yet been run or measured in this repository.
