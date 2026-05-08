"""Raw-JAX CNN benchmark foundation for MNIST-shaped image classification.

The default dataset is deterministic synthetic data shaped like MNIST images:
``[N, 28, 28, 1]`` with labels in ``[0, 9]``. The training code uses only JAX
operations so it can run locally now and later be reused on Google Cloud TPU.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
from jax import lax

ArrayTree: TypeAlias = dict[str, dict[str, jax.Array]]

IMAGE_SHAPE = (28, 28, 1)
NUM_CLASSES = 10
DEFAULT_STEPS = 5
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_WARMUP_STEPS = 1
DEFAULT_OUTPUT_DIR = Path("runs/cnn_mnist")
SUPPORTED_DATASETS = ("synthetic", "mnist", "fashion_mnist")


def init_cnn_params(seed: int = 0, dtype: jnp.dtype = jnp.float32) -> ArrayTree:
    """Initialize parameters for a small handwritten CNN.

    Shapes assume NHWC inputs with image shape ``[28, 28, 1]``:
    conv1 -> average pool -> conv2 -> average pool -> dense hidden -> logits.
    """
    key = jax.random.PRNGKey(seed)
    conv1_key, conv2_key, dense1_key, logits_key = jax.random.split(key, 4)

    return {
        "conv1": {
            "kernel": _he_normal(
                conv1_key, (3, 3, 1, 8), fan_in=3 * 3 * 1, dtype=dtype
            ),
            "bias": jnp.zeros((8,), dtype=dtype),
        },
        "conv2": {
            "kernel": _he_normal(
                conv2_key, (3, 3, 8, 16), fan_in=3 * 3 * 8, dtype=dtype
            ),
            "bias": jnp.zeros((16,), dtype=dtype),
        },
        "dense1": {
            "kernel": _he_normal(
                dense1_key, (7 * 7 * 16, 32), fan_in=7 * 7 * 16, dtype=dtype
            ),
            "bias": jnp.zeros((32,), dtype=dtype),
        },
        "logits": {
            "kernel": _he_normal(logits_key, (32, NUM_CLASSES), fan_in=32, dtype=dtype),
            "bias": jnp.zeros((NUM_CLASSES,), dtype=dtype),
        },
    }


def make_synthetic_mnist_data(
    num_examples: int, seed: int = 0, dtype: jnp.dtype = jnp.float32
) -> tuple[jax.Array, jax.Array]:
    """Create deterministic MNIST-shaped synthetic images and integer labels."""
    if num_examples < 1:
        msg = "num_examples must be at least 1"
        raise ValueError(msg)

    label_key, noise_key = jax.random.split(jax.random.PRNGKey(seed))
    labels = jax.random.randint(
        label_key, shape=(num_examples,), minval=0, maxval=NUM_CLASSES
    )
    templates = jax.vmap(_class_template)(labels).astype(dtype)
    noise = jax.random.normal(
        noise_key, shape=(num_examples, *IMAGE_SHAPE), dtype=dtype
    )
    images = jnp.clip(templates + 0.08 * noise, 0.0, 1.0)
    return images, labels.astype(jnp.int32)


def forward(params: ArrayTree, images: jax.Array) -> jax.Array:
    """Run the CNN forward pass and return logits with shape ``[batch, 10]``."""
    activations = _conv2d(images, params["conv1"]["kernel"], params["conv1"]["bias"])
    activations = jax.nn.relu(activations)
    activations = _average_pool2d(activations)

    activations = _conv2d(
        activations, params["conv2"]["kernel"], params["conv2"]["bias"]
    )
    activations = jax.nn.relu(activations)
    activations = _average_pool2d(activations)

    activations = activations.reshape((activations.shape[0], -1))
    activations = jnp.dot(activations, params["dense1"]["kernel"])
    activations = jax.nn.relu(activations + params["dense1"]["bias"])
    return jnp.dot(activations, params["logits"]["kernel"]) + params["logits"]["bias"]


@jax.jit
def training_step(
    params: ArrayTree, images: jax.Array, labels: jax.Array, learning_rate: float
) -> tuple[ArrayTree, jax.Array, jax.Array]:
    """Run one explicit SGD step and return updated params, loss, and accuracy."""

    def objective(current_params: ArrayTree) -> tuple[jax.Array, jax.Array]:
        return loss_and_accuracy(current_params, images, labels)

    (loss, accuracy), gradients = jax.value_and_grad(objective, has_aux=True)(params)
    updated_params = jax.tree_util.tree_map(
        lambda param, grad: param - learning_rate * grad, params, gradients
    )
    return updated_params, loss, accuracy


def loss_and_accuracy(
    params: ArrayTree, images: jax.Array, labels: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Return cross-entropy loss and classification accuracy for a batch."""
    logits = forward(params, images)
    one_hot_labels = jax.nn.one_hot(labels, NUM_CLASSES)
    loss = -jnp.mean(jnp.sum(one_hot_labels * jax.nn.log_softmax(logits), axis=-1))
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == labels)
    return loss, accuracy


def run_cnn_mnist_benchmark(
    *,
    dataset: str = "synthetic",
    data_dir: str | Path = "data/raw",
    steps: int = DEFAULT_STEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    seed: int = 0,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    platform_label: str = "local",
    log_every: int = 1,
) -> dict[str, Any]:
    """Run a local CNN training benchmark and write JSON metrics."""
    _validate_benchmark_args(
        dataset=dataset,
        steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        log_every=log_every,
    )

    total_steps = warmup_steps + steps
    params = init_cnn_params(seed=seed)
    images, labels = _load_dataset_batches(
        dataset=dataset,
        data_dir=Path(data_dir),
        total_steps=total_steps,
        batch_size=batch_size,
        seed=seed,
    )

    initial_loss, initial_accuracy = loss_and_accuracy(params, images[0], labels[0])
    initial_loss.block_until_ready()
    initial_accuracy.block_until_ready()

    for step_index in range(warmup_steps):
        params, loss, accuracy = training_step(
            params, images[step_index], labels[step_index], learning_rate
        )
        _block_training_outputs(params, loss, accuracy)

    timed_step_durations: list[float] = []
    final_loss = initial_loss
    final_accuracy = initial_accuracy
    for offset in range(steps):
        batch_index = warmup_steps + offset
        start_time = time.perf_counter()
        params, final_loss, final_accuracy = training_step(
            params, images[batch_index], labels[batch_index], learning_rate
        )
        _block_training_outputs(params, final_loss, final_accuracy)
        duration = time.perf_counter() - start_time
        timed_step_durations.append(duration)

        step_number = offset + 1
        if step_number == 1 or step_number == steps or step_number % log_every == 0:
            print(
                "step "
                f"{step_number}/{steps}: "
                f"loss={float(final_loss):.4f}, "
                f"accuracy={float(final_accuracy):.3f}, "
                f"step_time={duration:.4f}s"
            )

    total_training_time = sum(timed_step_durations)
    average_step_time = total_training_time / steps
    examples_per_second = batch_size * steps / total_training_time
    metrics_path = Path(output_dir) / "cnn_mnist_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, Any] = {
        "backend": jax.default_backend(),
        "devices": _device_summaries(),
        "platform_label": platform_label,
        "dataset_name": dataset,
        "seed": seed,
        "batch_size": batch_size,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "learning_rate": learning_rate,
        "total_training_time_seconds": total_training_time,
        "average_step_time_seconds": average_step_time,
        "examples_per_second": examples_per_second,
        "initial_loss": float(initial_loss),
        "final_loss": float(final_loss),
        "initial_accuracy": float(initial_accuracy),
        "final_accuracy": float(final_accuracy),
        "output_artifact_path": str(metrics_path),
    }

    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


def format_benchmark_summary(metrics: dict[str, Any]) -> str:
    """Format benchmark metrics as concise terminal output."""
    device_names = ", ".join(device["repr"] for device in metrics["devices"])
    return "\n".join(
        [
            (
                "CNN MNIST benchmark "
                f"({metrics['dataset_name']}, {metrics['platform_label']})"
            ),
            f"backend: {metrics['backend']}",
            f"devices: {device_names}",
            (
                "loss: "
                f"{metrics['initial_loss']:.4f} -> {metrics['final_loss']:.4f}; "
                f"accuracy: {metrics['final_accuracy']:.3f}"
            ),
            (
                "timing: "
                f"{metrics['average_step_time_seconds']:.4f}s/step, "
                f"{metrics['examples_per_second']:.1f} examples/s"
            ),
            f"metrics: {metrics['output_artifact_path']}",
        ]
    )


def _he_normal(
    key: jax.Array, shape: tuple[int, ...], *, fan_in: int, dtype: jnp.dtype
) -> jax.Array:
    scale = jnp.sqrt(jnp.array(2.0 / fan_in, dtype=dtype))
    return jax.random.normal(key, shape=shape, dtype=dtype) * scale


def _class_template(label: jax.Array) -> jax.Array:
    rows = jnp.arange(IMAGE_SHAPE[0], dtype=jnp.float32)[:, None]
    columns = jnp.arange(IMAGE_SHAPE[1], dtype=jnp.float32)[None, :]
    label_float = label.astype(jnp.float32)

    row_center = 4.0 + (label_float % 5.0) * 5.0
    column_center = 7.0 + jnp.floor(label_float / 5.0) * 14.0
    gaussian = jnp.exp(
        -((rows - row_center) ** 2 + (columns - column_center) ** 2) / (2.0 * 2.5**2)
    )
    vertical_line = jnp.where(jnp.abs(columns - column_center) <= 1.0, 0.35, 0.0)
    horizontal_line = jnp.where(jnp.abs(rows - row_center) <= 1.0, 0.20, 0.0)
    image = jnp.clip(gaussian + vertical_line + horizontal_line, 0.0, 1.0)
    return image[..., None]


def _conv2d(images: jax.Array, kernel: jax.Array, bias: jax.Array) -> jax.Array:
    convolution = lax.conv_general_dilated(
        images,
        kernel,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return convolution + bias


def _average_pool2d(images: jax.Array, pool_size: int = 2) -> jax.Array:
    batch_size, height, width, channels = images.shape
    pooled_height = height // pool_size
    pooled_width = width // pool_size
    trimmed = images[:, : pooled_height * pool_size, : pooled_width * pool_size, :]
    windows = trimmed.reshape(
        (batch_size, pooled_height, pool_size, pooled_width, pool_size, channels)
    )
    return jnp.mean(windows, axis=(2, 4))


def _load_dataset_batches(
    *,
    dataset: str,
    data_dir: Path,
    total_steps: int,
    batch_size: int,
    seed: int,
) -> tuple[jax.Array, jax.Array]:
    num_examples = total_steps * batch_size
    if dataset == "synthetic":
        images, labels = make_synthetic_mnist_data(num_examples, seed=seed)
    else:
        _raise_missing_local_dataset(dataset=dataset, data_dir=data_dir)

    image_batches = images.reshape((total_steps, batch_size, *IMAGE_SHAPE))
    label_batches = labels.reshape((total_steps, batch_size))
    return image_batches, label_batches


def _raise_missing_local_dataset(dataset: str, data_dir: Path) -> None:
    expected_dir = data_dir / dataset
    msg = (
        f"{dataset!r} is reserved for future local-file dataset support. "
        f"No downloader is used by this benchmark. Expected local files under "
        f"{expected_dir}."
    )
    raise FileNotFoundError(msg)


def _validate_benchmark_args(
    *,
    dataset: str,
    steps: int,
    batch_size: int,
    learning_rate: float,
    warmup_steps: int,
    log_every: int,
) -> None:
    if dataset not in SUPPORTED_DATASETS:
        supported = ", ".join(SUPPORTED_DATASETS)
        msg = f"unsupported dataset {dataset!r}; choose one of: {supported}"
        raise ValueError(msg)
    if steps < 1:
        msg = "steps must be at least 1"
        raise ValueError(msg)
    if batch_size < 1:
        msg = "batch_size must be at least 1"
        raise ValueError(msg)
    if learning_rate <= 0:
        msg = "learning_rate must be positive"
        raise ValueError(msg)
    if warmup_steps < 0:
        msg = "warmup_steps must be non-negative"
        raise ValueError(msg)
    if log_every < 1:
        msg = "log_every must be at least 1"
        raise ValueError(msg)


def _block_training_outputs(
    params: ArrayTree, loss: jax.Array, accuracy: jax.Array
) -> None:
    first_param = params["conv1"]["kernel"]
    first_param.block_until_ready()
    loss.block_until_ready()
    accuracy.block_until_ready()


def _device_summaries() -> list[dict[str, Any]]:
    return [
        {
            "platform": device.platform,
            "device_kind": device.device_kind,
            "id": device.id,
            "repr": str(device),
        }
        for device in jax.devices()
    ]
