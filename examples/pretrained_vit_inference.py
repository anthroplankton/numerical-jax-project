"""Benchmark pretrained ViT inference with JAX/Flax.

Usage:
    uv run --group pretrained python examples/pretrained_vit_inference.py \
      --jax-platform cpu \
      --image examples/assets/chihuahua_pet_licorice.jpg \
      --output runs/vit-inference/metrics.json

The first run downloads pretrained model weights and image-processor files from
Hugging Face unless they already exist in the local cache.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

DEFAULT_MODEL_NAME = "google/vit-base-patch16-224"
DEFAULT_OUTPUT_PATH = Path("runs/vit-inference/metrics.json")
DEFAULT_BATCH_SIZE = 1
DEFAULT_WARMUP_STEPS = 1
DEFAULT_BENCHMARK_STEPS = 5
DEFAULT_JAX_PLATFORM = "cpu"
JAX_PLATFORM_CHOICES = ("default", "cpu", "cuda", "tpu")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the ViT inference benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Hugging Face model name. Default: {DEFAULT_MODEL_NAME}.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to one local image file used as the benchmark input.",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of repeated images per inference step. Default: {DEFAULT_BATCH_SIZE}.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=_nonnegative_int,
        default=DEFAULT_WARMUP_STEPS,
        help=(
            "Number of untimed inference steps before benchmarking. "
            f"Default: {DEFAULT_WARMUP_STEPS}."
        ),
    )
    parser.add_argument(
        "--benchmark-steps",
        type=_positive_int,
        default=DEFAULT_BENCHMARK_STEPS,
        help=(f"Number of timed inference steps. Default: {DEFAULT_BENCHMARK_STEPS}."),
    )
    parser.add_argument(
        "--jax-platform",
        choices=JAX_PLATFORM_CHOICES,
        default=DEFAULT_JAX_PLATFORM,
        help=(
            "JAX platform request. Use 'default' to leave JAX_PLATFORMS unset. "
            f"Default: {DEFAULT_JAX_PLATFORM}."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path for JSON metrics. Default: {DEFAULT_OUTPUT_PATH}.",
    )
    return parser.parse_args(argv)


def run_vit_inference_benchmark(
    *,
    model_name: str,
    image_path: Path,
    batch_size: int,
    warmup_steps: int,
    benchmark_steps: int,
    selected_jax_platform: str = DEFAULT_JAX_PLATFORM,
) -> dict[str, Any]:
    """Run local JAX/Flax ViT inference and return benchmark metrics."""
    if not image_path.exists():
        msg = f"image path does not exist: {image_path}"
        raise FileNotFoundError(msg)

    apply_jax_platform(selected_jax_platform)

    try:
        import jax
        import jax.numpy as jnp
        from PIL import Image
        from transformers import AutoImageProcessor, FlaxViTForImageClassification
    except ImportError as exc:
        msg = (
            "Missing pretrained demo dependencies. Install or run with the optional "
            "dependency group, for example: "
            "`uv run --group pretrained python examples/pretrained_vit_inference.py ...`"
        )
        raise RuntimeError(msg) from exc

    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = FlaxViTForImageClassification.from_pretrained(model_name)

    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        processed = image_processor(images=rgb_image, return_tensors="np")

    pixel_values = jnp.asarray(processed["pixel_values"])
    batched_pixel_values = jnp.repeat(pixel_values, repeats=batch_size, axis=0)
    params = model.params

    @jax.jit
    def inference_step(model_params: Mapping[str, Any], inputs: Any) -> Any:
        return model(pixel_values=inputs, params=model_params, train=False).logits

    logits = None
    for _ in range(warmup_steps):
        logits = inference_step(params, batched_pixel_values)
        logits.block_until_ready()

    step_times: list[float] = []
    for _ in range(benchmark_steps):
        start_time = time.perf_counter()
        logits = inference_step(params, batched_pixel_values)
        logits.block_until_ready()
        step_times.append(time.perf_counter() - start_time)

    if logits is None:
        msg = "benchmark did not produce logits"
        raise RuntimeError(msg)

    predicted_index = int(jnp.argmax(logits[0]).item())
    predicted_label = lookup_label(model.config.id2label, predicted_index)

    return build_metrics(
        model_name=model_name,
        selected_jax_platform=selected_jax_platform,
        backend=jax.default_backend(),
        devices=summarize_devices(jax.devices()),
        input_shape=batched_pixel_values.shape,
        batch_size=batch_size,
        warmup_steps=warmup_steps,
        benchmark_steps=benchmark_steps,
        step_times=step_times,
        predicted_index=predicted_index,
        predicted_label=predicted_label,
    )


def build_metrics(
    *,
    model_name: str,
    selected_jax_platform: str,
    backend: str,
    devices: Sequence[Mapping[str, Any]],
    input_shape: Sequence[int],
    batch_size: int,
    warmup_steps: int,
    benchmark_steps: int,
    step_times: Sequence[float],
    predicted_index: int,
    predicted_label: str,
) -> dict[str, Any]:
    """Build the JSON-serializable metrics payload."""
    mean_step_time = fmean(step_times)
    return {
        "model_name": model_name,
        "selected_jax_platform": selected_jax_platform,
        "backend": backend,
        "devices": list(devices),
        "input_shape": [int(dimension) for dimension in input_shape],
        "batch_size": batch_size,
        "warmup_steps": warmup_steps,
        "benchmark_steps": benchmark_steps,
        "mean_step_time_sec": mean_step_time,
        "throughput_images_per_sec": batch_size / mean_step_time,
        "predicted_index": predicted_index,
        "predicted_label": predicted_label,
    }


def summarize_devices(devices: Sequence[Any]) -> list[dict[str, Any]]:
    """Return a JSON-serializable summary of JAX devices."""
    return [
        {
            "platform": device.platform,
            "device_kind": device.device_kind,
            "id": device.id,
            "repr": str(device),
        }
        for device in devices
    ]


def apply_jax_platform(selected_jax_platform: str) -> None:
    """Apply the requested JAX platform before JAX is imported."""
    if selected_jax_platform not in JAX_PLATFORM_CHOICES:
        choices = ", ".join(JAX_PLATFORM_CHOICES)
        msg = f"unsupported JAX platform {selected_jax_platform!r}; choose one of: {choices}"
        raise ValueError(msg)
    if selected_jax_platform != "default":
        os.environ["JAX_PLATFORMS"] = selected_jax_platform


def lookup_label(id2label: Mapping[Any, str], predicted_index: int) -> str:
    """Look up a class label from a Hugging Face id2label mapping."""
    return (
        id2label.get(predicted_index)
        or id2label.get(str(predicted_index))
        or str(predicted_index)
    )


def write_metrics(metrics: Mapping[str, Any], output_path: Path) -> None:
    """Write benchmark metrics as formatted JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")


def _positive_int(raw_value: str) -> int:
    value = int(raw_value)
    if value < 1:
        msg = "must be at least 1"
        raise argparse.ArgumentTypeError(msg)
    return value


def _nonnegative_int(raw_value: str) -> int:
    value = int(raw_value)
    if value < 0:
        msg = "must be non-negative"
        raise argparse.ArgumentTypeError(msg)
    return value


def main() -> None:
    """Run the CLI entry point."""
    args = parse_args()
    metrics = run_vit_inference_benchmark(
        model_name=args.model,
        image_path=args.image,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        benchmark_steps=args.benchmark_steps,
        selected_jax_platform=args.jax_platform,
    )
    write_metrics(metrics, args.output)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"metrics: {args.output}")


if __name__ == "__main__":
    main()
