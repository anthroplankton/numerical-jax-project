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
import shlex
import subprocess
import sys
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
DEFAULT_TOP_K = 5
JAX_PLATFORM_CHOICES = ("default", "cpu", "cuda", "tpu")
BATCH_SHARDING_CHOICES = ("none", "data")
DEFAULT_BATCH_SHARDING = "none"
DEFAULT_MESH_AXIS_NAME = "data"
DEFAULT_MIN_SHARD_DEVICES = 2
DEFAULT_PRIVATE_IMAGE_DIR = Path("data/local/demo2_vit_images")
PUBLIC_EXAMPLE_MANIFEST_PATH = Path("examples/assets/manifest.txt")
DATA_LOCAL_DIR = Path("data/local")
GIT_METADATA_TIMEOUT_SEC = 2.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the ViT inference benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Hugging Face model name. Default: {DEFAULT_MODEL_NAME}.",
    )
    image_source = parser.add_mutually_exclusive_group(required=True)
    image_source.add_argument(
        "--image",
        type=Path,
        help="Path to one local image file used as the benchmark input.",
    )
    image_source.add_argument(
        "--image-manifest",
        type=Path,
        help=(
            "Path to a local text manifest with one image path per line. "
            "Relative paths are resolved from the manifest directory."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of repeated images per inference step. Default: {DEFAULT_BATCH_SIZE}.",
    )
    parser.add_argument(
        "--batch-sharding",
        choices=BATCH_SHARDING_CHOICES,
        default=DEFAULT_BATCH_SHARDING,
        help=(
            "Batch-axis sharding mode. Use 'data' for explicit data sharding "
            f"over a named device mesh. Default: {DEFAULT_BATCH_SHARDING}."
        ),
    )
    parser.add_argument(
        "--mesh-axis-name",
        default=DEFAULT_MESH_AXIS_NAME,
        help=f"Device mesh axis name for --batch-sharding data. Default: {DEFAULT_MESH_AXIS_NAME}.",
    )
    parser.add_argument(
        "--require-multiple-devices",
        action="store_true",
        help=(
            "Fail unless at least --min-shard-devices JAX devices are visible. "
            "Intended for TPU/manual multi-device checks."
        ),
    )
    parser.add_argument(
        "--min-shard-devices",
        type=_positive_int,
        default=DEFAULT_MIN_SHARD_DEVICES,
        help=(
            "Minimum visible JAX devices required for --batch-sharding data or "
            f"--require-multiple-devices. Default: {DEFAULT_MIN_SHARD_DEVICES}."
        ),
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
        "--top-k",
        type=_positive_int,
        default=DEFAULT_TOP_K,
        help=f"Number of top predictions to include. Default: {DEFAULT_TOP_K}.",
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
    top_k: int = DEFAULT_TOP_K,
    batch_sharding: str = DEFAULT_BATCH_SHARDING,
    mesh_axis_name: str = DEFAULT_MESH_AXIS_NAME,
    require_multiple_devices: bool = False,
    min_shard_devices: int = DEFAULT_MIN_SHARD_DEVICES,
) -> dict[str, Any]:
    """Run local JAX/Flax ViT inference and return benchmark metrics."""
    return run_vit_inference_benchmarks(
        model_name=model_name,
        image_paths=[image_path],
        batch_size=batch_size,
        warmup_steps=warmup_steps,
        benchmark_steps=benchmark_steps,
        selected_jax_platform=selected_jax_platform,
        top_k=top_k,
        batch_sharding=batch_sharding,
        mesh_axis_name=mesh_axis_name,
        require_multiple_devices=require_multiple_devices,
        min_shard_devices=min_shard_devices,
    )


def run_vit_inference_benchmarks(
    *,
    model_name: str,
    image_paths: Sequence[Path],
    batch_size: int,
    warmup_steps: int,
    benchmark_steps: int,
    selected_jax_platform: str = DEFAULT_JAX_PLATFORM,
    top_k: int = DEFAULT_TOP_K,
    manifest_path: Path | None = None,
    batch_sharding: str = DEFAULT_BATCH_SHARDING,
    mesh_axis_name: str = DEFAULT_MESH_AXIS_NAME,
    require_multiple_devices: bool = False,
    min_shard_devices: int = DEFAULT_MIN_SHARD_DEVICES,
) -> dict[str, Any]:
    """Run local JAX/Flax ViT inference for one or more image paths."""
    if not image_paths:
        msg = "at least one image path is required"
        raise ValueError(msg)
    if manifest_path is None and len(image_paths) != 1:
        msg = "multiple image paths require manifest_path so output mode is explicit"
        raise ValueError(msg)

    missing_paths = [
        image_path for image_path in image_paths if not image_path.exists()
    ]
    if missing_paths:
        missing = ", ".join(str(image_path) for image_path in missing_paths)
        msg = f"image path does not exist: {missing}"
        raise FileNotFoundError(msg)

    apply_jax_platform(selected_jax_platform)

    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        msg = (
            "Missing pretrained demo dependencies. Install or run with the optional "
            "dependency group, for example: "
            "`uv run --group pretrained python examples/pretrained_vit_inference.py ...`"
        )
        raise RuntimeError(msg) from exc

    from jax_tpu_project.sharding import BatchShardingConfig, resolve_batch_sharding

    sharding_config = BatchShardingConfig(
        mode=batch_sharding,
        mesh_axis_name=mesh_axis_name,
        min_shard_devices=min_shard_devices,
        require_multiple_devices=require_multiple_devices,
    )
    resolved_sharding = resolve_batch_sharding(
        sharding_config,
        global_batch_size=batch_size,
        jax_module=jax,
    )

    try:
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
    params = model.params

    def inference_step_impl(model_params: Mapping[str, Any], inputs: Any) -> Any:
        return model(pixel_values=inputs, params=model_params, train=False).logits

    def build_inference_step(*, include_output_sharding: bool) -> Any:
        if not resolved_sharding.enabled:
            return jax.jit(inference_step_impl)

        jit_kwargs = {"in_shardings": (None, resolved_sharding.image_sharding)}
        if include_output_sharding:
            jit_kwargs["out_shardings"] = resolved_sharding.logits_sharding
        return jax.jit(inference_step_impl, **jit_kwargs)

    explicit_output_sharding = resolved_sharding.enabled
    resolved_sharding = resolved_sharding.with_jit_sharding_status(
        explicit_input_sharding=resolved_sharding.enabled,
        explicit_output_sharding=explicit_output_sharding,
    )
    inference_step = build_inference_step(
        include_output_sharding=explicit_output_sharding
    )

    def run_inference_step(model_params: Mapping[str, Any], inputs: Any) -> Any:
        nonlocal explicit_output_sharding, inference_step, resolved_sharding
        try:
            return inference_step(model_params, inputs)
        except Exception as exc:
            if not (resolved_sharding.enabled and explicit_output_sharding):
                raise
            explicit_output_sharding = False
            resolved_sharding = resolved_sharding.with_jit_sharding_status(
                explicit_input_sharding=True,
                explicit_output_sharding=False,
                fallback=(
                    "explicit logits out_shardings failed with "
                    f"{type(exc).__name__}; reran with explicit input sharding only"
                ),
            )
            inference_step = build_inference_step(include_output_sharding=False)
            return inference_step(model_params, inputs)

    if manifest_path is None:
        image_path = image_paths[0]
        with Image.open(image_path) as image:
            processed = image_processor(
                images=image.convert("RGB"), return_tensors="np"
            )

        pixel_values = jnp.asarray(processed["pixel_values"])
        batched_pixel_values = jnp.repeat(pixel_values, repeats=batch_size, axis=0)
        batched_pixel_values = resolved_sharding.shard_image_batch(
            batched_pixel_values,
            jax_module=jax,
        )

        logits = None
        for _ in range(warmup_steps):
            logits = run_inference_step(params, batched_pixel_values)
            logits.block_until_ready()

        step_times: list[float] = []
        for _ in range(benchmark_steps):
            start_time = time.perf_counter()
            logits = run_inference_step(params, batched_pixel_values)
            logits.block_until_ready()
            step_times.append(time.perf_counter() - start_time)

        if logits is None:
            msg = "benchmark did not produce logits"
            raise RuntimeError(msg)

        prediction = format_top_k_predictions_from_logits(
            logits=logits[0],
            top_k=top_k,
            id2label=model.config.id2label,
            jax_module=jax,
        )
        image_result = build_image_metrics(
            image_path=image_path,
            input_shape=batched_pixel_values.shape,
            batch_size=batch_size,
            benchmark_steps=benchmark_steps,
            step_times=step_times,
            prediction=prediction,
        )
        image_result["predicted_index"] = prediction["top1_index"]
        image_result["predicted_label"] = prediction["top1_label"]

        return build_run_metrics(
            model_name=model_name,
            selected_jax_platform=selected_jax_platform,
            backend=jax.default_backend(),
            devices=summarize_devices(jax.devices()),
            batch_size=batch_size,
            warmup_steps=warmup_steps,
            benchmark_steps=benchmark_steps,
            image_results=[image_result],
            manifest_path=None,
            sharding_metadata=resolved_sharding.metadata,
        )

    rgb_images = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            rgb_images.append(image.convert("RGB"))
    processed = image_processor(images=rgb_images, return_tensors="np")
    pixel_values = jnp.asarray(processed["pixel_values"])
    batch_specs = build_manifest_batch_specs(
        num_images=len(image_paths),
        batch_size=batch_size,
    )
    input_batches = [
        resolved_sharding.shard_image_batch(
            pad_manifest_batch(
                pixel_values[batch_spec["start_index"] : batch_spec["end_index"]],
                padding_count=batch_spec["padding_count"],
                jnp_module=jnp,
            ),
            jax_module=jax,
        )
        for batch_spec in batch_specs
    ]

    for _ in range(warmup_steps):
        for input_batch in input_batches:
            logits = run_inference_step(params, input_batch)
            logits.block_until_ready()

    batch_step_times: list[float] = []
    latest_batch_logits: list[Any] = []
    for _ in range(benchmark_steps):
        latest_batch_logits = []
        for input_batch in input_batches:
            start_time = time.perf_counter()
            logits = run_inference_step(params, input_batch)
            logits.block_until_ready()
            batch_step_times.append(time.perf_counter() - start_time)
            latest_batch_logits.append(logits)

    if not latest_batch_logits:
        msg = "benchmark did not produce logits"
        raise RuntimeError(msg)

    image_results = []
    for batch_index, (batch_spec, batch_logits) in enumerate(
        zip(batch_specs, latest_batch_logits, strict=True)
    ):
        for position_in_batch in range(batch_spec["real_size"]):
            image_index = batch_spec["start_index"] + position_in_batch
            prediction = format_top_k_predictions_from_logits(
                logits=batch_logits[position_in_batch],
                top_k=top_k,
                id2label=model.config.id2label,
                jax_module=jax,
            )
            image_result = build_manifest_image_metrics(
                image_path=image_paths[image_index],
                input_shape=pixel_values[image_index].shape,
                batch_index=batch_index,
                position_in_batch=position_in_batch,
                prediction=prediction,
            )
            image_result["predicted_index"] = prediction["top1_index"]
            image_result["predicted_label"] = prediction["top1_label"]
            image_results.append(image_result)

    return build_run_metrics(
        model_name=model_name,
        selected_jax_platform=selected_jax_platform,
        backend=jax.default_backend(),
        devices=summarize_devices(jax.devices()),
        batch_size=batch_size,
        warmup_steps=warmup_steps,
        benchmark_steps=benchmark_steps,
        image_results=image_results,
        manifest_path=manifest_path,
        input_shape=input_batches[0].shape,
        processing_mode="batched_manifest",
        num_batches=len(input_batches),
        num_padded_images=sum(
            batch_spec["padding_count"] for batch_spec in batch_specs
        ),
        last_batch_policy="pad_with_last_image",
        step_times=batch_step_times,
        sharding_metadata=resolved_sharding.metadata,
    )


def resolve_image_paths(
    image_path: Path | None, image_manifest: Path | None
) -> list[Path]:
    """Resolve a single image or local manifest into benchmark image paths."""
    if image_path is not None:
        return [image_path]
    if image_manifest is not None:
        return read_image_manifest(image_manifest)
    msg = "either image_path or image_manifest is required"
    raise ValueError(msg)


def read_image_manifest(manifest_path: Path) -> list[Path]:
    """Read a local image manifest with one image path per non-comment line."""
    manifest_dir = manifest_path.parent
    image_paths = [
        resolved_path
        for line in manifest_path.read_text().splitlines()
        if (resolved_path := resolve_manifest_line(line, manifest_dir)) is not None
    ]
    if not image_paths:
        msg = f"image manifest contains no image paths: {manifest_path}"
        raise ValueError(msg)
    return image_paths


def resolve_manifest_line(line: str, manifest_dir: Path) -> Path | None:
    """Resolve one image-manifest line without opening the image."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    image_path = Path(stripped).expanduser()
    if not image_path.is_absolute():
        image_path = manifest_dir / image_path
    return image_path


def private_image_manifest_example_path() -> Path:
    """Return the recommended local-only manifest path for private live demos."""
    return DEFAULT_PRIVATE_IMAGE_DIR / "manifest.txt"


def classify_manifest_path(manifest_path: Path) -> str:
    """Classify a manifest path for report metadata without reading images."""
    repo_relative_path = to_repo_relative_path(manifest_path)
    if repo_relative_path == PUBLIC_EXAMPLE_MANIFEST_PATH:
        return "public_example"
    if path_is_relative_to(repo_relative_path, DATA_LOCAL_DIR):
        return "local_private"
    return "local_manifest"


def to_repo_relative_path(path: Path) -> Path:
    """Return a repository-relative path when the path is inside the checkout."""
    try:
        return path.resolve().relative_to(Path.cwd().resolve())
    except (OSError, ValueError):
        return path


def path_is_relative_to(path: Path, prefix: Path) -> bool:
    """Return whether path is below prefix without raising ValueError."""
    try:
        path.relative_to(prefix)
    except ValueError:
        return False
    return True


def build_manifest_batch_specs(
    *, num_images: int, batch_size: int
) -> list[dict[str, int]]:
    """Return batch slices and final-batch padding counts for manifest mode."""
    if num_images < 1:
        msg = "num_images must be at least 1"
        raise ValueError(msg)
    if batch_size < 1:
        msg = "batch_size must be at least 1"
        raise ValueError(msg)

    batch_specs = []
    for start_index in range(0, num_images, batch_size):
        end_index = min(start_index + batch_size, num_images)
        real_size = end_index - start_index
        batch_specs.append(
            {
                "start_index": start_index,
                "end_index": end_index,
                "real_size": real_size,
                "padding_count": batch_size - real_size,
            }
        )
    return batch_specs


def pad_manifest_batch(
    pixel_values: Any, *, padding_count: int, jnp_module: Any
) -> Any:
    """Pad a final manifest batch to the configured batch size."""
    if padding_count == 0:
        return pixel_values
    if padding_count < 0:
        msg = "padding_count must be non-negative"
        raise ValueError(msg)
    padding_values = jnp_module.repeat(pixel_values[-1:], repeats=padding_count, axis=0)
    return jnp_module.concatenate([pixel_values, padding_values], axis=0)


def build_image_metrics(
    *,
    image_path: Path,
    input_shape: Sequence[int],
    batch_size: int,
    benchmark_steps: int,
    step_times: Sequence[float],
    prediction: Mapping[str, Any],
) -> dict[str, Any]:
    """Build JSON-serializable metrics for one image benchmark run."""
    mean_step_time = fmean(step_times)
    total_timed_inference = sum(step_times)
    throughput_counted_images = batch_size * benchmark_steps
    return {
        "image_path": str(image_path),
        "input_shape": [int(dimension) for dimension in input_shape],
        "batch_size": batch_size,
        "benchmark_steps": benchmark_steps,
        "mean_step_time_sec": mean_step_time,
        "total_timed_inference_sec": total_timed_inference,
        "throughput_counted_images": throughput_counted_images,
        "throughput_images_per_sec": throughput_counted_images / total_timed_inference,
        **dict(prediction),
    }


def build_manifest_image_metrics(
    *,
    image_path: Path,
    input_shape: Sequence[int],
    batch_index: int,
    position_in_batch: int,
    prediction: Mapping[str, Any],
) -> dict[str, Any]:
    """Build JSON-serializable prediction metrics for one manifest image."""
    return {
        "image_path": str(image_path),
        "input_shape": [int(dimension) for dimension in input_shape],
        "batch_index": batch_index,
        "position_in_batch": position_in_batch,
        **dict(prediction),
    }


def build_run_metrics(
    *,
    model_name: str,
    selected_jax_platform: str,
    backend: str,
    devices: Sequence[Mapping[str, Any]],
    batch_size: int,
    warmup_steps: int,
    benchmark_steps: int,
    image_results: Sequence[Mapping[str, Any]],
    manifest_path: Path | None,
    input_shape: Sequence[int] | None = None,
    processing_mode: str | None = None,
    num_batches: int | None = None,
    num_padded_images: int = 0,
    last_batch_policy: str | None = None,
    step_times: Sequence[float] | None = None,
    sharding_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the final JSON payload for single-image or manifest mode."""
    if not image_results:
        msg = "image_results must not be empty"
        raise ValueError(msg)

    common = {
        "model_name": model_name,
        "selected_jax_platform": selected_jax_platform,
        "backend": backend,
        "devices": list(devices),
        "batch_size": batch_size,
        "warmup_steps": warmup_steps,
        "benchmark_steps": benchmark_steps,
        "sharding": dict(sharding_metadata or {}),
    }
    if manifest_path is None:
        return {
            "mode": "single_image",
            "processing_mode": "repeated_single_image",
            "num_images": 1,
            "num_batches": 1,
            "timed_batch_runs": benchmark_steps,
            "num_padded_images": 0,
            "last_batch_policy": "none",
            **common,
            **dict(image_results[0]),
        }

    if input_shape is None:
        msg = "input_shape is required for manifest metrics"
        raise ValueError(msg)
    if processing_mode is None:
        msg = "processing_mode is required for manifest metrics"
        raise ValueError(msg)
    if num_batches is None:
        msg = "num_batches is required for manifest metrics"
        raise ValueError(msg)
    if step_times is None:
        msg = "step_times are required for manifest metrics"
        raise ValueError(msg)

    total_timed_inference = sum(step_times)
    num_images = len(image_results)
    timed_batch_runs = len(step_times)
    throughput_counted_images = num_images * benchmark_steps
    return {
        "mode": "image_manifest",
        **common,
        "manifest_path": str(manifest_path),
        "manifest_kind": classify_manifest_path(manifest_path),
        "input_shape": [int(dimension) for dimension in input_shape],
        "processing_mode": processing_mode,
        "num_images": num_images,
        "num_batches": num_batches,
        "timed_batch_runs": timed_batch_runs,
        "num_padded_images": num_padded_images,
        "last_batch_policy": last_batch_policy,
        "mean_step_time_sec": fmean(step_times),
        "total_timed_inference_sec": total_timed_inference,
        "throughput_counted_images": throughput_counted_images,
        "throughput_images_per_sec": throughput_counted_images / total_timed_inference,
        "image_results": list(image_results),
    }


def format_top_k_predictions_from_logits(
    *,
    logits: Any,
    top_k: int,
    id2label: Mapping[Any, str],
    jax_module: Any,
) -> dict[str, Any]:
    """Format top-k predictions from one logits row."""
    top_count = min(top_k, int(logits.shape[-1]))
    probabilities = jax_module.nn.softmax(logits)
    top_scores, top_indices = jax_module.lax.top_k(probabilities, top_count)
    return format_top_k_predictions(
        indices=[int(index) for index in top_indices.tolist()],
        scores=[float(score) for score in top_scores.tolist()],
        id2label=id2label,
    )


def format_top_k_predictions(
    *,
    indices: Sequence[int],
    scores: Sequence[float],
    id2label: Mapping[Any, str],
) -> dict[str, Any]:
    """Format top-k prediction indices and scores as JSON-serializable fields."""
    if not indices:
        msg = "at least one prediction index is required"
        raise ValueError(msg)
    top_predictions = [
        {
            "index": int(index),
            "label": lookup_label(id2label, int(index)),
            "score": float(score),
        }
        for index, score in zip(indices, scores, strict=True)
    ]
    top1 = top_predictions[0]
    return {
        "top1_index": top1["index"],
        "top1_label": top1["label"],
        "top5": top_predictions,
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


def collect_git_metadata(cwd: Path | None = None) -> dict[str, Any]:
    """Collect privacy-safe Git metadata for the current checkout if available."""
    workdir = Path.cwd() if cwd is None else cwd
    commit_output = _run_git_command(["rev-parse", "HEAD"], cwd=workdir)
    commit = _nonempty_stripped_output(commit_output)
    branch_output = _run_git_command(
        ["symbolic-ref", "--quiet", "--short", "HEAD"],
        cwd=workdir,
    )
    dirty_output = _run_git_command(["status", "--short"], cwd=workdir)

    return {
        "git_commit": commit,
        "git_branch": _nonempty_stripped_output(branch_output),
        "git_dirty": None if dirty_output is None else bool(dirty_output.strip()),
    }


def _run_git_command(args: Sequence[str], *, cwd: Path) -> str | None:
    """Run one Git command, returning stdout on success and None on failure."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            shell=False,
            timeout=GIT_METADATA_TIMEOUT_SEC,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return None
    return completed.stdout


def _nonempty_stripped_output(output: str | None) -> str | None:
    """Return stripped command output, or None for missing/empty output."""
    if output is None:
        return None
    value = output.strip()
    return value or None


def add_cli_run_metadata(
    metrics: Mapping[str, Any],
    *,
    argv: Sequence[str],
    output_path: Path,
    git_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Add reproducibility metadata available only at CLI execution time."""
    resolved_git_metadata = (
        collect_git_metadata() if git_metadata is None else dict(git_metadata)
    )
    return {
        **dict(metrics),
        "command_used": shlex.join(["python", *argv]),
        "output_path": str(output_path),
        **resolved_git_metadata,
    }


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
    image_paths = resolve_image_paths(args.image, args.image_manifest)
    metrics = run_vit_inference_benchmarks(
        model_name=args.model,
        image_paths=image_paths,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        benchmark_steps=args.benchmark_steps,
        selected_jax_platform=args.jax_platform,
        top_k=args.top_k,
        manifest_path=args.image_manifest,
        batch_sharding=args.batch_sharding,
        mesh_axis_name=args.mesh_axis_name,
        require_multiple_devices=args.require_multiple_devices,
        min_shard_devices=args.min_shard_devices,
    )
    metrics = add_cli_run_metadata(
        metrics,
        argv=sys.argv,
        output_path=args.output,
    )
    write_metrics(metrics, args.output)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"metrics: {args.output}")


if __name__ == "__main__":
    main()
