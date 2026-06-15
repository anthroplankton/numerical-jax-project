"""Run a small Demo 2 ViT classifier-head fine-tuning smoke workflow.

Usage:
    uv run --group pretrained --group training python \
      examples/demo2_pretrained_vit_finetune.py \
      --jax-platform cpu \
      --train-manifest data/local/imagenette2-320/train/manifest_train_64.txt \
      --eval-manifest data/local/imagenette2-320/val/manifest_val_64.txt

This extension keeps the pretrained ViT backbone frozen, trains only the
classifier head, and stores only the head parameters, optimizer state, current
step, and minimal metadata in Orbax checkpoints. It is a smoke workflow for
JAX/TPU training, checkpointing, and resume behavior, not an accuracy benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

import numpy as np

from pretrained_vit_inference import (
    BATCH_SHARDING_CHOICES,
    DEFAULT_BATCH_SHARDING,
    DEFAULT_JAX_PLATFORM,
    DEFAULT_MESH_AXIS_NAME,
    DEFAULT_MIN_SHARD_DEVICES,
    DEFAULT_MODEL_NAME,
    DEFAULT_TOP_K,
    JAX_PLATFORM_CHOICES,
    apply_jax_platform,
    build_manifest_batch_specs,
    collect_git_metadata,
    format_top_k_predictions_from_logits,
    pad_manifest_batch,
    summarize_devices,
    write_metrics,
)

MODE = "demo2_vit_head_finetune"
TRAINABLE_SCOPE = "classifier_head_only"
FROZEN_SCOPE = "vit_backbone"
DEFAULT_OUTPUT_DIR = Path("runs/vit-finetune/demo2_vit_head_finetune")
DEFAULT_CHECKPOINT_DIR = DEFAULT_OUTPUT_DIR / "checkpoints"
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_MAX_STEPS = 20
DEFAULT_MIN_TRAIN_SECONDS = 0.0
DEFAULT_CHECKPOINT_EVERY_STEPS = 10
DEFAULT_CHECKPOINT_EVERY_SECONDS = 30.0
DEFAULT_EVAL_EVERY_STEPS = 0
DEFAULT_SEED = 0
CHECKPOINT_METADATA_BYTES = 8192

IMAGENETTE_LABEL_TO_IMAGENET_INDEX = {
    "n01440764": 0,  # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}

STOP_REQUESTED = False
STOP_SIGNAL_NAME: str | None = None


@dataclass(frozen=True)
class LabeledImage:
    """One image manifest entry with an ImageNet class index."""

    image_path: Path
    label_id: str
    label_index: int


@dataclass(frozen=True)
class TrainingBatch:
    """One padded training or evaluation batch."""

    pixel_values: Any
    labels: Any
    mask: Any
    start_index: int
    real_size: int
    padding_count: int


@dataclass(frozen=True)
class HeadTrainState:
    """Trainable head state kept outside the frozen ViT backbone."""

    head_params: Any
    optimizer_state: Any
    step: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the Demo 2 fine-tuning extension."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Hugging Face model name. Default: {DEFAULT_MODEL_NAME}.",
    )
    parser.add_argument(
        "--train-manifest",
        required=True,
        type=Path,
        help="Path-only Imagenette training manifest with class directories.",
    )
    parser.add_argument(
        "--eval-manifest",
        required=True,
        type=Path,
        help="Path-only Imagenette eval manifest with class directories.",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Training batch size. Default: {DEFAULT_BATCH_SIZE}.",
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
        help=(
            "Device mesh axis name for --batch-sharding data. "
            f"Default: {DEFAULT_MESH_AXIS_NAME}."
        ),
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
        "--learning-rate",
        type=_positive_float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Classifier-head learning rate. Default: {DEFAULT_LEARNING_RATE}.",
    )
    parser.add_argument(
        "--max-steps",
        type=_positive_int,
        default=DEFAULT_MAX_STEPS,
        help=(
            "Maximum optimizer steps. With --min-train-seconds > 0 this is an "
            "upper bound for a time-controlled smoke run. "
            f"Default: {DEFAULT_MAX_STEPS}."
        ),
    )
    parser.add_argument(
        "--min-train-seconds",
        type=_nonnegative_float,
        default=DEFAULT_MIN_TRAIN_SECONDS,
        help=(
            "Minimum training-loop duration for time-controlled smoke runs. "
            f"Default: {DEFAULT_MIN_TRAIN_SECONDS}."
        ),
    )
    parser.add_argument(
        "--checkpoint-every-steps",
        type=_nonnegative_int,
        default=DEFAULT_CHECKPOINT_EVERY_STEPS,
        help=(
            "Save an Orbax checkpoint every N completed steps; 0 disables. "
            f"Default: {DEFAULT_CHECKPOINT_EVERY_STEPS}."
        ),
    )
    parser.add_argument(
        "--checkpoint-every-seconds",
        type=_nonnegative_float,
        default=DEFAULT_CHECKPOINT_EVERY_SECONDS,
        help=(
            "Save an Orbax checkpoint after N seconds since the previous save; "
            f"0 disables. Default: {DEFAULT_CHECKPOINT_EVERY_SECONDS}."
        ),
    )
    parser.add_argument(
        "--eval-every-steps",
        type=_nonnegative_int,
        default=DEFAULT_EVAL_EVERY_STEPS,
        help=(
            "Write eval_metrics.csv every N completed steps; 0 disables "
            "extra periodic evals. Initial and final eval rows are still "
            f"written. Default: {DEFAULT_EVAL_EVERY_STEPS}."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help=f"Orbax checkpoint directory. Default: {DEFAULT_CHECKPOINT_DIR}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for summary and metric artifacts. Default: {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest Orbax checkpoint in --checkpoint-dir.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Write predictions_before.json and predictions_after.json.",
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
        "--reinit-head",
        action="store_true",
        help=(
            "Randomly reinitialize only the classifier head before training. "
            "The frozen ViT backbone and pretrained default path are unchanged."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for --reinit-head. Default: {DEFAULT_SEED}.",
    )
    parser.add_argument(
        "--top-k",
        type=_positive_int,
        default=DEFAULT_TOP_K,
        help=f"Number of top predictions to save. Default: {DEFAULT_TOP_K}.",
    )
    return parser.parse_args(argv)


def read_labeled_manifest(manifest_path: Path) -> list[LabeledImage]:
    """Read a path-only Imagenette manifest and derive labels from directories."""
    manifest_dir = manifest_path.parent
    entries = [
        entry
        for line in manifest_path.read_text().splitlines()
        if (entry := resolve_labeled_manifest_line(line, manifest_dir)) is not None
    ]
    if not entries:
        msg = f"image manifest contains no image paths: {manifest_path}"
        raise ValueError(msg)
    return entries


def resolve_labeled_manifest_line(line: str, manifest_dir: Path) -> LabeledImage | None:
    """Resolve one path-only manifest line into a labeled image entry."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    image_path = Path(stripped).expanduser()
    if not image_path.is_absolute():
        image_path = manifest_dir / image_path
    label_id = image_path.parent.name
    try:
        label_index = IMAGENETTE_LABEL_TO_IMAGENET_INDEX[label_id]
    except KeyError as exc:
        known = ", ".join(sorted(IMAGENETTE_LABEL_TO_IMAGENET_INDEX))
        msg = (
            f"unsupported Imagenette label directory {label_id!r} for {image_path}; "
            f"expected one of: {known}"
        )
        raise ValueError(msg) from exc
    return LabeledImage(
        image_path=image_path,
        label_id=label_id,
        label_index=label_index,
    )


def validate_manifest_paths(entries: Sequence[LabeledImage]) -> None:
    """Fail before model loading if any manifest image path is missing."""
    missing_paths = [
        entry.image_path for entry in entries if not entry.image_path.exists()
    ]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        msg = f"image path does not exist: {missing}"
        raise FileNotFoundError(msg)


def count_labels(entries: Sequence[LabeledImage]) -> dict[str, int]:
    """Return deterministic label-id counts for manifest-skew analysis."""
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.label_id] = counts.get(entry.label_id, 0) + 1
    return dict(sorted(counts.items()))


def pad_label_batch(labels: Any, *, padding_count: int, jnp_module: Any) -> Any:
    """Pad labels to match a padded image batch."""
    if padding_count == 0:
        return labels
    if padding_count < 0:
        msg = "padding_count must be non-negative"
        raise ValueError(msg)
    padding_values = jnp_module.repeat(labels[-1:], repeats=padding_count, axis=0)
    return jnp_module.concatenate([labels, padding_values], axis=0)


def build_batch_mask(*, real_size: int, padding_count: int, jnp_module: Any) -> Any:
    """Return a 1D mask where real examples are 1 and padded examples are 0."""
    if real_size < 1:
        msg = "real_size must be at least 1"
        raise ValueError(msg)
    if padding_count < 0:
        msg = "padding_count must be non-negative"
        raise ValueError(msg)
    real_values = jnp_module.ones((real_size,), dtype=jnp_module.float32)
    if padding_count == 0:
        return real_values
    padding_values = jnp_module.zeros((padding_count,), dtype=jnp_module.float32)
    return jnp_module.concatenate([real_values, padding_values], axis=0)


def should_continue_training(
    *, step: int, max_steps: int, elapsed_sec: float, min_train_seconds: float
) -> bool:
    """Return whether the smoke training loop should continue."""
    if step >= max_steps:
        return False
    if min_train_seconds > 0:
        return elapsed_sec < min_train_seconds
    return True


def should_save_checkpoint(
    *,
    step: int,
    now: float,
    last_checkpoint_time: float,
    checkpoint_every_steps: int,
    checkpoint_every_seconds: float,
    stop_requested: bool = False,
) -> bool:
    """Return whether periodic or emergency checkpointing should run."""
    if step < 1:
        return False
    if stop_requested:
        return True
    if checkpoint_every_steps > 0 and step % checkpoint_every_steps == 0:
        return True
    return (
        checkpoint_every_seconds > 0
        and now - last_checkpoint_time >= checkpoint_every_seconds
    )


def should_run_periodic_eval(
    *, step: int, start_step: int, eval_every_steps: int
) -> bool:
    """Return whether an extra eval row should be written for this step."""
    return eval_every_steps > 0 and step > start_step and step % eval_every_steps == 0


def reinitialize_classifier_head(
    head_params: Any,
    *,
    seed: int,
    jax_module: Any,
    jnp_module: Any,
) -> Any:
    """Reinitialize only classifier-head leaves for learning-curve smoke plots."""
    leaves, treedef = jax_module.tree_util.tree_flatten(head_params)
    key = jax_module.random.PRNGKey(seed)
    new_leaves = []
    for leaf in leaves:
        key, subkey = jax_module.random.split(key)
        if getattr(leaf, "ndim", 0) >= 2:
            values = 0.02 * jax_module.random.truncated_normal(
                subkey,
                lower=-2.0,
                upper=2.0,
                shape=leaf.shape,
                dtype=leaf.dtype,
            )
            new_leaves.append(values)
        else:
            new_leaves.append(jnp_module.zeros_like(leaf))
    return jax_module.tree_util.tree_unflatten(treedef, new_leaves)


def encode_checkpoint_metadata(metadata: Mapping[str, Any]) -> np.ndarray:
    """Encode metadata into a fixed-size uint8 array for Orbax PyTree storage."""
    payload = json.dumps(dict(metadata), sort_keys=True).encode("utf-8")
    if len(payload) > CHECKPOINT_METADATA_BYTES:
        msg = (
            "checkpoint metadata is too large for the fixed-size metadata field: "
            f"{len(payload)} > {CHECKPOINT_METADATA_BYTES}"
        )
        raise ValueError(msg)
    encoded = np.zeros((CHECKPOINT_METADATA_BYTES,), dtype=np.uint8)
    encoded[: len(payload)] = np.frombuffer(payload, dtype=np.uint8)
    return encoded


def decode_checkpoint_metadata(encoded: Any) -> dict[str, Any]:
    """Decode metadata written by encode_checkpoint_metadata."""
    data = np.asarray(encoded, dtype=np.uint8)
    zero_positions = np.flatnonzero(data == 0)
    end = int(zero_positions[0]) if len(zero_positions) else len(data)
    if end == 0:
        return {}
    return json.loads(data[:end].tobytes().decode("utf-8"))


def validate_restored_checkpoint_metadata(
    restored_metadata: Mapping[str, Any],
    expected_metadata: Mapping[str, Any],
) -> None:
    """Validate checkpoint identity fields before resuming training."""
    checked_fields = ("mode", "model_name", "trainable_scope", "frozen_scope")
    mismatches = [
        field
        for field in checked_fields
        if restored_metadata.get(field) != expected_metadata.get(field)
    ]
    if mismatches:
        details = ", ".join(
            (
                f"{field}: restored={restored_metadata.get(field)!r}, "
                f"expected={expected_metadata.get(field)!r}"
            )
            for field in mismatches
        )
        msg = f"checkpoint metadata mismatch: {details}"
        raise ValueError(msg)


def build_checkpoint_metadata(
    *,
    model_name: str,
    train_manifest: Path,
    eval_manifest: Path,
    batch_size: int,
    learning_rate: float,
    eval_every_steps: int,
    reinit_head: bool,
    seed: int,
    label_ids: Sequence[str],
    git_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the minimal metadata stored with classifier-head checkpoints."""
    return {
        "mode": MODE,
        "model_name": model_name,
        "trainable_scope": TRAINABLE_SCOPE,
        "frozen_scope": FROZEN_SCOPE,
        "train_manifest": str(train_manifest),
        "eval_manifest": str(eval_manifest),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "eval_every_steps": eval_every_steps,
        "reinit_head": reinit_head,
        "seed": seed,
        "label_ids": list(label_ids),
        **dict(git_metadata),
    }


def build_checkpoint_item(
    *,
    head_params: Any,
    optimizer_state: Any,
    step: int,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the Orbax-managed checkpoint payload."""
    return {
        "head_params": head_params,
        "optimizer_state": optimizer_state,
        "step": np.asarray(step, dtype=np.int64),
        "metadata_json": encode_checkpoint_metadata(metadata),
    }


def checkpoint_contains_allowed_keys(item: Mapping[str, Any]) -> bool:
    """Return whether a checkpoint payload excludes frozen backbone state."""
    return set(item) == {"head_params", "optimizer_state", "step", "metadata_json"}


def install_signal_handlers() -> None:
    """Install a minimal SIGTERM handler for emergency checkpointing."""

    def request_stop(signum: int, _frame: Any) -> None:
        global STOP_REQUESTED, STOP_SIGNAL_NAME
        STOP_REQUESTED = True
        STOP_SIGNAL_NAME = signal.Signals(signum).name

    signal.signal(signal.SIGTERM, request_stop)


def reset_signal_state() -> None:
    """Clear in-process signal state before a new fine-tuning run."""
    global STOP_REQUESTED, STOP_SIGNAL_NAME
    STOP_REQUESTED = False
    STOP_SIGNAL_NAME = None


def resolve_run_paths(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve run artifact directories before Orbax checkpoint creation."""
    args.output_dir = args.output_dir.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    return args


def create_checkpoint_manager(checkpoint_dir: Path) -> Any:
    """Create an Orbax CheckpointManager for the head-only state."""
    from orbax import checkpoint as ocp

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    try:
        options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
    except TypeError:
        options = ocp.CheckpointManagerOptions(max_to_keep=3)
    return ocp.CheckpointManager(checkpoint_dir, options=options)


def save_training_checkpoint(
    manager: Any,
    *,
    step: int,
    head_params: Any,
    optimizer_state: Any,
    metadata: Mapping[str, Any],
) -> int:
    """Save one Orbax checkpoint and wait for async writes to finish."""
    from orbax import checkpoint as ocp

    item = build_checkpoint_item(
        head_params=head_params,
        optimizer_state=optimizer_state,
        step=step,
        metadata=metadata,
    )
    try:
        manager.save(step, args=ocp.args.StandardSave(item))
    except TypeError:
        manager.save(step, item)
    manager.wait_until_finished()
    return step


def restore_latest_checkpoint(
    manager: Any,
    *,
    target: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Restore the latest Orbax checkpoint into a reference PyTree."""
    latest_step = manager.latest_step()
    if latest_step is None:
        return None

    from orbax import checkpoint as ocp

    try:
        restored = manager.restore(latest_step, args=ocp.args.StandardRestore(target))
    except TypeError:
        restored = manager.restore(latest_step, items=target)
    return dict(restored)


def load_pixel_values(entries: Sequence[LabeledImage], image_processor: Any) -> Any:
    """Load and preprocess manifest images for the ViT model."""
    from PIL import Image

    rgb_images = []
    for entry in entries:
        with Image.open(entry.image_path) as image:
            rgb_images.append(image.convert("RGB"))
    processed = image_processor(images=rgb_images, return_tensors="np")
    return processed["pixel_values"]


def build_batches(
    *,
    pixel_values: Any,
    labels: Any,
    batch_size: int,
    jnp_module: Any,
) -> list[TrainingBatch]:
    """Build padded image, label, and mask batches."""
    specs = build_manifest_batch_specs(
        num_images=int(labels.shape[0]),
        batch_size=batch_size,
    )
    batches = []
    for spec in specs:
        batch_pixels = pixel_values[spec["start_index"] : spec["end_index"]]
        batch_labels = labels[spec["start_index"] : spec["end_index"]]
        batches.append(
            TrainingBatch(
                pixel_values=pad_manifest_batch(
                    batch_pixels,
                    padding_count=spec["padding_count"],
                    jnp_module=jnp_module,
                ),
                labels=pad_label_batch(
                    batch_labels,
                    padding_count=spec["padding_count"],
                    jnp_module=jnp_module,
                ),
                mask=build_batch_mask(
                    real_size=spec["real_size"],
                    padding_count=spec["padding_count"],
                    jnp_module=jnp_module,
                ),
                start_index=spec["start_index"],
                real_size=spec["real_size"],
                padding_count=spec["padding_count"],
            )
        )
    return batches


def shard_training_batches(
    batches: Sequence[TrainingBatch],
    *,
    resolved_sharding: Any,
    jax_module: Any,
) -> list[TrainingBatch]:
    """Place padded training/eval batch arrays on resolved batch shardings."""
    return [
        TrainingBatch(
            pixel_values=resolved_sharding.shard_image_batch(
                batch.pixel_values,
                jax_module=jax_module,
            ),
            labels=resolved_sharding.shard_label_batch(
                batch.labels,
                jax_module=jax_module,
            ),
            mask=resolved_sharding.shard_mask_batch(
                batch.mask,
                jax_module=jax_module,
            ),
            start_index=batch.start_index,
            real_size=batch.real_size,
            padding_count=batch.padding_count,
        )
        for batch in batches
    ]


def write_metrics_header(metrics_path: Path) -> None:
    """Create the per-step CSV metric artifact."""
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "step",
                "loss",
                "accuracy",
                "step_time_sec",
                "examples_per_second",
                "checkpoint_saved",
            ]
        )


def append_metrics_row(metrics_path: Path, row: Mapping[str, Any]) -> None:
    """Append one per-step metric row."""
    with metrics_path.open("a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                row["step"],
                row["loss"],
                row["accuracy"],
                row["step_time_sec"],
                row["examples_per_second"],
                row["checkpoint_saved"],
            ]
        )


def write_eval_metrics_header(metrics_path: Path) -> None:
    """Create the periodic eval CSV metric artifact."""
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "eval_loss", "eval_accuracy"])


def append_eval_metrics_row(
    metrics_path: Path,
    *,
    step: int,
    metrics: Mapping[str, float],
) -> None:
    """Append one eval metric row."""
    with metrics_path.open("a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([step, metrics["loss"], metrics["accuracy"]])


def append_log(output_dir: Path, message: str) -> None:
    """Append one line to train.log."""
    log_path = output_dir / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as file:
        file.write(message.rstrip() + "\n")


def build_summary(
    *,
    model_name: str,
    selected_jax_platform: str,
    backend: str,
    devices: Sequence[Mapping[str, Any]],
    train_manifest: Path,
    eval_manifest: Path,
    train_examples: int,
    eval_examples: int,
    train_label_counts: Mapping[str, int],
    eval_label_counts: Mapping[str, int],
    batch_size: int,
    learning_rate: float,
    eval_every_steps: int,
    reinit_head: bool,
    seed: int,
    start_step: int,
    final_step: int,
    resumed_from_checkpoint: bool,
    checkpoint_dir: Path,
    latest_checkpoint_step: int | None,
    initial_loss: float,
    final_loss: float,
    step_times: Sequence[float],
    total_train_examples: int,
    total_runtime_sec: float,
    interrupted: bool,
    stop_signal_name: str | None,
    git_metadata: Mapping[str, Any],
    sharding_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the run summary JSON payload."""
    total_step_time = sum(step_times)
    examples_per_second = (
        total_train_examples / total_step_time if total_step_time > 0 else None
    )
    return {
        "mode": MODE,
        "model_name": model_name,
        "trainable_scope": TRAINABLE_SCOPE,
        "frozen_scope": FROZEN_SCOPE,
        "selected_jax_platform": selected_jax_platform,
        "backend": backend,
        "devices": list(devices),
        "train_manifest": str(train_manifest),
        "eval_manifest": str(eval_manifest),
        "train_examples": train_examples,
        "eval_examples": eval_examples,
        "train_label_counts": dict(train_label_counts),
        "eval_label_counts": dict(eval_label_counts),
        "num_train_classes": len(train_label_counts),
        "num_eval_classes": len(eval_label_counts),
        "batch_size": batch_size,
        "sharding": dict(sharding_metadata or {}),
        "learning_rate": learning_rate,
        "eval_every_steps": eval_every_steps,
        "reinit_head": reinit_head,
        "seed": seed,
        "start_step": start_step,
        "final_step": final_step,
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "checkpoint_path": str(checkpoint_dir),
        "latest_checkpoint_step": latest_checkpoint_step,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "mean_step_time_sec": fmean(step_times) if step_times else None,
        "examples_per_second": examples_per_second,
        "timing_scope": {
            "mean_step_time_sec": (
                "training-step execution time only; excludes checkpoint write time"
            ),
            "examples_per_second": (
                "training examples divided by accumulated training-step time; "
                "excludes checkpoint write time"
            ),
            "total_runtime_sec": (
                "full run wall time including setup, evaluation, checkpointing, "
                "prediction writing, and summary writing"
            ),
        },
        "total_runtime_sec": total_runtime_sec,
        "interrupted": interrupted,
        "stop_signal_name": stop_signal_name,
        **dict(git_metadata),
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


def _positive_float(raw_value: str) -> float:
    value = float(raw_value)
    if value <= 0:
        msg = "must be greater than 0"
        raise argparse.ArgumentTypeError(msg)
    return value


def _nonnegative_float(raw_value: str) -> float:
    value = float(raw_value)
    if value < 0:
        msg = "must be non-negative"
        raise argparse.ArgumentTypeError(msg)
    return value


def run_finetune(args: argparse.Namespace) -> dict[str, Any]:
    """Run the fine-tuning smoke workflow and return the summary payload."""
    reset_signal_state()
    args = resolve_run_paths(args)
    apply_jax_platform(args.jax_platform)

    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        msg = (
            "Missing Demo 2 fine-tuning dependencies. Install or run with the "
            "optional dependency groups, for example: "
            "`uv run --group pretrained --group training python "
            "examples/demo2_pretrained_vit_finetune.py ...`"
        )
        raise RuntimeError(msg) from exc

    from jax_tpu_project.sharding import BatchShardingConfig, resolve_batch_sharding

    sharding_config = BatchShardingConfig(
        mode=args.batch_sharding,
        mesh_axis_name=args.mesh_axis_name,
        min_shard_devices=args.min_shard_devices,
        require_multiple_devices=args.require_multiple_devices,
    )
    resolved_sharding = resolve_batch_sharding(
        sharding_config,
        global_batch_size=args.batch_size,
        jax_module=jax,
    )
    explicit_prediction_output_sharding = (
        resolved_sharding.enabled and args.save_predictions
    )
    resolved_sharding = resolved_sharding.with_jit_sharding_status(
        explicit_input_sharding=resolved_sharding.enabled,
        explicit_output_sharding=explicit_prediction_output_sharding,
    )
    if not explicit_prediction_output_sharding:
        resolved_sharding = resolved_sharding.with_metadata_updates(
            logits_partition_spec=None,
        )

    install_signal_handlers()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_entries = read_labeled_manifest(args.train_manifest)
    eval_entries = read_labeled_manifest(args.eval_manifest)
    validate_manifest_paths(train_entries)
    validate_manifest_paths(eval_entries)
    label_ids = sorted({entry.label_id for entry in [*train_entries, *eval_entries]})
    train_label_counts = count_labels(train_entries)
    eval_label_counts = count_labels(eval_entries)
    git_metadata = collect_git_metadata()

    try:
        import optax
        from transformers import AutoImageProcessor, FlaxViTForImageClassification
    except ImportError as exc:
        msg = (
            "Missing Demo 2 fine-tuning dependencies. Install or run with the "
            "optional dependency groups, for example: "
            "`uv run --group pretrained --group training python "
            "examples/demo2_pretrained_vit_finetune.py ...`"
        )
        raise RuntimeError(msg) from exc

    image_processor = AutoImageProcessor.from_pretrained(args.model)
    model = FlaxViTForImageClassification.from_pretrained(args.model)
    params = model.params
    if "vit" not in params or "classifier" not in params:
        msg = "expected pretrained ViT params to contain 'vit' and 'classifier'"
        raise ValueError(msg)
    backbone_params = params["vit"]
    initial_head_params = params["classifier"]
    if args.reinit_head:
        initial_head_params = reinitialize_classifier_head(
            initial_head_params,
            seed=args.seed,
            jax_module=jax,
            jnp_module=jnp,
        )

    train_pixel_values = jnp.asarray(load_pixel_values(train_entries, image_processor))
    eval_pixel_values = jnp.asarray(load_pixel_values(eval_entries, image_processor))
    train_labels = jnp.asarray(
        np.asarray([entry.label_index for entry in train_entries], dtype=np.int32)
    )
    eval_labels = jnp.asarray(
        np.asarray([entry.label_index for entry in eval_entries], dtype=np.int32)
    )
    train_batches = build_batches(
        pixel_values=train_pixel_values,
        labels=train_labels,
        batch_size=args.batch_size,
        jnp_module=jnp,
    )
    eval_batches = build_batches(
        pixel_values=eval_pixel_values,
        labels=eval_labels,
        batch_size=args.batch_size,
        jnp_module=jnp,
    )
    train_batches = shard_training_batches(
        train_batches,
        resolved_sharding=resolved_sharding,
        jax_module=jax,
    )
    eval_batches = shard_training_batches(
        eval_batches,
        resolved_sharding=resolved_sharding,
        jax_module=jax,
    )

    optimizer = optax.adam(args.learning_rate)
    state = HeadTrainState(
        head_params=initial_head_params,
        optimizer_state=optimizer.init(initial_head_params),
        step=0,
    )
    checkpoint_metadata = build_checkpoint_metadata(
        model_name=args.model,
        train_manifest=args.train_manifest,
        eval_manifest=args.eval_manifest,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_every_steps=args.eval_every_steps,
        reinit_head=args.reinit_head,
        seed=args.seed,
        label_ids=label_ids,
        git_metadata=git_metadata,
    )
    checkpoint_manager = create_checkpoint_manager(args.checkpoint_dir)
    resumed_from_checkpoint = False
    restored_checkpoint_step = None
    if args.resume:
        target = build_checkpoint_item(
            head_params=state.head_params,
            optimizer_state=state.optimizer_state,
            step=state.step,
            metadata=checkpoint_metadata,
        )
        restored = restore_latest_checkpoint(checkpoint_manager, target=target)
        if restored is not None:
            state = HeadTrainState(
                head_params=restored["head_params"],
                optimizer_state=restored["optimizer_state"],
                step=int(np.asarray(restored["step"])),
            )
            restored_metadata = decode_checkpoint_metadata(restored["metadata_json"])
            validate_restored_checkpoint_metadata(
                restored_metadata,
                checkpoint_metadata,
            )
            resumed_from_checkpoint = True
            restored_checkpoint_step = checkpoint_manager.latest_step()

    def train_step_impl(
        head_params: Any,
        optimizer_state: Any,
        pixel_batch: Any,
        label_batch: Any,
        mask: Any,
    ) -> tuple[Any, Any, Any, Any]:
        def loss_fn(candidate_head_params: Any) -> tuple[Any, Any]:
            logits = model(
                pixel_values=pixel_batch,
                params={"vit": backbone_params, "classifier": candidate_head_params},
                train=False,
            ).logits
            losses = optax.softmax_cross_entropy_with_integer_labels(
                logits,
                label_batch,
            )
            loss = jnp.sum(losses * mask) / jnp.sum(mask)
            return loss, logits

        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(head_params)
        updates, optimizer_state = optimizer.update(
            grads,
            optimizer_state,
            head_params,
        )
        head_params = optax.apply_updates(head_params, updates)
        correct = jnp.sum((jnp.argmax(logits, axis=-1) == label_batch) * mask)
        accuracy = correct / jnp.sum(mask)
        return head_params, optimizer_state, loss, accuracy

    def build_train_step() -> Any:
        if not resolved_sharding.enabled:
            return jax.jit(train_step_impl)
        return jax.jit(
            train_step_impl,
            in_shardings=(
                None,
                None,
                resolved_sharding.image_sharding,
                resolved_sharding.label_sharding,
                resolved_sharding.mask_sharding,
            ),
        )

    def eval_step_impl(
        head_params: Any,
        pixel_batch: Any,
        label_batch: Any,
        mask: Any,
    ) -> tuple[Any, Any]:
        logits = model(
            pixel_values=pixel_batch,
            params={"vit": backbone_params, "classifier": head_params},
            train=False,
        ).logits
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, label_batch)
        loss = jnp.sum(losses * mask) / jnp.sum(mask)
        correct = jnp.sum((jnp.argmax(logits, axis=-1) == label_batch) * mask)
        return loss, correct

    def build_eval_step() -> Any:
        if not resolved_sharding.enabled:
            return jax.jit(eval_step_impl)
        return jax.jit(
            eval_step_impl,
            in_shardings=(
                None,
                resolved_sharding.image_sharding,
                resolved_sharding.label_sharding,
                resolved_sharding.mask_sharding,
            ),
        )

    def predict_step_impl(head_params: Any, pixel_batch: Any) -> Any:
        return model(
            pixel_values=pixel_batch,
            params={"vit": backbone_params, "classifier": head_params},
            train=False,
        ).logits

    def build_predict_step() -> Any:
        if not resolved_sharding.enabled:
            return jax.jit(predict_step_impl)
        jit_kwargs = {
            "in_shardings": (
                None,
                resolved_sharding.image_sharding,
            )
        }
        if explicit_prediction_output_sharding:
            jit_kwargs["out_shardings"] = resolved_sharding.logits_sharding
        return jax.jit(predict_step_impl, **jit_kwargs)

    train_step = build_train_step()
    eval_step = build_eval_step()
    predict_step = build_predict_step()

    def evaluate(head_params: Any) -> dict[str, float]:
        total_loss = 0.0
        total_correct = 0.0
        total_examples = 0
        for batch in eval_batches:
            loss, correct = eval_step(
                head_params,
                batch.pixel_values,
                batch.labels,
                batch.mask,
            )
            loss.block_until_ready()
            total_loss += float(loss) * batch.real_size
            total_correct += float(correct)
            total_examples += batch.real_size
        return {
            "loss": total_loss / total_examples,
            "accuracy": total_correct / total_examples,
        }

    def collect_predictions(head_params: Any) -> list[dict[str, Any]]:
        predictions = []
        for batch in eval_batches:
            try:
                logits = predict_step(head_params, batch.pixel_values)
            except Exception as exc:
                if resolved_sharding.enabled:
                    msg = (
                        "sharded prediction step failed with --batch-sharding "
                        "data enabled; prediction collection is not falling "
                        "back to an unsharded path"
                    )
                    raise RuntimeError(msg) from exc
                raise
            logits.block_until_ready()
            for offset in range(batch.real_size):
                entry = eval_entries[batch.start_index + offset]
                prediction = format_top_k_predictions_from_logits(
                    logits=logits[offset],
                    top_k=args.top_k,
                    id2label=model.config.id2label,
                    jax_module=jax,
                )
                predictions.append(
                    {
                        "image_path": str(entry.image_path),
                        "label_id": entry.label_id,
                        "label_index": entry.label_index,
                        **prediction,
                    }
                )
        return predictions

    run_start_time = time.perf_counter()
    append_log(
        args.output_dir,
        f"start_step={state.step} resumed_from_checkpoint={resumed_from_checkpoint}",
    )
    initial_eval = evaluate(state.head_params)
    if args.save_predictions:
        write_metrics(
            {"mode": MODE, "predictions": collect_predictions(state.head_params)},
            args.output_dir / "predictions_before.json",
        )

    metrics_path = args.output_dir / "metrics.csv"
    eval_metrics_path = args.output_dir / "eval_metrics.csv"
    write_metrics_header(metrics_path)
    write_eval_metrics_header(eval_metrics_path)
    append_eval_metrics_row(
        eval_metrics_path,
        step=state.step,
        metrics=initial_eval,
    )
    last_eval_metrics_step = state.step
    step_times: list[float] = []
    total_train_examples = 0
    latest_checkpoint_step = restored_checkpoint_step
    train_loop_start_time = time.perf_counter()
    last_checkpoint_time = train_loop_start_time
    start_step = state.step

    while should_continue_training(
        step=state.step,
        max_steps=args.max_steps,
        elapsed_sec=time.perf_counter() - train_loop_start_time,
        min_train_seconds=args.min_train_seconds,
    ):
        batch = train_batches[state.step % len(train_batches)]
        step_start = time.perf_counter()
        head_params, optimizer_state, loss, accuracy = train_step(
            state.head_params,
            state.optimizer_state,
            batch.pixel_values,
            batch.labels,
            batch.mask,
        )
        loss.block_until_ready()
        step_time = time.perf_counter() - step_start
        state = HeadTrainState(
            head_params=head_params,
            optimizer_state=optimizer_state,
            step=state.step + 1,
        )
        step_times.append(step_time)
        total_train_examples += batch.real_size

        now = time.perf_counter()
        checkpoint_saved = should_save_checkpoint(
            step=state.step,
            now=now,
            last_checkpoint_time=last_checkpoint_time,
            checkpoint_every_steps=args.checkpoint_every_steps,
            checkpoint_every_seconds=args.checkpoint_every_seconds,
            stop_requested=STOP_REQUESTED,
        )
        if checkpoint_saved:
            latest_checkpoint_step = save_training_checkpoint(
                checkpoint_manager,
                step=state.step,
                head_params=state.head_params,
                optimizer_state=state.optimizer_state,
                metadata=checkpoint_metadata,
            )
            last_checkpoint_time = now
        append_metrics_row(
            metrics_path,
            {
                "step": state.step,
                "loss": float(loss),
                "accuracy": float(accuracy),
                "step_time_sec": step_time,
                "examples_per_second": batch.real_size / step_time,
                "checkpoint_saved": checkpoint_saved,
            },
        )
        append_log(
            args.output_dir,
            (
                f"step={state.step} loss={float(loss):.6f} "
                f"accuracy={float(accuracy):.6f} step_time_sec={step_time:.6f} "
                f"checkpoint_saved={checkpoint_saved}"
            ),
        )
        if not STOP_REQUESTED and should_run_periodic_eval(
            step=state.step,
            start_step=start_step,
            eval_every_steps=args.eval_every_steps,
        ):
            eval_metrics = evaluate(state.head_params)
            append_eval_metrics_row(
                eval_metrics_path,
                step=state.step,
                metrics=eval_metrics,
            )
            last_eval_metrics_step = state.step
        if STOP_REQUESTED:
            break

    if latest_checkpoint_step != state.step:
        latest_checkpoint_step = save_training_checkpoint(
            checkpoint_manager,
            step=state.step,
            head_params=state.head_params,
            optimizer_state=state.optimizer_state,
            metadata=checkpoint_metadata,
        )

    final_eval = evaluate(state.head_params)
    if last_eval_metrics_step != state.step:
        append_eval_metrics_row(
            eval_metrics_path,
            step=state.step,
            metrics=final_eval,
        )
    if args.save_predictions:
        write_metrics(
            {"mode": MODE, "predictions": collect_predictions(state.head_params)},
            args.output_dir / "predictions_after.json",
        )

    total_runtime_sec = time.perf_counter() - run_start_time
    summary = build_summary(
        model_name=args.model,
        selected_jax_platform=args.jax_platform,
        backend=jax.default_backend(),
        devices=summarize_devices(jax.devices()),
        train_manifest=args.train_manifest,
        eval_manifest=args.eval_manifest,
        train_examples=len(train_entries),
        eval_examples=len(eval_entries),
        train_label_counts=train_label_counts,
        eval_label_counts=eval_label_counts,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_every_steps=args.eval_every_steps,
        reinit_head=args.reinit_head,
        seed=args.seed,
        start_step=start_step,
        final_step=state.step,
        resumed_from_checkpoint=resumed_from_checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        latest_checkpoint_step=latest_checkpoint_step,
        initial_loss=float(initial_eval["loss"]),
        final_loss=float(final_eval["loss"]),
        step_times=step_times,
        total_train_examples=total_train_examples,
        total_runtime_sec=total_runtime_sec,
        interrupted=STOP_REQUESTED,
        stop_signal_name=STOP_SIGNAL_NAME,
        git_metadata=git_metadata,
        sharding_metadata=resolved_sharding.metadata,
    )
    write_metrics(summary, args.output_dir / "summary.json")
    append_log(
        args.output_dir,
        f"final_step={state.step} interrupted={STOP_REQUESTED}",
    )
    return summary


def main() -> None:
    """Run the CLI entry point."""
    args = parse_args()
    summary = run_finetune(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"summary: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
