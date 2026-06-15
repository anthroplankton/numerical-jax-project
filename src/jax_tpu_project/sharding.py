"""Helpers for explicit batch-axis JAX sharding.

The helpers keep JAX runtime objects separate from JSON-safe metadata so example
scripts can report what was requested and what was resolved without leaking
environment-specific private context.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np

BatchShardingMode = Literal["none", "data"]

BATCH_SHARDING_CHOICES = ("none", "data")
DEFAULT_BATCH_SHARDING: BatchShardingMode = "none"
DEFAULT_MESH_AXIS_NAME = "data"
DEFAULT_MIN_SHARD_DEVICES = 2
IMAGE_PARTITION_SPEC_RANK = 4
LOGITS_PARTITION_SPEC_RANK = 2
BATCH_VECTOR_PARTITION_SPEC_RANK = 1


@dataclass(frozen=True)
class BatchShardingConfig:
    """Requested batch-sharding options from a CLI or caller."""

    mode: BatchShardingMode = DEFAULT_BATCH_SHARDING
    mesh_axis_name: str = DEFAULT_MESH_AXIS_NAME
    min_shard_devices: int = DEFAULT_MIN_SHARD_DEVICES
    require_multiple_devices: bool = False

    def __post_init__(self) -> None:
        if self.mode not in BATCH_SHARDING_CHOICES:
            choices = ", ".join(BATCH_SHARDING_CHOICES)
            msg = f"unsupported batch sharding mode {self.mode!r}; choose one of: {choices}"
            raise ValueError(msg)
        if not self.mesh_axis_name:
            msg = "mesh_axis_name must not be empty"
            raise ValueError(msg)
        if self.min_shard_devices < 1:
            msg = "min_shard_devices must be at least 1"
            raise ValueError(msg)


@dataclass(frozen=True)
class ResolvedBatchSharding:
    """Resolved runtime sharding objects plus grouped JSON-safe metadata."""

    enabled: bool
    mode: BatchShardingMode
    mesh: Any | None
    image_sharding: Any | None
    logits_sharding: Any | None
    label_sharding: Any | None
    mask_sharding: Any | None
    metadata: dict[str, Any]

    def shard_image_batch(self, image_batch: Any, *, jax_module: Any) -> Any:
        """Place an image batch on the configured sharding when enabled."""
        if not self.enabled:
            return image_batch
        return jax_module.device_put(image_batch, self.image_sharding)

    def shard_label_batch(self, label_batch: Any, *, jax_module: Any) -> Any:
        """Place a label batch on the configured batch-vector sharding."""
        if not self.enabled:
            return label_batch
        return jax_module.device_put(label_batch, self.label_sharding)

    def shard_mask_batch(self, mask_batch: Any, *, jax_module: Any) -> Any:
        """Place a mask batch on the configured batch-vector sharding."""
        if not self.enabled:
            return mask_batch
        return jax_module.device_put(mask_batch, self.mask_sharding)

    def with_metadata_updates(self, **updates: Any) -> "ResolvedBatchSharding":
        """Return a copy with JSON metadata fields updated."""
        return replace(self, metadata={**self.metadata, **updates})

    def with_jit_sharding_status(
        self,
        *,
        explicit_input_sharding: bool,
        explicit_output_sharding: bool,
        fallback: str | None = None,
    ) -> "ResolvedBatchSharding":
        """Return a copy with metadata updated for the actual jit sharding level."""
        if explicit_input_sharding and explicit_output_sharding:
            jit_sharding_level = "input_and_output"
        elif explicit_input_sharding:
            jit_sharding_level = "input_only"
        else:
            jit_sharding_level = "none"

        metadata = {
            **self.metadata,
            "explicit_jit_shardings": (
                explicit_input_sharding or explicit_output_sharding
            ),
            "explicit_jit_input_sharding": explicit_input_sharding,
            "explicit_jit_output_sharding": explicit_output_sharding,
            "jit_sharding_level": jit_sharding_level,
            "fallback": fallback,
        }
        return replace(self, metadata=metadata)


def resolve_batch_sharding(
    config: BatchShardingConfig,
    *,
    global_batch_size: int,
    jax_module: Any | None = None,
) -> ResolvedBatchSharding:
    """Validate and resolve batch-axis sharding for the active JAX runtime."""
    if jax_module is None:
        import jax as jax_module

    jax_device_count = int(jax_module.device_count())
    jax_local_device_count = int(jax_module.local_device_count())

    if config.mode == "none":
        validate_multiple_device_guard(
            require_multiple_devices=config.require_multiple_devices,
            visible_device_count=jax_device_count,
            min_shard_devices=config.min_shard_devices,
        )
        return ResolvedBatchSharding(
            enabled=False,
            mode=config.mode,
            mesh=None,
            image_sharding=None,
            logits_sharding=None,
            label_sharding=None,
            mask_sharding=None,
            metadata=build_sharding_metadata(
                config=config,
                enabled=False,
                global_batch_size=global_batch_size,
                jax_device_count=jax_device_count,
                jax_local_device_count=jax_local_device_count,
            ),
        )

    validate_data_sharding_device_count(
        visible_device_count=jax_device_count,
        min_shard_devices=config.min_shard_devices,
    )
    validate_batch_divisibility(
        batch_size=global_batch_size,
        shard_device_count=jax_device_count,
    )

    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    devices = np.asarray(jax_module.devices(), dtype=object)
    mesh = Mesh(devices, (config.mesh_axis_name,))
    image_spec = PartitionSpec(config.mesh_axis_name, None, None, None)
    logits_spec = PartitionSpec(config.mesh_axis_name, None)
    batch_vector_spec = PartitionSpec(config.mesh_axis_name)
    image_sharding = NamedSharding(mesh, image_spec)
    logits_sharding = NamedSharding(mesh, logits_spec)
    batch_vector_sharding = NamedSharding(mesh, batch_vector_spec)

    return ResolvedBatchSharding(
        enabled=True,
        mode=config.mode,
        mesh=mesh,
        image_sharding=image_sharding,
        logits_sharding=logits_sharding,
        label_sharding=batch_vector_sharding,
        mask_sharding=batch_vector_sharding,
        metadata=build_sharding_metadata(
            config=config,
            enabled=True,
            global_batch_size=global_batch_size,
            jax_device_count=jax_device_count,
            jax_local_device_count=jax_local_device_count,
            mesh_device_count=jax_device_count,
            per_device_batch_size=global_batch_size // jax_device_count,
            image_partition_spec=format_partition_spec(
                config.mesh_axis_name,
                rank=IMAGE_PARTITION_SPEC_RANK,
            ),
            logits_partition_spec=format_partition_spec(
                config.mesh_axis_name,
                rank=LOGITS_PARTITION_SPEC_RANK,
            ),
            label_partition_spec=format_partition_spec(
                config.mesh_axis_name,
                rank=BATCH_VECTOR_PARTITION_SPEC_RANK,
            ),
            mask_partition_spec=format_partition_spec(
                config.mesh_axis_name,
                rank=BATCH_VECTOR_PARTITION_SPEC_RANK,
            ),
            explicit_jit_input_sharding=True,
            explicit_jit_output_sharding=True,
        ),
    )


def validate_multiple_device_guard(
    *,
    require_multiple_devices: bool,
    visible_device_count: int,
    min_shard_devices: int,
) -> None:
    """Validate the general multiple-device runtime guard."""
    if require_multiple_devices and visible_device_count < min_shard_devices:
        msg = (
            "--require-multiple-devices was passed, but visible JAX devices="
            f"{visible_device_count} is below --min-shard-devices={min_shard_devices}. "
            "Use a multi-device runtime or remove --require-multiple-devices."
        )
        raise ValueError(msg)


def validate_data_sharding_device_count(
    *,
    visible_device_count: int,
    min_shard_devices: int,
) -> None:
    """Validate that data sharding has enough visible devices."""
    if visible_device_count < min_shard_devices:
        msg = (
            "--batch-sharding data requires multiple visible JAX devices, but "
            f"visible JAX devices={visible_device_count} is below "
            f"--min-shard-devices={min_shard_devices}. Use a multi-device TPU "
            "runtime or run with --batch-sharding none."
        )
        raise ValueError(msg)


def validate_batch_divisibility(*, batch_size: int, shard_device_count: int) -> None:
    """Validate that the global batch can be divided across the device mesh."""
    if shard_device_count < 1:
        msg = "shard_device_count must be at least 1"
        raise ValueError(msg)
    if batch_size % shard_device_count != 0:
        msg = (
            "--batch-sharding data requires batch_size to be divisible by the "
            f"mesh device count; batch_size={batch_size}, "
            f"mesh_device_count={shard_device_count}. Choose a batch size "
            "divisible by the mesh size or use --batch-sharding none."
        )
        raise ValueError(msg)


def build_sharding_metadata(
    *,
    config: BatchShardingConfig,
    enabled: bool,
    global_batch_size: int,
    jax_device_count: int,
    jax_local_device_count: int,
    mesh_device_count: int | None = None,
    per_device_batch_size: int | None = None,
    image_partition_spec: str | None = None,
    logits_partition_spec: str | None = None,
    label_partition_spec: str | None = None,
    mask_partition_spec: str | None = None,
    explicit_jit_input_sharding: bool = False,
    explicit_jit_output_sharding: bool = False,
    fallback: str | None = None,
) -> dict[str, Any]:
    """Build grouped JSON-safe metadata for the run output."""
    if enabled and mesh_device_count is None:
        msg = "mesh_device_count is required when sharding is enabled"
        raise ValueError(msg)

    if explicit_jit_input_sharding and explicit_jit_output_sharding:
        jit_sharding_level = "input_and_output"
    elif explicit_jit_input_sharding:
        jit_sharding_level = "input_only"
    else:
        jit_sharding_level = "none"

    return {
        "enabled": enabled,
        "mode": config.mode,
        "mesh_axis_name": config.mesh_axis_name,
        "min_shard_devices": config.min_shard_devices,
        "require_multiple_devices": config.require_multiple_devices,
        "mesh_shape": (
            {config.mesh_axis_name: int(mesh_device_count)}
            if mesh_device_count is not None
            else {}
        ),
        "mesh_device_count": mesh_device_count,
        "jax_device_count": jax_device_count,
        "jax_local_device_count": jax_local_device_count,
        "global_batch_size": global_batch_size,
        "per_device_batch_size": per_device_batch_size,
        "image_partition_spec": image_partition_spec,
        "logits_partition_spec": logits_partition_spec,
        "label_partition_spec": label_partition_spec,
        "mask_partition_spec": mask_partition_spec,
        "explicit_jit_shardings": (
            explicit_jit_input_sharding or explicit_jit_output_sharding
        ),
        "explicit_jit_input_sharding": explicit_jit_input_sharding,
        "explicit_jit_output_sharding": explicit_jit_output_sharding,
        "jit_sharding_level": jit_sharding_level,
        "fallback": fallback,
    }


def format_partition_spec(mesh_axis_name: str, *, rank: int) -> str:
    """Return a stable report-facing string for a batch-axis PartitionSpec."""
    if rank < 1:
        msg = "rank must be at least 1"
        raise ValueError(msg)
    remaining_axes = ", ".join("None" for _ in range(rank - 1))
    if remaining_axes:
        return f"PartitionSpec({mesh_axis_name!r}, {remaining_axes})"
    return f"PartitionSpec({mesh_axis_name!r})"
