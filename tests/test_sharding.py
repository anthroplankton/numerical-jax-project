from __future__ import annotations

import json

import pytest

from jax_tpu_project.sharding import (
    BatchShardingConfig,
    build_sharding_metadata,
    format_partition_spec,
    resolve_batch_sharding,
    validate_batch_divisibility,
)


class FakeJax:
    def __init__(self, *, device_count: int, local_device_count: int | None = None):
        self._device_count = device_count
        self._local_device_count = (
            device_count if local_device_count is None else local_device_count
        )

    def device_count(self) -> int:
        return self._device_count

    def local_device_count(self) -> int:
        return self._local_device_count


def test_none_mode_metadata_is_json_safe() -> None:
    config = BatchShardingConfig(mode="none", mesh_axis_name="data")

    resolved = resolve_batch_sharding(
        config,
        global_batch_size=4,
        jax_module=FakeJax(device_count=1),
    )

    assert resolved.enabled is False
    assert resolved.image_sharding is None
    assert resolved.logits_sharding is None
    assert resolved.label_sharding is None
    assert resolved.mask_sharding is None
    assert resolved.metadata["enabled"] is False
    assert resolved.metadata["mode"] == "none"
    assert resolved.metadata["mesh_axis_name"] == "data"
    assert resolved.metadata["mesh_shape"] == {}
    assert resolved.metadata["mesh_device_count"] is None
    assert resolved.metadata["jax_device_count"] == 1
    assert resolved.metadata["jax_local_device_count"] == 1
    assert resolved.metadata["global_batch_size"] == 4
    assert resolved.metadata["per_device_batch_size"] is None
    assert resolved.metadata["label_partition_spec"] is None
    assert resolved.metadata["mask_partition_spec"] is None
    assert resolved.metadata["explicit_jit_shardings"] is False
    json.dumps(resolved.metadata, sort_keys=True)


def test_data_mode_fails_on_single_visible_device() -> None:
    config = BatchShardingConfig(mode="data", min_shard_devices=2)

    with pytest.raises(ValueError, match="--batch-sharding data requires"):
        resolve_batch_sharding(
            config,
            global_batch_size=4,
            jax_module=FakeJax(device_count=1),
        )


def test_require_multiple_devices_fails_when_none_mode_has_one_device() -> None:
    config = BatchShardingConfig(
        mode="none",
        min_shard_devices=2,
        require_multiple_devices=True,
    )

    with pytest.raises(ValueError, match="--require-multiple-devices"):
        resolve_batch_sharding(
            config,
            global_batch_size=4,
            jax_module=FakeJax(device_count=1),
        )


def test_batch_divisibility_error_mentions_batch_size_and_device_count() -> None:
    with pytest.raises(ValueError) as exc_info:
        validate_batch_divisibility(batch_size=10, shard_device_count=4)

    message = str(exc_info.value)
    assert "batch_size=10" in message
    assert "mesh_device_count=4" in message
    assert "--batch-sharding none" in message


def test_data_mode_metadata_records_resolved_runtime_facts() -> None:
    config = BatchShardingConfig(
        mode="data",
        mesh_axis_name="data",
        min_shard_devices=2,
        require_multiple_devices=True,
    )

    metadata = build_sharding_metadata(
        config=config,
        enabled=True,
        global_batch_size=16,
        jax_device_count=4,
        jax_local_device_count=4,
        mesh_device_count=4,
        per_device_batch_size=4,
        image_partition_spec=format_partition_spec("data", rank=4),
        logits_partition_spec=format_partition_spec("data", rank=2),
        label_partition_spec=format_partition_spec("data", rank=1),
        mask_partition_spec=format_partition_spec("data", rank=1),
        explicit_jit_input_sharding=True,
        explicit_jit_output_sharding=True,
    )

    assert metadata == {
        "enabled": True,
        "mode": "data",
        "mesh_axis_name": "data",
        "min_shard_devices": 2,
        "require_multiple_devices": True,
        "mesh_shape": {"data": 4},
        "mesh_device_count": 4,
        "jax_device_count": 4,
        "jax_local_device_count": 4,
        "global_batch_size": 16,
        "per_device_batch_size": 4,
        "image_partition_spec": "PartitionSpec('data', None, None, None)",
        "logits_partition_spec": "PartitionSpec('data', None)",
        "label_partition_spec": "PartitionSpec('data')",
        "mask_partition_spec": "PartitionSpec('data')",
        "explicit_jit_shardings": True,
        "explicit_jit_input_sharding": True,
        "explicit_jit_output_sharding": True,
        "jit_sharding_level": "input_and_output",
        "fallback": None,
    }
    json.dumps(metadata, sort_keys=True)


def test_format_partition_spec_supports_batch_vector_specs() -> None:
    assert format_partition_spec("data", rank=1) == "PartitionSpec('data')"


def test_local_none_mode_leaves_array_available_for_jax_jit() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    config = BatchShardingConfig(mode="none")
    resolved = resolve_batch_sharding(
        config,
        global_batch_size=1,
        jax_module=jax,
    )

    @jax.jit
    def add_one(values):
        return values + 1

    values = jnp.asarray([[1.0]])
    placed = resolved.shard_image_batch(values, jax_module=jax)
    labels = resolved.shard_label_batch(jnp.asarray([1]), jax_module=jax)
    mask = resolved.shard_mask_batch(jnp.asarray([1.0]), jax_module=jax)
    result = add_one(placed)

    assert result.tolist() == [[2.0]]
    assert labels.tolist() == [1]
    assert mask.tolist() == [1.0]
