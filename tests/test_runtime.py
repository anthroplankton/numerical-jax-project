from __future__ import annotations

import json

from jax_tpu_project.runtime import get_device_summary


def test_get_device_summary_is_json_serializable() -> None:
    summary = get_device_summary()

    json.dumps(summary)


def test_get_device_summary_contains_device_fields() -> None:
    summary = get_device_summary()

    assert isinstance(summary["default_backend"], str)
    assert isinstance(summary["devices"], list)
    assert summary["devices"]

    for device in summary["devices"]:
        assert isinstance(device["platform"], str)
        assert isinstance(device["device_kind"], str)
        assert isinstance(device["id"], int)
        assert isinstance(device["repr"], str)
