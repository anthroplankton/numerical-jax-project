"""Runtime utilities for inspecting the local JAX environment."""

from __future__ import annotations

from typing import Any

import jax


def get_device_summary() -> dict[str, Any]:
    """Return a JSON-serializable summary of the active JAX runtime devices."""
    return {
        "default_backend": jax.default_backend(),
        "devices": [
            {
                "platform": device.platform,
                "device_kind": device.device_kind,
                "id": device.id,
                "repr": str(device),
            }
            for device in jax.devices()
        ],
    }
