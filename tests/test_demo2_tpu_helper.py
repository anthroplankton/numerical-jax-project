from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "demo2_tpu_run_when_active.sh"


def test_demo2_tpu_helper_has_valid_bash_syntax() -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not available")

    subprocess.run([bash, "-n", str(SCRIPT_PATH)], check=True)


def test_demo2_tpu_helper_documents_safe_cloud_lifecycle() -> None:
    text = SCRIPT_PATH.read_text()

    assert "queued-resources create" not in text
    assert "--delete-after" in text
    assert "PROJECT_ID" in text
    assert "ZONE" in text
    assert "TPU_NAME" in text
    assert "QUEUED_RESOURCE_ID" in text
    assert "REPO_URL" in text
    assert "BRANCH" in text
    assert "demo2_cloud_public_examples_tpu_b4.json" in text
    assert "demo2_local_cpu_vs_cloud_tpu_public_examples_b4_compare.json" in text
    assert "values are intentionally not logged" in text
    assert "trap on_exit EXIT" in text
    assert "cleanup_printed" in text
    assert "require_non_negative_integer" in text
    assert "require_positive_integer" not in text
    assert "REPO_URL must not contain embedded credentials" in text
    assert "git remote set-url origin" in text
    assert 'git pull --ff-only origin "$BRANCH"' in text
    assert 'backend != "tpu"' in text
    assert "No JAX devices are visible" in text
