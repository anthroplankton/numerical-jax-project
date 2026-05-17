from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "compare_vit_results.py"


def load_compare_module():
    spec = importlib.util.spec_from_file_location("compare_vit_results", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload))


def test_summarize_single_image_result_includes_comparison_fields(tmp_path) -> None:
    module = load_compare_module()
    result_path = tmp_path / "cpu.json"
    write_json(
        result_path,
        {
            "mode": "single_image",
            "command_used": "python examples/pretrained_vit_inference.py --image image.jpg",
            "output_path": "runs/vit-inference/cpu.json",
            "model_name": "google/vit-base-patch16-224",
            "selected_jax_platform": "cpu",
            "backend": "cpu",
            "devices": [{"platform": "cpu"}],
            "image_path": "image.jpg",
            "batch_size": 1,
            "benchmark_steps": 5,
            "total_timed_inference_sec": 2.0,
            "mean_step_time_sec": 0.4,
            "throughput_images_per_sec": 2.5,
            "predicted_label": "Chihuahua",
        },
    )

    summary = module.summarize_result(result_path, module.load_metrics(result_path))

    assert summary["source_path"] == str(result_path)
    assert summary["command_used"].startswith("python examples/pretrained")
    assert summary["output_path"] == "runs/vit-inference/cpu.json"
    assert summary["input_image"] == "image.jpg"
    assert summary["input_manifest"] is None
    assert summary["num_images"] == 1
    assert summary["backend"] == "cpu"
    assert summary["batch_size"] == 1
    assert summary["total_runtime_sec"] == 2.0
    assert summary["throughput_images_per_sec"] == 2.5
    assert summary["per_image_time_sec"] == pytest.approx(0.4)


def test_build_comparison_computes_speedup_from_existing_json(tmp_path) -> None:
    module = load_compare_module()
    cpu_path = tmp_path / "cpu.json"
    tpu_path = tmp_path / "tpu.json"
    write_json(
        cpu_path,
        {
            "mode": "single_image",
            "backend": "cpu",
            "image_path": "image.jpg",
            "batch_size": 1,
            "throughput_images_per_sec": 2.0,
        },
    )
    write_json(
        tpu_path,
        {
            "mode": "single_image",
            "backend": "tpu",
            "image_path": "image.jpg",
            "batch_size": 1,
            "throughput_images_per_sec": 8.0,
        },
    )

    comparison = module.build_comparison([cpu_path, tpu_path])

    assert comparison["baseline_source_path"] == str(cpu_path)
    assert comparison["results"][0]["backend"] == "cpu"
    assert comparison["results"][1]["backend"] == "tpu"
    assert comparison["comparisons"][0]["throughput_speedup_vs_baseline"] == 1.0
    assert comparison["comparisons"][1]["throughput_speedup_vs_baseline"] == 4.0
