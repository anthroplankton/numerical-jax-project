from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SCRIPT_PATH = SCRIPTS_DIR / "generate_vit_summary_tables.py"


def load_summary_module():
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(
        "generate_vit_summary_tables", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_vit_summary_tables"] = module
    spec.loader.exec_module(module)
    return module


def write_result_json(
    path: Path,
    *,
    backend: str,
    batch_size: int,
    num_images: int,
    throughput: float,
    command_used: str | None = None,
    git_commit: str | None = None,
    git_dirty: bool | None = None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "mode": "image_manifest",
                "processing_mode": "batched_manifest",
                "backend": backend,
                "devices": [{"platform": backend, "device_kind": backend}],
                "batch_size": batch_size,
                "num_images": num_images,
                "num_batches": max(num_images // batch_size, 1),
                "num_padded_images": 0,
                "benchmark_steps": 5,
                "mean_step_time_sec": 0.5,
                "total_timed_inference_sec": 2.5,
                "throughput_images_per_sec": throughput,
                "git_commit": git_commit,
                "git_dirty": git_dirty,
                "command_used": command_used
                or "python examples/pretrained_vit_inference.py --benchmark-steps 5",
            }
        )
    )


def test_discover_records_classifies_known_raw_results_and_skips_comparison_json(
    tmp_path,
) -> None:
    module = load_summary_module()
    input_dir = tmp_path / "runs"
    input_dir.mkdir()
    write_result_json(
        input_dir / "demo2_local_imagenette320_val64_cpu_b1.json",
        backend="cpu",
        batch_size=1,
        num_images=64,
        throughput=2.0,
        git_commit="0123456789abcdef",
        git_dirty=False,
    )
    write_result_json(
        input_dir / "demo2_cloud_imagenette320_val64_tpu_b4.json",
        backend="tpu",
        batch_size=4,
        num_images=64,
        throughput=20.0,
    )
    (
        input_dir / "demo2_local_cpu_vs_cloud_tpu_public_examples_b4_compare.json"
    ).write_text(json.dumps({"comparisons": [], "results": []}))

    records = module.discover_records(input_dir)

    assert [record.path.name for record in records] == [
        "demo2_local_imagenette320_val64_cpu_b1.json",
        "demo2_cloud_imagenette320_val64_tpu_b4.json",
    ]
    assert records[0].key.dataset == "val64"
    assert records[0].key.machine == "local_cpu"
    assert records[0].key.batch_size == 1
    assert records[0].summary["git_commit"] == "0123456789abcdef"
    assert records[0].summary["git_dirty"] is False


def test_write_summary_tables_outputs_ratios_missing_runs_and_scope_note(
    tmp_path,
) -> None:
    module = load_summary_module()
    input_dir = tmp_path / "runs"
    output_dir = tmp_path / "report-results"
    input_dir.mkdir()
    write_result_json(
        input_dir / "demo2_local_imagenette320_val64_cpu_b1.json",
        backend="cpu",
        batch_size=1,
        num_images=64,
        throughput=2.0,
    )
    write_result_json(
        input_dir / "demo2_local_imagenette320_val64_cpu_b4.json",
        backend="cpu",
        batch_size=4,
        num_images=64,
        throughput=6.0,
    )
    write_result_json(
        input_dir / "demo2_external_ryzen7735hs_wsl_imagenette320_val64_cpu_b1.json",
        backend="cpu",
        batch_size=1,
        num_images=64,
        throughput=4.0,
    )
    write_result_json(
        input_dir / "demo2_cloud_imagenette320_val64_tpu_b1.json",
        backend="tpu",
        batch_size=1,
        num_images=64,
        throughput=20.0,
        command_used=(
            "python examples/pretrained_vit_inference.py --jax-platform tpu "
            "--output runs/vit-inference/demo2_asus_a16_private_tpu_b1.json"
        ),
    )
    write_result_json(
        input_dir / "demo2_local_public_examples_cpu_b1.json",
        backend="cpu",
        batch_size=1,
        num_images=5,
        throughput=1.0,
    )
    write_result_json(
        input_dir / "demo2_cloud_public_examples_tpu_b4.json",
        backend="tpu",
        batch_size=4,
        num_images=5,
        throughput=100.0,
    )

    module.write_summary_tables(module.discover_records(input_dir), output_dir)

    overview = (output_dir / "demo2_imagenette320_overview.md").read_text()
    batch_scaling = (output_dir / "demo2_imagenette320_batch_scaling.md").read_text()
    cpu_vs_tpu = (output_dir / "demo2_imagenette320_cpu_vs_tpu.md").read_text()
    cpu_machine = (output_dir / "demo2_cpu_machine_comparison.md").read_text()
    public_summary = (output_dir / "demo2_public_examples_summary.md").read_text()

    assert "ViT inference only; no training" in overview
    assert "Command used" not in overview
    assert (
        "python examples/pretrained_vit_inference.py --jax-platform tpu" not in overview
    )
    assert "demo2_asus_a16_private_tpu_b1.json" not in overview
    assert "demo2_cloud_imagenette320_val64_tpu_b1.json" in overview
    assert "public examples" not in overview
    assert "public examples" not in batch_scaling
    assert "public examples" not in cpu_vs_tpu
    assert "public examples" not in cpu_machine
    assert "not run" in overview
    assert (
        "| Imagenette 320 val64 | local CPU | 2 | 6 | 3x | not run | n/a |"
        in batch_scaling
    )
    assert "| Imagenette 320 val64 | 1 | 2 | 4 | 20 | 10x | 5x | 2x |" in cpu_vs_tpu
    assert "| Imagenette 320 val64 | 1 | 2 | 4 | 2x |" in cpu_machine
    assert "Imagenette 320 full validation" not in cpu_machine
    assert "small reproducible smoke/demo evidence" in public_summary
    assert "not part of the Imagenette 320 benchmark set" in public_summary
    assert "| public examples | local CPU | 1 | run | cpu | cpu | 5 |" in public_summary
    assert "| public examples | cloud TPU | 4 | run | tpu | tpu | 5 |" in public_summary
    assert "| public examples | cloud TPU | 1 | not run |" in public_summary
    assert (
        "| public examples | external Ryzen 7735HS WSL CPU | 8 | not run |"
        in public_summary
    )
