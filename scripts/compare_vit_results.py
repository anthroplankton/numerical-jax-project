"""Compare Demo 2 pretrained ViT JSON result files.

This helper is local-only: it reads existing JSON metrics and does not require
JAX, TPU access, model weights, network access, or image files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for result comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result_paths",
        nargs="+",
        type=Path,
        help="Demo 2 JSON result files to compare. The first file is the baseline.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for a JSON comparison summary.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        help="Optional path for a report-ready Markdown benchmark table.",
    )
    return parser.parse_args(argv)


def load_metrics(path: Path) -> dict[str, Any]:
    """Load one Demo 2 JSON metrics file."""
    return json.loads(path.read_text())


def summarize_result(path: Path, metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Extract comparable fields from one Demo 2 metrics payload."""
    mode = infer_mode(metrics)
    num_images = metrics.get("num_images")
    if num_images is None and mode == "single_image":
        num_images = 1

    throughput = metrics.get("throughput_images_per_sec")
    per_image_time = None
    if is_number(throughput) and throughput > 0:
        per_image_time = 1 / throughput
    total_runtime = infer_total_runtime_sec(metrics)
    timed_batch_runs = infer_timed_batch_runs(metrics, mode)
    num_batches = metrics.get("num_batches")
    if num_batches is None and mode == "single_image":
        num_batches = 1
    num_padded_images = metrics.get("num_padded_images")
    if num_padded_images is None and mode == "single_image":
        num_padded_images = 0

    return {
        "source_path": str(path),
        "command_used": metrics.get("command_used"),
        "output_path": metrics.get("output_path", str(path)),
        "mode": mode,
        "processing_mode": metrics.get("processing_mode"),
        "model_name": metrics.get("model_name"),
        "selected_jax_platform": metrics.get("selected_jax_platform"),
        "backend": metrics.get("backend"),
        "devices": metrics.get("devices", []),
        "input_image": metrics.get("image_path"),
        "input_manifest": metrics.get("manifest_path"),
        "manifest_kind": metrics.get("manifest_kind"),
        "batch_size": metrics.get("batch_size"),
        "num_images": num_images,
        "num_batches": num_batches,
        "timed_batch_runs": timed_batch_runs,
        "num_padded_images": num_padded_images,
        "benchmark_steps": metrics.get("benchmark_steps"),
        "total_runtime_sec": total_runtime,
        "mean_step_time_sec": metrics.get("mean_step_time_sec"),
        "throughput_counted_images": infer_throughput_counted_images(
            metrics, total_runtime
        ),
        "throughput_images_per_sec": throughput,
        "per_image_time_sec": per_image_time,
        "predicted_label": metrics.get("predicted_label"),
    }


def infer_mode(metrics: Mapping[str, Any]) -> str | None:
    """Infer the result mode for current and legacy Demo 2 JSON payloads."""
    mode = metrics.get("mode")
    if isinstance(mode, str):
        return mode
    if metrics.get("manifest_path") is not None or metrics.get("image_results"):
        return "image_manifest"
    if metrics.get("image_path") is not None or metrics.get("predicted_label"):
        return "single_image"
    return None


def infer_total_runtime_sec(metrics: Mapping[str, Any]) -> float | None:
    """Return total timed runtime, deriving it for legacy curated artifacts."""
    total_runtime = metrics.get("total_timed_inference_sec")
    if is_number(total_runtime):
        return float(total_runtime)

    mean_step_time = metrics.get("mean_step_time_sec")
    timed_batch_runs = metrics.get("timed_batch_runs") or metrics.get("benchmark_steps")
    if is_number(mean_step_time) and is_number(timed_batch_runs):
        return float(mean_step_time) * float(timed_batch_runs)
    return None


def infer_timed_batch_runs(metrics: Mapping[str, Any], mode: str | None) -> int | None:
    """Infer timed model-call count for current and legacy payloads."""
    timed_batch_runs = metrics.get("timed_batch_runs")
    if isinstance(timed_batch_runs, int):
        return timed_batch_runs
    benchmark_steps = metrics.get("benchmark_steps")
    if mode == "single_image" and isinstance(benchmark_steps, int):
        return benchmark_steps
    return None


def infer_throughput_counted_images(
    metrics: Mapping[str, Any], total_runtime: float | None
) -> float | None:
    """Infer the image count represented by throughput when possible."""
    counted_images = metrics.get("throughput_counted_images")
    if is_number(counted_images):
        return float(counted_images)
    throughput = metrics.get("throughput_images_per_sec")
    if is_number(throughput) and total_runtime is not None:
        return float(throughput) * total_runtime
    return None


def is_number(value: object) -> bool:
    """Return whether value is a JSON number, excluding booleans."""
    return isinstance(value, int | float) and not isinstance(value, bool)


def build_comparison(result_paths: Sequence[Path]) -> dict[str, Any]:
    """Build a JSON-serializable comparison summary."""
    if not result_paths:
        msg = "at least one result path is required"
        raise ValueError(msg)
    summaries = [summarize_result(path, load_metrics(path)) for path in result_paths]
    baseline = summaries[0]
    baseline_throughput = baseline.get("throughput_images_per_sec")

    comparisons = []
    for summary in summaries:
        throughput = summary.get("throughput_images_per_sec")
        speedup = None
        if (
            is_number(throughput)
            and is_number(baseline_throughput)
            and baseline_throughput > 0
        ):
            speedup = throughput / baseline_throughput
        comparisons.append(
            {
                "source_path": summary["source_path"],
                "baseline_source_path": baseline["source_path"],
                "throughput_speedup_vs_baseline": speedup,
            }
        )

    return {
        "baseline_source_path": baseline["source_path"],
        "results": summaries,
        "comparisons": comparisons,
    }


def build_markdown_table(comparison: Mapping[str, Any]) -> str:
    """Build a report-ready Markdown table from a comparison summary."""
    speedups = {
        item["source_path"]: item.get("throughput_speedup_vs_baseline")
        for item in comparison.get("comparisons", [])
    }
    rows = [
        [
            "Result",
            "Mode",
            "Backend",
            "Batch",
            "Images",
            "Padded",
            "Mean step (s)",
            "Throughput (img/s)",
            "Speedup",
        ],
        ["---", "---", "---", "---:", "---:", "---:", "---:", "---:", "---:"],
    ]
    for result in comparison.get("results", []):
        source_path = result.get("source_path")
        rows.append(
            [
                Path(str(source_path)).name,
                result.get("mode"),
                result.get("backend"),
                result.get("batch_size"),
                result.get("num_images"),
                result.get("num_padded_images"),
                format_markdown_number(result.get("mean_step_time_sec")),
                format_markdown_number(result.get("throughput_images_per_sec")),
                format_speedup(speedups.get(source_path)),
            ]
        )
    return "\n".join(
        "| " + " | ".join(format_markdown_cell(cell) for cell in row) + " |"
        for row in rows
    )


def format_speedup(speedup: object) -> str:
    """Format a speedup value for a Markdown table."""
    if not is_number(speedup):
        return ""
    return f"{float(speedup):.3g}x"


def format_markdown_number(value: object) -> str:
    """Format a JSON number compactly for a Markdown table."""
    if not is_number(value):
        return ""
    return f"{float(value):.6g}"


def format_markdown_cell(value: object) -> str:
    """Format and escape one Markdown table cell."""
    if value is None:
        return ""
    return str(value).replace("|", "\\|")


def write_comparison(comparison: Mapping[str, Any], output_path: Path) -> None:
    """Write the comparison summary as formatted JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")


def write_markdown_table(comparison: Mapping[str, Any], output_path: Path) -> None:
    """Write a report-ready Markdown comparison table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_markdown_table(comparison) + "\n")


def main() -> None:
    """Run the CLI entry point."""
    args = parse_args()
    comparison = build_comparison(args.result_paths)
    if args.output is not None:
        write_comparison(comparison, args.output)
    if args.markdown_output is not None:
        write_markdown_table(comparison, args.markdown_output)
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
