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
    return parser.parse_args(argv)


def load_metrics(path: Path) -> dict[str, Any]:
    """Load one Demo 2 JSON metrics file."""
    return json.loads(path.read_text())


def summarize_result(path: Path, metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Extract comparable fields from one Demo 2 metrics payload."""
    mode = metrics.get("mode")
    num_images = metrics.get("num_images")
    if num_images is None and mode == "single_image":
        num_images = 1

    throughput = metrics.get("throughput_images_per_sec")
    per_image_time = None
    if isinstance(throughput, int | float) and throughput > 0:
        per_image_time = 1 / throughput

    return {
        "source_path": str(path),
        "command_used": metrics.get("command_used"),
        "output_path": metrics.get("output_path", str(path)),
        "mode": mode,
        "model_name": metrics.get("model_name"),
        "selected_jax_platform": metrics.get("selected_jax_platform"),
        "backend": metrics.get("backend"),
        "devices": metrics.get("devices", []),
        "input_image": metrics.get("image_path"),
        "input_manifest": metrics.get("manifest_path"),
        "batch_size": metrics.get("batch_size"),
        "num_images": num_images,
        "num_batches": metrics.get("num_batches"),
        "num_padded_images": metrics.get("num_padded_images"),
        "benchmark_steps": metrics.get("benchmark_steps"),
        "total_runtime_sec": metrics.get("total_timed_inference_sec"),
        "mean_step_time_sec": metrics.get("mean_step_time_sec"),
        "throughput_images_per_sec": throughput,
        "per_image_time_sec": per_image_time,
        "predicted_label": metrics.get("predicted_label"),
    }


def build_comparison(result_paths: Sequence[Path]) -> dict[str, Any]:
    """Build a JSON-serializable comparison summary."""
    summaries = [summarize_result(path, load_metrics(path)) for path in result_paths]
    baseline = summaries[0]
    baseline_throughput = baseline.get("throughput_images_per_sec")

    comparisons = []
    for summary in summaries:
        throughput = summary.get("throughput_images_per_sec")
        speedup = None
        if (
            isinstance(throughput, int | float)
            and isinstance(baseline_throughput, int | float)
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


def write_comparison(comparison: Mapping[str, Any], output_path: Path) -> None:
    """Write the comparison summary as formatted JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")


def main() -> None:
    """Run the CLI entry point."""
    args = parse_args()
    comparison = build_comparison(args.result_paths)
    if args.output is not None:
        write_comparison(comparison, args.output)
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
