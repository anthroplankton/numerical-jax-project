"""Generate Demo 2 ViT report summary tables from existing JSON artifacts.

This helper is local-only: it reads existing raw JSON metrics under
``runs/vit-inference/`` and writes curated Markdown summaries under
``report/results/``. It does not require JAX, TPU access, model weights,
network access, image files, or cloud credentials.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from compare_vit_results import is_number, load_metrics, summarize_result

# Fixed Demo 2 report matrix dimensions, not a generic benchmark configuration system.
IMAGENETTE_DATASET_ORDER = ("val64", "val256", "valfull")
PUBLIC_EXAMPLES_DATASET_ORDER = ("public_examples",)
CPU_MACHINE_DATASET_ORDER = ("val64", "val256")
ALL_DATASET_ORDER = PUBLIC_EXAMPLES_DATASET_ORDER + IMAGENETTE_DATASET_ORDER
BATCH_SIZE_ORDER = (1, 4, 8)

DATASET_LABELS = {
    "public_examples": "public examples",
    "val64": "Imagenette 320 val64",
    "val256": "Imagenette 320 val256",
    "valfull": "Imagenette 320 full validation",
}

MACHINE_LABELS = {
    "local_cpu": "local CPU",
    "external_cpu": "external Ryzen 7735HS WSL CPU",
    "cloud_tpu": "cloud TPU",
}

MACHINE_ORDER = tuple(MACHINE_LABELS)

SCOPE_NOTE = (
    "Scope: ViT inference only; no training; no dataset-level accuracy "
    "evaluation; not a full benchmark study; no universal TPU speedup claim."
)

RESULT_PATTERNS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (
        re.compile(r"^demo2_local_public_examples_cpu_b(?P<batch>\d+)\.json$"),
        "public_examples",
        "local_cpu",
    ),
    (
        re.compile(
            r"^demo2_external_ryzen7735hs_wsl_public_examples_cpu_b"
            r"(?P<batch>\d+)\.json$"
        ),
        "public_examples",
        "external_cpu",
    ),
    (
        re.compile(r"^demo2_cloud_public_examples_tpu_b(?P<batch>\d+)\.json$"),
        "public_examples",
        "cloud_tpu",
    ),
    (
        re.compile(
            r"^demo2_local_imagenette320_(?P<dataset>val64|val256)_cpu_b"
            r"(?P<batch>\d+)\.json$"
        ),
        "",
        "local_cpu",
    ),
    (
        re.compile(
            r"^demo2_external_ryzen7735hs_wsl_imagenette320_"
            r"(?P<dataset>val64|val256)_cpu_b(?P<batch>\d+)\.json$"
        ),
        "",
        "external_cpu",
    ),
    (
        re.compile(
            r"^demo2_cloud_imagenette320_(?P<dataset>val64|val256|valfull)_"
            r"tpu_b(?P<batch>\d+)\.json$"
        ),
        "",
        "cloud_tpu",
    ),
)


@dataclass(frozen=True)
class ResultKey:
    """Stable identity for one Demo 2 result row."""

    dataset: str
    machine: str
    batch_size: int


@dataclass(frozen=True)
class ResultRecord:
    """One classified raw JSON artifact and its extracted summary."""

    key: ResultKey
    path: Path
    summary: Mapping[str, Any]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for summary generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("runs/vit-inference"),
        help="Directory containing raw Demo 2 ViT JSON result artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("report/results"),
        help="Directory where generated Markdown summary tables should be written.",
    )
    return parser.parse_args(argv)


def classify_result_path(path: Path) -> ResultKey | None:
    """Classify a raw JSON filename into the summary matrix dimensions."""
    filename = path.name
    for pattern, fixed_dataset, machine in RESULT_PATTERNS:
        match = pattern.match(filename)
        if match is None:
            continue
        dataset = fixed_dataset or match.group("dataset")
        batch_size = int(match.group("batch"))
        if dataset in ALL_DATASET_ORDER and batch_size in BATCH_SIZE_ORDER:
            return ResultKey(dataset=dataset, machine=machine, batch_size=batch_size)
    return None


def discover_records(input_dir: Path) -> list[ResultRecord]:
    """Read and summarize all recognized raw JSON result files."""
    if not input_dir.exists():
        msg = f"input directory does not exist: {input_dir}"
        raise FileNotFoundError(msg)

    records: list[ResultRecord] = []
    for path in sorted(input_dir.glob("*.json"), key=lambda item: item.name):
        key = classify_result_path(path)
        if key is None:
            continue
        metrics = load_metrics(path)
        summary = apply_summary_fallbacks(summarize_result(path, metrics), metrics)
        records.append(ResultRecord(key=key, path=path, summary=summary))
    return sorted(records, key=record_sort_key)


def apply_summary_fallbacks(
    summary: Mapping[str, Any], metrics: Mapping[str, Any]
) -> dict[str, Any]:
    """Fill missing aggregate counts without including per-image predictions."""
    result = dict(summary)
    image_results = metrics.get("image_results")
    if result.get("num_images") is None and isinstance(image_results, list):
        result["num_images"] = len(image_results)

    num_images = result.get("num_images")
    batch_size = result.get("batch_size")
    if (
        result.get("num_batches") is None
        and is_number(num_images)
        and is_number(batch_size)
        and batch_size > 0
    ):
        result["num_batches"] = math.ceil(float(num_images) / float(batch_size))

    num_batches = result.get("num_batches")
    if (
        result.get("num_padded_images") is None
        and is_number(num_images)
        and is_number(num_batches)
        and is_number(batch_size)
    ):
        padded = int(num_batches) * int(batch_size) - int(num_images)
        result["num_padded_images"] = max(padded, 0)
    return result


def record_sort_key(record: ResultRecord) -> tuple[int, int, int, str]:
    """Return the deterministic sort key for records."""
    return (
        ALL_DATASET_ORDER.index(record.key.dataset),
        MACHINE_ORDER.index(record.key.machine),
        BATCH_SIZE_ORDER.index(record.key.batch_size),
        record.path.name,
    )


def build_index(records: Sequence[ResultRecord]) -> dict[ResultKey, ResultRecord]:
    """Build a lookup table keyed by dataset, machine/backend, and batch."""
    return {record.key: record for record in records}


def render_overview(
    index: Mapping[ResultKey, ResultRecord], dataset_order: Sequence[str]
) -> str:
    """Render the grouped result overview table."""
    rows = [
        [
            "Dataset",
            "Machine/backend",
            "Batch",
            "Status",
            "Backend",
            "Device kind",
            "Images",
            "Batches",
            "Padded",
            "Steps",
            "Mean step (s)",
            "Total runtime (s)",
            "Throughput (img/s)",
            "Git commit",
            "Git dirty",
            "Result JSON",
            "Command used",
        ],
        [
            "---",
            "---",
            "---:",
            "---",
            "---",
            "---",
            "---:",
            "---:",
            "---:",
            "---:",
            "---:",
            "---:",
            "---:",
            "---",
            "---",
            "---",
            "---",
        ],
    ]
    for dataset in dataset_order:
        for machine in MACHINE_ORDER:
            for batch_size in BATCH_SIZE_ORDER:
                record = index.get(ResultKey(dataset, machine, batch_size))
                rows.append(build_overview_row(dataset, machine, batch_size, record))
    return render_document("Demo 2 Imagenette 320 Result Overview", rows)


def build_overview_row(
    dataset: str, machine: str, batch_size: int, record: ResultRecord | None
) -> list[object]:
    """Build one overview table row."""
    if record is None:
        return [
            DATASET_LABELS[dataset],
            MACHINE_LABELS[machine],
            batch_size,
            "not run",
            "n/a",
            "n/a",
            "n/a",
            "n/a",
            "n/a",
            "n/a",
            "n/a",
            "n/a",
            "n/a",
            "n/a",
            "n/a",
            "not run",
            "not run",
        ]
    summary = record.summary
    return [
        DATASET_LABELS[dataset],
        MACHINE_LABELS[machine],
        batch_size,
        "run",
        value_or_na(summary.get("backend")),
        value_or_na(first_device_kind(summary)),
        value_or_na(summary.get("num_images")),
        value_or_na(summary.get("num_batches")),
        value_or_na(summary.get("num_padded_images")),
        value_or_na(summary.get("benchmark_steps")),
        format_number_or_na(summary.get("mean_step_time_sec")),
        format_number_or_na(summary.get("total_runtime_sec")),
        format_number_or_na(summary.get("throughput_images_per_sec")),
        format_git_commit(summary.get("git_commit")),
        value_or_na(summary.get("git_dirty")),
        record.path.name,
        value_or_na(summary.get("command_used")),
    ]


def render_batch_scaling(
    index: Mapping[ResultKey, ResultRecord], dataset_order: Sequence[str]
) -> str:
    """Render throughput scaling relative to b1 for each dataset and machine."""
    rows = [
        [
            "Dataset",
            "Machine/backend",
            "b1 throughput",
            "b4 throughput",
            "b4 vs b1",
            "b8 throughput",
            "b8 vs b1",
        ],
        ["---", "---", "---:", "---:", "---:", "---:", "---:"],
    ]
    for dataset in dataset_order:
        for machine in MACHINE_ORDER:
            base = throughput(index, dataset, machine, 1)
            rows.append(
                [
                    DATASET_LABELS[dataset],
                    MACHINE_LABELS[machine],
                    format_throughput(index, dataset, machine, 1),
                    format_throughput(index, dataset, machine, 4),
                    format_ratio(ratio(throughput(index, dataset, machine, 4), base)),
                    format_throughput(index, dataset, machine, 8),
                    format_ratio(ratio(throughput(index, dataset, machine, 8), base)),
                ]
            )
    return render_document("Demo 2 Imagenette 320 Batch Scaling", rows)


def render_cpu_vs_tpu(
    index: Mapping[ResultKey, ResultRecord], dataset_order: Sequence[str]
) -> str:
    """Render cross-device throughput ratio matrix."""
    rows = [
        [
            "Dataset",
            "Batch",
            "Local CPU throughput",
            "External CPU throughput",
            "Cloud TPU throughput",
            "TPU / local CPU",
            "TPU / external CPU",
            "External CPU / local CPU",
        ],
        ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
    ]
    for dataset in dataset_order:
        for batch_size in BATCH_SIZE_ORDER:
            local = throughput(index, dataset, "local_cpu", batch_size)
            external = throughput(index, dataset, "external_cpu", batch_size)
            cloud = throughput(index, dataset, "cloud_tpu", batch_size)
            rows.append(
                [
                    DATASET_LABELS[dataset],
                    batch_size,
                    format_throughput(index, dataset, "local_cpu", batch_size),
                    format_throughput(index, dataset, "external_cpu", batch_size),
                    format_throughput(index, dataset, "cloud_tpu", batch_size),
                    format_ratio(ratio(cloud, local)),
                    format_ratio(ratio(cloud, external)),
                    format_ratio(ratio(external, local)),
                ]
            )
    return render_document("Demo 2 Imagenette 320 CPU vs TPU Summary", rows)


def render_cpu_machine_comparison(index: Mapping[ResultKey, ResultRecord]) -> str:
    """Render local CPU versus external CPU throughput comparison."""
    rows = [
        [
            "Dataset",
            "Batch",
            "Local CPU throughput",
            "External Ryzen 7735HS WSL throughput",
            "External / local",
            "Local result JSON",
            "External result JSON",
        ],
        ["---", "---:", "---:", "---:", "---:", "---", "---"],
    ]
    for dataset in CPU_MACHINE_DATASET_ORDER:
        for batch_size in BATCH_SIZE_ORDER:
            local = index.get(ResultKey(dataset, "local_cpu", batch_size))
            external = index.get(ResultKey(dataset, "external_cpu", batch_size))
            rows.append(
                [
                    DATASET_LABELS[dataset],
                    batch_size,
                    format_throughput(index, dataset, "local_cpu", batch_size),
                    format_throughput(index, dataset, "external_cpu", batch_size),
                    format_ratio(
                        ratio(
                            throughput(index, dataset, "external_cpu", batch_size),
                            throughput(index, dataset, "local_cpu", batch_size),
                        )
                    ),
                    local.path.name if local is not None else "not run",
                    external.path.name if external is not None else "not run",
                ]
            )
    return render_document("Demo 2 Imagenette 320 CPU Machine Comparison", rows)


def render_public_examples_summary(index: Mapping[ResultKey, ResultRecord]) -> str:
    """Render public-example smoke/demo evidence separately from Imagenette."""
    rows = [
        [
            "Dataset",
            "Machine/backend",
            "Batch",
            "Status",
            "Backend",
            "Device kind",
            "Images",
            "Batches",
            "Padded",
            "Steps",
            "Mean step (s)",
            "Total runtime (s)",
            "Throughput (img/s)",
            "Result JSON",
        ],
        [
            "---",
            "---",
            "---:",
            "---",
            "---",
            "---",
            "---:",
            "---:",
            "---:",
            "---:",
            "---:",
            "---:",
            "---:",
            "---",
        ],
    ]
    for dataset in PUBLIC_EXAMPLES_DATASET_ORDER:
        for machine in MACHINE_ORDER:
            for batch_size in BATCH_SIZE_ORDER:
                record = index.get(ResultKey(dataset, machine, batch_size))
                rows.append(
                    build_public_examples_row(dataset, machine, batch_size, record)
                )
    note = (
        "Public examples are small reproducible smoke/demo evidence and are not "
        "part of the Imagenette 320 benchmark set."
    )
    return render_document("Demo 2 Public Examples Summary", rows, extra_note=note)


def build_public_examples_row(
    dataset: str, machine: str, batch_size: int, record: ResultRecord | None
) -> list[object]:
    """Build one public-example summary row."""
    if record is None:
        return [
            DATASET_LABELS[dataset],
            MACHINE_LABELS[machine],
            batch_size,
            "not run",
            "n/a",
            "n/a",
            "n/a",
            "n/a",
            "n/a",
            "n/a",
            "n/a",
            "n/a",
            "n/a",
            "not run",
        ]
    summary = record.summary
    return [
        DATASET_LABELS[dataset],
        MACHINE_LABELS[machine],
        batch_size,
        "run",
        value_or_na(summary.get("backend")),
        value_or_na(first_device_kind(summary)),
        value_or_na(summary.get("num_images")),
        value_or_na(summary.get("num_batches")),
        value_or_na(summary.get("num_padded_images")),
        value_or_na(summary.get("benchmark_steps")),
        format_number_or_na(summary.get("mean_step_time_sec")),
        format_number_or_na(summary.get("total_runtime_sec")),
        format_number_or_na(summary.get("throughput_images_per_sec")),
        record.path.name,
    ]


def render_document(
    title: str, rows: Sequence[Sequence[object]], extra_note: str | None = None
) -> str:
    """Render one complete generated Markdown document."""
    note = f"\n{extra_note}\n" if extra_note is not None else ""
    return (
        f"# {title}\n\n"
        "Generated by `scripts/generate_vit_summary_tables.py` from existing "
        "raw JSON artifacts under `runs/vit-inference/`.\n\n"
        f"{SCOPE_NOTE}\n"
        f"{note}\n"
        f"{render_markdown_table(rows)}\n"
    )


def render_markdown_table(rows: Sequence[Sequence[object]]) -> str:
    """Render a Markdown table with escaped cells."""
    return "\n".join(
        "| " + " | ".join(format_markdown_cell(cell) for cell in row) + " |"
        for row in rows
    )


def first_device_kind(summary: Mapping[str, Any]) -> object:
    """Return the first recorded device kind, if available."""
    devices = summary.get("devices")
    if isinstance(devices, list) and devices:
        first_device = devices[0]
        if isinstance(first_device, Mapping):
            return first_device.get("device_kind")
    return None


def throughput(
    index: Mapping[ResultKey, ResultRecord],
    dataset: str,
    machine: str,
    batch_size: int,
) -> float | None:
    """Return throughput for an existing result, if numeric."""
    record = index.get(ResultKey(dataset, machine, batch_size))
    if record is None:
        return None
    value = record.summary.get("throughput_images_per_sec")
    if is_number(value):
        return float(value)
    return None


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Return a ratio only when both values exist and denominator is positive."""
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def format_throughput(
    index: Mapping[ResultKey, ResultRecord],
    dataset: str,
    machine: str,
    batch_size: int,
) -> str:
    """Format throughput while distinguishing missing runs from missing metrics."""
    record = index.get(ResultKey(dataset, machine, batch_size))
    if record is None:
        return "not run"
    return format_number_or_na(record.summary.get("throughput_images_per_sec"))


def format_ratio(value: float | None) -> str:
    """Format a throughput ratio."""
    if value is None:
        return "n/a"
    return f"{value:.3g}x"


def format_number_or_na(value: object) -> str:
    """Format a JSON number compactly for Markdown."""
    if not is_number(value):
        return "n/a"
    return f"{float(value):.6g}"


def format_git_commit(value: object) -> str:
    """Format a Git commit for compact report tables."""
    if not isinstance(value, str) or not value:
        return "n/a"
    return value[:12]


def value_or_na(value: object) -> object:
    """Return a display value for nullable summary fields."""
    if value is None:
        return "n/a"
    return value


def format_markdown_cell(value: object) -> str:
    """Format and escape one Markdown table cell."""
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_summary_tables(records: Sequence[ResultRecord], output_dir: Path) -> None:
    """Write all curated summary Markdown tables."""
    index = build_index(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "demo2_imagenette320_overview.md": render_overview(
            index, IMAGENETTE_DATASET_ORDER
        ),
        "demo2_imagenette320_batch_scaling.md": render_batch_scaling(
            index, IMAGENETTE_DATASET_ORDER
        ),
        "demo2_imagenette320_cpu_vs_tpu.md": render_cpu_vs_tpu(
            index, IMAGENETTE_DATASET_ORDER
        ),
        "demo2_cpu_machine_comparison.md": render_cpu_machine_comparison(index),
        "demo2_public_examples_summary.md": render_public_examples_summary(index),
    }
    for filename, content in outputs.items():
        (output_dir / filename).write_text(content)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the CLI entry point."""
    args = parse_args(argv)
    records = discover_records(args.input_dir)
    write_summary_tables(records, args.output_dir)
    print(
        f"Generated Demo 2 ViT summary tables from {len(records)} raw result artifacts."
    )


if __name__ == "__main__":
    main()
