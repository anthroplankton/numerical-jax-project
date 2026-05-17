# Report Results

This directory contains curated, small result artifacts that can be referenced
from the course report.

## Demo 2: ViT Local CPU Baseline

The Demo 2 JSON files are local CPU baseline evidence for pretrained ViT
inference with `google/vit-base-patch16-224`:

- `demo2_vit_local_cpu_b1.json`
- `demo2_vit_local_cpu_b4.json`
- `demo2_vit_local_cpu_b8.json`

These runs used the sample image:

```text
examples/assets/chihuahua_pet_licorice.jpg
```

The benchmark repeats the same single image along the batch dimension and times
JAX/Flax inference after warmup steps. These files are useful for local runtime
and throughput comparison, but they are not dataset-level accuracy evaluation.

Generate a report-ready Markdown table from these existing JSON files:

```bash
uv run python scripts/compare_vit_results.py \
  report/results/demo2_vit_local_cpu_b1.json \
  report/results/demo2_vit_local_cpu_b4.json \
  report/results/demo2_vit_local_cpu_b8.json \
  --markdown-output runs/vit-inference/demo2_local_cpu_table.md
```

The current curated JSON files are legacy local CPU artifacts and may not
include every field emitted by newer Demo 2 runs. The comparison helper infers
stable summary fields where possible without changing the original artifacts.

These artifacts are not GPU results and not TPU results. TPU execution,
monitoring, cleanup, and local-vs-TPU comparison remain planned work.

## Demo 2: Imagenette 320 Local CPU Tables

Imagenette 320 (`imagenette2-320`) is the recommended optional local benchmark
dataset for later Demo 2 work. Keep the dataset and generated manifests under
ignored `data/local/imagenette2-320/`. After creating
`data/local/imagenette2-320/val/manifest_val_64.txt` and running the `b1`,
`b4`, and `b8` CPU benchmarks documented in `docs/pretrained_vit_demo.md`,
generate a report-ready table with:

```bash
uv run python scripts/compare_vit_results.py \
  runs/vit-inference/demo2_imagenette320_val64_cpu_b1.json \
  runs/vit-inference/demo2_imagenette320_val64_cpu_b4.json \
  runs/vit-inference/demo2_imagenette320_val64_cpu_b8.json \
  --markdown-output report/results/demo2_imagenette320_val64_cpu.md
```

Do not commit Imagenette images or local manifests under `data/local/`. Commit
the Markdown table only after the JSON inputs represent intentional curated
local CPU evidence.
