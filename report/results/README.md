# Report Results

This directory contains curated, small result artifacts that can be referenced
from the course report.

## Demo 2: Current Curated Result Artifacts

This directory currently contains small, report-ready Demo 2 local CPU
artifacts. They are local evidence only: they are not TPU results and they are
not classification accuracy evaluations.

### Legacy Single-Image JSON Artifacts

The original Demo 2 JSON files are legacy single-image repeated-batch local CPU
artifacts for pretrained ViT inference with `google/vit-base-patch16-224`:

- `demo2_vit_local_cpu_b1.json`
- `demo2_vit_local_cpu_b4.json`
- `demo2_vit_local_cpu_b8.json`

These runs used the sample image:

```text
examples/assets/chihuahua_pet_licorice.jpg
```

These runs repeat the same sample image along the batch dimension and time
JAX/Flax inference after warmup steps. They are useful for local runtime and
throughput comparison, but they are not dataset-level accuracy evaluation.

The report-ready table generated from these legacy JSON files is:

- `demo2_vit_single_image_local_cpu.md`

Regenerate it with:

```bash
uv run python scripts/compare_vit_results.py \
  report/results/demo2_vit_local_cpu_b1.json \
  report/results/demo2_vit_local_cpu_b4.json \
  report/results/demo2_vit_local_cpu_b8.json \
  --markdown-output report/results/demo2_vit_single_image_local_cpu.md
```

The current curated JSON files are legacy local CPU artifacts and may not
include every field emitted by newer Demo 2 runs. The comparison helper infers
stable summary fields where possible without changing the original artifacts.

### Imagenette 320 Local CPU Tables

Imagenette 320 (`imagenette2-320`) is the recommended optional local benchmark
dataset for later Demo 2 work. Keep the dataset and generated manifests under
ignored `data/local/imagenette2-320/`.

- `demo2_imagenette320_val64_cpu.md`: primary current local Imagenette CPU
  benchmark table using `manifest_val_64.txt`.
- `demo2_imagenette320_val256_cpu.md`: extended local Imagenette CPU benchmark
  table using `manifest_val_256.txt`.

Regenerate the primary val64 table after the JSON files exist with:

```bash
uv run python scripts/compare_vit_results.py \
  runs/vit-inference/demo2_imagenette320_val64_cpu_b1.json \
  runs/vit-inference/demo2_imagenette320_val64_cpu_b4.json \
  runs/vit-inference/demo2_imagenette320_val64_cpu_b8.json \
  --markdown-output report/results/demo2_imagenette320_val64_cpu.md
```

Raw JSON outputs under `runs/vit-inference/` are not committed by default.
Curate small report tables under this directory when they are ready to cite.

### External-Machine Naming

Keep standard curated baseline names generic, such as
`demo2_imagenette320_val64_cpu.md`. Exploratory external-machine artifacts may
include a neutral machine or environment label, for example
`demo2_asus_a16_ryzen7735hs_wsl_imagenette320_val64_cpu.md`.

CPU artifacts should not include a GPU model such as `rx7600s`. Reserve
`rx7600s` and `rocm` labels for explicit ROCm/GPU sanity-check artifacts, and
only when actual ROCm/GPU evidence exists.

### Private Local Live-Demo Table

- `demo2_private_local_cpu.md`: private local live-demo evidence from local
  images under ignored `data/local/`. This is qualitative local evidence, not a
  public reproducible benchmark dataset and not an accuracy evaluation.

Do not commit Imagenette images, private images, local manifests, raw logs, or
`runs/` artifacts. TPU execution, monitoring, cleanup, and local-vs-TPU
comparison remain planned work.
