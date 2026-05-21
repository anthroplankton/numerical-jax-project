# Report Results

This directory contains curated, small result artifacts that can be referenced
from the course report.

## Demo 2: Current Curated Result Artifacts

This directory contains report-ready Markdown summaries for Demo 2 CPU evidence.
The raw JSON benchmark outputs live under ignored `runs/vit-inference/` and are
treated as generated local artifacts. Do not normally commit raw JSON outputs.

Each Markdown table in this directory should correspond to real JSON artifacts
and should be generated with `scripts/compare_vit_results.py --markdown-output`.
For example:

```bash
uv run python scripts/compare_vit_results.py \
  runs/vit-inference/demo2_local_public_examples_cpu_b1.json \
  runs/vit-inference/demo2_local_public_examples_cpu_b4.json \
  runs/vit-inference/demo2_local_public_examples_cpu_b8.json \
  --markdown-output report/results/demo2_local_public_examples_cpu.md
```

These tables summarize inference timing and throughput. They are not
classification accuracy evaluations, GPU results, or TPU results.

### Primary Local CPU Tables

The local-machine CPU tables are the primary current-machine evidence:

- `demo2_local_public_examples_cpu.md`: public example image set, `b1`, `b4`,
  and `b8`.
- `demo2_local_imagenette320_val64_cpu.md`: local Imagenette 320 validation
  manifest with 64 images, `b1`, `b4`, and `b8`.
- `demo2_local_imagenette320_val256_cpu.md`: local Imagenette 320 validation
  manifest with 256 images, `b1`, `b4`, and `b8`.
- `demo2_local_private_examples_cpu.md`: private local live-demo image set,
  `b1`, `b4`, and `b8`.

The Imagenette and private inputs live under ignored `data/local/` paths. Do not
commit Imagenette images, private images, or local manifests. For private local
examples, only the curated Markdown table may be committed.

### Supplementary External CPU Tables

The external Ryzen 7735HS WSL tables are supplementary CPU evidence. Keep them
separate from the local-machine tables:

- `demo2_external_ryzen7735hs_wsl_public_examples_cpu.md`: public example image
  set, `b1` and `b4` only. External public `b8` is pending and should not be
  fabricated.
- `demo2_external_ryzen7735hs_wsl_imagenette320_val64_cpu.md`: Imagenette 320
  validation manifest with 64 images, `b1`, `b4`, and `b8`.
- `demo2_external_ryzen7735hs_wsl_imagenette320_val256_cpu.md`: Imagenette 320
  validation manifest with 256 images, `b1`, `b4`, and `b8`.

External-machine labels should stay neutral and environment-based, such as
`external_ryzen7735hs_wsl`. CPU artifacts should not include a GPU model such as
`rx7600s`. Reserve `rx7600s` and `rocm` labels for explicit ROCm/GPU
sanity-check artifacts, and only when actual ROCm/GPU evidence exists.

### Pending TPU Evidence

TPU execution, TPU JSON artifacts, cloud monitoring evidence, cleanup evidence,
and CPU-vs-TPU comparison tables remain pending. Do not claim CPU-vs-TPU
comparison results until a real TPU JSON artifact exists.
