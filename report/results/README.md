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

These artifacts are not GPU results and not TPU results. TPU execution,
monitoring, cleanup, and local-vs-TPU comparison remain planned work.
