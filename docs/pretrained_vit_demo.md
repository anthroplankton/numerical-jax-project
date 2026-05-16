# Demo 2: Pretrained ViT Inference Benchmark

Demo 2 is a local compatibility spike for running pretrained Vision Transformer
inference with JAX/Flax. The default model is
`google/vit-base-patch16-224` from Hugging Face.

Because of current course presentation constraints, this is the primary demo
path for the project. Demo 1 remains preserved as background raw-JAX CNN work,
and Demo 3 remains optional future work.

This demo is inference-only. It does not fine-tune the model, and it does not
claim TPU execution has been completed. The next planned step is a conservative
Google Cloud TPU VM workflow for running the same inference benchmark with
`--jax-platform tpu`.

## Setup

The pretrained dependencies are optional and live in the `pretrained` dependency
group:

```bash
uv sync --group pretrained
```

The first run downloads the image processor and model weights from Hugging Face
unless they are already present in the local cache. This can take time and uses
network bandwidth and disk space.

## Sample Image

The repository includes a small public-domain sample image for reproducible
classroom demos:

```text
examples/assets/chihuahua_pet_licorice.jpg
```

The image is `chihuahua_pet_licorice.jpg` from Wikimedia Commons. It is included
only as a small, stable input image for Demo 2. Do not commit private images,
large datasets, model caches, or downloaded model weights.

## Run Command

Run a small local benchmark:

```bash
uv run --group pretrained python examples/pretrained_vit_inference.py \
  --jax-platform cpu \
  --image examples/assets/chihuahua_pet_licorice.jpg \
  --batch-size 1 \
  --warmup-steps 1 \
  --benchmark-steps 5 \
  --output runs/vit-inference/metrics.json
```

For a larger repeated-image batch:

```bash
uv run --group pretrained python examples/pretrained_vit_inference.py \
  --jax-platform cpu \
  --image examples/assets/chihuahua_pet_licorice.jpg \
  --batch-size 4 \
  --warmup-steps 2 \
  --benchmark-steps 10 \
  --output runs/vit-inference/batch4_metrics.json
```

## Expected Output

The script prints JSON metrics and writes the same payload to `--output`.
Expected fields are:

```json
{
  "model_name": "google/vit-base-patch16-224",
  "selected_jax_platform": "cpu",
  "backend": "cpu",
  "devices": [
    {
      "platform": "cpu",
      "device_kind": "cpu",
      "id": 0,
      "repr": "TFRT_CPU_0"
    }
  ],
  "input_shape": [1, 3, 224, 224],
  "batch_size": 1,
  "warmup_steps": 1,
  "benchmark_steps": 5,
  "mean_step_time_sec": 0.123,
  "throughput_images_per_sec": 8.13,
  "predicted_index": 285,
  "predicted_label": "Egyptian cat"
}
```

The exact backend, device list, prediction, and timing values depend on the
machine, installed JAX build, image content, and local cache state.

## Model Notes

- `google/vit-base-patch16-224` is a pretrained ViT image-classification model.
- The script uses `AutoImageProcessor` for preprocessing and
  `FlaxViTForImageClassification` for JAX/Flax inference.
- The input image is converted to RGB, preprocessed to model tensor format, then
  repeated along the batch dimension according to `--batch-size`.
- The benchmark uses warmup steps before timed steps and calls
  `block_until_ready()` so asynchronous JAX execution is included in the timing.
- The `--jax-platform` option accepts `default`, `cpu`, `cuda`, or `tpu`. The
  classroom command uses `cpu` for stable local evidence. The actual backend and
  devices are still recorded from JAX at runtime.

## Observed Local CPU Runs

The following curated local CPU artifacts were produced after the model download
succeeded:

| Artifact | Batch size | Predicted label | Mean step time | Throughput |
| --- | ---: | --- | ---: | ---: |
| `report/results/demo2_vit_local_cpu_b1.json` | 1 | Chihuahua | 0.18744530999993003 s | 5.334889413879565 images/s |
| `report/results/demo2_vit_local_cpu_b4.json` | 4 | Chihuahua | 0.8838171279999187 s | 4.5258231293299485 images/s |
| `report/results/demo2_vit_local_cpu_b8.json` | 8 | Chihuahua | 1.973294752699894 s | 4.054133316401044 images/s |

These runs are single-image repeated-batch inference baselines, not
dataset-level accuracy evaluation.

## Local CUDA Limitation

On the laptop used for this local spike, a simple JAX GPU matrix multiplication
worked. The ViT-like convolution path failed during cuDNN autotuning, so local
CUDA is not used as Demo 2 benchmark evidence. This project does not claim GPU
ViT inference success for Demo 2.

## Limitations

- This is a local compatibility spike, not a final benchmark result.
- The default pytest suite does not download the model or require Hugging Face
  network access.
- First-run download time is not part of the timed inference loop.
- The same single image is repeated for batch benchmarking, so throughput is a
  systems-oriented inference measurement rather than a dataset-level evaluation.
- TPU execution, monitoring, cleanup, and local-vs-TPU comparison remain planned
  work and have not been completed by this demo.

## Next Planned Step

Prepare and test a Google Cloud TPU VM workflow for Demo 2 using
`examples/pretrained_vit_inference.py --jax-platform tpu`, then capture TPU
metrics, logs, monitoring notes, and cleanup evidence if resources are created.
