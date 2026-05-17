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

## Private Local Image Set

For a live classroom demo with private photos, keep local-only files under:

```text
data/local/demo2_vit_images/
```

This path is ignored by Git through `data/local/`, so private photos and the
local manifest should not be committed.

Recommended naming convention:

```text
data/local/demo2_vit_images/
├── demo2_private_001.jpg
├── demo2_private_002.jpg
├── demo2_private_003.jpg
└── manifest.txt
```

Use a manifest when you want a stable demo order. The manifest is a plain text
file with one image path per line. Blank lines and lines starting with `#` are
ignored. Relative paths are resolved from the manifest directory:

```text
# data/local/demo2_vit_images/manifest.txt
demo2_private_001.jpg
demo2_private_002.jpg
demo2_private_003.jpg
```

Run the private local set:

```bash
uv run --group pretrained python examples/pretrained_vit_inference.py \
  --jax-platform cpu \
  --image-manifest data/local/demo2_vit_images/manifest.txt \
  --batch-size 4 \
  --warmup-steps 1 \
  --benchmark-steps 5 \
  --output runs/vit-inference/demo2_private_local.json
```

The public `--image examples/assets/chihuahua_pet_licorice.jpg` path remains the
reproducible command for other users.

The private manifest workflow is for a local live demo only. It is useful for
showing qualitative pretrained predictions on a few local images, but it is not
a public benchmark dataset and it is not an accuracy benchmark unless explicit
ground-truth labels and top-k evaluation are added later.

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
Expected fields are shown below. The `top5` examples are abbreviated for
readability; the default output contains five entries unless `--top-k` changes
that count.

```json
{
  "mode": "single_image",
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
  "total_timed_inference_sec": 0.615,
  "throughput_images_per_sec": 8.13,
  "top1_index": 151,
  "top1_label": "Chihuahua",
  "top5": [
    {
      "index": 151,
      "label": "Chihuahua",
      "score": 0.9
    }
  ],
  "predicted_index": 151,
  "predicted_label": "Chihuahua"
}
```

The exact backend, device list, prediction, and timing values depend on the
machine, installed JAX build, image content, and local cache state. Manifest
runs use a different top-level schema so first-image predictions are not
mistaken for whole-run predictions:

```json
{
  "mode": "image_manifest",
  "manifest_path": "data/local/demo2_vit_images/manifest.txt",
  "input_shape": [4, 3, 224, 224],
  "processing_mode": "batched_manifest",
  "num_images": 10,
  "num_batches": 3,
  "timed_batch_runs": 15,
  "num_padded_images": 2,
  "last_batch_policy": "pad_with_last_image",
  "mean_step_time_sec": 0.2,
  "total_timed_inference_sec": 3.0,
  "throughput_images_per_sec": 16.67,
  "image_results": [
    {
      "image_path": "data/local/demo2_vit_images/demo2_chihuahua_pet_licorice_public.jpg",
      "input_shape": [3, 224, 224],
      "batch_index": 0,
      "position_in_batch": 0,
      "top1_index": 151,
      "top1_label": "Chihuahua",
      "top5": [
        {
          "index": 151,
          "label": "Chihuahua",
          "score": 0.9
        }
      ]
    }
  ]
}
```

Each `image_results` entry includes that image's input shape, batch index,
position inside its batch, and qualitative top-k predictions. Manifest timing is
reported at the batch/run level rather than as separate per-image timing.

## Model Notes

- `google/vit-base-patch16-224` is a pretrained ViT image-classification model.
- The script uses `AutoImageProcessor` for preprocessing and
  `FlaxViTForImageClassification` for JAX/Flax inference.
- In single-image mode, the input image is converted to RGB, preprocessed to
  model tensor format, then repeated along the batch dimension according to
  `--batch-size`.
- `--image` runs one image. `--image-manifest` runs a small local image set and
  writes per-image results under `image_results`.
- Manifest mode loads all manifest images, preprocesses each image to
  `[3, 224, 224]`, stacks them into batches shaped
  `[batch_size, 3, 224, 224]`, and runs one model call per batch.
- The final partial manifest batch is padded by repeating its last real image so
  every timed model call has the same batch shape. Padded entries are ignored
  for predictions and throughput counts real manifest images only.
- Prediction fields are qualitative pretrained model outputs. The script records
  `top1_index`, `top1_label`, and a default five-entry `top5` list with class
  indices, labels, and scores.
- `--top-k` controls how many entries are written to the `top5` list; the
  default is 5.
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
- Single-image mode repeats the same image for batch benchmarking. Manifest mode
  uses real mixed-image batches, with final-batch padding if needed. Both modes
  remain systems-oriented inference measurements rather than dataset-level
  evaluations.
- Private manifest runs are qualitative local demonstrations, not public
  benchmark datasets or classification-accuracy measurements.
- TPU execution, monitoring, cleanup, and local-vs-TPU comparison remain planned
  work and have not been completed by this demo.

## Next Planned Step

Prepare and test a Google Cloud TPU VM workflow for Demo 2 using
`examples/pretrained_vit_inference.py --jax-platform tpu`, then capture TPU
metrics, logs, monitoring notes, and cleanup evidence if resources are created.
