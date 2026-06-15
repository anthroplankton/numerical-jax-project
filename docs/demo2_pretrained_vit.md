# Demo 2: Pretrained ViT Inference And Head Fine-Tuning

Demo 2 is now a Vision Transformer workflow with JAX/Flax. Its completed
evidence is an inference benchmark workflow.
It uses raw JSON outputs under ignored `runs/vit-inference/`, curated
report-ready Markdown tables under `report/results/`, manifest batching, public
example assets, supplementary external CPU evidence, a completed Google Cloud
TPU public-example smoke run, and retrieved Imagenette 320 TPU inference
artifacts. The default model is `google/vit-base-patch16-224` from Hugging Face.
The optional head fine-tuning extension writes generated smoke-run artifacts
under ignored `runs/vit-finetune/`.

Because of current course presentation constraints, this is the primary demo
path for the project. Demo 1 remains preserved as background raw-JAX CNN work,
and Demo 3 remains optional future work.

TPU inference evidence now includes a small public-example smoke run plus
Imagenette 320 `val64`, `val256`, and `val_full` inference timing tables. The
optional fine-tuning extension is part of Demo 2, not Demo 3, and has produced
classifier-head-only TPU smoke workflow evidence with GCS checkpoint
restore/resume. It should still be described narrowly: workflow,
checkpoint/resume, and TPU execution evidence, not full ViT fine-tuning, not a
dataset-level accuracy evaluation, and not a full controlled benchmark. For
actual TPU execution, use
`cloud/demo2_tpu_quickstart.md`. For resource variants, monitoring/evidence
guidance, cleanup discipline, and course evidence appendices, see
`cloud/demo2_pretrained_vit_tpu_workflow.md`.

## Setup

The pretrained dependencies are optional and live in the `pretrained` dependency
group:

```bash
uv sync --frozen --group pretrained
```

The optional fine-tuning extension also uses the `training` dependency group:

```bash
uv sync --frozen --group pretrained --group training
```

The first run downloads the image processor and model weights from Hugging Face
unless they are already present in the local cache. This can take time and uses
network bandwidth and disk space.

## Fresh Benchmark Machine Setup

Run from an Ubuntu or WSL terminal at the repository root:

```bash
uv sync --frozen --group dev --group pretrained
bash scripts/check_jax_device.sh
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

`scripts/check_jax_device.sh` defaults `JAX_PLATFORMS` to `cpu` unless the
caller already set it. This keeps the local CPU sanity path explicit and
avoids misleading GPU plugin initialization during CPU-only checks on machines
with unrelated GPU drivers installed.

After the environment checks pass, run a public five-image manifest smoke
benchmark:

```bash
uv run --group pretrained python examples/pretrained_vit_inference.py \
  --jax-platform cpu \
  --image-manifest examples/assets/manifest.txt \
  --batch-size 4 \
  --warmup-steps 1 \
  --benchmark-steps 1 \
  --output runs/vit-inference/demo2_public_examples_smoke_cpu_b4.json
```

This smoke filename is temporary local output. It is not automatically included
in the generated report summaries, which use the curated
`demo2_local_public_examples_cpu_b*.json` artifact naming scheme.

Optional Imagenette benchmarks require separately prepared local data. Current
curated local and supplementary external CPU tables exist, but project scripts
and tests do not download Imagenette, and pytest/CI should not depend on that
dataset.

## Public Example Images

The repository includes five public Wikimedia Commons images for reproducible
public demos:

```text
examples/assets/chihuahua_pet_licorice.jpg
examples/assets/adelie_penguins_brooding.jpg
examples/assets/doge_homemade_meme.jpg
examples/assets/polar_bear_zoo_face.jpg
examples/assets/black_cat_staring_closeup.jpg
examples/assets/manifest.txt
```

The single-image smoke test still uses `chihuahua_pet_licorice.jpg`. The public
manifest runs the five-image public example set after cloning the repository.
These images are qualitative Demo 2 inputs, not a public benchmark dataset or an
accuracy benchmark. Do not commit private images, large datasets, model caches,
or downloaded model weights.

## Private Local Image Set

For a private local demo with private photos, keep local-only files under:

```text
data/local/demo2_vit_images/
```

This path is ignored by Git through `data/local/`, so private photos and the
local manifest should not be committed.

The current local live-demo manifest is expected to contain 15 images: the
original local set, the local banana image, and local copies of the four public
Wikimedia examples.

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
  --output runs/vit-inference/demo2_local_private_examples_cpu_b4.json
```

The public `--image examples/assets/chihuahua_pet_licorice.jpg` path remains the
reproducible command for other users.

The private manifest workflow is for a local live demo only. It is useful for
showing qualitative pretrained predictions on a few local images, but it is not
a public benchmark dataset and it is not an accuracy benchmark unless explicit
ground-truth labels and top-k evaluation are added later.

## Optional Local Imagenette 320 Benchmark Data

Imagenette 320 (`imagenette2-320`) is the recommended local benchmark dataset
for optional Demo 2 CPU evidence. Current curated local and supplementary
external CPU Markdown tables exist for `val64` and `val256`, but the workflow is
still local-only: the repository does not download Imagenette automatically,
default tests do not depend on it, and CI should not require it. Keep extracted
files and generated manifests under ignored `data/local/imagenette2-320/`:

```text
data/local/imagenette2-320/
```

Before running the Imagenette commands below, the validation directory must
already exist:

```text
data/local/imagenette2-320/val
```

### Download and extract

Run from the repository root. This downloads the official fastai Imagenette 320
archive into ignored local storage and extracts it under `data/local/`:

```bash
mkdir -p data/local
curl -L --fail --show-error -o data/local/imagenette2-320.tgz https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar -xzf data/local/imagenette2-320.tgz -C data/local
test -d data/local/imagenette2-320/val
find data/local/imagenette2-320/val -type f | wc -l
```

Imagenette files remain under ignored `data/local/`, and generated manifests
remain under ignored `data/local/imagenette2-320/`. Raw JSON benchmark outputs
remain under ignored `runs/vit-inference/`. Tests and CI must not require
Imagenette, and project scripts still do not automatically download it:
`scripts/build_image_manifest.py` only scans existing local images and does not
create the Imagenette directory tree.

Build manifests before running benchmarks. Use these names for the validation
split:

```text
data/local/imagenette2-320/val/manifest_val_64.txt
data/local/imagenette2-320/val/manifest_val_256.txt
data/local/imagenette2-320/val/manifest_val_full.txt
```

Lightweight 64-image manifest:

```bash
uv run python scripts/build_image_manifest.py \
  data/local/imagenette2-320/val \
  --output data/local/imagenette2-320/val/manifest_val_64.txt \
  --limit 64

wc -l data/local/imagenette2-320/val/manifest_val_64.txt
head data/local/imagenette2-320/val/manifest_val_64.txt
```

Larger 256-image manifest:

```bash
uv run python scripts/build_image_manifest.py \
  data/local/imagenette2-320/val \
  --output data/local/imagenette2-320/val/manifest_val_256.txt \
  --limit 256

wc -l data/local/imagenette2-320/val/manifest_val_256.txt
head data/local/imagenette2-320/val/manifest_val_256.txt
```

Optional full validation manifest:

```bash
uv run python scripts/build_image_manifest.py \
  data/local/imagenette2-320/val \
  --output data/local/imagenette2-320/val/manifest_val_full.txt

wc -l data/local/imagenette2-320/val/manifest_val_full.txt
head data/local/imagenette2-320/val/manifest_val_full.txt
```

The helper scans local image files only. It does not download data, decode
images, require labels, or add files to Git. Do not commit Imagenette images or
the generated manifests under `data/local/`. The generated manifest includes
one header line, so `wc -l` reports one more line than the `--limit` image
count.

For cloud TPU Imagenette runs, preserve the same repository-relative paths on
the TPU VM. Either download and extract Imagenette 320 on the TPU VM, or copy a
prepared local `data/local/imagenette2-320/` directory to the TPU VM. The
benchmark commands expect:

```text
data/local/imagenette2-320/val
data/local/imagenette2-320/val/manifest_val_64.txt
data/local/imagenette2-320/val/manifest_val_256.txt
data/local/imagenette2-320/val/manifest_val_full.txt
```

Keep `data/local/`, generated manifests, dataset files, model caches, and raw
JSON benchmark outputs uncommitted.

## Optional Classifier-Head Fine-Tuning Extension

`examples/demo2_pretrained_vit_finetune.py` is a small Demo 2 training smoke
extension. It keeps the pretrained ViT backbone frozen and updates only the
classifier head. Orbax manages checkpoint/resume, and the checkpoint payload is
limited to classifier-head parameters, optimizer state, current step, and small
metadata. It does not save the frozen ViT backbone or Hugging Face model cache.

This extension is not a new Demo 3, not full ViT fine-tuning, not a model
quality study, and not an Imagenette accuracy benchmark. Any accuracy-like
numbers written during the smoke run are helper metrics for the listed manifest
only.

Prepare small path-only Imagenette manifests from existing local data. The
script derives labels from Imagenette class directory names such as
`n01440764/` and maps them to the corresponding ImageNet class indices used by
`google/vit-base-patch16-224`. For report-friendly learning curves, prefer the
balanced manifest mode so the tiny smoke input is less class-skewed:

```bash
uv run python scripts/build_image_manifest.py \
  data/local/imagenette2-320/train \
  --output data/local/imagenette2-320/train/manifest_train_balanced_50.txt \
  --per-class-limit 5

uv run python scripts/build_image_manifest.py \
  data/local/imagenette2-320/val \
  --output data/local/imagenette2-320/val/manifest_val_balanced_50.txt \
  --per-class-limit 5
```

The original global-limit form is still available when class balance is not the
goal:

```bash
uv run python scripts/build_image_manifest.py \
  data/local/imagenette2-320/train \
  --output data/local/imagenette2-320/train/manifest_train_64.txt \
  --limit 64

uv run python scripts/build_image_manifest.py \
  data/local/imagenette2-320/val \
  --output data/local/imagenette2-320/val/manifest_val_64.txt \
  --limit 64
```

These generated manifests are small smoke inputs. Even the balanced form is not
a dataset-level evaluation protocol; it only makes class distribution easier to
inspect in `summary.json`.

Run a local CPU smoke test:

```bash
uv run --group pretrained --group training python examples/demo2_pretrained_vit_finetune.py \
  --jax-platform cpu \
  --train-manifest data/local/imagenette2-320/train/manifest_train_balanced_50.txt \
  --eval-manifest data/local/imagenette2-320/val/manifest_val_balanced_50.txt \
  --batch-size 8 \
  --learning-rate 0.001 \
  --max-steps 20 \
  --checkpoint-every-steps 10 \
  --checkpoint-every-seconds 30 \
  --eval-every-steps 5 \
  --checkpoint-dir runs/vit-finetune/demo2_local_balanced50_cpu/checkpoints \
  --output-dir runs/vit-finetune/demo2_local_balanced50_cpu \
  --save-predictions
```

By default the script starts from the pretrained classifier head. For an
optional learning-curve demonstration, add `--reinit-head --seed 0` to
randomly reinitialize only the classifier head while keeping the ViT backbone
frozen. This mode is useful for plotting a clearer loss curve; it is not the
default evidence path and is not a model-quality claim.

Expected generated artifacts:

```text
runs/vit-finetune/demo2_local_balanced50_cpu/summary.json
runs/vit-finetune/demo2_local_balanced50_cpu/metrics.csv
runs/vit-finetune/demo2_local_balanced50_cpu/eval_metrics.csv
runs/vit-finetune/demo2_local_balanced50_cpu/predictions_before.json
runs/vit-finetune/demo2_local_balanced50_cpu/predictions_after.json
runs/vit-finetune/demo2_local_balanced50_cpu/train.log
runs/vit-finetune/demo2_local_balanced50_cpu/checkpoints/
```

`summary.json` records the mode, model name, trainable scope
`classifier_head_only`, frozen scope `vit_backbone`, backend/devices, manifests,
label counts, class counts, batch size, learning rate, `eval_every_steps`,
`reinit_head`, seed, start/final step, resume status, checkpoint path,
initial/final loss, step timing, throughput, total runtime, and privacy-safe Git
metadata when available. `metrics.csv` is the per-step training CSV.
`eval_metrics.csv` has `step`, `eval_loss`, and `eval_accuracy` rows for the
initial state, final state, and optional periodic eval points. In
`summary.json`, `mean_step_time_sec` and `examples_per_second` measure
training-step execution time and exclude checkpoint write time, while
`total_runtime_sec` includes setup, evaluation, checkpointing, prediction
writing, and summary writing overhead.

For notebook-based report plots, load the local ignored artifacts directly:
`summary.json`, `metrics.csv`, `eval_metrics.csv`,
`predictions_before.json`, and `predictions_after.json`. Commit only curated
derived summaries or small report-ready Markdown under `report/results/`, not
raw checkpoints, logs, datasets, model caches, or generated notebook outputs.
Near-zero loss in a tiny `train64`/`val64` or balanced smoke setup can happen if
the selected subset is easy, class-skewed, or already aligned with the
pretrained ImageNet classifier head; do not interpret it as Imagenette
dataset-level accuracy.

For TPU runs, use absolute `RUN_DIR` and `CKPT_DIR` values before passing
`--output-dir` and `--checkpoint-dir`. Orbax writes local checkpoint files
first; the GCS workflow in `cloud/demo2_tpu_quickstart.md` copies those
checkpoint directories to durable storage for resume after spot interruption,
maintenance, or TPU VM deletion risk. Durable GCS copies should remain outside
Git along with raw logs, predictions, datasets, and model caches.

## Imagenette 320 Local CPU Benchmark Commands

Use `manifest_val_64.txt` as a lightweight documented local benchmark input.
These runs measure model inference after preprocessing; they do not include full
dataset loading time and they do not compute Imagenette accuracy.

Val64 `b1`:

```bash
uv run --group pretrained python examples/pretrained_vit_inference.py \
  --jax-platform cpu \
  --image-manifest data/local/imagenette2-320/val/manifest_val_64.txt \
  --batch-size 1 \
  --warmup-steps 1 \
  --benchmark-steps 5 \
  --output runs/vit-inference/demo2_local_imagenette320_val64_cpu_b1.json
```

Val64 `b4`:

```bash
uv run --group pretrained python examples/pretrained_vit_inference.py \
  --jax-platform cpu \
  --image-manifest data/local/imagenette2-320/val/manifest_val_64.txt \
  --batch-size 4 \
  --warmup-steps 1 \
  --benchmark-steps 5 \
  --output runs/vit-inference/demo2_local_imagenette320_val64_cpu_b4.json
```

Val64 `b8`:

```bash
uv run --group pretrained python examples/pretrained_vit_inference.py \
  --jax-platform cpu \
  --image-manifest data/local/imagenette2-320/val/manifest_val_64.txt \
  --batch-size 8 \
  --warmup-steps 1 \
  --benchmark-steps 5 \
  --output runs/vit-inference/demo2_local_imagenette320_val64_cpu_b8.json
```

When local runtime is acceptable, `manifest_val_256.txt` can use the same `b1`,
`b4`, and `b8` command pattern by replacing `val64` paths and output filenames
with `val256`. The current pre-TPU progress report uses the `val256` curated
tables as the main CPU benchmark evidence because b1/b4/b8 all use 256 real
images with 0 padded images.

Generate a report-ready local CPU table after the JSON files exist:

```bash
uv run python scripts/compare_vit_results.py \
  runs/vit-inference/demo2_local_imagenette320_val64_cpu_b1.json \
  runs/vit-inference/demo2_local_imagenette320_val64_cpu_b4.json \
  runs/vit-inference/demo2_local_imagenette320_val64_cpu_b8.json \
  --markdown-output report/results/demo2_local_imagenette320_val64_cpu.md
```

Only commit the generated Markdown table if it is intentionally curated for the
report. Do not commit raw `runs/` outputs, Imagenette images, or local manifests.

## External-Machine Artifact Names

Standard curated table names should include the evidence scope, input set, and
platform, for example `demo2_local_imagenette320_val64_cpu.md`. Supplementary
external-machine artifacts should use a neutral environment label, such as
`demo2_external_ryzen7735hs_wsl_imagenette320_val64_cpu.md`.

The current supplementary external public examples table has `b1` and `b4`
only. External public `b8` is pending and should not be fabricated.

For CPU artifacts, avoid including a GPU model in the file name. Reserve labels
such as `rx7600s` and `rocm` for explicit ROCm/GPU sanity-check artifacts with
actual ROCm/GPU evidence. Docker or ROCm setup is optional AMD GPU
sanity-check work, not part of the default local CPU benchmark setup.

## Run Command

Run the formal local public examples `b1` benchmark:

```bash
uv run --group pretrained python examples/pretrained_vit_inference.py \
  --jax-platform cpu \
  --image-manifest examples/assets/manifest.txt \
  --batch-size 1 \
  --warmup-steps 1 \
  --benchmark-steps 5 \
  --output runs/vit-inference/demo2_local_public_examples_cpu_b1.json
```

Run the formal public five-image manifest `b4` benchmark:

```bash
uv run --group pretrained python examples/pretrained_vit_inference.py \
  --jax-platform cpu \
  --image-manifest examples/assets/manifest.txt \
  --batch-size 4 \
  --warmup-steps 1 \
  --benchmark-steps 5 \
  --output runs/vit-inference/demo2_local_public_examples_cpu_b4.json
```

## Expected Output

The script prints JSON metrics and writes the same payload to `--output`.
Expected fields are shown below. The `top5` examples are abbreviated for
readability; the default output contains five entries unless `--top-k` changes
that count.

```json
{
  "mode": "single_image",
  "processing_mode": "repeated_single_image",
  "command_used": "python examples/pretrained_vit_inference.py --jax-platform cpu ...",
  "output_path": "runs/vit-inference/metrics.json",
  "git_commit": "0123456789abcdef0123456789abcdef01234567",
  "git_branch": "feat/demo2-tpu-evidence",
  "git_dirty": false,
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
  "num_images": 1,
  "num_batches": 1,
  "timed_batch_runs": 5,
  "num_padded_images": 0,
  "last_batch_policy": "none",
  "mean_step_time_sec": 0.123,
  "total_timed_inference_sec": 0.615,
  "throughput_counted_images": 5,
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
runs use different top-level result fields so first-image predictions are not
mistaken for whole-run predictions:

```json
{
  "mode": "image_manifest",
  "command_used": "python examples/pretrained_vit_inference.py --jax-platform cpu ...",
  "output_path": "runs/vit-inference/demo2_local_private_examples_cpu_b4.json",
  "git_commit": "0123456789abcdef0123456789abcdef01234567",
  "git_branch": "feat/demo2-tpu-evidence",
  "git_dirty": false,
  "manifest_path": "data/local/demo2_vit_images/manifest.txt",
  "manifest_kind": "local_private",
  "input_shape": [4, 3, 224, 224],
  "processing_mode": "batched_manifest",
  "num_images": 15,
  "num_batches": 4,
  "timed_batch_runs": 20,
  "num_padded_images": 1,
  "last_batch_policy": "pad_with_last_image",
  "mean_step_time_sec": 0.2,
  "total_timed_inference_sec": 4.0,
  "throughput_counted_images": 75,
  "throughput_images_per_sec": 18.75,
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
New CLI-generated JSON outputs also include `command_used`, `output_path`, and
privacy-safe Git provenance fields (`git_commit`, `git_branch`, and
`git_dirty`) so local CPU and TPU artifacts can be compared with clearer
provenance. `git_commit` is the observed checkout from `git rev-parse HEAD`, and
`git_dirty` is derived from `git status --short` without storing the full status
output. The Git fields are nullable: legacy artifacts, runs outside a Git
checkout, or environments without Git may report `null` values.
Curated Markdown tables under `report/results/` summarize selected fields from
the raw JSON files.

## Result JSON Fields

CPU and TPU comparison should rely on these stable top-level fields when
present:

- `mode` and `processing_mode`
- `model_name`, `selected_jax_platform`, actual `backend`, and `devices`
- `git_commit`, `git_branch`, and `git_dirty`
- `sharding`, which records requested batch-sharding settings and resolved
  runtime facts
- `batch_size`, `warmup_steps`, `benchmark_steps`, `timed_batch_runs`
- `num_images`, `num_batches`, `num_padded_images`, and `last_batch_policy`
- `mean_step_time_sec`, `total_timed_inference_sec`,
  `throughput_counted_images`, and `throughput_images_per_sec`
- `image_path` for single-image mode or `manifest_path` and `manifest_kind` for
  manifest mode
- `image_results` for manifest mode, with per-image qualitative predictions

For `b1` single-image mode, `batch_size` is 1, `num_images` is 1,
`num_batches` is 1, `timed_batch_runs` equals `benchmark_steps`, and
`num_padded_images` is 0. For the public manifest `b4` command, the five-image
manifest creates two batches and pads the final one with three repeated copies
of the last real image. Padded entries are excluded from predictions and
throughput counts.

## Explicit Batch-Axis Sharding Option

The previous Demo 2 TPU workflow showed TPU backend execution, but it did not
yet show explicit multi-device JAX sharding. The inference script now includes a
planned manual validation path for batch-axis data sharding:

- `--batch-sharding none|data`: defaults to `none`; `data` uses explicit
  batch-axis sharding for image batches.
- `--mesh-axis-name`: names the one-dimensional device mesh axis; defaults to
  `data`.
- `--require-multiple-devices`: fails if the requested runtime does not expose
  enough JAX devices.
- `--min-shard-devices`: minimum visible JAX devices for data sharding or the
  multiple-device guard; defaults to 2.

When `--batch-sharding data` is selected, image batches shaped
`[batch_size, 3, 224, 224]` use `PartitionSpec('data', None, None, None)` and
logits shaped `[batch_size, num_classes]` try to use
`PartitionSpec('data', None)`. Model parameters remain unsharded. The global
batch size must be divisible by the mesh device count.

Planned TPU VM inference command after a multi-device TPU resource is prepared:

```bash
uv run --group pretrained python examples/pretrained_vit_inference.py \
  --jax-platform tpu \
  --image-manifest examples/assets/manifest.txt \
  --batch-size 4 \
  --batch-sharding data \
  --mesh-axis-name data \
  --require-multiple-devices \
  --min-shard-devices 2 \
  --warmup-steps 1 \
  --benchmark-steps 5 \
  --output runs/vit-inference/demo2_sharded_public_examples_tpu_b4.json
```

The generated JSON includes a top-level `sharding` object with the requested
mode, mesh axis name, device counts, mesh shape, per-device batch size,
partition specs, and whether explicit jit shardings were applied. This is a
planned evidence target until an actual sharded TPU JSON artifact is generated
and retrieved; do not describe explicit sharded TPU execution as completed based
on this command alone.

Expected manifest metadata for the current image sets:

| Image set | Images | Batch size | Num batches | Padded images |
| --- | ---: | ---: | ---: | ---: |
| Public `examples/assets/manifest.txt` | 5 | 1 | 5 | 0 |
| Public `examples/assets/manifest.txt` | 5 | 4 | 2 | 3 |
| Local `data/local/demo2_vit_images/manifest.txt` | 15 | 1 | 15 | 0 |
| Local `data/local/demo2_vit_images/manifest.txt` | 15 | 4 | 4 | 1 |

## Model Notes

- `google/vit-base-patch16-224` is a pretrained ViT image-classification model.
- The script uses `AutoImageProcessor` for preprocessing and
  `FlaxViTForImageClassification` for JAX/Flax inference.
- The optional fine-tuning script uses the same model family and updates only
  the classifier head. The ViT backbone stays frozen during the training smoke
  loop.
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
  documented local commands use `cpu` for stable local evidence. The actual
  backend and devices are still recorded from JAX at runtime.

## Curated CPU Result Tables

The following report-ready Markdown tables summarize real JSON artifacts from
`runs/vit-inference/`:

- `report/results/demo2_local_public_examples_cpu.md`
- `report/results/demo2_external_ryzen7735hs_wsl_public_examples_cpu.md`
- `report/results/demo2_local_imagenette320_val64_cpu.md`
- `report/results/demo2_external_ryzen7735hs_wsl_imagenette320_val64_cpu.md`
- `report/results/demo2_local_imagenette320_val256_cpu.md`
- `report/results/demo2_external_ryzen7735hs_wsl_imagenette320_val256_cpu.md`
- `report/results/demo2_local_private_examples_cpu.md`
- `report/results/demo2_local_cpu_vs_cloud_tpu_public_examples_b4.md`

For grouped report-ready summaries, start with
`report/results/README.md`. The generated summary set includes
`demo2_imagenette320_overview.md`, `demo2_imagenette320_batch_scaling.md`,
`demo2_imagenette320_cpu_vs_tpu.md`, `demo2_cpu_machine_comparison.md`, and
`demo2_public_examples_summary.md`.

Local CPU tables are the primary current-machine evidence. External Ryzen 7735HS
WSL CPU tables are supplementary and kept separate. The
`demo2_local_cpu_vs_cloud_tpu_public_examples_b4.md` table is the first local
CPU `b4` versus cloud TPU `b4` smoke-run comparison for the public five-image
manifest. It should be interpreted as smoke-run evidence only: five images,
batch size 4, three padded final-batch entries, a short benchmark loop, no
dataset-level accuracy evaluation, and no controlled hardware benchmark.

For the historical pre-TPU progress report, the `val256` tables were the main
CPU benchmark evidence; `val64` remains a lighter supporting path. The external
public examples table currently has `b1` and `b4` only; external public `b8` is
pending.

## Report Table Generation Patterns

Use two different comparison patterns for Demo 2 result artifacts.

The current curated CPU tables use a vertical, within-group batch-size
comparison. This compares `b1`, `b4`, and `b8` within the same evidence scope,
input set, and platform:

```text
runs/vit-inference/demo2_<scope>_<input_set>_<platform>_b1.json
runs/vit-inference/demo2_<scope>_<input_set>_<platform>_b4.json
runs/vit-inference/demo2_<scope>_<input_set>_<platform>_b8.json
  -> report/results/demo2_<scope>_<input_set>_<platform>.md
```

For example:

```bash
uv run python scripts/compare_vit_results.py \
  runs/vit-inference/demo2_local_public_examples_cpu_b1.json \
  runs/vit-inference/demo2_local_public_examples_cpu_b4.json \
  runs/vit-inference/demo2_local_public_examples_cpu_b8.json \
  --markdown-output report/results/demo2_local_public_examples_cpu.md
```

If a batch size is intentionally unavailable, omit that JSON and document the
missing batch as pending. This is the current external public examples case:
`b1` and `b4` exist, while external public `b8` is pending.

The current horizontal comparison compares local CPU and cloud TPU for the same
public input set and batch size. Generate it only after a real TPU JSON artifact
exists and has been retrieved:

```bash
uv run python scripts/compare_vit_results.py \
  runs/vit-inference/demo2_local_public_examples_cpu_b4.json \
  runs/vit-inference/demo2_cloud_public_examples_tpu_b4.json \
  --output runs/vit-inference/demo2_local_cpu_vs_cloud_tpu_public_examples_b4_compare.json \
  --markdown-output report/results/demo2_local_cpu_vs_cloud_tpu_public_examples_b4.md
```

The JSON comparison output remains an ignored/generated artifact under
`runs/vit-inference/`. Only the intentionally curated report-ready Markdown table
belongs under `report/results/`. The current table reports about 1931.76x
throughput speedup for this specific small public smoke-run comparison; it
should not be generalized to TPU performance overall.

Do not mix external Ryzen 7735HS WSL CPU artifacts into the primary local CPU vs
cloud TPU table. External CPU evidence is supplementary cross-machine CPU
evidence. The current three-way public-example smoke/demo view belongs in the
separate `report/results/demo2_public_examples_summary.md`; it should not be
treated as the primary local-vs-TPU result because it mixes hardware, OS/WSL,
cache, and thermal variables.

## Local CUDA Limitation

On the laptop used for this local spike, a simple JAX GPU matrix multiplication
worked. The ViT-like convolution path failed during cuDNN autotuning, so local
CUDA is not used as Demo 2 benchmark evidence. This project does not claim GPU
ViT inference success for Demo 2.

## Compare Result JSON Files

Use the local comparison helper after a TPU JSON artifact has been retrieved:

```bash
uv run python scripts/compare_vit_results.py \
  runs/vit-inference/demo2_local_public_examples_cpu_b4.json \
  runs/vit-inference/demo2_cloud_public_examples_tpu_b4.json \
  --output runs/vit-inference/demo2_local_cpu_vs_cloud_tpu_public_examples_b4_compare.json \
  --markdown-output report/results/demo2_local_cpu_vs_cloud_tpu_public_examples_b4.md
```

The helper only reads existing JSON files. It summarizes command metadata and
Git provenance when available, input image or manifest, backend, devices, batch
size, image count, total timed runtime, throughput, derived per-image time, and
output path. The optional Markdown output is intended for report-ready benchmark
tables and keeps the main performance table focused on timing, throughput, and
speedup. Use the comparison JSON output when detailed provenance fields are
needed.

## Limitations

- The completed TPU evidence includes a public-example smoke run and Imagenette
  320 inference timing tables, not a final local-vs-TPU benchmark study.
- The default pytest suite does not download the model or require Hugging Face
  network access.
- First-run download time is not part of the timed inference loop.
- Single-image mode repeats the same image for batch benchmarking. Manifest mode
  uses real mixed-image batches, with final-batch padding if needed. Both modes
  remain systems-oriented inference measurements rather than dataset-level
  evaluations.
- The TPU smoke run used five public example images, batch size 4, one warmup
  step, five benchmark steps, and `num_padded_images = 3`.
- The Imagenette TPU tables are ViT inference timing evidence for retrieved
  JSON artifacts. They do not train or fine-tune the model, compute Imagenette
  accuracy, or establish a universal TPU speedup claim.
- The optional fine-tuning extension is a classifier-head-only workflow smoke
  run. It should not be described as full ViT fine-tuning, a large benchmark, or
  model-quality evaluation.
- Private manifest runs are qualitative local demonstrations, not public
  benchmark datasets or classification-accuracy measurements.
- Dataset-level accuracy evaluation, longer benchmark loops, monitoring
  analysis, and controlled hardware comparison remain future work.

## Next Planned Step

Extend the TPU evidence only when a larger controlled comparison is actually
executed. The likely next benchmark step is a more controlled CPU-vs-TPU
comparison with longer timing loops, recorded environment metadata, monitoring
notes, and the same cleanup discipline.
