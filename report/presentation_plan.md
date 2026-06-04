# Demo 2 Pretrained ViT Inference: Presentation Scope and TPU Plan

> Historical note: this was a pre-TPU presentation plan. It is superseded by
> the current Demo 2 TPU evidence, Imagenette 320 inference artifacts, and
> generated summaries under `report/results/`. Keep this file as planning
> history, not as the current project status.

## Purpose

This repository is a course project for Numerical Computation with JAX. The
current presentation plan focuses on one complete path: build a reproducible
local CPU baseline for pretrained ViT inference with JAX/Flax, then prepare the
same workflow for planned Google Cloud TPU execution.

The primary model is `google/vit-base-patch16-224`. The goal is to demonstrate
JAX runtime awareness, benchmark timing, JSON evidence collection, and a
conservative path toward TPU comparison without claiming TPU results before a
real cloud run exists.

## Current Presentation Scope

Demo 2 is the primary presentation target:

- pretrained ViT inference with JAX/Flax
- local CPU baseline evidence
- planned Google Cloud TPU VM workflow

Demo 1 is preserved as raw-JAX CNN background/foundation work. It remains useful
for explaining the project history and lower-level JAX training concepts, but it
is not the current presentation focus.

Demo 3 is preserved as optional future work for a larger pretrained-model or
Gemma-like cloud workflow. It is not part of the current presentation scope.

The current presentation does not attempt to complete all three demos.

## Completed Work

- Demo 2 script exists: `examples/pretrained_vit_inference.py`.
- The script uses Hugging Face `AutoImageProcessor` for preprocessing.
- The script uses `FlaxViTForImageClassification` for JAX/Flax inference.
- The default model is `google/vit-base-patch16-224`.
- Five public example assets are checked in under `examples/assets/`.
- `examples/assets/manifest.txt` provides a reproducible public five-image
  manifest.
- Single-image mode repeats one image to configurable batch sizes.
- Manifest mode supports true mixed-image batches.
- Manifest final-batch padding repeats the last real image; padded entries are
  ignored for predictions and throughput, and `num_padded_images` records the
  padding count.
- The benchmark uses warmup steps, timed steps, and `block_until_ready()`.
- The benchmark writes JSON metrics.
- The local comparison helper exists: `scripts/compare_vit_results.py`.
- The pre-TRC TPU workflow documentation exists:
  `cloud/demo2_pretrained_vit_tpu_workflow.md`.
- Lightweight pytest tests exist and do not download model weights.
- Local CPU inference succeeded during manual checking.

## Current Evidence

Curated local CPU JSON artifacts:

- `report/results/demo2_vit_local_cpu_b1.json`
- `report/results/demo2_vit_local_cpu_b4.json`
- `report/results/demo2_vit_local_cpu_b8.json`

These artifacts record local CPU inference for the checked-in sample image. The
predicted label was `Chihuahua`.

These are single-image repeated-batch inference measurements. They are useful as
local runtime and throughput evidence, but they are not dataset-level accuracy
evaluation, not GPU results, and not TPU results.

## Local CUDA Limitation

On the laptop used for local checking, simple JAX GPU matrix multiplication
worked, but the ViT-like convolution path failed during cuDNN autotuning.
Therefore local CUDA is documented as a laptop environment limitation and is not
used as Demo 2 benchmark evidence.

## Planned Google Cloud TPU Work

The planned TPU workflow is documented in
`cloud/demo2_pretrained_vit_tpu_workflow.md`. The next step is to run the same Demo 2
script on a TPU VM with:

```bash
--jax-platform tpu
```

Planned evidence to collect from a real TPU attempt:

- backend/device output
- JSON benchmark metrics
- terminal logs
- monitoring notes or screenshots if available
- cleanup evidence

TPU execution is planned, not completed.

## Proposed In-Class Demonstration

The proposed in-class demo is a high-level workflow demonstration, not a
finished slide-by-slide deck:

1. Briefly show the repository structure and the Demo 2 files.
2. Run the local CPU Demo 2 command.
3. Show the JSON output and predicted label.
4. Explain the curated local CPU evidence under `report/results/`.
5. Explain the planned TPU workflow and evidence requirements.

Stable local CPU command:

```bash
uv run --group pretrained python examples/pretrained_vit_inference.py \
  --jax-platform cpu \
  --image examples/assets/chihuahua_pet_licorice.jpg \
  --batch-size 1 \
  --warmup-steps 1 \
  --benchmark-steps 5 \
  --output runs/vit-inference/demo2_cpu_b1.json
```

## Risks And Fallback Plan

- If model download or network access is unavailable during class, use the
  existing `report/results/` JSON artifacts.
- If using private live-demo photos, keep them under ignored
  `data/local/demo2_vit_images/` and use the manifest workflow rather
  than committing the files. Treat those predictions as qualitative examples,
  not public benchmark or accuracy evidence. The manifest workflow uses true
  mixed-image batches. If the final batch is partial, it is padded by repeating
  its last real image; padded entries are ignored for predictions and
  throughput, and `num_padded_images` records the padding count.
- If TPU quota or access is unavailable, present the TPU workflow as planned
  work and do not fabricate results.
- If local CPU runtime is slow, reduce benchmark steps for live demonstration
  and distinguish that run from the curated baseline artifacts.
- If CUDA is asked about, explain it as a local hardware/environment limitation:
  simple JAX GPU matmul worked, but the ViT-like convolution path failed during
  cuDNN autotuning.

## Instructor Feedback Requested

- Is focusing the current presentation on Demo 2 only acceptable?
- Is the TPU workflow documentation enough before the actual TPU run?
- Does the final submission need actual TPU metrics, or is a documented TPU
  workflow plus local CPU evidence acceptable for the current milestone?
- Is local CPU versus TPU inference comparison sufficient for the course
  project once TPU execution is available?
