# Current Project Status

## Current Presentation Scope

因為課程時間與展示限制，目前簡報與 demo 主軸收斂到 **Demo 2:
pretrained ViT workflow with JAX/Flax**。

目前主要展示路徑：

- model：`google/vit-base-patch16-224`
- runtime：JAX/Flax
- demo script：`examples/pretrained_vit_inference.py`
- stable local backend：local CPU
- public image set：5 tracked images under `examples/assets/`
- single-image smoke input：`examples/assets/chihuahua_pet_licorice.jpg`
- raw JSON benchmark outputs：ignored/generated `runs/vit-inference/`
- optional fine-tuning outputs：ignored/generated `runs/vit-finetune/`
- curated report-ready Markdown tables：`report/results/`
- Google Cloud / TRC setup state：TRC confirmation 已收到；第一個 Demo 2 TPU
  public-example smoke run、artifact retrieval、comparison table、cleanup
  verification，以及 Imagenette 320 TPU inference tables 都已有 privacy-safe
  report record
- optional Demo 2 fine-tuning TPU state：classifier-head-only workflow 已在 TPU
  上觀察到 first run、spot/maintenance interruption、GCS checkpoint copies，
  以及 replacement TPU VM restore/resume；raw artifacts 仍維持在 ignored
  `runs/vit-finetune/` 或暫存 GCS，不進 Git
- explicit JAX sharding state：Demo 2 inference 與 optional classifier-head
  fine-tuning now expose planned explicit batch-axis sharding paths；目前尚未有
  completed sharded TPU fine-tuning artifact
- report-ready setup/evidence record：`report/google_cloud_trc_setup.md`
- remaining benchmark work after TPU inference evidence：若時間與 quota 允許，
  可再規劃 controlled hardware comparison、較完整 monitoring evidence，或明確
  定義過的 dataset-level accuracy evaluation；目前 Imagenette workflow 仍維持
  local-only data preparation，不自動下載、不進 pytest/CI，也不提交 `data/local/`
  內容

TPU execution evidence now exists for both a small public-example smoke run and
Imagenette 320 validation-manifest inference timing. It should not be described
as training, a full benchmark study, dataset-level accuracy evaluation,
controlled hardware comparison, or a universal TPU speedup claim. The optional
classifier-head fine-tuning extension has also produced TPU smoke workflow
evidence: first run on `v6e-1` spot in `us-east1-d`, real spot or maintenance
interruption after that run, GCS checkpoint copies at steps `15100`, `15120`,
and `15140`, and successful resume on `v6e-1` spot in `europe-west4-a` with
`backend=tpu`, `resumed_from_checkpoint=true`, `start_step=15140`,
`final_step=51538`, `trainable_scope=classifier_head_only`, and
`frozen_scope=vit_backbone`. This is checkpoint/resume and TPU execution
workflow evidence, not full ViT fine-tuning or accuracy benchmark evidence.

## What The Repository Currently Does

已完成的共用 project foundation：

- Python package 使用 `src/jax_tpu_project/` layout。
- Project setup 使用 `uv`、`pyproject.toml`、`uv.lock`。
- Ruff and pytest 設定已放在 `pyproject.toml`。
- JAX runtime/device sanity check 已實作：
  - `src/jax_tpu_project/runtime.py`
  - `src/jax_tpu_project/cli.py`
  - `scripts/check_jax_device.sh`
  - local sanity script defaults `JAX_PLATFORMS` to `cpu` unless the caller
    already set it, so CPU evidence remains explicit on benchmark machines

已完成的 Demo 2 重點：

- Optional `pretrained` dependency group for Transformers/Flax/Pillow/Hugging
  Face workflow。
- Pretrained ViT inference script:
  - `examples/pretrained_vit_inference.py`
- Optional Demo 2 classifier-head fine-tuning script:
  - `examples/demo2_pretrained_vit_finetune.py`
  - freezes the ViT backbone and trains only the classifier head
  - exposes explicit batch-axis sharding flags matching inference:
    `--batch-sharding`, `--mesh-axis-name`, `--require-multiple-devices`, and
    `--min-shard-devices`
  - uses Orbax checkpoint/resume for head params, optimizer state, step, and
    minimal metadata only
  - writes report-friendly `summary.json`, `metrics.csv`, and
    `eval_metrics.csv`; `summary.json` includes train/eval label counts,
    class counts, and sharding metadata so tiny-manifest skew and resolved
    runtime sharding settings are visible
  - generated outputs belong under ignored `runs/vit-finetune/`
- Local CPU public example images:
  - `examples/assets/chihuahua_pet_licorice.jpg`
  - `examples/assets/adelie_penguins_brooding.jpg`
  - `examples/assets/doge_homemade_meme.jpg`
  - `examples/assets/polar_bear_zoo_face.jpg`
  - `examples/assets/black_cat_staring_closeup.jpg`
  - `examples/assets/manifest.txt`
  - `examples/assets/README.md`
- Optional private live-demo image set support:
  - recommended local-only path: `data/local/demo2_vit_images/`
  - optional manifest: `data/local/demo2_vit_images/manifest.txt`
  - current expected local manifest size: 15 images
  - manifest mode uses real mixed-image batches
  - the final partial batch is padded by repeating its last real image; padded
    entries are ignored for predictions and throughput, and `num_padded_images`
    records the padding count
  - qualitative live predictions only; not a public dataset or accuracy benchmark
- Demo 2 documentation:
  - `docs/demo2_pretrained_vit.md`
- TPU documentation:
  - `cloud/demo2_tpu_quickstart.md`
  - `cloud/demo2_pretrained_vit_tpu_workflow.md`
- Local result comparison helper:
  - `scripts/compare_vit_results.py`
  - optional `--markdown-output` for report-ready benchmark tables
- Local image manifest builder:
  - `scripts/build_image_manifest.py`
  - intended for existing local-only image directories such as optional
    Imagenette 320 data under ignored `data/local/imagenette2-320/`
  - does not download datasets or inspect image contents
  - requires `data/local/imagenette2-320/val` to exist before Imagenette
    manifests can be generated
- Curated Demo 2 result tables:
  - local public examples, Imagenette val64, Imagenette val256, and private
    local examples
  - supplementary external Ryzen 7735HS WSL public examples, Imagenette val64,
    and Imagenette val256
  - cloud TPU Imagenette val64, val256, and full validation-manifest inference
    tables
  - local CPU versus cloud TPU public-example smoke comparison table
  - external public examples currently include `b1` and `b4` only; external
    public `b8` remains pending
- Google Cloud / TRC setup status:
  - local CPU Demo 2 and JSON comparison helper are prepared
  - reusable Cloud TPU quickstart is documented separately from course-specific
    TRC setup/evidence records
  - Cloud TPU workflow reference is documented with placeholders, resource
    variants, cleanup guidance, and first smoke-run evidence appendix
  - course-specific setup, TPU smoke-run evidence, Imagenette TPU evidence, and
    cleanup status are summarized in `report/google_cloud_trc_setup.md`
  - exact TPU checkout commit was not preserved in the available report notes;
    do not substitute a later local commit SHA
- Lightweight tests that do not download model weights:
  - `tests/test_pretrained_vit_inference.py`

## Demo 2 CPU Evidence

Manual local checks completed:

- Hugging Face model download succeeded for `google/vit-base-patch16-224`。
- Local CPU inference succeeded。
- Prediction for the sample image was `Chihuahua`。
- Raw JSON benchmark outputs are generated under ignored `runs/vit-inference/`
  and are not committed by default。
- Curated Markdown tables under `report/results/` are generated from real JSON
  artifacts. Flat comparison tables use
  `scripts/compare_vit_results.py --markdown-output`; grouped summary tables
  use `scripts/generate_vit_summary_tables.py`。

Primary local-machine CPU tables:

- `report/results/demo2_local_public_examples_cpu.md`
- `report/results/demo2_local_imagenette320_val64_cpu.md`
- `report/results/demo2_local_imagenette320_val256_cpu.md`
- `report/results/demo2_local_private_examples_cpu.md`

Supplementary external Ryzen 7735HS WSL CPU tables:

- `report/results/demo2_external_ryzen7735hs_wsl_public_examples_cpu.md`
  contains `b1` and `b4` only; external public `b8` is pending.
- `report/results/demo2_external_ryzen7735hs_wsl_imagenette320_val64_cpu.md`
- `report/results/demo2_external_ryzen7735hs_wsl_imagenette320_val256_cpu.md`

These are CPU inference timing/throughput summaries. They are not dataset-level
accuracy evaluations, not GPU results, and not TPU results. External CPU
evidence is supplementary and should stay separate from local-machine evidence.
For the historical pre-TPU progress report, Imagenette `val256` was the main
CPU benchmark evidence because b1/b4/b8 all used 256 real images with 0 padded
images. Imagenette `val64` remains a lightweight documented benchmark path and
supporting CPU result.
The private local table is live-demo evidence, not a public reproducible
benchmark dataset.
Private manifest runs follow the same qualitative-inference framing unless
explicit labels and top-k evaluation are added later, but they now measure true
batched manifest inference over the listed images.

## Demo 2 TPU Inference Evidence

The first successful TPU artifact is:

- `runs/vit-inference/demo2_cloud_public_examples_tpu_b4.json`

The report-ready public-example comparison table is:

- `report/results/demo2_local_cpu_vs_cloud_tpu_public_examples_b4.md`

This run used `examples/pretrained_vit_inference.py --jax-platform tpu` with
`examples/assets/manifest.txt`, batch size 4, one warmup step, five benchmark
steps, and output path
`runs/vit-inference/demo2_cloud_public_examples_tpu_b4.json`.

Key recorded fields:

- backend：`tpu`
- JSON-visible device kind：`TPU v6 lite`
- mode：`image_manifest`
- manifest kind：`public_example`
- num_images：5
- batch_size：4
- num_batches：2
- num_padded_images：3
- timed_batch_runs：10
- total timed inference：約 0.01098252 秒
- throughput：約 2276.3446 images/sec

The generated comparison table reports about 1931.76x throughput speedup versus
the local CPU `b4` artifact for this specific smoke run. This statement is
limited to the five-image public smoke run. It is not a general TPU speed claim,
not dataset-level accuracy evaluation, and not a full controlled hardware
benchmark.

Imagenette 320 TPU inference JSON artifacts have also been retrieved under
ignored `runs/vit-inference/` for:

- `val64`: `demo2_cloud_imagenette320_val64_tpu_b1.json`,
  `demo2_cloud_imagenette320_val64_tpu_b4.json`, and
  `demo2_cloud_imagenette320_val64_tpu_b8.json`
- `val256`: `demo2_cloud_imagenette320_val256_tpu_b1.json`,
  `demo2_cloud_imagenette320_val256_tpu_b4.json`, and
  `demo2_cloud_imagenette320_val256_tpu_b8.json`
- `val_full`: `demo2_cloud_imagenette320_valfull_tpu_b1.json`,
  `demo2_cloud_imagenette320_valfull_tpu_b4.json`, and
  `demo2_cloud_imagenette320_valfull_tpu_b8.json`

The curated TPU Imagenette tables are:

- `report/results/demo2_cloud_imagenette320_val64_tpu.md`
- `report/results/demo2_cloud_imagenette320_val256_tpu.md`
- `report/results/demo2_cloud_imagenette320_valfull_tpu.md`

For grouped report-ready summaries, start with
`report/results/README.md`. The generated summary set includes
`demo2_imagenette320_overview.md`, `demo2_imagenette320_batch_scaling.md`,
`demo2_imagenette320_cpu_vs_tpu.md`, `demo2_cpu_machine_comparison.md`, and
`demo2_public_examples_summary.md`.

These tables report backend `tpu`, JSON-visible device kind `TPU v6 lite`,
batch sizes `b1` / `b4` / `b8`, one warmup step, five benchmark steps, and
Imagenette validation-manifest inference timing. The full validation manifest
contains 3925 images; `b4` and `b8` have 3 padded final-batch entries. The
curated TPU tables report throughput in the approximate range of 1812 to 4370
images/sec for the retrieved artifacts.

The Imagenette TPU evidence is still ViT inference only. It is not model
training, not fine-tuning, not dataset-level accuracy evaluation, not a full
controlled benchmark study, and not a universal TPU speedup claim.

New Demo 2 JSON outputs include stable result fields for mode, processing mode,
batch size, image count, batch count, padding count, timed batch runs, timing,
throughput, backend/devices, and manifest kind when applicable.

## Local CUDA Limitation

On the laptop used for local checks, simple JAX GPU matrix multiplication
worked, but the ViT-like convolution path failed during cuDNN autotuning.
Therefore local CUDA is documented as a limitation and is not used as Demo 2
benchmark evidence.

## Demo 1 Status

Demo 1 remains in the repository as preserved background/foundation work.

Implemented:

- raw-JAX hand-written CNN benchmark foundation:
  - `src/jax_tpu_project/cnn_mnist.py`
  - `examples/cnn_mnist_benchmark.py`
  - `tests/test_cnn_mnist.py`
- deterministic synthetic MNIST-shaped data path
- JSON metrics output for local smoke runs

Not completed:

- real MNIST/Fashion-MNIST local-file loader
- curated Demo 1 result under `report/results/`
- Demo 1 TPU execution and local-vs-TPU comparison

Demo 1 is not the current presentation focus.

## Demo 3 Status

Demo 3 remains optional future work for a larger pretrained-model or Gemma-like
cloud workflow. It is not the current presentation focus and has no completed
code, model access notes, cloud run, TPU run, or result artifacts.

## Current Validation Scope

Default tests remain lightweight and offline:

- `tests/test_runtime.py`
- `tests/test_cnn_mnist.py`
- `tests/test_pretrained_vit_inference.py`
- `tests/test_demo2_pretrained_vit_finetune.py`
- `tests/test_compare_vit_results.py`
- `tests/test_generate_vit_summary_tables.py`
- `tests/test_demo2_tpu_helper.py`
- `tests/test_build_image_manifest.py`

The Demo 2 tests check argument parsing, metrics helper behavior, generated
summary-table behavior, optional helper script syntax/static safety, and
platform environment helper behavior. They do not require GPU, TPU, network
access, Hugging Face access, image opening, or model weight download.

## Next Technical Milestones

1. Preserve TPU evidence without committing raw JSON under
   `runs/vit-inference/`; commit only curated Markdown result tables.
2. Keep `report/results/demo2_local_cpu_vs_cloud_tpu_public_examples_b4.md` as
   the curated table for the first CPU-vs-TPU public-example smoke comparison.
3. Keep the three cloud TPU Imagenette tables as inference timing evidence, not
   accuracy evaluation or training evidence.
4. Preserve the optional Demo 2 classifier-head fine-tuning TPU smoke evidence
   without committing raw `runs/vit-finetune/` artifacts, checkpoints, logs,
   datasets, model caches, or GCS objects.
5. If time and quota allow, plan a controlled local-vs-TPU comparison with
   longer benchmark loops, recorded commit SHA, environment metadata,
   monitoring notes, and cleanup verification.
6. Keep Imagenette 320 preparation under ignored `data/local/imagenette2-320/`;
   do not add automatic downloads or pytest/CI dependencies.
7. Keep Demo 1 and Demo 3 preserved as future work unless course scope changes.
