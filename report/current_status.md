# Current Project Status

## Current Presentation Scope

因為課程時間與展示限制，目前簡報與 demo 主軸收斂到 **Demo 2:
pretrained ViT inference benchmark with JAX/Flax**。

目前主要展示路徑：

- model：`google/vit-base-patch16-224`
- runtime：JAX/Flax
- demo script：`examples/pretrained_vit_inference.py`
- stable classroom backend：local CPU
- public image set：5 tracked images under `examples/assets/`
- single-image smoke input：`examples/assets/chihuahua_pet_licorice.jpg`
- raw JSON benchmark outputs：ignored/generated `runs/vit-inference/`
- curated report-ready Markdown tables：`report/results/`
- Google Cloud / TRC setup state：dedicated Google Cloud project 已建立，
  billing 已 linked，budget alerts 已設定，Cloud TPU API 已啟用，
  project number 已提交到 TRC form；TRC confirmation 已收到；第一個 Demo 2
  TPU public-example smoke run 已完成，artifact 已取回，CPU-vs-TPU comparison
  table 已產生，cleanup 已完成並確認 selected zone 沒有剩餘 queued resource
  或 TPU VM
- report-ready setup record：`report/google_cloud_trc_setup.md`
- next benchmark work after first TPU smoke evidence：若時間與 quota 允許，
  再規劃較完整的 Imagenette TPU run 或 controlled hardware comparison；
  Imagenette 320
  (`imagenette2-320`) 已有 local/external CPU curated Markdown tables，但
  dataset workflow 仍維持 local-only，不自動下載、不進 pytest/CI，也不提交
  `data/local/` 內容

First TPU execution evidence now exists, but it is limited to a small
public-example smoke run. It should not be described as a full benchmark study,
dataset-level accuracy evaluation, or controlled hardware comparison.

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
- Local CPU classroom public images:
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
- Curated Demo 2 CPU tables:
  - local public examples, Imagenette val64, Imagenette val256, and private
    local examples
  - supplementary external Ryzen 7735HS WSL public examples, Imagenette val64,
    and Imagenette val256
  - external public examples currently include `b1` and `b4` only; external
    public `b8` remains pending
- Google Cloud / TRC setup status:
  - local CPU Demo 2 and JSON comparison helper are prepared
  - reusable Cloud TPU quickstart is documented separately from course-specific
    TRC setup/evidence records
  - Cloud TPU workflow reference is documented with placeholders, resource
    variants, cleanup guidance, and first smoke-run evidence appendix
  - dedicated Google Cloud project was created outside the repository
  - project ID and project number were verified with `gcloud projects describe`
    and kept private
  - billing account was linked outside the repository
  - budget alerts were configured:
    `numerical-jax-first-warning` at 10 USD and
    `numerical-jax-main-limit` at 60 USD
  - Cloud TPU API was enabled outside the repository
  - project number was submitted to TRC and TRC confirmation has been received
  - setup record is documented in `report/google_cloud_trc_setup.md`
  - initial v4 queued resource in `us-central2-b` remained in
    `WAITING_FOR_RESOURCES` for several days and was abandoned
  - successful smoke run used a TRC spot queued resource in `us-east1-d` with
    Google Cloud accelerator type `v6e-1`, runtime `v2-alpha-tpuv6e`, and
    JSON-visible device kind `TPU v6 lite`
  - successful TPU run used branch `feat/demo2-tpu-evidence`; the exact TPU
    checkout commit was not preserved in the available report notes
  - Demo 2 TPU JSON artifact was generated and retrieved:
    `runs/vit-inference/demo2_cloud_public_examples_tpu_b4.json`
  - CPU-vs-TPU comparison table was generated:
    `report/results/demo2_local_cpu_vs_cloud_tpu_public_examples_b4.md`
  - cleanup completed after artifact retrieval; queued-resource deletion
    succeeded, and both queued-resource list and TPU-VM list returned zero items
    in `us-east1-d`
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
  artifacts with `scripts/compare_vit_results.py --markdown-output`。

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

## Demo 2 TPU Smoke Evidence

The first successful TPU artifact is:

- `runs/vit-inference/demo2_cloud_public_examples_tpu_b4.json`

The report-ready comparison table is:

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
not an Imagenette result, not dataset-level accuracy evaluation, and not a full
controlled hardware benchmark.

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
- `tests/test_compare_vit_results.py`
- `tests/test_build_image_manifest.py`

The Demo 2 tests check argument parsing, metrics helper behavior, and platform
environment helper behavior. They do not require GPU, TPU, network access,
Hugging Face access, image opening, or model weight download.

## Next Technical Milestones

1. Preserve the first TPU smoke-run evidence without committing raw JSON under
   `runs/vit-inference/`.
2. Keep `report/results/demo2_local_cpu_vs_cloud_tpu_public_examples_b4.md` as
   the curated table for this first CPU-vs-TPU smoke comparison.
3. If time and quota allow, plan a broader Imagenette TPU run or controlled
   hardware comparison with longer benchmark loops, recorded commit SHA,
   environment metadata, monitoring notes, and the same cleanup verification.
4. Keep Imagenette 320 preparation under ignored `data/local/imagenette2-320/`;
   do not add automatic downloads or pytest/CI dependencies.
5. Keep Demo 1 and Demo 3 preserved as future work unless course scope changes.
