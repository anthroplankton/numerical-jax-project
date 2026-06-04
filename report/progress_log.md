# Project Progress Log

## Phase 1: Initial Python/JAX Scaffold

- Date / phase label：初始 scaffold；對應 git commit `4f6001b chore: bootstrap initial JAX project scaffold`
- What was done：
  - 建立 course project repository 的基本 Python project structure。
  - 使用 `uv`、`pyproject.toml`、`uv.lock` 管理 project 與 dependencies。
  - 建立 `src/jax_tpu_project/` package。
  - 建立 JAX runtime/device summary utility and CLI。
  - 加入 `scripts/check_jax_device.sh` 作為本機 JAX device sanity check。
  - 加入 pytest runtime tests。
  - 建立 `.gitignore`，排除 caches、virtual environments、credentials、generated runs、raw data、model weights、uncurated cloud outputs。
- Files or modules added：
  - `AGENTS.md`
  - `README.md`
  - `pyproject.toml`
  - `uv.lock`
  - `.gitignore`
  - `scripts/check_jax_device.sh`
  - `src/jax_tpu_project/__init__.py`
  - `src/jax_tpu_project/cli.py`
  - `src/jax_tpu_project/runtime.py`
  - `tests/test_runtime.py`
- Commands/checks known from repository：
  - README documents `bash scripts/check_jax_device.sh`
  - README documents `uv run python -m jax_tpu_project.cli devices`
  - `pyproject.toml` configures pytest test path as `tests`
- Current evidence/results：
  - Runtime utility returns JSON-serializable backend/device information.
  - Tests check `default_backend` and visible device fields.
- Limitations：
  - Scaffold only; no training demo in this phase.
  - No cloud TPU workflow or benchmark evidence.
- Next planned step：
  - Add a small raw-JAX training-oriented demo suitable for local smoke testing and future TPU comparison.

## Phase 2: Demo 1 Raw-JAX CNN Benchmark Foundation

- Date / phase label：CNN benchmark foundation；對應 git commit `b4c8abf feat: add hand-written MNIST CNN benchmark`
- What was done：
  - 建立 hand-written CNN benchmark for MNIST-shaped image classification。
  - 使用 raw JAX primitives，不使用 Flax、Optax、PyTorch、TensorFlow。
  - 實作 deterministic synthetic MNIST-shaped data generator。
  - 實作 CNN parameter initialization、forward pass、cross-entropy loss、accuracy、`jax.jit` training step、explicit SGD update。
  - 實作 benchmark runner，包含 warmup steps、timed steps、terminal logging、JSON metrics output。
  - 加入 CLI example script `examples/cnn_mnist_benchmark.py`。
  - 更新 README with Demo 1 commands and expected metrics。
  - 加入 CNN tests for shapes, forward pass, training smoke behavior, metrics writing, and deterministic seed behavior。
- Files or modules added/updated：
  - `src/jax_tpu_project/cnn_mnist.py`
  - `examples/cnn_mnist_benchmark.py`
  - `tests/test_cnn_mnist.py`
  - `src/jax_tpu_project/__init__.py`
  - `README.md`
- Commands/checks known from repository：
  - Quick smoke benchmark:
    ```bash
    uv run python examples/cnn_mnist_benchmark.py \
      --dataset synthetic \
      --steps 3 \
      --batch-size 16 \
      --seed 0 \
      --output-dir runs/smoke \
      --platform-label local
    ```
  - Longer local benchmark:
    ```bash
    uv run python examples/cnn_mnist_benchmark.py \
      --dataset synthetic \
      --steps 50 \
      --batch-size 64 \
      --learning-rate 0.05 \
      --warmup-steps 5 \
      --seed 0 \
      --output-dir runs/local-cnn-mnist \
      --platform-label local
    ```
- Current evidence/results：
  - Existing ignored generated artifact: `runs/smoke/cnn_mnist_metrics.json`。
  - That artifact records `dataset_name: synthetic`, `backend: cpu`, `platform_label: local`, `steps: 3`, `warmup_steps: 1`, `batch_size: 16`, `seed: 0`。
  - The artifact includes loss, accuracy, total training time, average step time, examples per second, backend, devices, and output artifact path。
- Limitations：
  - Current completed benchmark data path is synthetic only。
  - `mnist` and `fashion_mnist` are accepted as CLI choices but currently reserved for future local-file dataset support and raise `FileNotFoundError`。
  - No Google Cloud TPU run has been completed in this repository。
  - No cloud monitoring screenshots, TPU logs, or local-vs-TPU comparison artifacts exist yet。
  - Existing `runs/` artifact is ignored and should not be treated as curated final report evidence until copied or summarized intentionally under `report/results/` with context。
- Next planned step：
  - Implement real MNIST/Fashion-MNIST local-file loading, then produce curated local benchmark artifacts before moving to TPU execution.

## Phase 3: Presentation Planning and Current Status Documentation

- Date / phase label：2026-05-08 presentation planning documentation
- What was done：
  - Created `report/` for course-report materials.
  - Added and revised a Traditional Chinese presentation plan with a broader three-demo roadmap.
  - Added this progress log grounded in repository files, git history, tests, and available generated metrics.
  - Added a current project status summary for local demo usage, tested behavior, missing work, and next milestones.
  - Historical note: at that phase, Demo 2 and Demo 3 were still planned work, not implemented work.
- Files or modules added：
  - `report/presentation_plan.md`
  - `report/progress_log.md`
  - `report/current_status.md`
- Commands/checks run during this documentation task：
  - `rg --files`
  - `git status --short`
  - `git log --oneline --decorate -10`
  - `git show --stat --oneline --decorate HEAD`
  - `git show --stat --oneline --decorate 4f6001b`
  - Inspected README, pyproject, source modules, examples, tests, scripts, `.gitignore`, and `runs/smoke/cnn_mnist_metrics.json`
- Current evidence/results：
  - At that phase, documentation separated completed local scaffold/Demo 1 CNN foundation from planned real dataset work, planned TPU benchmark, planned Demo 2, and optional planned Demo 3。
- Limitations：
  - This phase is documentation-only。
  - No new source code, tests, cloud workflows, dependencies, Docker files, CI, or notebooks were added。
  - At that phase, no real MNIST/Fashion-MNIST, TPU, Demo 2, or Demo 3 results were claimed。
- Next planned step at that phase：
  - Run or curate a local Demo 1 benchmark result under `report/results/`, then implement real dataset loading and prepare TPU workflow documentation.

## Phase 4: Demo 2 Pretrained ViT Inference Local CPU Baseline

- Date / phase label：2026-05-17 Demo 2 scope update and local CPU baseline
- What was changed：
  - Demo 2 scope changed from a generic Flax + Optax or existing JAX training-stack comparison to a pretrained ViT inference benchmark.
  - Selected default model：`google/vit-base-patch16-224`。
  - Added an optional `pretrained` dependency group for the ViT demo only, keeping pretrained dependencies out of the default environment。
  - Added a local inference benchmark script using Hugging Face `AutoImageProcessor` and `FlaxViTForImageClassification`。
  - Added `--jax-platform` to the ViT script with choices `default`, `cpu`, `cuda`, and `tpu`; the default is `cpu` for stable local runs。
  - Added documentation for local setup, first-run model download behavior, expected metrics, and limitations。
  - Added lightweight tests for argument parsing and metrics helpers that do not download model weights。
  - Added a public example image set under `examples/assets/` for reproducible public demos。
  - Added support for a local-only private image manifest under ignored `data/local/demo2_vit_images/` for live demos。
  - The public example manifest now contains 5 tracked images。
  - The local live-demo manifest is expected to contain 15 images, including the local banana image and local copies of the four public Wikimedia examples。
- Files or modules added/updated：
  - `pyproject.toml`
  - `examples/pretrained_vit_inference.py`
  - `examples/assets/chihuahua_pet_licorice.jpg`
  - `examples/assets/adelie_penguins_brooding.jpg`
  - `examples/assets/doge_homemade_meme.jpg`
  - `examples/assets/polar_bear_zoo_face.jpg`
  - `examples/assets/black_cat_staring_closeup.jpg`
  - `examples/assets/manifest.txt`
  - `examples/assets/README.md`
  - `docs/demo2_pretrained_vit.md`
  - `report/results/README.md`
  - legacy single-image local CPU JSON artifacts under `report/results/`,
    later superseded by the regenerated artifact policy in Phase 5.5
  - `tests/test_pretrained_vit_inference.py`
  - `report/progress_log.md`
- Current expected output：
  - Single-image mode writes JSON metrics with model name, selected JAX platform, actual JAX backend, devices, input shape, batch size, warmup steps, benchmark steps, timing, throughput, top-k predictions, predicted class index, and predicted label。
  - Manifest mode writes run-level aggregate timing fields plus `image_results` for per-image top-k qualitative predictions, without presenting the first image prediction as a whole-run prediction。
  - Manifest mode now uses true mixed-image batches shaped `[batch_size, 3, 224, 224]`; the final partial batch is padded by repeating its last real image and padded entries are ignored for predictions。
- Current evidence/results：
  - Hugging Face model download for `google/vit-base-patch16-224` succeeded during manual local checking。
  - Local CPU ViT inference succeeded using the sample image `examples/assets/chihuahua_pet_licorice.jpg`。
  - At that phase, legacy local CPU JSON artifacts under `report/results/`
    recorded `b1`, `b4`, and `b8` single-image repeated-batch timings. These
    legacy JSON artifacts were later superseded by the current policy: raw JSON
    under ignored `runs/vit-inference/` and curated Markdown tables under
    `report/results/`。
- Limitations：
  - This phase created the local CPU Demo 2 baseline and compatibility evidence, not final local-vs-TPU benchmark evidence。
  - The first real run downloads model weights and processor files from Hugging Face unless they are already cached。
  - The default pytest suite does not download the pretrained model。
  - Private local demo photos and their manifest should stay under ignored `data/local/` and should not be committed。
  - Private manifest predictions are qualitative local live-demo outputs, not a public benchmark dataset or classification-accuracy evaluation。
  - Manifest throughput counts real manifest images only, so final-batch padding is included as runtime overhead but not counted as extra real images; `num_padded_images` records the padding count。
  - Simple JAX GPU matrix multiplication worked on the laptop, but the ViT-like convolution path failed during cuDNN autotuning; local CUDA is therefore recorded as a limitation, not as completed Demo 2 benchmark evidence。
  - TPU execution, monitoring, cleanup, and local-vs-TPU comparison for Demo 2 remain planned work and have not been completed。
- Next planned step at that phase：
  - Use the local CPU artifacts as Demo 2 baseline evidence, then plan TPU execution, monitoring, cleanup, and comparison separately without adding cloud automation before the workflow is ready. Phase 5 later addressed TPU workflow documentation and comparison-helper preparation; real TPU execution remains planned。

## Phase 5: Current Presentation Scope And Demo 2 TPU Workflow Planning

- Date / phase label：2026-05-17 presentation scope update
- What changed：
  - Because of course constraints, the current presentation/demo focus narrowed to Demo 2 only。
  - Demo 2 remains the pretrained ViT inference benchmark using JAX/Flax and `google/vit-base-patch16-224`。
  - Demo 1 is preserved in the repository as background/foundation work, but it is not the current presentation focus。
  - Demo 3 is preserved as optional future work, but it is not the current presentation focus。
  - Added documentation for a conservative Google Cloud TPU VM workflow for Demo 2。
  - Added a local JSON comparison helper for Demo 2 result files; it compares existing files only and does not require TPU access。
  - Added pre-TRC Google Cloud guidance：先建立 dedicated Google Cloud project、在本機記錄 project ID / project number、提交 project number 到 TRC form，並在 TRC confirmation received 之前不建立 TPU resources；後續 Phase 5.6 已更新為 TRC confirmation received。
  - 明確記錄目前沒有建立 Google Cloud resources，沒有執行 TPU VM run，也沒有 CPU-vs-TPU result collection。
- Files or modules added/updated：
  - `README.md`
  - `docs/demo2_pretrained_vit.md`
  - `cloud/demo2_pretrained_vit_tpu_workflow.md`
  - `scripts/compare_vit_results.py`
  - `report/current_status.md`
  - `report/presentation_plan.md`
  - `report/progress_log.md`
- Evidence/results at that phase：
  - Demo 2 local CPU baseline exists under `report/results/`。
  - The curated local CPU artifacts remain the only completed Demo 2 benchmark evidence。
  - Local CUDA remains a documented laptop limitation, not a completed benchmark。
- Limitations：
  - Demo 2 TPU execution, monitoring, artifact retrieval, cleanup, and local-vs-TPU comparison are planned but not completed。
  - At that phase, TRC project-number submission was still an external next
    step and was not stored in the repository。
  - The TPU workflow document is documentation-only and uses placeholders such as `<PROJECT_ID>`, `<PROJECT_NUMBER>`, `<ZONE>`, `<TPU_NAME>`, `<ACCELERATOR_TYPE>`, `<RUNTIME_VERSION>`, `<REPO_URL>`, and `<BRANCH>`。
  - Free-trial credits should not be consumed unless TRC is delayed or unavailable and the run plan plus cleanup command are ready。
- Next planned step：
  - At that phase, submit the Google Cloud project number to TRC outside the
    repository, wait for TRC confirmation, and continue Demo 2
    documentation/evidence preparation while waiting. Use Imagenette 320
    (`imagenette2-320`) as the recommended optional local benchmark dataset for
    later Demo 2 work, while keeping it under ignored `data/local/` and out of
    pytest/CI。
  - Use `cloud/demo2_pretrained_vit_tpu_workflow.md` to prepare a controlled TPU VM attempt, then record actual commands, metrics, logs, monitoring notes, cleanup evidence, and comparison output only after a real run occurs。

## Phase 5.1: Demo 2 Benchmark Asset And Result Field Stabilization

- Date / phase label：2026-05-17 Demo 2 benchmark asset/result-field stabilization
- What changed：
  - Stabilized new Demo 2 JSON outputs with explicit fields for mode,
    processing mode, batch size, image count, batch count, padding count, timed
    batch runs, throughput counted images, timing, backend/devices, and manifest
    kind。
  - Kept legacy curated local CPU JSON artifacts unchanged, while updating the
    comparison helper to infer stable summary fields where possible。
  - Added report-ready Markdown table output to `scripts/compare_vit_results.py`
    through `--markdown-output`。
  - Added `scripts/build_image_manifest.py` for deterministic manifests from
    existing local image directories, including optional local Imagenette 320
    data under ignored `data/local/imagenette2-320/`。
  - Documented formal Demo 2 local CPU `b1` and public manifest `b4` commands,
    including final-batch padding behavior。
- Files or modules added/updated：
  - `examples/pretrained_vit_inference.py`
  - `scripts/compare_vit_results.py`
  - `scripts/build_image_manifest.py`
  - `tests/test_pretrained_vit_inference.py`
  - `tests/test_compare_vit_results.py`
  - `tests/test_build_image_manifest.py`
  - `README.md`
  - `docs/demo2_pretrained_vit.md`
  - `report/results/README.md`
  - `report/current_status.md`
  - `report/progress_log.md`
- Commands/checks run：
  - `uv run ruff check .` passed。
  - `uv run ruff format --check .` passed after formatting touched Python
    files。
  - `uv run pytest` passed with 40 tests after `uv` cache-lock escalation was
    required。
  - `git diff --check` passed。
- Limitations：
  - This phase did not download Imagenette 320, run TPU, create cloud resources,
    or generate new benchmark evidence。
  - Imagenette 320 remains local-only optional benchmark preparation unless a
    later run produces curated artifacts and documentation。

## Phase 5.2: Demo 2 Local CPU Benchmark Tables

- Date / phase label：2026-05-18 Demo 2 local CPU benchmark table organization
- What changed：
  - Added curated Markdown tables under `report/results/` for legacy
    single-image local CPU results, Imagenette 320 val64, Imagenette 320 val256,
    and private local live-demo results。
  - Updated `report/results/README.md` to distinguish legacy JSON artifacts,
    public report-ready tables, extended Imagenette evidence, and private
    local-only evidence。
  - Clarified that raw JSON files under `runs/vit-inference/` are not committed
    by default。
- Evidence/results at that phase：
  - At that phase, local CPU Markdown tables existed for legacy single-image,
    Imagenette val64, Imagenette val256, and private local evidence. These names
    were later superseded by the regenerated scope-specific table set recorded in
    Phase 5.5。
- Limitations：
  - These are local CPU tables only, not TPU results。
  - The tables summarize inference timing/throughput, not classification
    accuracy evaluation。
  - Private local evidence is not a public reproducible benchmark dataset。

## Phase 5.3: Demo 2 Benchmark Setup Clarification

- Date / phase label：2026-05-18 Demo 2 external-machine setup clarification
- What changed：
  - Updated the local JAX device sanity script so it defaults
    `JAX_PLATFORMS` to `cpu` unless the caller already set a platform。
  - Documented the fresh Ubuntu/WSL benchmark-machine setup path, including
    dependency sync, local CPU device sanity check, Ruff, pytest, and a public
    five-image manifest smoke run。
  - Clarified that Imagenette 320 must be manually downloaded/extracted before
    building local manifests, and that scripts/tests do not download it。
  - Added neutral external-machine artifact naming guidance for CPU vs
    ROCm/GPU exploratory outputs。
- Current evidence/results：
  - This phase records documentation and setup behavior only。
  - No new benchmark result, GPU/ROCm result, cloud run, or TPU run is claimed。

## Phase 5.4: Google Cloud / TRC Onboarding Setup Record

- Date / phase label：2026-05-21 Google Cloud / TRC onboarding setup record
- What changed：
  - Recorded the external Google Cloud / TRC setup state as report-ready
    material。
  - Documented that a dedicated Google Cloud project was created outside the
    repository。
  - Documented that project ID and project number were verified with:
    `gcloud projects describe <PROJECT_ID> --format="table(projectId,name,projectNumber)"`。
  - Documented that billing was linked, budget alerts were configured, Cloud TPU
    API was enabled, and the project number was submitted to TRC。
  - Kept all private identifiers out of the repository by using placeholders
    such as `<PROJECT_ID>` and `<PROJECT_NUMBER>`。
- Files added/updated：
  - `report/google_cloud_trc_setup.md`
  - `report/current_status.md`
  - `report/progress_log.md`
  - `README.md`
  - `cloud/demo2_pretrained_vit_tpu_workflow.md`
- Current evidence/results：
  - Google Cloud project setup, billing link, budget alerts, Cloud TPU API
    enablement, and TRC project-number submission are recorded as completed
    external setup steps。
  - This phase was later superseded by Phase 5.6, which records TRC
    confirmation received。
- Limitations：
  - This phase records external setup only。
  - No Cloud TPU VM was created。
  - No TPU execution, TPU metrics, cloud benchmark result, monitoring screenshot,
    cleanup evidence, or CPU-vs-TPU comparison exists yet。
- Next planned step：
  - Phase 5.6 records the updated TRC confirmation status. After that update,
    use `cloud/demo2_pretrained_vit_tpu_workflow.md` for a controlled manual
    TPU VM attempt and record real execution evidence only after it occurs。

## Phase 5.5: Demo 2 Regenerated CPU Artifact Set

- Date / phase label：2026-05-21 Demo 2 regenerated CPU artifact alignment
- What changed：
  - Regenerated the current Demo 2 Markdown result tables from raw JSON artifacts。
  - Aligned the artifact policy：raw JSON benchmark outputs live under ignored
    `runs/vit-inference/`; curated report-ready Markdown tables live under
    `report/results/`。
  - Separated primary local-machine CPU evidence from supplementary external
    Ryzen 7735HS WSL CPU evidence。
  - Kept private input files and manifests local-only under ignored `data/local/`；
    only the curated private table is report-ready。
- Current evidence/results：
  - Local public examples table：
    `report/results/demo2_local_public_examples_cpu.md`。
  - Local Imagenette tables：
    `report/results/demo2_local_imagenette320_val64_cpu.md` and
    `report/results/demo2_local_imagenette320_val256_cpu.md`。
  - Local private examples table：
    `report/results/demo2_local_private_examples_cpu.md`。
  - Supplementary external Ryzen 7735HS WSL public examples table：
    `report/results/demo2_external_ryzen7735hs_wsl_public_examples_cpu.md`。
    It currently contains `b1` and `b4` only; external public `b8` is pending。
  - Supplementary external Ryzen 7735HS WSL Imagenette tables：
    `report/results/demo2_external_ryzen7735hs_wsl_imagenette320_val64_cpu.md`
    and
    `report/results/demo2_external_ryzen7735hs_wsl_imagenette320_val256_cpu.md`。
- Limitations：
  - These are CPU inference timing/throughput tables only, not TPU results and
    not classification-accuracy evaluation。
  - External CPU evidence is supplementary and should not be merged into the
    primary local-machine evidence。
  - TPU execution, TPU JSON artifacts, cloud monitoring evidence, cleanup
    evidence, and CPU-vs-TPU comparison remain pending。

## Phase 5.6: TRC Confirmation Received Documentation Update

- Date / phase label：2026-05-28 TRC confirmation received documentation update
- What changed：
  - Updated current project documentation to state that TRC confirmation has
    been received。
  - Kept real project IDs, project numbers, billing account IDs, TRC email
    content, screenshots, credentials, and private cloud identifiers out of the
    repository。
  - Kept all TPU workflow values as placeholders such as `<PROJECT_ID>`,
    `<ZONE>`, `<TPU_NAME>`, `<ACCELERATOR_TYPE>`, and `<RUNTIME_VERSION>`。
- Files updated：
  - `README.md`
  - `report/current_status.md`
  - `report/google_cloud_trc_setup.md`
  - `report/progress_log.md`
  - `cloud/demo2_pretrained_vit_tpu_workflow.md`
- Current evidence/results：
  - This is a documentation-only status update。
  - TRC confirmation has been received。
  - No Cloud TPU VM has been created。
  - No TPU benchmark has been run。
  - No TPU JSON, monitoring evidence, cleanup evidence, or CPU-vs-TPU
    comparison exists yet。
- Next planned step：
  - Review the documented TPU workflow, placeholders, quota/cost assumptions,
    and cleanup plan before any manually confirmed TPU VM attempt。

## Phase 5.7: First Demo 2 TPU Smoke Run Evidence

- Date / phase label：2026-06-02 first Demo 2 TPU public-example smoke run
- What changed：
  - TRC confirmation 後，完成第一個 real Demo 2 Cloud TPU smoke run。
  - Successful run 使用 `examples/pretrained_vit_inference.py` with
    `--jax-platform tpu`、`--image-manifest examples/assets/manifest.txt`、
    `--batch-size 4`、`--warmup-steps 1`、`--benchmark-steps 5`。
  - TPU JSON artifact 已產生並取回到 ignored/generated path：
    `runs/vit-inference/demo2_cloud_public_examples_tpu_b4.json`。
  - 使用 local CPU baseline：
    `runs/vit-inference/demo2_local_public_examples_cpu_b4.json`。
  - 使用 `scripts/compare_vit_results.py` 產生 curated Markdown comparison：
    `report/results/demo2_local_cpu_vs_cloud_tpu_public_examples_b4.md`。
  - Cleanup completed after artifact retrieval；queued resource deletion
    succeeded，`gcloud compute tpus queued-resources list` 與
    `gcloud compute tpus tpu-vm list` 在 selected zone 都回傳 0 items。
- Resource / environment notes：
  - Successful resource 使用 TRC spot quota。
  - Zone：`us-east1-d`。
  - Google Cloud accelerator type：`v6e-1`。
  - Runtime version：`v2-alpha-tpuv6e`。
  - JSON-visible device kind：`TPU v6 lite`。
  - Branch：`feat/demo2-tpu-evidence`。
  - Exact TPU checkout commit 沒有保存在目前可用的 report notes 中；不可用
    post-edit local commit SHA 代替，這是 reproducibility limitation。
- Operational note：
  - Initial v4 queued resource in `us-central2-b` remained in
    `WAITING_FOR_RESOURCES` for several days and was abandoned。
  - Successful smoke run used the smaller v6e TRC spot queued resource in
    `us-east1-d`。
  - This is an availability / queue behavior note, not a performance claim。
- Current evidence/results：
  - TPU JSON reported backend `tpu`、device kind `TPU v6 lite`、mode
    `image_manifest`、manifest kind `public_example`、num_images 5、batch_size
    4、num_batches 2、num_padded_images 3、timed_batch_runs 10。
  - TPU JSON reported total timed inference around 0.01098252 sec and
    throughput around 2276.3446 images/sec。
  - Generated comparison table reports about 1931.76x throughput speedup versus
    this local CPU `b4` public-example artifact for this specific smoke run。
- Limitations：
  - This is a small public smoke run only。
  - It uses five public example images, batch size 4, final-batch padding
    (`num_padded_images = 3`), and a short benchmark loop。
  - It is not dataset-level accuracy evaluation。
  - It is not a full controlled hardware benchmark。
  - It should not be described as TPU generally being 1931x faster。
  - Raw JSON under `runs/vit-inference/` remains ignored/generated and should not
    be committed in this edit；only the curated Markdown table belongs under
    `report/results/`。
- Next planned step：
  - At that phase, the next planned step was broader TPU evidence or a
    controlled local-vs-TPU comparison. Phase 5.9 later added Imagenette TPU
    inference tables; controlled comparison and monitoring evidence remain
    future work。

## Phase 5.8: Demo 2 TPU Documentation Architecture Split

- Date / phase label：2026-06-03 Demo 2 TPU documentation architecture split
- What changed：
  - Created `cloud/demo2_tpu_quickstart.md` as the reusable user-facing TPU
    execution quickstart。
  - Refactored `cloud/demo2_pretrained_vit_tpu_workflow.md` into a reference
    document for resource variants, monitoring/evidence guidance, cleanup
    discipline, troubleshooting, and the course smoke-run appendix。
  - Clarified that TRC spot quota was the course project's funding/quota path,
    not a requirement of the Demo 2 code。
  - Kept `report/google_cloud_trc_setup.md` as the course-specific Google
    Cloud / TRC setup and evidence record。
- Claim boundary：
  - The completed TPU result remains a five-image public smoke-run comparison,
    not a full benchmark study, not dataset-level accuracy evaluation, and not a
    controlled hardware comparison。

## Phase 5.9: Demo 2 Imagenette TPU Inference Tables

- Date / phase label：2026-06-04 Demo 2 Imagenette 320 TPU inference evidence
- What changed：
  - Retrieved TPU JSON artifacts for Imagenette 320 inference are now reflected
    in the report surface。
  - Artifact families covered：
    - `val64`: `b1`, `b4`, `b8`
    - `val256`: `b1`, `b4`, `b8`
    - `val_full`: `b1`, `b4`, `b8`
  - Raw JSON artifacts remain under ignored/generated
    `runs/vit-inference/` and are not committed。
  - Generated curated Markdown TPU tables under `report/results/` with
    `scripts/compare_vit_results.py --markdown-output`。
  - Added documented recursive retrieval command using
    `gcloud compute tpus tpu-vm scp --recurse` for copying the whole
    `runs/vit-inference/` folder from a TPU VM。
- Files added：
  - `report/results/demo2_cloud_imagenette320_val64_tpu.md`
  - `report/results/demo2_cloud_imagenette320_val256_tpu.md`
  - `report/results/demo2_cloud_imagenette320_valfull_tpu.md`
- Files updated：
  - `README.md`
  - `cloud/demo2_tpu_quickstart.md`
  - `cloud/demo2_pretrained_vit_tpu_workflow.md`
  - `docs/demo2_pretrained_vit.md`
  - `report/current_status.md`
  - `report/google_cloud_trc_setup.md`
  - `report/results/README.md`
  - `report/progress_log.md`
- Current evidence/results：
  - TPU Imagenette tables report backend `tpu` and JSON-visible device kind
    `TPU v6 lite`。
  - `val64` table covers 64 images with no padding for `b1`, `b4`, and `b8`。
  - `val256` table covers 256 images with no padding for `b1`, `b4`, and `b8`。
  - `val_full` table covers 3925 images; `b4` and `b8` include 3 padded
    final-batch entries。
- Limitations：
  - This phase did not run `gcloud`, create/delete cloud resources, rerun TPU
    benchmarks, commit, or push。
  - The tables are ViT inference timing evidence only。
  - They are not training results, not fine-tuning, not dataset-level accuracy
    evaluation, not a full controlled benchmark study, and not a universal TPU
    speedup claim。

## Phase 5.10: Demo 2 Grouped Result Summaries

- Date / phase label：2026-06-04 Demo 2 generated grouped summary tables
- What changed：
  - Added generated report summaries with
    `scripts/generate_vit_summary_tables.py`。
  - Imagenette 320 summary files now stay focused on Imagenette evidence, while
    `report/results/demo2_public_examples_summary.md` records public-example
    smoke/demo evidence separately。
- Claim boundary：
  - These summaries are ViT inference evidence only；they are not training,
    dataset-level accuracy evaluation, a full benchmark study, or a universal
    TPU speedup claim。

## Phase 5.11: Demo 2 Optional TPU Run-When-Active Helper

- Date / phase label：2026-06-04 Demo 2 optional TPU run-when-active helper
- What changed：
  - Added `scripts/demo2_tpu_run_when_active.sh` as an optional
    run-when-active helper for an existing queued resource。
  - The helper waits for `ACTIVE`, runs the existing public-example TPU `b4`
    smoke command, retrieves the JSON artifact, runs the existing local
    comparison command, and prints cleanup instructions。
  - Updated `cloud/demo2_tpu_quickstart.md` with a short optional-helper
    section while keeping the manual quickstart as the primary workflow。
- Claim boundary：
  - This was a documentation/script addition only；it did not run `gcloud`,
    create/delete cloud resources, rerun TPU benchmarks, or add new TPU
    evidence。

## Phase 5.12: Demo 2 Optional ViT Classifier-Head Fine-Tuning Extension

- Date / phase label：2026-06-05 Demo 2 classifier-head fine-tuning workflow
  support
- What changed：
  - Added `examples/demo2_pretrained_vit_finetune.py` as an optional Demo 2
    extension, not a new Demo 3。
  - The script uses the existing pretrained ViT model path, freezes the ViT
    backbone, trains only the classifier head, and writes generated artifacts
    under ignored `runs/vit-finetune/`。
  - Added Orbax checkpoint/resume support for classifier-head parameters,
    optimizer state, current step, and minimal metadata only。
  - Added lightweight helper tests that avoid TPU, Google Cloud credentials,
    model downloads, Imagenette data, and Orbax-heavy runtime。
  - Added the `training` dependency group for `optax` and `orbax-checkpoint`。
  - Updated README, Demo 2 docs, TPU quickstart/reference docs, current status,
    and result-artifact guidance。
- Claim boundary：
  - This phase did not run model downloads, TPU jobs, `gcloud` commands, cloud
    resource creation/deletion, commits, or pushes。
  - No TPU fine-tuning evidence exists yet from this phase。
  - The extension is a checkpoint/resume and TPU training smoke workflow, not
    full ViT fine-tuning, not a large benchmark, and not dataset-level accuracy
    evaluation。

## Planned Phases

### Phase 6: Real MNIST/Fashion-MNIST and Curated Local Result

- Status：planned, not completed。
- Goal：extend Demo 1 from synthetic MNIST-shaped data to real MNIST and Fashion-MNIST local-file data。
- Planned work：
  - Add local-file dataset loading without making network access required for default tests。
  - Keep synthetic mode as the default smoke path。
  - Produce a curated local result under `report/results/` with command, environment, configuration, and metrics。
  - Add or update tests for small dataset-loading behavior where practical。
- Evidence needed before marking complete：
  - Source code for loader。
  - Tests or smoke checks。
  - Curated local benchmark artifact and documented command。

### Phase 7: TPU Workflow and Local-vs-TPU Comparison for Demo 1

- Status：planned, not completed。
- Goal：run Demo 1 on Google Cloud TPU and compare against local execution using reproducible evidence。
- Planned work：
  - Document TPU VM setup, repository checkout/update, environment setup, run command, monitoring, output retrieval, and cleanup。
  - Run CNN benchmark with a clear `platform_label` such as `tpu`。
  - Capture JSON metrics, terminal logs, backend/devices, and monitoring evidence where available。
  - Generate local-vs-TPU comparison table or figure。
- Evidence needed before marking complete：
  - TPU run metrics。
  - Cloud logs or screenshots。
  - Cleanup notes。
  - Comparison artifact grounded in saved metrics。

### Phase 8: Broader Demo 2 TPU Benchmark Evidence

- Status：partially completed by Phase 5.7 public smoke-run evidence and Phase
  5.9 Imagenette TPU inference tables；controlled comparison work remains
  planned。
- Goal：extend the first Demo 2 public-example TPU smoke evidence into a broader
  Google Cloud TPU evidence set only if time, quota, and evidence quality are
  sufficient。
- Planned work：
  - Preserve the first public-example TPU smoke evidence without overstating it。
  - Preserve the Imagenette TPU inference tables without describing them as
    training, accuracy evaluation, or universal speedup evidence。
  - If another TPU run is justified, record exact commit SHA, backend/device
    output, JSON metrics, terminal logs, monitoring notes, cleanup evidence,
    and deletion verification。
  - Prefer a controlled CPU-vs-TPU comparison with longer timing loops over
    repeating another tiny smoke run。
  - Compare TPU metrics against matching local CPU artifacts with
    `scripts/compare_vit_results.py` after TPU artifacts are retrieved locally。
- Evidence needed before marking complete：
  - Controlled comparison metrics beyond the first five-image smoke run and
    basic Imagenette TPU inference tables。
  - Exact commit / environment metadata。
  - Monitoring notes or privacy-safe screenshots if available。
  - Cleanup evidence for any additional resources。
  - Local-vs-TPU comparison artifact grounded in saved metrics。

### Phase 9: Optional Demo 3 Pretrained/Gemma Cloud Demo

- Status：optional planned work, not completed。
- Goal：show how a pretrained-model cloud workflow differs from the smaller training demos, only if access and scope are manageable。
- Planned work：
  - Review model source, license, access requirements, and usage constraints。
  - Decide whether the demo is inference-only, fine-tuning, or workflow demonstration。
  - Estimate dependencies, hardware, memory, quota, and setup time。
  - Prepare fallback material if access, quota, hardware, or time is unavailable。
- Evidence needed before marking complete：
  - Model selection and access/license notes。
  - Setup and run commands。
  - Cloud or local execution evidence。
  - Clear fallback or limitation notes if not executed。
