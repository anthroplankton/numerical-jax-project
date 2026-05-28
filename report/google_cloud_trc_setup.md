# Google Cloud / TRC Setup Record

## Purpose

本文件記錄 Demo 2 pretrained ViT image inference benchmark 進入 Google
Cloud TPU / TRC execution 之前，已完成的 Google Cloud onboarding setup。
這份紀錄的用途是提供 course report 可以引用的 setup evidence，並保留
後續 CPU-vs-TPU comparison 的脈絡。

本文件只記錄 setup 狀態，不代表 TPU VM 已建立，也不代表 TPU benchmark
已完成。目前已完成的 benchmark evidence 仍是 local CPU Demo 2 artifacts。

## Privacy And Placeholders

為避免暴露 private cloud identifiers，本文件只使用 placeholders：

- `<PROJECT_ID>`：dedicated Google Cloud project ID。
- `<PROJECT_NUMBER>`：提交給 TRC 的 Google Cloud project number。
- `<ZONE>`：未來 TPU VM zone。
- `<TPU_NAME>`：未來 TPU VM resource name。
- `<ACCELERATOR_TYPE>`：未來 TPU accelerator type。
- `<RUNTIME_VERSION>`：未來 TPU VM runtime version。

本 repository 不記錄 real project number、real billing account ID、
credentials、service account keys、private TRC form URL，或未經遮蔽的
private cloud screenshots。

## Completed External Setup Steps

以下步驟已在 Google Cloud Console、`gcloud` CLI 或 TRC short form 中由
使用者於 repository 外部完成：

1. 建立 dedicated Google Cloud project，用於本 Numerical Computation
   with JAX course project。
2. 使用下列指令確認 project ID 與 project number：

   ```bash
   gcloud projects describe <PROJECT_ID> --format="table(projectId,name,projectNumber)"
   ```

   這個指令會以 table 欄位顯示 `projectId`、`name`、`projectNumber`。
   實際 project number 應保存在 private local notes 或 Google Cloud Console
   中，不提交到 repository。

3. 建立 billing account，並將 billing account linked 到 `<PROJECT_ID>`。
   Billing account ID 不記錄於 repository。
4. 建立 budget alerts 作為成本提醒：

   | Budget alert name | Threshold / amount |
   | --- | ---: |
   | `numerical-jax-first-warning` | 10 USD |
   | `numerical-jax-main-limit` | 60 USD |

5. 啟用 Cloud TPU API。
6. 將 `<PROJECT_NUMBER>` 提交到 TRC short form。
7. TRC confirmation 已收到；TRC email content、project-specific identifiers
   與 private cloud details 不記錄於 repository。

## Current Setup State

- TRC confirmation 已收到。
- repository 只記錄 confirmation status，不記錄 TRC email content、real
  project ID、real project number、billing account ID、credentials 或 private
  cloud identifiers。
- 尚未建立 Cloud TPU VM。
- 尚未執行 `examples/pretrained_vit_inference.py --jax-platform tpu`。
- 尚未產生 cloud TPU benchmark JSON、logs、monitoring notes 或 screenshots。
- 尚未完成 local CPU vs TPU comparison。

因此，目前不能宣稱 TPU execution、TPU performance result、cloud benchmark
或 CPU-vs-TPU comparison 已完成。

## Not Done Yet

以下工作尚未進行，需在建立 TPU VM 前再次確認 placeholders、quota、cost
風險與 cleanup plan：

- 選定 `<ZONE>`、`<TPU_NAME>`、`<ACCELERATOR_TYPE>` 與 `<RUNTIME_VERSION>`。
- 建立或啟動 TPU VM。
- 在 TPU VM 上 clone 或 update repository。
- 在 TPU VM 上安裝 pretrained demo dependencies 與 TPU-compatible JAX。
- 在 TPU VM 上確認 JAX backend/devices。
- 在 TPU VM 上執行 Demo 2 benchmark。
- 下載或整理 TPU result artifacts。
- 擷取 monitoring evidence 或 redacted screenshots。
- 刪除 TPU VM 並記錄 cleanup evidence。
- 使用 `scripts/compare_vit_results.py` 產生 CPU-vs-TPU comparison。

## Cost-Control Principles

- 雖然 TRC confirmation 已收到，在 zone、accelerator type、run command、
  artifact retrieval command 與 cleanup command 都明確之前，不建立 TPU
  resources。
- 任何 TPU VM 建立前，都先確認 cleanup command：

  ```bash
  gcloud compute tpus tpu-vm delete <TPU_NAME> --zone=<ZONE>
  ```

- 先以短時間 smoke run 驗證 backend/device visibility、model loading 與 JSON
  output，再考慮較完整的 benchmark run。
- 保留 Google Cloud Console 的 budget alerts 作為提醒，但不把 budget alert
  視為硬性 resource shutdown mechanism。
- 不提交 billing details、credentials、service account keys、`.env` files、
  model caches、private screenshots 或 uncurated cloud logs。
- Cloud resources 建立、啟動、停止或刪除都應由使用者手動確認，不由 repo
  automation 自動執行。

## Next Planned Steps Before TPU VM Creation

1. 根據 TRC confirmation 與 Google Cloud Console 狀態，確認可用的
   `<ZONE>`、`<ACCELERATOR_TYPE>` 與 `<RUNTIME_VERSION>`。
2. Review `cloud/demo2_pretrained_vit_tpu_workflow.md`，確認 create、verify、run、
   retrieve、compare、cleanup commands 都仍符合當時環境。
3. 在 local machine 先執行 local preflight checks，包括 `git status`、
   Ruff、pytest，以及確認 local CPU Demo 2 artifacts。
4. 建立 TPU VM 前，再次確認 cleanup command 與成本風險。
5. 在 TPU VM 上執行 JAX backend/device verification。
6. 以 `--jax-platform tpu` 執行 Demo 2 pretrained ViT inference benchmark。
7. 取回 TPU JSON metrics、terminal logs、monitoring notes，並在必要時加入
   redacted screenshots。
8. 刪除 TPU VM，確認沒有遺留 running resources。
9. 使用 `scripts/compare_vit_results.py` 比較 local CPU 與 TPU result files。
10. 將實際 TPU execution、metrics、cleanup 與 comparison evidence 補到
    `report/results/`、`report/progress_log.md` 或 final report material。
