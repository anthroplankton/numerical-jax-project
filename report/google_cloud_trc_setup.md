# Google Cloud / TRC Setup Record

## Purpose

本文件記錄 Demo 2 pretrained ViT image inference benchmark 的 Google
Cloud onboarding setup、TRC confirmation，以及第一個 Cloud TPU public-example
smoke run 的 privacy-safe 狀態。這份紀錄的用途是提供 course report 可以引用
的 setup / execution evidence，並保留 CPU-vs-TPU comparison 的脈絡。

本文件不是 generic user quickstart。可重用的 TPU execution instructions
請看 `cloud/demo2_tpu_quickstart.md`；較完整的 resource / evidence reference
請看 `cloud/demo2_pretrained_vit_tpu_workflow.md`。

TRC 是本課程專案使用的 quota / funding path，不是程式本身的需求。其他使用者只要
有其他有效的 Google Cloud TPU quota / funding path，也可以依照 quickstart 執行
Demo 2 TPU smoke run。

目前已完成的 TPU evidence 是 small public smoke run，不是 full controlled
benchmark study，也不是 dataset-level accuracy evaluation。

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
- 第一個 Demo 2 TPU public-example smoke run 已完成。
- 成功的 TPU resource 使用：
  - zone：`us-east1-d`
  - accelerator type：`v6e-1`
  - runtime version：`v2-alpha-tpuv6e`
  - quota/funding type：TRC spot quota
  - JSON-visible device kind：`TPU v6 lite`
- 成功的 TPU run 使用 branch：`feat/demo2-tpu-evidence`。
- exact TPU checkout commit 沒有保存在目前可用的 report notes 中；不可用
  post-edit local commit SHA 代替，這是 reproducibility limitation。
- TPU JSON artifact 已產生並取回：
  `runs/vit-inference/demo2_cloud_public_examples_tpu_b4.json`。
- CPU-vs-TPU comparison Markdown table 已產生：
  `report/results/demo2_local_cpu_vs_cloud_tpu_public_examples_b4.md`。
- Cleanup 已完成；queued resource deletion succeeded，且
  `gcloud compute tpus queued-resources list` 與
  `gcloud compute tpus tpu-vm list` 在 selected zone 回傳 0 items。

因此，目前可以宣稱已完成第一個 TPU smoke run 與 artifact retrieval /
comparison / cleanup verification。不能宣稱 full benchmark study、
dataset-level accuracy evaluation、controlled hardware comparison，或 TPU
generally 1931x faster。

## Operational Notes

- Initial attempt：一個 v4 queued resource 在 `us-central2-b` 維持
  `WAITING_FOR_RESOURCES` 數日後放棄。這是 availability / queue behavior
  note，不是 performance claim。
- Successful attempt：改用較小的 v6e TRC spot queued resource，在
  `us-east1-d` 完成 public-example smoke run。
- Network / subnet note：成功的 first smoke run 使用 selected region 的
  default VPC network 與 default subnet。Earlier resource attempts showed that
  subnet availability can matter; missing regional default subnet can block TPU
  resource creation。除非有明確且安全的需求，不在 repository 記錄 private IPs、
  subnet CIDR ranges、hostnames 或 detailed network topology。

## Remaining Work

以下工作仍屬後續延伸，不應在尚未執行前寫成 completed evidence：

- 保存更完整的 branch / exact commit / environment metadata。
- 擷取可公開的 monitoring evidence 或 redacted screenshots。
- 執行較長 benchmark loop 或 Imagenette TPU benchmark。
- 進行 controlled local-vs-TPU hardware comparison。
- 加入 dataset-level accuracy evaluation，若後續定義 labels / top-k
  evaluation workflow。

## Cost-Control Principles

- 雖然第一個 TPU smoke run 已完成，未來任何 TPU resource 建立前仍需先確認
  zone、accelerator type、run command、artifact retrieval command 與 cleanup
  command。
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

## Next Planned Steps

1. 保留第一個 TPU smoke-run evidence，但不要提交 raw JSON under
   `runs/vit-inference/`。
2. 將 curated comparison table 保留在：
   `report/results/demo2_local_cpu_vs_cloud_tpu_public_examples_b4.md`。
3. 若後續再建立 TPU resource，重新 review
   `cloud/demo2_pretrained_vit_tpu_workflow.md`，確認 create、verify、run、
   retrieve、compare、cleanup commands 都仍符合當時環境。
4. 未來 run 應保存 exact commit SHA、backend/device verification、artifact
   retrieval、monitoring notes、cleanup transcript 與 deletion verification。
5. 若時間與 quota 允許，再執行 Imagenette TPU benchmark 或更 controlled 的
   local-vs-TPU comparison。
