# Numerical Computation with JAX 目前進度報告

> Snapshot date: 2026-05-22  
> Status: pre-TPU execution, awaiting TRC confirmation  
> Scope: Demo 2 local CPU benchmark evidence and TPU workflow preparation

## 1. 專案目標與目前進度

此專案是 Numerical Computation with JAX 的課程專案，目標是用 Python 與
JAX 建立可執行、可重現、可留下證據的 benchmark workflow，並把同一套流程從
local CPU 延伸到 Google Cloud TPU。現階段主軸是 **Demo 2: pretrained ViT
inference benchmark with JAX/Flax**，使用 `google/vit-base-patch16-224` 做 image
inference。

此專案的核心設計是 local-to-TPU workflow：local CPU baseline -> TPU VM setup ->
JAX TPU backend verification -> TPU benchmark run -> artifact retrieval ->
monitoring / observability analysis -> cleanup -> CPU-vs-TPU comparison。目前已完成
local CPU Demo 2 workflow、Imagenette CPU benchmark evidence、JSON metrics output、
Markdown result-table generation，以及 Google Cloud / TRC 的 pre-execution 文件準備。

TPU execution 仍因等待 **TRC confirmation** 尚未進行。此報告尚未有 TPU benchmark、TPU metrics、monitoring analysis、cleanup transcript
或 CPU-vs-TPU comparison。

## 2. Demo 2 ViT benchmark workflow

Demo 2 的主要 example 程式是 `examples/pretrained_vit_inference.py`。它使用
Hugging Face `AutoImageProcessor` 做 preprocessing，並用
`FlaxViTForImageClassification` 在 JAX/Flax backend 上執行 pretrained ViT
inference。這個 demo 是 inference-only，沒有 fine-tuning，也不是 dataset-level
accuracy evaluation。

此專案目前支援單張圖片與 `--image-manifest` 兩種輸入。manifest mode 會讀取多張圖片
並組成 mixed-image batches；最後一個不滿的 batch 會重複最後一張真實圖片做
padding，並在 metrics 中記錄 `num_padded_images`。padding 圖片不計入 prediction
與 throughput 的真實圖片數，因此可以避免把補齊 batch 的資料誤算進 benchmark 結果。

benchmark workflow 包含 `warmup` 與 timed loop，並在 timed inference 後使用
`block_until_ready()`，避免 JAX asynchronous execution 讓 timing 太早停止。其中
`warmup` 不作為正式 benchmark 數據，而是先讓 JAX/Flax execution path 完成必要的
初始化、compilation 相關開銷或 cache warm-up。後面的 timed loop 才用來統計
throughput，這樣可以減少第一次執行造成的偏差。每次執行會記錄 selected JAX
platform、實際 backend、devices、batch size、num images、padding、mean step time、
total timed inference time、throughput，並輸出 JSON artifact。

完成 benchmark 後，`scripts/compare_vit_results.py` 會把既有 JSON artifacts 整理成
report-ready Markdown table，因此同一套結果格式可以用在 local CPU 與後續 TPU
comparison。Demo 2 的 example 程式也已提供 `--jax-platform tpu` 參數，作為後續在
TPU VM 上重跑相同 benchmark workflow 的入口。等 TRC confirmation 後，預計取回 TPU
JSON artifact，並把 application-level JSON metrics 與 Google Cloud
infrastructure-level monitoring metrics 一起分析。

## 3. 目前 CPU benchmark 結果與詮釋

目前最適合作為主要 CPU evidence 的結果是 Imagenette 320 validation manifest
`val256`。

| Machine / environment | Batch size | Images | Padded | Mean step (s) | Throughput | Within-scope speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| older local laptop CPU | 1 | 256 | 0 | 0.449014 | 2.2271 img/s | 1x |
| older local laptop CPU | 4 | 256 | 0 | 1.51115 | 2.6470 img/s | 1.19x |
| older local laptop CPU | 8 | 256 | 0 | 2.53903 | 3.1508 img/s | 1.41x |
| newer external CPU machine | 1 | 256 | 0 | 0.0834544 | 11.9826 img/s | 1x |
| newer external CPU machine | 4 | 256 | 0 | 0.284359 | 14.0667 img/s | 1.17x |
| newer external CPU machine | 8 | 256 | 0 | 0.580369 | 13.7843 img/s | 1.15x |

older local laptop CPU 的 `val256` 顯示 batch size 從 b1 到 b8 時，throughput 從
2.2271 img/s 提升到 3.1508 img/s。newer external CPU machine 整體 throughput
明顯較高，但 b4 與 b8 數值接近，也提醒 batch size 增加不一定會帶來單調提升。這個
結果主要是 supplementary evidence，表示同一套 workflow 可以在另一個 CPU 環境重跑；
兩台機器年分、CPU、OS/WSL、cache state、thermal condition 都不同，因此這部分目前
只能作為背景脈絡，不適合作為嚴格公平的 controlled hardware benchmark。

batch size 比較也是此專案設計中的一部分。理論上，TPU 這類 accelerator 較有機會在
較大的 batch size 下發揮平行化優勢，並攤平部分固定開銷；但 throughput 不一定會隨
batch size 單調增加，仍會受到 model shape、memory、compilation、padding、input
pipeline、host-device transfer 與實際 hardware utilization 影響。因此，目前 CPU 端的
batch-size benchmark 比較適合作為 baseline 與測量框架，真正的 CPU-vs-TPU 差異需要
等 TPU 實際執行後再判斷。

## 4. JAX-specific implementation

Demo 2 使用 pretrained ViT model，因此 model architecture 與 pretrained weights
不是此專案自行訓練或設計。此階段主要完成的是把 JAX/Flax pretrained inference
包成可重現的 execution 與 benchmark workflow，包含 batching、padding、warmup、
timed loop、asynchronous timing control、backend/device metadata、JSON artifacts，
以及 Markdown result-table generation。

首先，benchmark 測量的是 jitted inference path，但目前沒有 non-jit ablation，因此不把
效能差異歸因於 `jax.jit` 本身。比較重要的是，同一個 jitted path 會先跑 warmup，再進入
timed loop，並用 `block_until_ready()` 等 JAX computation 真正完成。

```python
@jax.jit
def inference_step(model_params: Mapping[str, Any], inputs: Any) -> Any:
    return model(pixel_values=inputs, params=model_params, train=False).logits

for _ in range(warmup_steps):
    for input_batch in input_batches:
        logits = inference_step(params, input_batch)
        logits.block_until_ready()

for _ in range(benchmark_steps):
    for input_batch in input_batches:
        start_time = time.perf_counter()
        logits = inference_step(params, input_batch)
        logits.block_until_ready()
```

因為 JAX 可能 asynchronous dispatch，若沒有 `block_until_ready()`，timing 可能只量到
派送工作到 backend 的時間，而不是實際 inference 完成時間。

第二，batch size 會改變實際傳入模型的 input tensor shape。manifest mode 會把圖片
分成 batches，最後一個不滿的 batch 會用最後一張真實圖片 padding 到固定 batch shape；
metrics 會記錄 `num_padded_images`，throughput 則只計算真實圖片。這讓同一個 batch
size 下的 benchmark path 對 JAX/XLA execution 比較乾淨。

```python
batch_specs = build_manifest_batch_specs(
    num_images=len(image_paths),
    batch_size=batch_size,
)
input_batches = [
    pad_manifest_batch(
        pixel_values[batch_spec["start_index"] : batch_spec["end_index"]],
        padding_count=batch_spec["padding_count"],
        jnp_module=jnp,
    )
    for batch_spec in batch_specs
]
```

第三，每次 benchmark 也會把 selected JAX platform、實際 backend、devices、batch
size、warmup steps、benchmark steps、num images、num batches、padding、mean step
time、total timed inference time 與 throughput 寫入 JSON artifacts。這些 metadata
讓 local CPU artifacts 與未來 TPU artifacts 可以用同一個 result-table workflow
整理與比較。

Demo 1 仍保留為 raw-JAX CNN training foundation，可補充說明專案並非只有呼叫
pretrained model；但本次進度報告主線仍以 Demo 2 的 ViT inference benchmark 與
TPU workflow 準備為主。

## 5. TRC / TPU 目前狀態

Google Cloud preparation 已經開始。repository 中記錄的狀態包含：dedicated Google
Cloud project 已建立，billing 已 linked，budget alerts 已設定，Cloud TPU API 已啟用，
project number 已提交到 TRC short form，並且已撰寫 pre-TRC TPU VM workflow 文件。
這些內容使用 placeholders，報告中不放入 project ID、project number、billing
details、account details 或其他 private Google Cloud identifiers。

TPU 部分是此專案的重要實驗階段，不是附帶項目。預計流程是：

1. 在 local CPU 先保留可重現 baseline。
2. 建立或連線到 TPU VM，並記錄 branch / commit / runtime。
3. 在 TPU VM 上確認 JAX TPU backend 與 devices。
4. 用 `--jax-platform tpu` 執行 Demo 2 benchmark。
5. 取回 TPU JSON artifact。
6. 收集 monitoring / observability evidence。
7. 清理 TPU resources 並保留 cleanup evidence。
8. 產生 local CPU vs TPU comparison table。

目前正等待 **TRC confirmation**。TPU VM 建立、JAX TPU backend/device
verification、Demo 2 TPU benchmark、TPU JSON artifact retrieval、monitoring /
observability analysis、cleanup transcript，以及 CPU-vs-TPU comparison 都仍是
pending work。

後續監控比較理想的 evidence 會分成三層：

- Application-level evidence：benchmark JSON、throughput、mean step time、
  backend、devices、batch size、num images。
- Infrastructure-level evidence：Google Cloud monitoring metrics、TPU
  utilization、idle time、memory usage、CPU usage、runtime logs、resource
  lifecycle、cleanup record。
- Report-level interpretation：確認 TPU workload 是否真的正確執行、是否有明顯
  idle time 或 overhead，以及 observed performance difference 是否有意義。


## 6. 限制與下一步

目前的主要限制是 CPU baseline 已完成，但 TPU execution 尚未完成，所以還不能回答
local CPU 與 TPU 的實際 performance 差異。Imagenette benchmark 也只量測 inference
timing 與 throughput，沒有計算 classification accuracy。hardware comparison 目前只
能作為背景描述，因為 repository 沒有可靠、完整的 CPU model、RAM、OS/WSL、JAX/JAXLIB
metadata 可形成 controlled comparison。

下一步是等待 TRC confirmation，確認 zone、accelerator type、
runtime version、成本與 cleanup plan 後，依照 `cloud/demo2_pretrained_vit_tpu_workflow.md`
執行短 TPU smoke run。實際執行完成後，才應整理 TPU JSON metrics、terminal logs、
Google Cloud monitoring / observability evidence、cleanup evidence，並用
`scripts/compare_vit_results.py` 產生 local CPU vs TPU comparison table。
