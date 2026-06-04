# Demo 2 Cloud TPU ViT Head Fine-Tuning GCS Resume Smoke Evidence

This is a curated, report-facing summary for the optional Demo 2 classifier-head
fine-tuning workflow. It summarizes observed TPU checkpoint/resume evidence and
does not include raw logs, checkpoints, model caches, datasets, or GCS objects.

## Scope

- Demo: Demo 2 pretrained ViT classifier-head fine-tuning extension.
- Model family: `google/vit-base-patch16-224`.
- Trainable scope: `classifier_head_only`.
- Frozen scope: `vit_backbone`.
- Checkpoint payload: classifier head parameters, optimizer state, current step,
  and minimal metadata only.
- Evidence type: TPU training smoke workflow, durable checkpoint copy, and
  restore/resume evidence.
- Not evidence for: full ViT fine-tuning, dataset-level accuracy, model quality,
  or a controlled hardware benchmark.

## Observed Workflow Summary

| Item | Curated value |
| --- | --- |
| First run TPU shape | `v6e-1` spot, `us-east1-d` |
| Interruption evidence | Real spot or maintenance interruption after the first run |
| Durable checkpoint copies | GCS checkpoint steps `15100`, `15120`, `15140` |
| Resume TPU shape | `v6e-1` spot, `europe-west4-a` |
| Resume backend | `tpu` |
| Resumed from checkpoint | `true` |
| Resume start step | `15140` |
| Resume final step | `51538` |
| Trainable scope | `classifier_head_only` |
| Frozen scope | `vit_backbone` |

The available curated summary does not record `initial_loss`, `final_loss`,
`mean_step_time_sec`, `examples_per_second`, or raw `metrics.csv` rows. Do not
fabricate those values from memory or logs that are not included in this
reduced report artifact.

## Interpretation

This evidence demonstrates that the optional Demo 2 workflow can run on TPU,
copy local Orbax checkpoints to durable GCS storage, restore the classifier-head
training state on a replacement TPU VM, and continue from the restored step.
Near-zero loss in the tiny `train64`/`val64` smoke setup would not imply
dataset-level accuracy; the manifest limit may select a small and class-skewed
subset.
