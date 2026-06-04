#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/demo2_tpu_run_when_active.sh [--delete-after] [--poll-seconds N] [--max-wait-seconds N]

Optional run-when-active helper for Demo 2 TPU execution. This script does not
create queued resources. It monitors an existing queued resource until ACTIVE,
runs the public-example TPU b4 smoke command on the TPU VM, retrieves the JSON
artifact, runs the local CPU-vs-TPU comparison, and prints cleanup instructions.

Required environment variables:
  PROJECT_ID
  ZONE
  TPU_NAME
  QUEUED_RESOURCE_ID
  REPO_URL
  BRANCH

Options:
  --delete-after        Delete the queued resource after retrieval/comparison.
  --poll-seconds N      Poll interval while waiting for ACTIVE. Default: 20.
  --max-wait-seconds N  Stop waiting after N seconds. Default: 3600.
                        Use 0 to wait without a timeout.
  -h, --help            Show this help text.
USAGE
}

delete_after=false
poll_seconds=20
max_wait_seconds=3600

while (($# > 0)); do
  case "$1" in
    --delete-after)
      delete_after=true
      shift
      ;;
    --poll-seconds)
      poll_seconds="${2:?--poll-seconds requires a value}"
      shift 2
      ;;
    --max-wait-seconds)
      max_wait_seconds="${2:?--max-wait-seconds requires a value}"
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

require_non_negative_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "$name must be a non-negative integer, got: $value" >&2
    exit 2
  fi
}

require_non_negative_integer "--poll-seconds" "$poll_seconds"
require_non_negative_integer "--max-wait-seconds" "$max_wait_seconds"
if ((poll_seconds == 0)); then
  echo "--poll-seconds must be greater than 0" >&2
  exit 2
fi

required_vars=(
  PROJECT_ID
  ZONE
  TPU_NAME
  QUEUED_RESOURCE_ID
  REPO_URL
  BRANCH
)

missing_vars=()
for var_name in "${required_vars[@]}"; do
  if [[ -z "${!var_name:-}" ]]; then
    missing_vars+=("$var_name")
  fi
done

if ((${#missing_vars[@]} > 0)); then
  echo "Missing required environment variables: ${missing_vars[*]}" >&2
  usage >&2
  exit 2
fi

if [[ "$REPO_URL" =~ ^https?://[^/]+@ ]]; then
  echo "REPO_URL must not contain embedded credentials." >&2
  exit 2
fi

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud is required for this helper." >&2
  exit 127
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required locally for the comparison step." >&2
  exit 127
fi

mkdir -p runs/vit-inference
log_path="runs/vit-inference/demo2_tpu_run_when_active_$(date -u +%Y%m%dT%H%M%SZ).log"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$log_path"
}

log_command() {
  log "COMMAND: $*"
}

normalize_queued_resource_state() {
  local normalized="$1"
  normalized="${normalized//$'\r'/}"
  normalized="${normalized//$'\n'/}"
  while [[ "$normalized" == [[:space:]]* ]]; do
    normalized="${normalized#?}"
  done
  while [[ "$normalized" == *[[:space:]] ]]; do
    normalized="${normalized%?}"
  done
  normalized="${normalized#state=}"
  printf '%s\n' "$normalized"
}

print_cleanup_instructions() {
  cleanup_printed=true
  cat <<'CLEANUP'

Cleanup commands to run explicitly after reviewing artifacts:

gcloud compute tpus queued-resources delete "$QUEUED_RESOURCE_ID" \
  --project "$PROJECT_ID" \
  --zone "$ZONE" \
  --force \
  --quiet

gcloud compute tpus queued-resources list \
  --project "$PROJECT_ID" \
  --zone "$ZONE"

gcloud compute tpus tpu-vm list \
  --project "$PROJECT_ID" \
  --zone "$ZONE"
CLEANUP
}

shell_quote() {
  printf '%q' "$1"
}

helper_started=false
cleanup_printed=false

on_exit() {
  local exit_code=$?
  set +e
  if ((exit_code != 0)) &&
    [[ "$helper_started" == true ]] &&
    [[ "$cleanup_printed" != true ]]; then
    log "Helper failed with exit code $exit_code. Cleanup instructions follow."
    print_cleanup_instructions
  fi
  exit "$exit_code"
}

trap on_exit EXIT

helper_started=true
log "Demo 2 TPU helper started. Log stores sanitized command templates only."
log "Required environment variables are set; values are intentionally not logged."
log "This helper does not create queued resources."

local_cpu_json="runs/vit-inference/demo2_local_public_examples_cpu_b4.json"
cloud_tpu_json="runs/vit-inference/demo2_cloud_public_examples_tpu_b4.json"
comparison_json="runs/vit-inference/demo2_local_cpu_vs_cloud_tpu_public_examples_b4_compare.json"
comparison_md="report/results/demo2_local_cpu_vs_cloud_tpu_public_examples_b4.md"

if [[ ! -f "$local_cpu_json" ]]; then
  log "Missing local CPU baseline JSON: $local_cpu_json"
  cat >&2 <<'BASELINE'
Create the local baseline first from the repository root:

uv run --group pretrained python examples/pretrained_vit_inference.py \
  --jax-platform cpu \
  --image-manifest examples/assets/manifest.txt \
  --batch-size 4 \
  --warmup-steps 1 \
  --benchmark-steps 5 \
  --output runs/vit-inference/demo2_local_public_examples_cpu_b4.json
BASELINE
  exit 1
fi

start_epoch="$(date +%s)"
while true; do
  log_command 'gcloud compute tpus queued-resources describe "$QUEUED_RESOURCE_ID" --project "$PROJECT_ID" --zone "$ZONE" --format="value(state)"'
  raw_state="$(
    gcloud compute tpus queued-resources describe "$QUEUED_RESOURCE_ID" \
      --project "$PROJECT_ID" \
      --zone "$ZONE" \
      --format="value(state)"
  )"
  state="$(normalize_queued_resource_state "${raw_state:-UNKNOWN}")"
  state="${state:-UNKNOWN}"
  log "Queued resource state: $state"

  case "$state" in
    ACTIVE)
      break
      ;;
    WAITING_FOR_RESOURCES | PROVISIONING | CREATING)
      ;;
    *)
      log "Unexpected queued resource state; stopping before TPU execution."
      print_cleanup_instructions
      exit 1
      ;;
  esac

  if ((max_wait_seconds > 0)); then
    now_epoch="$(date +%s)"
    elapsed_seconds=$((now_epoch - start_epoch))
    if ((elapsed_seconds >= max_wait_seconds)); then
      log "Timed out waiting for ACTIVE after ${elapsed_seconds}s."
      print_cleanup_instructions
      exit 124
    fi
  fi

  sleep "$poll_seconds"
done

log_command 'gcloud compute tpus tpu-vm describe "$TPU_NAME" --project "$PROJECT_ID" --zone "$ZONE" --format="value(name,state)"'
gcloud compute tpus tpu-vm describe "$TPU_NAME" \
  --project "$PROJECT_ID" \
  --zone "$ZONE" \
  --format="value(name,state)"

remote_script="$(cat <<'REMOTE'
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

if [[ ! -d numerical-jax-project/.git ]]; then
  git clone --origin origin "$REPO_URL" numerical-jax-project
fi

cd numerical-jax-project
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REPO_URL"
else
  git remote add origin "$REPO_URL"
fi
git fetch origin --prune
if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  git switch "$BRANCH"
else
  git switch --create "$BRANCH" --track "origin/$BRANCH"
fi
git pull --ff-only origin "$BRANCH"
git status --short --branch
git rev-parse HEAD

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv --version
uv sync --group pretrained

uv pip install -U "jax[tpu]" \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

uv run python - <<'PY'
import jax

backend = jax.default_backend()
devices = jax.devices()
print("jax_version=", jax.__version__)
print("default_backend=", backend)
print("device_count=", jax.device_count())
print("local_device_count=", jax.local_device_count())
print("devices=", devices)
if backend != "tpu":
    raise SystemExit(f"Expected JAX default backend 'tpu', got {backend!r}")
if not devices:
    raise SystemExit("No JAX devices are visible")
PY

uv run --group pretrained python examples/pretrained_vit_inference.py \
  --jax-platform tpu \
  --image-manifest examples/assets/manifest.txt \
  --batch-size 4 \
  --warmup-steps 1 \
  --benchmark-steps 5 \
  --output runs/vit-inference/demo2_cloud_public_examples_tpu_b4.json

ls -lh runs/vit-inference/demo2_cloud_public_examples_tpu_b4.json
REMOTE
)"

remote_command="REPO_URL=$(shell_quote "$REPO_URL") BRANCH=$(shell_quote "$BRANCH") bash -lc $(shell_quote "$remote_script")"

log_command 'gcloud compute tpus tpu-vm ssh "$TPU_NAME" --project "$PROJECT_ID" --zone "$ZONE" --command "<clone/update repo, verify TPU backend, run Demo 2 public b4 command>"'
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --project "$PROJECT_ID" \
  --zone "$ZONE" \
  --command "$remote_command"

log_command 'gcloud compute tpus tpu-vm scp "$TPU_NAME":~/numerical-jax-project/runs/vit-inference/demo2_cloud_public_examples_tpu_b4.json runs/vit-inference/demo2_cloud_public_examples_tpu_b4.json --project "$PROJECT_ID" --zone "$ZONE"'
gcloud compute tpus tpu-vm scp \
  "$TPU_NAME":~/numerical-jax-project/runs/vit-inference/demo2_cloud_public_examples_tpu_b4.json \
  "$cloud_tpu_json" \
  --project "$PROJECT_ID" \
  --zone "$ZONE"

log_command 'uv run python scripts/compare_vit_results.py runs/vit-inference/demo2_local_public_examples_cpu_b4.json runs/vit-inference/demo2_cloud_public_examples_tpu_b4.json --output runs/vit-inference/demo2_local_cpu_vs_cloud_tpu_public_examples_b4_compare.json --markdown-output report/results/demo2_local_cpu_vs_cloud_tpu_public_examples_b4.md'
uv run python scripts/compare_vit_results.py \
  "$local_cpu_json" \
  "$cloud_tpu_json" \
  --output "$comparison_json" \
  --markdown-output "$comparison_md"

if "$delete_after"; then
  log "Deletion explicitly requested with --delete-after."
  log_command 'gcloud compute tpus queued-resources delete "$QUEUED_RESOURCE_ID" --project "$PROJECT_ID" --zone "$ZONE" --force --quiet'
  gcloud compute tpus queued-resources delete "$QUEUED_RESOURCE_ID" \
    --project "$PROJECT_ID" \
    --zone "$ZONE" \
    --force \
    --quiet

  log_command 'gcloud compute tpus queued-resources list --project "$PROJECT_ID" --zone "$ZONE"'
  gcloud compute tpus queued-resources list \
    --project "$PROJECT_ID" \
    --zone "$ZONE"

  log_command 'gcloud compute tpus tpu-vm list --project "$PROJECT_ID" --zone "$ZONE"'
  gcloud compute tpus tpu-vm list \
    --project "$PROJECT_ID" \
    --zone "$ZONE"
else
  log "Deletion was not requested. Review artifacts, then clean up manually."
  print_cleanup_instructions
fi

log "Demo 2 TPU helper finished."
log "Local log path: $log_path"
