#!/usr/bin/env bash
set -euo pipefail

# Shared persistent artifacts; source synchronization is external to this job.
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
B="${WARM_BASE:-/mnt/afs/task3_2/L202500276_lwz}"
PROJECT="${WARM_PROJECT:-$B/projects/WARM}"
PY="${WARM_TABLE4_PYTHON:-$B/envs/warm/bin/python}"
LEGACY="${WARM_TABLE4_LEGACY_CODE:-$B/projects/WARM_evaluations/code/c4763a975298de6f00939360551616af7902d57a}"
EVAL_ROOT="${WARM_TABLE4_EVAL_ROOT:-$B/projects/WARM_evaluations/results/step_019100/libero_10}"
OUTPUT="${WARM_TABLE4_OUTPUT:-$B/projects/WARM_evaluations/table4_libero_20260925/formal-$(date +%Y%m%d-%H%M%S)}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBGL_ALWAYS_SOFTWARE=1 OMP_NUM_THREADS=1
# CUDA and Mesa EGL use separate device namespaces. Both Python entrypoints
# install the explicit process-local software-EGL selector before LIBERO import.
unset MUJOCO_EGL_DEVICE_ID
export WARM_TABLE4_EGL_DEVICE_ID="${WARM_TABLE4_EGL_DEVICE_ID:-0}"
# Only the trusted official LIBERO initial-state files need legacy NumPy pickle.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$B/projects/WARM_evaluations/libero_config}"
export DIFFSYNTH_MODEL_BASE_PATH="$PROJECT/checkpoints" DIFFSYNTH_SKIP_DOWNLOAD=true
export HF_HOME="${HF_HOME:-$PROJECT/cache/huggingface}"
export TRITON_CACHE_DIR="/tmp/warm-table4-triton-${HOSTNAME:-node}-$$"
export TORCHINDUCTOR_CACHE_DIR="/tmp/warm-table4-inductor-${HOSTNAME:-node}-$$"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
test -x "$PY"
test -f "$LEGACY/experiments/libero/eval_libero_single.py"
if [[ -e "$OUTPUT" ]]; then
  echo "ERROR: output already exists; preserve evidence and choose a new output" >&2
  exit 2
fi
mkdir -p "$OUTPUT"
echo "table4_output=$OUTPUT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
N="${#GPUS[@]}"
if ((N < 1 || N > 4)); then
  echo "ERROR: expose between one and four H100 GPUs" >&2
  exit 2
fi
"$PY" -c 'import torch; assert torch.cuda.device_count() >= int(__import__("sys").argv[1]), "fewer visible GPUs than requested"' "$N"
REUSE=()
if [[ -n "${WARM_TABLE4_REUSE_SHARDS:-}" ]]; then
  IFS=':' read -r -a REUSE <<< "$WARM_TABLE4_REUSE_SHARDS"
fi
"$PY" "$CODE/scripts/plan_libero_table4_resume.py" --reuse-shards "${REUSE[@]}" \
  --workers "$N" --output "$OUTPUT/resume_plan.json"
mapfile -t EPISODE_GROUPS < <("$PY" -c 'import json,sys; p=json.load(open(sys.argv[1])); [print(",".join(map(str,x))) for x in p["episode_groups"]]' "$OUTPUT/resume_plan.json")
EXPECTED_SHA="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_sha256"] or "")' "$OUTPUT/resume_plan.json")"
N="${#EPISODE_GROUPS[@]}"
# Check every worker with its actual CUDA visibility, before loading any model.
for ((rank=0; rank<N; rank++)); do
  echo "table4_stage=renderer_and_restoration_check rank=$rank cuda=${GPUS[$rank]} egl=$WARM_TABLE4_EGL_DEVICE_ID"
  if CUDA_VISIBLE_DEVICES="${GPUS[$rank]}" "$PY" "$CODE/scripts/probe_libero_table4_environment.py" \
    --require-cuda --qualify-branches --output "$OUTPUT/environment-$rank" \
    > "$OUTPUT/environment-$rank.console.log" 2>&1; then
    echo "table4_environment=passed rank=$rank"
  else
    tail -n 60 "$OUTPUT/environment-$rank.console.log" >&2
    exit 2
  fi
done

PIDS=()
SHARDS=("${REUSE[@]}")
stop_children() {
  trap - INT TERM
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
  exit 130
}
trap stop_children INT TERM
for ((rank=0; rank<N; rank++)); do
  EPISODES="${EPISODE_GROUPS[$rank]}"
  SHARD="$OUTPUT/shard-$rank"
  SHARDS+=("$SHARD")
  CUDA_VISIBLE_DEVICES="${GPUS[$rank]}" "$PY" "$CODE/scripts/collect_libero_table4.py" \
    --legacy-code "$LEGACY" --project "$PROJECT" --eval-root "$EVAL_ROOT" \
    --output "$SHARD" --episodes "$EPISODES" --cohort "libero10-table4-s3407-v1" \
    --expected-checkpoint-sha256 "$EXPECTED_SHA" \
    --max-hours "${WARM_TABLE4_MAX_HOURS:-4}" > "$OUTPUT/shard-$rank.console.log" 2>&1 &
  PIDS+=("$!")
  echo "table4_worker rank=$rank gpu=${GPUS[$rank]} episodes=$EPISODES pid=$! log=$OUTPUT/shard-$rank.console.log"
done
while true; do
  LIVE=0
  for pid in "${PIDS[@]}"; do if kill -0 "$pid" 2>/dev/null; then LIVE=$((LIVE+1)); fi; done
  if ((LIVE == 0)); then break; fi
  echo "table4_heartbeat time=$(date -u +%FT%TZ) active_workers=$LIVE"
  for ((rank=0; rank<N; rank++)); do
    printf 'worker=%s last_log=' "$rank"
    tail -n 1 "$OUTPUT/shard-$rank.console.log" || true
  done
  sleep 30
done
FAILED=0
for pid in "${PIDS[@]}"; do
  if wait "$pid"; then :; else FAILED=1; fi
done
if ((FAILED)); then
  echo "ERROR: at least one shard failed; evidence preserved at $OUTPUT" >&2
  exit 2
fi
"$PY" "$CODE/scripts/merge_libero_table4.py" --shards "${SHARDS[@]}" --output "$OUTPUT/report"
echo "TABLE4_COMPLETE=$OUTPUT/report/table4_rows.tex"
