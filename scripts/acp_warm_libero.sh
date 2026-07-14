#!/bin/bash
# ACP entrypoint for WARM LIBERO experiments.
#
# 提交为 ACP 任务的 command script。整体流程分四种 RUN_KIND，按顺序推进：
#   1. download_dino      下载并 pin DINOv2-base 快照（一次性，需要联网）
#   2. prepare_artifacts  构建 M1/M2 不可变产物链（catalog/audit/统计/DINO+VAE
#                         特征/H=32 事件库/oracle 门/stride-1 候选缓存/contract）
#                         单卡 GPU 即可，跑一次即可，产物不可覆盖
#   3. oracle_check       打印 oracle 门报告（top-32 oracle 动作距离需比
#                         context top-1 低约 15-20% 才继续训练，否则先修检索）
#   4. train              正式训练（多卡，zero1/zero2 自动选择）
#
# 注意：prepare_artifacts 与 train 都要求 git 工作树干净（仓库脚本内部也会
# 强制检查），提交本脚本与其他改动后再启动正式任务。
#
# ============================================================================
# 消融实验参数配置总览（配合 TASK_NAME / SOURCE_POLICY 使用）
# ============================================================================
# 所有消融共享同一条 M1/M2 产物链（同一事件库/候选缓存/contract/基线权重），
# 只改 task 与 source policy，保证初始化、数据顺序、高斯采样、产物身份配对：
#
#   (a) FastWAM 基线（无记忆，无 WARM 模块）：
#         TASK_NAME=libero_uncond_2cam224_1e-4
#         SOURCE_POLICY 忽略；不注入任何 warm_candidates/contract 覆盖
#   (b) M2 source-only 高斯零假设（Gaussian null，禁止读库/DINO）：
#         TASK_NAME=libero_warm_source_2cam224_1e-4  SOURCE_POLICY=gaussian_null
#   (c) M2 source-only 检索 source（context top-1 作为流匹配 source）：
#         TASK_NAME=libero_warm_source_2cam224_1e-4  SOURCE_POLICY=fixed_context_top1
#   (d) M2 训练期上界诊断（禁止用于推理/rollout）：
#         TASK_NAME=libero_warm_source_2cam224_1e-4  SOURCE_POLICY=oracle_action_top1
#   (e) 完整 WARM（后果对齐重排 + 连续 gate + 事件适配器 + 语义桥）：
#         TASK_NAME=libero_warm_2cam224_1e-4         SOURCE_POLICY=fixed_context_top1
#
# 正式 M2.1 fixed/null 对比要求 (b)(c) 用同一 recipe 分别训练出两个带
# .training.json attestation 的 checkpoint，详见 docs/M2_ONLINE_RETRIEVAL.md。
# ============================================================================

set +e

# ==========================================
# 0. User-editable ACP config
# ==========================================

# TODO: WARM 项目目录（ACP 文件系统上的绝对路径）
PROJECT_DIR="${PROJECT_DIR:-/mnt/afs/task3_2/L202500276_lwz/projects/WARM}"

# TODO: conda 环境目录（path-based env，脚本会优先 conda activate，失败则注入 PATH）
CONDA_ENV_DIR="${CONDA_ENV_DIR:-/mnt/afs/task3_2/L202500276_lwz/envs/warm}"

# TODO: Wan/FastWAM/ActionDiT 权重根目录（configs 默认 ./checkpoints，保持一致即可）
DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${PROJECT_DIR}/checkpoints}"

# TODO: M1/M2 不可变产物链根目录。prepare_artifacts 会在其下创建 m1/ 与 m2/，
# 产物拒绝覆盖：如需重建，换一个全新路径（例如 libero_v2）。
WARM_ARTIFACT_ROOT="${WARM_ARTIFACT_ROOT:-/mnt/afs/task3_2/L202500276_lwz/projects/WARM_artifacts/libero_v1}"

# TODO: LIBERO LeRobot 数据根目录（其下须含四个 *_no_noops_lerobot 子目录）
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-${PROJECT_DIR}/data/libero_mujoco3.3.2}"

# TODO: FastWAM 基线 checkpoint（WARM 的 base_checkpoint_path，contract 会绑定其 SHA-256）
FASTWAM_BASE_CHECKPOINT="${FASTWAM_BASE_CHECKPOINT:-${PROJECT_DIR}/checkpoints/fastwam_release/libero_uncond_2cam224.pt}"

# TODO: DINOv2-base 本地快照目录（由 RUN_KIND=download_dino 下载生成）
WARM_DINO_CHECKPOINT="${WARM_DINO_CHECKPOINT:-${PROJECT_DIR}/checkpoints/dinov2-base}"

# TODO: DINO 的 40 位 Hub commit SHA。留空则自动读取 download_dino 写出的
# marker 文件 ${WARM_DINO_CHECKPOINT}.revision.json。
WARM_DINO_REVISION="${WARM_DINO_REVISION:-}"

# TODO: Wan2.2 VAE 单文件权重（不是目录）
WARM_VAE_CHECKPOINT="${WARM_VAE_CHECKPOINT:-${PROJECT_DIR}/checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors}"

# TODO: 持久化缓存目录（HF/torch/triton 等）
CACHE_ROOT="${CACHE_ROOT:-${PROJECT_DIR}/cache}"

# TODO: 若 ACP 提供节点本地 SSD/NVMe，优先改成本地路径，避免 AFS 上的
# HF datasets 文件锁在分布式启动时报 FileNotFoundError。
LOCAL_CACHE_ROOT="${LOCAL_CACHE_ROOT:-/tmp/${USER:-warm}/warm_cache}"

# TODO: 持久化日志根目录
LOG_ROOT="${LOG_ROOT:-${PROJECT_DIR}/tmp/acp_logs}"

# TODO: 本次 ACP 任务要做什么，见文件头说明。
# 可选值: download_dino | prepare_artifacts | oracle_check | train
RUN_KIND="${RUN_KIND:-train}"

# TODO: 训练任务名（configs/task/*.yaml 的文件名去掉 .yaml），消融组合见文件头。
# 可选值:
#   libero_uncond_2cam224_1e-4        FastWAM 基线 (a)
#   libero_warm_source_2cam224_1e-4   M2 source-only (b)(c)(d)
#   libero_warm_2cam224_1e-4          完整 WARM (e)
TASK_NAME="${TASK_NAME:-libero_warm_2cam224_1e-4}"

# TODO: source 策略，仅对 warm/warm_source 任务生效，消融组合见文件头。
# 可选值: fixed_context_top1 | gaussian_null | oracle_action_top1
SOURCE_POLICY="${SOURCE_POLICY:-fixed_context_top1}"

# TODO: 按 ACP 分配的 GPU 资源修改。NPROC_PER_NODE 必须等于可见 GPU 数。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

# TODO: DeepSpeed stage。auto: >=8 卡用 ZeRO1（对齐上游 FastWAM 论文配方），
# <8 卡用 ZeRO2（优化器/梯度分片更激进，更省显存）。也可强制填 1 或 2。
ZERO_STAGE="${ZERO_STAGE:-auto}"

# TODO: 显存/速度旋钮。batch_size 是每卡值，不是全局值。
# "task" 表示沿用 configs/task/*.yaml（warm 任务默认 batch_size=16, accum=1，
# 这是 8 卡 H100/A100 量级的配方）。
# 显存吃紧的参考组合（完整 WARM 含 33 帧 video pass，比 M2 更耗显存）：
#   8 卡:  PER_DEVICE_BATCH_SIZE=task GRADIENT_ACCUMULATION_STEPS=task
#   4 卡:  TARGET_GLOBAL_BATCH_SIZE=128 PER_DEVICE_BATCH_SIZE=8  GRADIENT_ACCUMULATION_STEPS=auto
#   更省:  PER_DEVICE_BATCH_SIZE=4 GRADIENT_ACCUMULATION_STEPS=auto
TARGET_GLOBAL_BATCH_SIZE="${TARGET_GLOBAL_BATCH_SIZE:-128}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-task}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-task}"

# TODO: MoT 混合注意力激活重算。"task" 沿用任务配置
# （完整 WARM 默认 true 且必须开启；M2/基线默认 false，OOM 时可设 true）。
MOT_CHECKPOINT_MIXED_ATTN="${MOT_CHECKPOINT_MIXED_ATTN:-task}"

# TODO: 运行时长与断点。长 ACP 任务建议设 MAX_STEPS 保证 walltime 内干净退出，
# 下一段用 RESUME 指向 runs/.../checkpoints/state/step_NNNNNN 续跑。
# 留 null 则沿用任务配置（warm 任务 num_epochs=10）。
MAX_STEPS="${MAX_STEPS:-null}"
RESUME="${RESUME:-null}"
NUM_EPOCHS="${NUM_EPOCHS:-null}"
LOG_EVERY="${LOG_EVERY:-null}"
SAVE_EVERY="${SAVE_EVERY:-null}"
EVAL_EVERY="${EVAL_EVERY:-null}"

# TODO: 正式实验保持 true（产物链与正式训练要求干净 git 工作树，且仓库内部
# 脚本会再次强制检查）。仅调试冒烟时可设 false。
REQUIRE_CLEAN_GIT="${REQUIRE_CLEAN_GIT:-true}"

# wandb（默认 offline，正式配方不联网记日志）
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-WARM}"
export WANDB_MODE="${WANDB_MODE:-offline}"

# HF offline。download_dino 会强制置 0，其余任务默认离线。
HF_OFFLINE="${HF_OFFLINE:-1}"

# TODO: HF 端点。集群通常无法直连 huggingface.co，默认走 hf-mirror.com 镜像；
# 仅 download_dino 使用（镜像与官方 hub 的 commit SHA 一致）。
HF_DOWNLOAD_ENDPOINT="${HF_DOWNLOAD_ENDPOINT:-https://hf-mirror.com}"

# 特征预计算旋钮（仅 prepare_artifacts 使用）
WARM_PRECOMPUTE_DEVICE="${WARM_PRECOMPUTE_DEVICE:-cuda}"
WARM_PRECOMPUTE_DTYPE="${WARM_PRECOMPUTE_DTYPE:-bfloat16}"
WARM_DINO_BATCH_SIZE="${WARM_DINO_BATCH_SIZE:-64}"
WARM_VAE_BATCH_SIZE="${WARM_VAE_BATCH_SIZE:-4}"

# 训练前是否先做 Hydra 全量解析预检（写 resolved_config.preflight.yaml）
PREFLIGHT_RESOLVE="${PREFLIGHT_RESOLVE:-true}"

# 日志扫描到可疑错误时是否强制失败退出码
STRICT_LOG_SCAN="${STRICT_LOG_SCAN:-false}"

# 追加的 Hydra 覆盖，空格分隔的 key=value（例如 "seed=17 model.memory_sigma=0.2"）
HYDRA_EXTRA_ARGS="${HYDRA_EXTRA_ARGS:-}"

# ==========================================
# 1. Helpers
# ==========================================

timestamp() {
  date +"%Y%m%d_%H%M%S"
}

require_no_todo() {
  local name="$1"
  local value="$2"
  if [[ "${value}" == *"TODO"* ]]; then
    echo "ERROR: ${name} still contains TODO placeholder: ${value}"
    return 1
  fi
  return 0
}

is_positive_integer() {
  [[ "${1}" =~ ^[1-9][0-9]*$ ]]
}

ceil_div() {
  echo $(( ($1 + $2 - 1) / $2 ))
}

validate_optional_positive_integer() {
  local name="$1" value="$2"
  if [[ -z "${value}" || "${value}" == "null" || "${value}" == "none" ]]; then
    return 0
  fi
  if ! is_positive_integer "${value}"; then
    echo "ERROR: ${name} must be a positive integer, null, or none. Got ${value}"
    return 1
  fi
}

validate_optional_nonnegative_integer() {
  local name="$1" value="$2"
  if [[ -z "${value}" || "${value}" == "null" || "${value}" == "none" ]]; then
    return 0
  fi
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: ${name} must be a non-negative integer, null, or none. Got ${value}"
    return 1
  fi
}

count_visible_devices() {
  local csv="${1// /}"
  if [[ -z "${csv}" || "${csv}" == "all" ]]; then
    echo "0"
    return 0
  fi
  local IFS=','
  read -r -a devices <<< "${csv}"
  echo "${#devices[@]}"
}

warn_if_gpu_request_mismatch() {
  if ! is_positive_integer "${NPROC_PER_NODE}"; then
    echo "ERROR: NPROC_PER_NODE must be a positive integer, got ${NPROC_PER_NODE}"
    return 1
  fi
  local visible_count
  visible_count="$(count_visible_devices "${CUDA_VISIBLE_DEVICES:-}")"
  if (( visible_count > 0 && NPROC_PER_NODE > visible_count )); then
    echo "ERROR: NPROC_PER_NODE=${NPROC_PER_NODE} exceeds CUDA_VISIBLE_DEVICES count=${visible_count} (${CUDA_VISIBLE_DEVICES})."
    return 1
  fi
  if (( visible_count > 0 && NPROC_PER_NODE != visible_count )); then
    echo "WARNING: NPROC_PER_NODE=${NPROC_PER_NODE} but CUDA_VISIBLE_DEVICES has ${visible_count} entries."
  fi
}

task_kind() {
  case "${TASK_NAME}" in
    libero_warm_source_*) echo "warm_source" ;;
    libero_warm_online*)  echo "invalid" ;;
    libero_warm_*)        echo "warm_full" ;;
    libero_uncond_*|libero_joint_*|libero_idm_*) echo "fastwam" ;;
    *) echo "unknown" ;;
  esac
}

resolve_training_batching() {
  if [[ "${PER_DEVICE_BATCH_SIZE}" == "task" ]]; then
    RESOLVED_BATCH_SIZE="task"
  else
    if ! is_positive_integer "${PER_DEVICE_BATCH_SIZE}"; then
      echo "ERROR: PER_DEVICE_BATCH_SIZE must be a positive integer or task, got ${PER_DEVICE_BATCH_SIZE}"
      return 1
    fi
    RESOLVED_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE}"
  fi

  if [[ "${GRADIENT_ACCUMULATION_STEPS}" == "task" ]]; then
    RESOLVED_GRADIENT_ACCUMULATION_STEPS="task"
  elif [[ "${GRADIENT_ACCUMULATION_STEPS}" == "auto" ]]; then
    if [[ "${RESOLVED_BATCH_SIZE}" == "task" ]]; then
      echo "ERROR: GRADIENT_ACCUMULATION_STEPS=auto requires an explicit PER_DEVICE_BATCH_SIZE."
      return 1
    fi
    if ! is_positive_integer "${TARGET_GLOBAL_BATCH_SIZE}"; then
      echo "ERROR: TARGET_GLOBAL_BATCH_SIZE must be a positive integer, got ${TARGET_GLOBAL_BATCH_SIZE}"
      return 1
    fi
    local micro_global=$(( RESOLVED_BATCH_SIZE * NPROC_PER_NODE ))
    RESOLVED_GRADIENT_ACCUMULATION_STEPS="$(ceil_div "${TARGET_GLOBAL_BATCH_SIZE}" "${micro_global}")"
  else
    if ! is_positive_integer "${GRADIENT_ACCUMULATION_STEPS}"; then
      echo "ERROR: GRADIENT_ACCUMULATION_STEPS must be a positive integer, task, or auto, got ${GRADIENT_ACCUMULATION_STEPS}"
      return 1
    fi
    RESOLVED_GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS}"
  fi

  if [[ "${RESOLVED_BATCH_SIZE}" == "task" || "${RESOLVED_GRADIENT_ACCUMULATION_STEPS}" == "task" ]]; then
    RESOLVED_EFFECTIVE_GLOBAL_BATCH_SIZE="task-config"
  else
    RESOLVED_EFFECTIVE_GLOBAL_BATCH_SIZE=$(( RESOLVED_BATCH_SIZE * NPROC_PER_NODE * RESOLVED_GRADIENT_ACCUMULATION_STEPS ))
  fi
}

resolve_zero_stage() {
  case "${ZERO_STAGE}" in
    1|2)
      RESOLVED_ZERO_STAGE="${ZERO_STAGE}"
      ;;
    auto)
      if (( NPROC_PER_NODE >= 8 )); then
        RESOLVED_ZERO_STAGE="1"
      else
        RESOLVED_ZERO_STAGE="2"
      fi
      ;;
    *)
      echo "ERROR: Unsupported ZERO_STAGE=${ZERO_STAGE}. Expected auto, 1, or 2."
      return 1
      ;;
  esac
  # WARM 的 train_zero{1,2}.sh 各自硬编码对应 accelerate 配置文件
  if [[ "${RESOLVED_ZERO_STAGE}" == "1" ]]; then
    TRAIN_SCRIPT="scripts/train_zero1.sh"
  else
    TRAIN_SCRIPT="scripts/train_zero2.sh"
  fi
}

append_optional_override() {
  local key="$1" value="$2"
  if [[ -n "${value}" && "${value}" != "null" && "${value}" != "none" && "${value}" != "task" ]]; then
    CMD+=("${key}=${value}")
  fi
}

validate_lerobot_dir() {
  local dir="$1"
  if [[ ! -f "${dir}/meta/tasks.jsonl" || ! -f "${dir}/meta/info.json" ]]; then
    echo "ERROR: Not a LeRobot dataset directory: ${dir}"
    return 1
  fi
}

require_clean_git() {
  if [[ "${REQUIRE_CLEAN_GIT}" != "true" ]]; then
    echo "WARNING: REQUIRE_CLEAN_GIT=false; this run is not admissible as formal evidence."
    return 0
  fi
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "ERROR: dirty git worktree. Commit changes first (formal artifact/training runs require a clean checkout)."
    git status --short | head -20
    return 1
  fi
}

resolve_dino_revision() {
  if [[ -n "${WARM_DINO_REVISION}" ]]; then
    return 0
  fi
  local marker="${WARM_DINO_CHECKPOINT}.revision.json"
  if [[ ! -f "${marker}" ]]; then
    echo "ERROR: WARM_DINO_REVISION is empty and marker not found: ${marker}"
    echo "Run this script once with RUN_KIND=download_dino first."
    return 1
  fi
  WARM_DINO_REVISION="$(python - "${marker}" <<'PY'
import json, sys
print(json.loads(open(sys.argv[1], encoding="utf-8").read())["revision"])
PY
)"
  if [[ ! "${WARM_DINO_REVISION}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: marker revision is not a 40-character commit SHA: ${WARM_DINO_REVISION}"
    return 1
  fi
}

run_and_log() {
  echo "========== COMMAND ARGV =========="
  local index=0
  for arg in "$@"; do
    printf '  [%03d] %q\n' "${index}" "${arg}"
    index=$((index + 1))
  done
  echo "=================================="
  "$@" 2>&1 | tee "${LOG_FILE}"
  return "${PIPESTATUS[0]}"
}

scan_log_for_errors() {
  [[ -f "${LOG_FILE}" ]] || return 1
  grep -qE \
    "Traceback|RuntimeError|ImportError|ModuleNotFoundError|CUDA out of memory|illegal memory access|NCCL error|Error executing job" \
    "${LOG_FILE}"
}

handle_termination() {
  echo "!! ACP SCRIPT TERMINATED: received $1 (scheduler walltime/preemption/manual stop)."
  exit 143
}

# ==========================================
# 2. Validate config
# ==========================================

require_no_todo PROJECT_DIR "${PROJECT_DIR}" || exit 2
require_no_todo CONDA_ENV_DIR "${CONDA_ENV_DIR}" || exit 2
require_no_todo WARM_ARTIFACT_ROOT "${WARM_ARTIFACT_ROOT}" || exit 2
require_no_todo LIBERO_DATA_ROOT "${LIBERO_DATA_ROOT}" || exit 2
require_no_todo FASTWAM_BASE_CHECKPOINT "${FASTWAM_BASE_CHECKPOINT}" || exit 2
require_no_todo LOG_ROOT "${LOG_ROOT}" || exit 2

validate_optional_positive_integer MAX_STEPS "${MAX_STEPS}" || exit 2
validate_optional_positive_integer NUM_EPOCHS "${NUM_EPOCHS}" || exit 2
validate_optional_nonnegative_integer LOG_EVERY "${LOG_EVERY}" || exit 2
validate_optional_nonnegative_integer SAVE_EVERY "${SAVE_EVERY}" || exit 2
validate_optional_nonnegative_integer EVAL_EVERY "${EVAL_EVERY}" || exit 2
warn_if_gpu_request_mismatch || exit 2

if [[ ! -d "${PROJECT_DIR}" ]]; then
  echo "ERROR: PROJECT_DIR does not exist: ${PROJECT_DIR}"
  exit 2
fi
if [[ ! -x "${CONDA_ENV_DIR}/bin/python" ]]; then
  echo "ERROR: Cannot find python in CONDA_ENV_DIR: ${CONDA_ENV_DIR}/bin/python"
  exit 2
fi

# ==========================================
# 3. Environment setup
# ==========================================

echo "========== INIT ENVIRONMENT =========="
cd "${PROJECT_DIR}" || exit 2

RUN_STAMP="$(timestamp)"
RUN_LABEL="${RUN_KIND}_${TASK_NAME}_${SOURCE_POLICY}_${RUN_STAMP}"
LOG_DIR="${LOG_ROOT}/${RUN_LABEL}"
LOG_FILE="${LOG_DIR}/console.log"
JOB_LOCAL_CACHE_ROOT="${LOCAL_CACHE_ROOT}/${RUN_LABEL}"
mkdir -p "${LOG_DIR}" "${JOB_LOCAL_CACHE_ROOT}/hf_datasets" "${JOB_LOCAL_CACHE_ROOT}/triton"

trap 'handle_termination SIGTERM' TERM
trap 'handle_termination SIGINT' INT

# conda activate（优先），失败则退化为 PATH 注入
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null
  conda activate "${CONDA_ENV_DIR}" 2>/dev/null \
    || echo "WARNING: conda activate failed; falling back to PATH injection."
fi
export PATH="${CONDA_ENV_DIR}/bin:${PATH}"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

export DIFFSYNTH_MODEL_BASE_PATH
export HF_HOME="${CACHE_ROOT}/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${JOB_LOCAL_CACHE_ROOT}/hf_datasets"
export TORCH_HOME="${CACHE_ROOT}/torch"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TRITON_CACHE_DIR="${JOB_LOCAL_CACHE_ROOT}/triton"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "${RUN_KIND}" == "download_dino" ]]; then
  HF_OFFLINE=0
fi
export HF_DATASETS_OFFLINE="${HF_OFFLINE}"
export TRANSFORMERS_OFFLINE="${HF_OFFLINE}"
export HF_HUB_OFFLINE="${HF_OFFLINE}"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo,eth0,bond0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

echo "PROJECT_DIR=${PROJECT_DIR}"
echo "CONDA python=$(command -v python)"
echo "RUN_KIND=${RUN_KIND}"
echo "TASK_NAME=${TASK_NAME}"
echo "SOURCE_POLICY=${SOURCE_POLICY}"
echo "WARM_ARTIFACT_ROOT=${WARM_ARTIFACT_ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "LOG_DIR=${LOG_DIR}"
echo "GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo unknown)"

# ==========================================
# 4. Build command per RUN_KIND
# ==========================================

M1="${WARM_ARTIFACT_ROOT}/m1"
M2="${WARM_ARTIFACT_ROOT}/m2"
EVENT_BANK="${M1}/banks/hybrid_h32"
TRAIN_CACHE="${M2}/candidates/hybrid_h32_train_k32"
DEV_CACHE="${M2}/candidates/hybrid_h32_dev_k32"
TRAIN_CONTRACT="${M2}/contracts/hybrid_h32_train_source.json"
DEV_CONTRACT="${M2}/contracts/hybrid_h32_dev_source.json"
TRAIN_STATS="${M1}/train_stats/dataset_stats.json"

EXTRA_ARGS=()
if [[ -n "${HYDRA_EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=(${HYDRA_EXTRA_ARGS})
fi

CMD=()
case "${RUN_KIND}" in

  # ------------------------------------------------------------------
  # 一次性：下载并 pin DINOv2-base 快照，写 revision marker
  # ------------------------------------------------------------------
  download_dino)
    export HF_ENDPOINT="${HF_DOWNLOAD_ENDPOINT}"
    export WARM_DINO_CHECKPOINT WARM_DINO_REVISION
    DOWNLOAD_SCRIPT="${LOG_DIR}/download_dino.py"
    cat > "${DOWNLOAD_SCRIPT}" <<'PY'
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

repo_id = "facebook/dinov2-base"
target = Path(os.environ["WARM_DINO_CHECKPOINT"])
requested = os.environ.get("WARM_DINO_REVISION", "").strip() or "main"

api = HfApi()
info = api.model_info(repo_id, revision=requested)
commit = info.sha
if not (isinstance(commit, str) and len(commit) == 40):
    raise SystemExit(f"cannot resolve a 40-character commit for {repo_id}@{requested}: {commit!r}")

if target.exists() and any(target.iterdir()):
    raise SystemExit(f"target already exists and is not empty: {target}")

snapshot_download(repo_id=repo_id, revision=commit, local_dir=str(target))

marker = target.parent / (target.name + ".revision.json")
marker.write_text(
    json.dumps({"repo_id": repo_id, "revision": commit}, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(f"pinned {repo_id} at revision {commit}")
print(f"snapshot: {target}")
print(f"marker:   {marker}")
PY
    CMD=(python "${DOWNLOAD_SCRIPT}")
    ;;

  # ------------------------------------------------------------------
  # 一次性：构建 M1/M2 不可变产物链（catalog -> audit -> train-only 统计
  # -> DINO+VAE 特征 -> H=32 事件库 -> oracle 门 -> stride-1 候选缓存
  # -> train/dev source contract）。单卡 GPU 足够。
  # ------------------------------------------------------------------
  prepare_artifacts)
    require_clean_git || exit 2
    resolve_dino_revision || exit 2
    for dataset in libero_spatial libero_object libero_goal libero_10; do
      validate_lerobot_dir "${LIBERO_DATA_ROOT}/${dataset}_no_noops_lerobot" || exit 2
    done
    for path in "${FASTWAM_BASE_CHECKPOINT}" "${WARM_VAE_CHECKPOINT}"; do
      if [[ ! -f "${path}" ]]; then
        echo "ERROR: required checkpoint not found: ${path}"
        exit 2
      fi
    done
    if [[ ! -e "${WARM_DINO_CHECKPOINT}" ]]; then
      echo "ERROR: DINO snapshot not found: ${WARM_DINO_CHECKPOINT} (run download_dino first)"
      exit 2
    fi
    export WARM_ARTIFACT_ROOT LIBERO_DATA_ROOT FASTWAM_BASE_CHECKPOINT
    export WARM_DINO_CHECKPOINT WARM_DINO_REVISION WARM_VAE_CHECKPOINT
    export WARM_PRECOMPUTE_DEVICE WARM_PRECOMPUTE_DTYPE
    export WARM_DINO_BATCH_SIZE WARM_VAE_BATCH_SIZE
    CMD=(bash scripts/prepare_warm_full_artifacts.sh)
    ;;

  # ------------------------------------------------------------------
  # 决策辅助：打印 oracle 门报告。top-32 oracle 动作距离应比 context top-1
  # 低约 15-20%，未达标先修检索表示/切分，不要继续烧 5B 训练时间。
  # ------------------------------------------------------------------
  oracle_check)
    if [[ ! -f "${M1}/oracle/hybrid_h32.json" ]]; then
      echo "ERROR: oracle report not found: ${M1}/oracle/hybrid_h32.json (run prepare_artifacts first)"
      exit 2
    fi
    CMD=(python -m json.tool "${M1}/oracle/hybrid_h32.json")
    ;;

  # ------------------------------------------------------------------
  # 训练。按 TASK_NAME 自动决定注入哪些 WARM 产物覆盖，见文件头消融说明。
  # ------------------------------------------------------------------
  train)
    KIND="$(task_kind)"
    if [[ "${KIND}" == "invalid" || "${KIND}" == "unknown" ]]; then
      echo "ERROR: TASK_NAME=${TASK_NAME} is not a trainable LIBERO task for this script."
      exit 2
    fi
    require_clean_git || exit 2
    resolve_training_batching || exit 2
    resolve_zero_stage || exit 2

    WARM_OVERRIDES=()
    if [[ "${KIND}" == "warm_source" || "${KIND}" == "warm_full" ]]; then
      for path in "${EVENT_BANK}" "${TRAIN_CACHE}" "${DEV_CACHE}" \
                  "${TRAIN_CONTRACT}" "${DEV_CONTRACT}" "${TRAIN_STATS}" \
                  "${M1}/libero_catalog.json" "${M1}/libero_audit.json"; do
        if [[ ! -e "${path}" ]]; then
          echo "ERROR: required immutable artifact not found: ${path} (run prepare_artifacts first)"
          exit 2
        fi
      done
      if [[ ! -f "${FASTWAM_BASE_CHECKPOINT}" ]]; then
        echo "ERROR: base checkpoint not found: ${FASTWAM_BASE_CHECKPOINT}"
        exit 2
      fi
      WARM_OVERRIDES+=(
        "model.source_policy=${SOURCE_POLICY}"
        "model.run_contract_path=${TRAIN_CONTRACT}"
        "model.validation_run_contract_path=${DEV_CONTRACT}"
        "model.base_checkpoint_path=${FASTWAM_BASE_CHECKPOINT}"
        "data.warm_candidates.train.bank_directory=${EVENT_BANK}"
        "data.warm_candidates.train.candidate_directory=${TRAIN_CACHE}"
        "data.warm_candidates.train.catalog_path=${M1}/libero_catalog.json"
        "data.warm_candidates.train.normalization_stats_path=${TRAIN_STATS}"
        "data.warm_candidates.train.audit_report_path=${M1}/libero_audit.json"
        "data.warm_candidates.val.bank_directory=${EVENT_BANK}"
        "data.warm_candidates.val.candidate_directory=${DEV_CACHE}"
        "data.warm_candidates.val.catalog_path=${M1}/libero_catalog.json"
        "data.warm_candidates.val.normalization_stats_path=${TRAIN_STATS}"
        "data.warm_candidates.val.audit_report_path=${M1}/libero_audit.json"
      )
    fi
    if [[ "${KIND}" == "warm_full" ]]; then
      for path in "${M1}/features/train_features.list" "${M1}/features/dev_features.list"; do
        if [[ ! -e "${path}" ]]; then
          echo "ERROR: required feature list not found: ${path}"
          exit 2
        fi
      done
      WARM_OVERRIDES+=(
        "data.warm_candidates.train.retrospective_feature_list=${M1}/features/train_features.list"
        "data.warm_candidates.val.retrospective_feature_list=${M1}/features/dev_features.list"
      )
    fi

    # Hydra 全量解析预检：失败则不启动多卡任务
    if [[ "${PREFLIGHT_RESOLVE}" == "true" ]]; then
      echo "========== PREFLIGHT RESOLVE =========="
      python scripts/train.py "task=${TASK_NAME}" \
        "${WARM_OVERRIDES[@]}" "${EXTRA_ARGS[@]}" \
        --cfg job --resolve > "${LOG_DIR}/resolved_config.preflight.yaml"
      PREFLIGHT_CODE=$?
      if [[ "${PREFLIGHT_CODE}" -ne 0 ]]; then
        echo "ERROR: Hydra preflight resolve failed (exit ${PREFLIGHT_CODE}); aborting before launch."
        exit "${PREFLIGHT_CODE}"
      fi
      echo "Preflight resolved config saved to ${LOG_DIR}/resolved_config.preflight.yaml"
    fi

    echo "========== TRAINING BATCHING =========="
    echo "ZERO_STAGE=${RESOLVED_ZERO_STAGE} (requested ${ZERO_STAGE})"
    echo "TRAIN_SCRIPT=${TRAIN_SCRIPT}"
    echo "PER_DEVICE_BATCH_SIZE=${RESOLVED_BATCH_SIZE}"
    echo "GRADIENT_ACCUMULATION_STEPS=${RESOLVED_GRADIENT_ACCUMULATION_STEPS}"
    echo "EFFECTIVE_GLOBAL_BATCH_SIZE=${RESOLVED_EFFECTIVE_GLOBAL_BATCH_SIZE}"

    export RUN_ID="${RUN_ID:-${RUN_STAMP}_${SOURCE_POLICY}}"
    CMD=(
      bash "${TRAIN_SCRIPT}" "${NPROC_PER_NODE}"
      "task=${TASK_NAME}"
      "wandb.enabled=${WANDB_ENABLED}"
      "wandb.project=${WANDB_PROJECT}"
      "wandb.mode=${WANDB_MODE}"
      "${WARM_OVERRIDES[@]}"
    )
    append_optional_override "batch_size" "${RESOLVED_BATCH_SIZE}"
    append_optional_override "gradient_accumulation_steps" "${RESOLVED_GRADIENT_ACCUMULATION_STEPS}"
    append_optional_override "model.mot_checkpoint_mixed_attn" "${MOT_CHECKPOINT_MIXED_ATTN}"
    append_optional_override "max_steps" "${MAX_STEPS}"
    append_optional_override "resume" "${RESUME}"
    append_optional_override "num_epochs" "${NUM_EPOCHS}"
    append_optional_override "log_every" "${LOG_EVERY}"
    append_optional_override "save_every" "${SAVE_EVERY}"
    append_optional_override "eval_every" "${EVAL_EVERY}"
    CMD+=("${EXTRA_ARGS[@]}")
    ;;

  *)
    echo "ERROR: Unsupported RUN_KIND=${RUN_KIND}"
    echo "Expected one of: download_dino, prepare_artifacts, oracle_check, train"
    exit 2
    ;;
esac

# ==========================================
# 5. Run and summarize
# ==========================================

echo "========== START JOB =========="
run_and_log "${CMD[@]}"
EXIT_CODE=$?

if scan_log_for_errors; then
  if [[ "${EXIT_CODE}" -ne 0 || "${STRICT_LOG_SCAN}" == "true" ]]; then
    echo "!! ERROR DETECTED: serious failure pattern found in ${LOG_FILE}; forcing EXIT_CODE=1"
    EXIT_CODE=1
  else
    echo "WARNING: suspicious failure pattern found in ${LOG_FILE}, but command exited 0; keeping EXIT_CODE=0."
  fi
fi

echo "========== JOB FINISHED =========="
echo "EXIT_CODE=${EXIT_CODE}"
echo "LOG_FILE=${LOG_FILE}"

exit "${EXIT_CODE}"
