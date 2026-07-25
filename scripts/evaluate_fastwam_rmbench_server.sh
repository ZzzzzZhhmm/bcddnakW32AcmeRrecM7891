#!/usr/bin/env bash
set -euo pipefail

# Evaluate the same-data no-memory FastWAM baseline with the pinned official
# RMBench evaluator.  This comparison never loads a WARM bank or contract.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=scripts/warm_server_common.sh
source "${SCRIPT_DIR}/warm_server_common.sh"

RMBENCH_CODE_REVISION="${RMBENCH_CODE_REVISION:-57ee09cbc6267bc36ca0ac2d8d1c5c3b245c112c}"
RMBENCH_DATASET_REVISION="${RMBENCH_DATASET_REVISION:-855e90e1213d150bf4889130e83398f107314681}"
warm_require_env \
  WARM_ARTIFACT_ROOT \
  RMBENCH_ROOT \
  RMBENCH_HF_REVISION_MARKER \
  FASTWAM_RMBENCH_CHECKPOINT \
  WARM_RMBENCH_ONLINE_CONTRACT \
  FASTWAM_RMBENCH_EVAL_ROOT
warm_require_sha40 "${RMBENCH_CODE_REVISION}" "RMBENCH_CODE_REVISION"
warm_require_sha40 "${RMBENCH_DATASET_REVISION}" "RMBENCH_DATASET_REVISION"
warm_register_safe_directory "${PROJECT_ROOT}"
warm_require_private_checkout
warm_require_read_only_external_checkout "${RMBENCH_ROOT}" "${RMBENCH_CODE_REVISION}"
warm_configure_offline_logging
warm_refuse_existing_output "${FASTWAM_RMBENCH_EVAL_ROOT}"

TRAIN_STATS="${WARM_ARTIFACT_ROOT}/m1/train_stats/dataset_stats.json"
warm_require_file_or_directory \
  "${RMBENCH_HF_REVISION_MARKER}" \
  "${FASTWAM_RMBENCH_CHECKPOINT}" \
  "${TRAIN_STATS}" \
  "${WARM_RMBENCH_ONLINE_CONTRACT}/seeds"

SUITE="${WARM_RMBENCH_SUITE:-official9}"
case "${SUITE}" in
  official9|pilot3) ;;
  *) warm_die "WARM_RMBENCH_SUITE must be official9 or pilot3" ;;
esac

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}src"
python -c \
  'import sys; from fastwam.benchmarks.rmbench import validate_hf_revision_marker; validate_hf_revision_marker(sys.argv[1], expected_revision=sys.argv[2])' \
  "${RMBENCH_HF_REVISION_MARKER}" "${RMBENCH_DATASET_REVISION}"

exec python experiments/rmbench/run_rmbench_manager.py \
  "ckpt=${FASTWAM_RMBENCH_CHECKPOINT}" \
  "EVALUATION.rmbench_root=${RMBENCH_ROOT}" \
  "EVALUATION.hf_revision_marker=${RMBENCH_HF_REVISION_MARKER}" \
  "EVALUATION.suite=${SUITE}" \
  "EVALUATION.output_dir=${FASTWAM_RMBENCH_EVAL_ROOT}" \
  "EVALUATION.policy_kind=fastwam_baseline" \
  "EVALUATION.policy_name=fastwam_policy" \
  "EVALUATION.policy_source=experiments/robotwin/fastwam_policy" \
  "EVALUATION.seed_protocol_root=${WARM_RMBENCH_ONLINE_CONTRACT}" \
  "EVALUATION.baseline_policy_overrides.sim_cfg_path=$(pwd)/configs/sim_rmbench.yaml" \
  "EVALUATION.baseline_policy_overrides.sim_task=rmbench_fastwam_online_3cam384" \
  "EVALUATION.baseline_policy_overrides.device=${WARM_EVAL_DEVICE:-cuda}" \
  "EVALUATION.baseline_policy_overrides.dataset_stats_path=${TRAIN_STATS}" \
  "EVALUATION.baseline_policy_overrides.action_horizon=32" \
  "EVALUATION.baseline_policy_overrides.replan_steps=10" \
  "EVALUATION.baseline_policy_overrides.num_inference_steps=${WARM_NUM_INFERENCE_STEPS:-10}" \
  "++EVALUATION.baseline_policy_overrides.evaluation_namespace=${WARM_EVALUATION_NAMESPACE:-warm-rmbench-full-v1}" \
  "MULTIRUN.num_gpus=${WARM_NUM_GPUS:-1}" \
  "MULTIRUN.max_tasks_per_gpu=${WARM_MAX_TASKS_PER_GPU:-1}" \
  "seed=${WARM_ROOT_SEED:-17}"
