#!/usr/bin/env bash
set -euo pipefail

# Run complete-WARM LIBERO evaluations serially on one GPU.  Every task keeps
# the immutable evidence layout owned by acp_warm_libero_eval.sh.  This driver
# only adds a persistent plan, completion markers, resume/skip behavior, and a
# final aggregate summary; it never modifies a completed evaluation root.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-/mnt/afs/task3_2/L202500276_lwz/envs/warm}"
WARM_EVAL_BASE="${WARM_EVAL_BASE:-/mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations}"
WARM_TRAIN_RUN_DIR="${WARM_TRAIN_RUN_DIR:-${PROJECT_DIR}/runs/libero_warm_2cam224_1e-4/warm-full-4xh100-zero1-numerics-20260718-110409}"
WARM_CHECKPOINT="${WARM_CHECKPOINT:-${WARM_TRAIN_RUN_DIR}/checkpoints/weights/step_019100.pt}"

WARM_EVAL_SUITES="${WARM_EVAL_SUITES:-libero_spatial libero_object libero_goal libero_10}"
WARM_EVAL_TASK_IDS="${WARM_EVAL_TASK_IDS:-0 1 2 3 4 5 6 7 8 9}"
WARM_EVAL_SEEDS="${WARM_EVAL_SEEDS:-17}"
WARM_EVAL_LABEL="${WARM_EVAL_LABEL:-formal-v1}"
WARM_EVAL_BATCH_ID="${WARM_EVAL_BATCH_ID:-${WARM_EVAL_LABEL}}"
WARM_EVAL_PREPARE="${WARM_EVAL_PREPARE:-false}"
WARM_EXPECTED_TRIALS="${WARM_EXPECTED_TRIALS:-50}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ -x "${CONDA_ENV_DIR}/bin/python" ]] \
  || fail "Python environment not found: ${CONDA_ENV_DIR}"
# The delegated entrypoint is intentionally invoked through ``bash`` below.
# Requiring its executable bit is therefore both unnecessary and brittle when
# a user copies the repository through a filesystem that does not preserve
# POSIX modes (the canonical repository records this file as 100644).
[[ -f "${SCRIPT_DIR}/acp_warm_libero_eval.sh" ]] \
  || fail "single-task evaluation entrypoint is missing"
[[ "${CUDA_VISIBLE_DEVICES}" != *,* ]] \
  || fail "serial evaluation uses one GPU; CUDA_VISIBLE_DEVICES must contain one device"
[[ "${WARM_EVAL_LABEL}" =~ ^[A-Za-z0-9._-]+$ ]] \
  || fail "WARM_EVAL_LABEL contains unsafe characters"
[[ "${WARM_EVAL_BATCH_ID}" =~ ^[A-Za-z0-9._-]+$ ]] \
  || fail "WARM_EVAL_BATCH_ID contains unsafe characters"
[[ "${WARM_EXPECTED_TRIALS}" =~ ^[1-9][0-9]*$ ]] \
  || fail "WARM_EXPECTED_TRIALS must be positive"
case "${WARM_EVAL_PREPARE}" in
  true|false) ;;
  *) fail "WARM_EVAL_PREPARE must be true or false" ;;
esac

checkpoint_name="$(basename "${WARM_CHECKPOINT}")"
if [[ ! "${checkpoint_name}" =~ ^step_([0-9]{6})\.pt$ ]]; then
  fail "WARM_CHECKPOINT must use the canonical step_NNNNNN.pt name"
fi
step_digits="${BASH_REMATCH[1]}"
step_label="step_${step_digits}"

# Accept comma-separated or whitespace-separated lists without invoking eval.
read -r -a suites <<< "${WARM_EVAL_SUITES//,/ }"
read -r -a task_ids <<< "${WARM_EVAL_TASK_IDS//,/ }"
read -r -a seeds <<< "${WARM_EVAL_SEEDS//,/ }"
(( ${#suites[@]} > 0 )) || fail "WARM_EVAL_SUITES is empty"
(( ${#task_ids[@]} > 0 )) || fail "WARM_EVAL_TASK_IDS is empty"
(( ${#seeds[@]} > 0 )) || fail "WARM_EVAL_SEEDS is empty"

for suite in "${suites[@]}"; do
  case "${suite}" in
    libero_spatial|libero_object|libero_goal|libero_10) ;;
    *) fail "unsupported suite in WARM_EVAL_SUITES: ${suite}" ;;
  esac
done
for task_id in "${task_ids[@]}"; do
  [[ "${task_id}" =~ ^[0-9]+$ ]] || fail "invalid task id: ${task_id}"
  (( task_id < 10 )) || fail "LIBERO suite task id must be in [0, 9]: ${task_id}"
done
for seed in "${seeds[@]}"; do
  [[ "${seed}" =~ ^[0-9]+$ ]] || fail "invalid seed: ${seed}"
done

batch_root="${WARM_EVAL_BASE}/batches/${step_label}/${WARM_EVAL_BATCH_ID}"
status_root="${batch_root}/status"
plan_path="${batch_root}/plan.tsv"
summary_path="${batch_root}/summary.json"
mkdir -p "${status_root}"

plan_tmp="$(mktemp "${batch_root}/.plan.XXXXXX")"
trap 'rm -f "${plan_tmp:-}"' EXIT
printf 'suite\ttask_id\tseed\tevaluation_root\tresult_json\n' > "${plan_tmp}"
for seed in "${seeds[@]}"; do
  for suite in "${suites[@]}"; do
    for task_id in "${task_ids[@]}"; do
      task_label="task_$(printf '%02d' "${task_id}")"
      eval_root="${WARM_EVAL_BASE}/results/${step_label}/${suite}/${task_label}/seed_${seed}/${WARM_EVAL_LABEL}"
      result_json="${eval_root}/results/${suite}/gpu0_task${task_id}_results.json"
      printf '%s\t%s\t%s\t%s\t%s\n' \
        "${suite}" "${task_id}" "${seed}" "${eval_root}" "${result_json}" \
        >> "${plan_tmp}"
    done
  done
done

if [[ -e "${plan_path}" ]]; then
  cmp -s "${plan_tmp}" "${plan_path}" \
    || fail "batch id already has a different plan: ${plan_path}"
  rm -f "${plan_tmp}"
else
  mv "${plan_tmp}" "${plan_path}"
fi

if [[ "${WARM_EVAL_PREPARE}" == true ]]; then
  echo "===== one-time evaluation prepare ====="
  PROJECT_DIR="${PROJECT_DIR}" \
  CONDA_ENV_DIR="${CONDA_ENV_DIR}" \
  WARM_EVAL_BASE="${WARM_EVAL_BASE}" \
  WARM_TRAIN_RUN_DIR="${WARM_TRAIN_RUN_DIR}" \
  WARM_CHECKPOINT="${WARM_CHECKPOINT}" \
  EVAL_ACTION=prepare \
  bash "${SCRIPT_DIR}/acp_warm_libero_eval.sh"
fi

validate_and_mark() {
  local result_json="$1"
  local marker="$2"
  local suite="$3"
  local task_id="$4"
  local seed="$5"
  "${CONDA_ENV_DIR}/bin/python" - \
    "${result_json}" "${marker}" "${suite}" "${task_id}" "${seed}" \
    "${WARM_EXPECTED_TRIALS}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

result_path = Path(sys.argv[1]).resolve()
marker_path = Path(sys.argv[2])
expected_suite = sys.argv[3]
expected_task = int(sys.argv[4])
expected_seed = int(sys.argv[5])
expected_trials = int(sys.argv[6])

with result_path.open(encoding="utf-8") as handle:
    value = json.load(handle)
if value.get("task_suite") != expected_suite:
    raise SystemExit("result task_suite does not match the serial plan")
if int(value.get("task_id", -1)) != expected_task:
    raise SystemExit("result task_id does not match the serial plan")
if int(value.get("total_episodes", -1)) != expected_trials:
    raise SystemExit("result does not contain the expected number of trials")
successes = int(value.get("successes", -1))
if not 0 <= successes <= expected_trials:
    raise SystemExit("result successes is outside the valid range")
episodes = value.get("warm_online_episodes")
if not isinstance(episodes, list) or len(episodes) != expected_trials:
    raise SystemExit("result contains incomplete WARM episode evidence")
if sorted(int(item.get("episode_index", -1)) for item in episodes) != list(range(expected_trials)):
    raise SystemExit("result episode indices are incomplete or duplicated")
header = value.get("warm_online_header")
if not isinstance(header, dict) or header.get("source_policy") != "fixed_context_top1":
    raise SystemExit("result is not a fixed-context complete-WARM evaluation")
if header.get("side") != "full_retrospection":
    raise SystemExit("result is not a full-retrospection evaluation")
if int(header.get("contract", {}).get("root_seed", -1)) != expected_seed:
    raise SystemExit("result root seed does not match the serial plan")
runtime_attestation = header.get("runtime_attestation", {})
training_commit = str(runtime_attestation.get("git_commit", ""))
compatibility_path = result_path.parents[2] / "evaluation_compatibility.json"
compatibility_sha256 = None
if compatibility_path.is_file():
    compatibility_bytes = compatibility_path.read_bytes()
    compatibility_sha256 = hashlib.sha256(compatibility_bytes).hexdigest()
    compatibility = json.loads(compatibility_bytes)
    if compatibility.get("schema") != "warm.evaluation-compatibility":
        raise SystemExit("evaluation compatibility evidence has an invalid schema")
    if int(compatibility.get("version", -1)) != 1:
        raise SystemExit("evaluation compatibility evidence has an invalid version")
    if compatibility.get("training_commit") != training_commit:
        raise SystemExit("evaluation compatibility training commit mismatch")
    patch_sha = str(compatibility.get("patch_file_sha256", ""))
    if len(patch_sha) != 64 or any(character not in "0123456789abcdef" for character in patch_sha):
        raise SystemExit("evaluation compatibility patch digest is invalid")
    effective_namespace = str(compatibility.get("effective_evaluation_namespace", ""))
    if not effective_namespace.endswith(f"-compat-{patch_sha[:12]}"):
        raise SystemExit("evaluation compatibility namespace does not bind the patch")
    encoded_namespace = json.dumps(
        {"evaluation_namespace": effective_namespace},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    namespace_sha = hashlib.sha256(encoded_namespace).hexdigest()
    if header.get("contract", {}).get("evaluation_namespace_sha256") != namespace_sha:
        raise SystemExit("result contract does not bind the compatibility namespace")
elif training_commit == "c4763a975298de6f00939360551616af7902d57a":
    raise SystemExit("affected checkpoint result is missing compatibility evidence")
digest = hashlib.sha256(result_path.read_bytes()).hexdigest()
marker = {
    "schema": "warm.libero-serial-evaluation-completion",
    "version": 1,
    "suite": expected_suite,
    "task_id": expected_task,
    "root_seed": expected_seed,
    "successes": successes,
    "total_episodes": expected_trials,
    "duration_seconds": float(value.get("duration", 0.0)),
    "result_path": str(result_path),
    "result_sha256": digest,
    "evaluation_compatibility_sha256": compatibility_sha256,
}
marker_path.parent.mkdir(parents=True, exist_ok=True)
temporary = marker_path.with_name(f".{marker_path.name}.{os.getpid()}.tmp")
temporary.write_text(
    json.dumps(marker, sort_keys=True, indent=2, allow_nan=False) + "\n",
    encoding="utf-8",
)
os.replace(temporary, marker_path)
print(
    f"validated {expected_suite}/task_{expected_task:02d}/seed_{expected_seed}: "
    f"{successes}/{expected_trials}, duration={marker['duration_seconds']:.1f}s"
)
PY
}

job_count=$(( ${#suites[@]} * ${#task_ids[@]} * ${#seeds[@]} ))
job_index=0
while IFS=$'\t' read -r suite task_id seed eval_root result_json; do
  [[ "${suite}" != suite ]] || continue
  job_index=$((job_index + 1))
  job_key="${suite}_task_$(printf '%02d' "${task_id}")_seed_${seed}"
  marker="${status_root}/${job_key}.success.json"
  echo "===== [${job_index}/${job_count}] ${suite} task=${task_id} seed=${seed} ====="

  if [[ -e "${marker}" ]]; then
    [[ -f "${result_json}" ]] \
      || fail "completion marker exists but result is missing: ${result_json}"
    validate_and_mark "${result_json}" "${marker}" "${suite}" "${task_id}" "${seed}"
    echo "skip: already completed"
    continue
  fi

  if [[ -e "${eval_root}" ]]; then
    if [[ -f "${result_json}" ]]; then
      echo "recovering completion marker from an already complete immutable result"
      validate_and_mark "${result_json}" "${marker}" "${suite}" "${task_id}" "${seed}"
      continue
    fi
    fail "incomplete immutable evaluation root exists: ${eval_root}; preserve it and rerun this task with a new WARM_EVAL_LABEL"
  fi

  PROJECT_DIR="${PROJECT_DIR}" \
  CONDA_ENV_DIR="${CONDA_ENV_DIR}" \
  WARM_EVAL_BASE="${WARM_EVAL_BASE}" \
  WARM_TRAIN_RUN_DIR="${WARM_TRAIN_RUN_DIR}" \
  WARM_CHECKPOINT="${WARM_CHECKPOINT}" \
  WARM_EVAL_ROOT="${eval_root}" \
  WARM_TASK_SUITE="${suite}" \
  WARM_TASK_ID="${task_id}" \
  WARM_ROOT_SEED="${seed}" \
  WARM_EVAL_LABEL="${WARM_EVAL_LABEL}" \
  EVAL_ACTION=run \
  bash "${SCRIPT_DIR}/acp_warm_libero_eval.sh"

  [[ -f "${result_json}" ]] \
    || fail "evaluator exited successfully but did not publish ${result_json}"
  validate_and_mark "${result_json}" "${marker}" "${suite}" "${task_id}" "${seed}"
done < "${plan_path}"

"${CONDA_ENV_DIR}/bin/python" - "${plan_path}" "${status_root}" "${summary_path}" <<'PY'
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

plan_path = Path(sys.argv[1])
status_root = Path(sys.argv[2])
summary_path = Path(sys.argv[3])
rows = []
for line in plan_path.read_text(encoding="utf-8").splitlines()[1:]:
    suite, task_id, seed, _, _ = line.split("\t")
    marker = status_root / f"{suite}_task_{int(task_id):02d}_seed_{seed}.success.json"
    if not marker.is_file():
        raise SystemExit(f"missing completion marker: {marker}")
    rows.append(json.loads(marker.read_text(encoding="utf-8")))

groups = defaultdict(lambda: {"successes": 0, "episodes": 0, "duration_seconds": 0.0})
for row in rows:
    key = f"{row['suite']}/seed_{row['root_seed']}"
    groups[key]["successes"] += int(row["successes"])
    groups[key]["episodes"] += int(row["total_episodes"])
    groups[key]["duration_seconds"] += float(row["duration_seconds"])
for value in groups.values():
    value["success_rate"] = value["successes"] / value["episodes"]

total_successes = sum(int(row["successes"]) for row in rows)
total_episodes = sum(int(row["total_episodes"]) for row in rows)
summary = {
    "schema": "warm.libero-serial-evaluation-summary",
    "version": 1,
    "completed_jobs": len(rows),
    "total_successes": total_successes,
    "total_episodes": total_episodes,
    "overall_success_rate": total_successes / total_episodes,
    "total_duration_seconds": sum(float(row["duration_seconds"]) for row in rows),
    "groups": dict(sorted(groups.items())),
    "results": rows,
}
temporary = summary_path.with_name(f".{summary_path.name}.{os.getpid()}.tmp")
temporary.write_text(
    json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n",
    encoding="utf-8",
)
os.replace(temporary, summary_path)
for key, value in sorted(groups.items()):
    print(
        f"{key}: {value['successes']}/{value['episodes']} "
        f"({100.0 * value['success_rate']:.2f}%)"
    )
print(
    f"overall: {total_successes}/{total_episodes} "
    f"({100.0 * summary['overall_success_rate']:.2f}%)"
)
print(f"summary={summary_path}")
PY

echo "SERIAL_EVAL_COMPLETE jobs=${job_count} summary=${summary_path}"
