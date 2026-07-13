#!/usr/bin/env bash

# Shared fail-closed guards for formal WARM GPU-server jobs.  This file is
# sourced by launchers; it intentionally performs no work at import time.

warm_die() {
  printf 'error: %s\n' "$*" >&2
  return 2
}

warm_require_env() {
  local name
  for name in "$@"; do
    if [[ -z "${!name:-}" ]]; then
      warm_die "${name} must be set" || return
    fi
  done
}

warm_require_file_or_directory() {
  local path
  for path in "$@"; do
    if [[ ! -e "${path}" ]]; then
      warm_die "required immutable input not found: ${path}" || return
    fi
  done
}

warm_require_sha40() {
  local value="$1"
  local label="$2"
  if [[ ! "${value}" =~ ^[0-9a-f]{40}$ ]]; then
    warm_die "${label} must be a lowercase 40-character commit" || return
  fi
}

warm_require_private_checkout() {
  local root expected_head remote_names fetch_url push_url expected_origin
  root="$(git rev-parse --show-toplevel 2>/dev/null)" || {
    warm_die "formal jobs must run inside the WARM Git checkout"
    return
  }
  cd "${root}"
  remote_names="$(git remote | LC_ALL=C sort)"
  if [[ "${remote_names}" != "origin" ]]; then
    warm_die "the formal checkout must have exactly one remote named origin" || return
  fi
  fetch_url="$(git remote get-url origin)"
  push_url="$(git remote get-url --push origin)"
  if [[ "${fetch_url}" != "${push_url}" ]]; then
    warm_die "origin fetch and push URLs must be identical" || return
  fi
  expected_origin="${WARM_EXPECTED_ORIGIN:-}"
  if [[ -n "${expected_origin}" ]]; then
    if [[ "${fetch_url}" != "${expected_origin}" ]]; then
      warm_die "origin fetch/push URLs must equal WARM_EXPECTED_ORIGIN (${expected_origin})" || return
    fi
  else
    case "${fetch_url}" in
      git@github.com:ZzzzzZhhmm/WARM.git|https://github.com/ZzzzzZhhmm/WARM.git) ;;
      *)
        warm_die "origin must be the private ZzzzzZhhmm/WARM SSH or HTTPS URL" || return
        ;;
    esac
  fi
  if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
    warm_die "formal jobs require a committed, clean WARM checkout" || return
  fi
  expected_head="${WARM_CODE_REVISION:-$(git rev-parse HEAD)}"
  warm_require_sha40 "${expected_head}" "WARM_CODE_REVISION" || return
  if [[ "$(git rev-parse HEAD)" != "${expected_head}" ]]; then
    warm_die "checkout HEAD does not match WARM_CODE_REVISION" || return
  fi
  export WARM_CODE_REVISION="${expected_head}"
  export WARM_REPOSITORY_ROOT="${root}"
}

warm_require_read_only_external_checkout() {
  local checkout="$1"
  local expected_revision="$2"
  local actual push_url
  warm_require_sha40 "${expected_revision}" "external code revision" || return
  if [[ ! -d "${checkout}/.git" ]]; then
    warm_die "external checkout is not a Git worktree: ${checkout}" || return
  fi
  actual="$(git -C "${checkout}" rev-parse HEAD)"
  if [[ "${actual}" != "${expected_revision}" ]]; then
    warm_die "external checkout revision mismatch: ${actual}" || return
  fi
  if [[ -n "$(git -C "${checkout}" status --porcelain --untracked-files=normal)" ]]; then
    warm_die "external checkout must be clean" || return
  fi
  push_url="$(git -C "${checkout}" remote get-url --push origin 2>/dev/null || true)"
  if [[ "${push_url}" != "DISABLED" ]]; then
    warm_die "external checkout push URL must be the literal DISABLED" || return
  fi
}

warm_configure_offline_logging() {
  export WANDB_MODE="${WANDB_MODE:-offline}"
  if [[ "${WANDB_MODE}" != "offline" && "${WANDB_MODE}" != "disabled" ]]; then
    if [[ "${WARM_ALLOW_NETWORK_LOGGING:-0}" != "1" ]]; then
      warm_die "network experiment logging requires WARM_ALLOW_NETWORK_LOGGING=1" || return
    fi
  fi
  # Formal runs consume local, revision-pinned snapshots.  Disallow accidental
  # model downloads even when the server has Internet access.
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
}

warm_refuse_existing_output() {
  local path="$1"
  if [[ -e "${path}" || -L "${path}" ]]; then
    warm_die "immutable output already exists: ${path}" || return
  fi
}
