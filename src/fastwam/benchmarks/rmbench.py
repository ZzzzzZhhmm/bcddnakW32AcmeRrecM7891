"""Fail-closed definition of the official RMBench evaluation protocol.

The benchmark implementation remains an external dependency.  WARM pins and
validates it, but neither vendors it nor writes tracked or untracked files in
its Git checkout.  Evaluation runs from a disposable WARM-owned overlay whose
read-only inputs are symbolic links to the pristine official checkout.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


RMBENCH_PROTOCOL_VERSION = "warm-rmbench-official9-v1"
RMBENCH_REPOSITORY = "https://github.com/RoboTwin-Platform/RMBench.git"
RMBENCH_CODE_REVISION = "57ee09cbc6267bc36ca0ac2d8d1c5c3b245c112c"
RMBENCH_HF_REPOSITORY = "TianxingChen/RMBench"
RMBENCH_HF_DATASET_REVISION = "855e90e1213d150bf4889130e83398f107314681"
RMBENCH_TASK_CONFIG = "demo_clean"
RMBENCH_INSTRUCTION_TYPE = "unseen"
RMBENCH_EPISODES_PER_TASK = 100


@dataclass(frozen=True)
class RMBenchTask:
    """One paper-reported RMBench task and its official step limit."""

    name: str
    memory_regime: str
    step_limit: int


# Paper order: five M(1) tasks followed by four M(n) tasks.  Helper tasks in
# `_eval_step_limit.yml` are deliberately excluded from the official suite.
RMBENCH_TASKS: tuple[RMBenchTask, ...] = (
    RMBenchTask("observe_and_pickup", "M(1)", 250),
    RMBenchTask("rearrange_blocks", "M(1)", 700),
    RMBenchTask("put_back_block", "M(1)", 500),
    RMBenchTask("swap_blocks", "M(1)", 1000),
    RMBenchTask("swap_T", "M(1)", 600),
    RMBenchTask("blocks_ranking_try", "M(n)", 3500),
    RMBenchTask("press_button", "M(n)", 1500),
    RMBenchTask("cover_blocks", "M(n)", 1500),
    RMBenchTask("battery_try", "M(n)", 1000),
)

# Small smoke/pilot suite chosen to cover state restoration, ordering, and
# retry-dependent memory.  It is not an alternative benchmark score.
RMBENCH_PILOT_TASK_NAMES: tuple[str, ...] = (
    "put_back_block",
    "rearrange_blocks",
    "battery_try",
)
RMBENCH_PILOT_TASKS: tuple[RMBenchTask, ...] = tuple(
    task for name in RMBENCH_PILOT_TASK_NAMES for task in RMBENCH_TASKS if task.name == name
)


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def task_manifest() -> dict[str, Any]:
    """Return the immutable, hashable protocol manifest."""

    return {
        "protocol_version": RMBENCH_PROTOCOL_VERSION,
        "official_code_revision": RMBENCH_CODE_REVISION,
        "hf_repository": RMBENCH_HF_REPOSITORY,
        "hf_dataset_revision": RMBENCH_HF_DATASET_REVISION,
        "task_config": RMBENCH_TASK_CONFIG,
        "instruction_type": RMBENCH_INSTRUCTION_TYPE,
        "episodes_per_task": RMBENCH_EPISODES_PER_TASK,
        "tasks": [asdict(task) for task in RMBENCH_TASKS],
        "pilot_task_names": list(RMBENCH_PILOT_TASK_NAMES),
    }


RMBENCH_TASK_MANIFEST_SHA256 = hashlib.sha256(
    _canonical_json_bytes(task_manifest())
).hexdigest()


def tasks_for_suite(suite: str) -> tuple[RMBenchTask, ...]:
    """Resolve an exact supported suite; arbitrary subsets are rejected."""

    normalized = str(suite).strip().lower()
    if normalized in {"official9", "official", "full"}:
        return RMBENCH_TASKS
    if normalized in {"pilot3", "pilot"}:
        return RMBENCH_PILOT_TASKS
    raise ValueError(
        f"Unsupported RMBench suite {suite!r}; expected 'official9' or 'pilot3'."
    )


def task_by_name(name: str) -> RMBenchTask:
    for task in RMBENCH_TASKS:
        if task.name == name:
            return task
    allowed = ", ".join(task.name for task in RMBENCH_TASKS)
    raise ValueError(f"Not an official RMBench task: {name!r}. Allowed: {allowed}")


def validate_protocol_pins(
    *,
    code_revision: str,
    hf_revision: str,
    manifest_sha256: str,
    task_config: str,
    episodes_per_task: int,
) -> None:
    """Reject accidental protocol drift before launching expensive jobs."""

    expected = {
        "code_revision": RMBENCH_CODE_REVISION,
        "hf_revision": RMBENCH_HF_DATASET_REVISION,
        "manifest_sha256": RMBENCH_TASK_MANIFEST_SHA256,
        "task_config": RMBENCH_TASK_CONFIG,
        "episodes_per_task": RMBENCH_EPISODES_PER_TASK,
    }
    actual = {
        "code_revision": str(code_revision),
        "hf_revision": str(hf_revision),
        "manifest_sha256": str(manifest_sha256),
        "task_config": str(task_config),
        "episodes_per_task": int(episodes_per_task),
    }
    mismatches = [
        f"{key}: expected {expected[key]!r}, got {actual[key]!r}"
        for key in expected
        if actual[key] != expected[key]
    ]
    if mismatches:
        raise RuntimeError("RMBench protocol pin mismatch: " + "; ".join(mismatches))


def _run_git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def _normalized_remote(url: str) -> str:
    return url.strip().rstrip("/").lower()


def validate_read_only_checkout(
    root: str | Path,
    *,
    expected_revision: str = RMBENCH_CODE_REVISION,
    require_push_disabled: bool = True,
) -> dict[str, Any]:
    """Validate a pinned external checkout without changing it.

    Tracked, staged, and untracked changes are all forbidden.  WARM executes
    the official evaluator through a separate runtime overlay, so the public
    checkout must remain a pristine read-only source tree.  A public push URL
    is forbidden by default, making accidental upstream writes fail closed.
    """

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"RMBench checkout not found: {root_path}")

    required = (
        "LICENSE",
        "script/eval_policy.py",
        "task_config/demo_clean.yml",
        "task_config/_eval_step_limit.yml",
        "policy",
    )
    missing = [item for item in required if not (root_path / item).exists()]
    if missing:
        raise FileNotFoundError(
            f"Invalid RMBench checkout {root_path}; missing: {', '.join(missing)}"
        )

    inside = _run_git(root_path, "rev-parse", "--is-inside-work-tree").stdout.strip()
    if inside != "true":
        raise RuntimeError(f"Not a Git worktree: {root_path}")
    head = _run_git(root_path, "rev-parse", "HEAD").stdout.strip()
    if head != expected_revision:
        raise RuntimeError(
            f"RMBench code revision mismatch: expected {expected_revision}, got {head}"
        )

    tracked_status = _run_git(
        root_path,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).stdout.strip()
    if tracked_status:
        raise RuntimeError(
            "RMBench checkout has tracked/staged changes or untracked files; evaluation requires a "
            f"read-only code checkout. Status:\n{tracked_status}"
        )

    fetch_url = _run_git(root_path, "remote", "get-url", "origin").stdout.strip()
    if _normalized_remote(fetch_url) != _normalized_remote(RMBENCH_REPOSITORY):
        raise RuntimeError(
            f"Unexpected RMBench origin: expected {RMBENCH_REPOSITORY}, got {fetch_url}"
        )

    push_proc = _run_git(
        root_path,
        "remote",
        "get-url",
        "--push",
        "origin",
        check=False,
    )
    push_url = push_proc.stdout.strip() if push_proc.returncode == 0 else ""
    push_disabled_values = {"", "disabled", "no_push", "no-push", "disabled://"}
    if require_push_disabled and push_url.strip().lower() not in push_disabled_values:
        raise RuntimeError(
            "RMBench checkout has an enabled push URL. Disable it before evaluation "
            "with: git -C <RMBENCH_ROOT> remote set-url --push origin DISABLED. "
            f"Current push URL: {push_url}"
        )

    _validate_official_task_limits(root_path / "task_config" / "_eval_step_limit.yml")
    _validate_demo_clean(root_path / "task_config" / "demo_clean.yml")
    return {
        "root": str(root_path),
        "head": head,
        "fetch_url": fetch_url,
        "push_url": push_url,
        "tracked_clean": True,
    }


def _simple_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load simple benchmark YAML while keeping this module dependency-light."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - project depends on PyYAML via Hydra
        raise RuntimeError("PyYAML is required to validate RMBench") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a YAML mapping in {path}")
    return payload


def _validate_official_task_limits(path: Path) -> None:
    payload = _simple_yaml_mapping(path)
    for task in RMBENCH_TASKS:
        actual = payload.get(task.name)
        if actual != task.step_limit:
            raise RuntimeError(
                f"Official step limit drift for {task.name}: "
                f"expected {task.step_limit}, got {actual!r}"
            )


def _validate_demo_clean(path: Path) -> None:
    payload = _simple_yaml_mapping(path)
    if payload.get("embodiment") != ["aloha-agilex"]:
        raise RuntimeError(
            f"demo_clean embodiment drift: expected ['aloha-agilex'], "
            f"got {payload.get('embodiment')!r}"
        )
    randomization = payload.get("domain_randomization")
    if not isinstance(randomization, dict):
        raise RuntimeError(f"Invalid demo_clean domain_randomization in {path}")
    forbidden_enabled = (
        "random_background",
        "cluttered_table",
        "random_table_height",
        "random_light",
    )
    enabled = [key for key in forbidden_enabled if bool(randomization.get(key, False))]
    if enabled:
        raise RuntimeError(f"demo_clean unexpectedly enables: {', '.join(enabled)}")
    nonzero_randomizers = (
        "random_head_camera_dis",
        "clean_background_rate",
        "crazy_random_light_rate",
    )
    # clean_background_rate is one in the official file; the other strengths
    # must be zero even if their parent toggle is false.
    expected_values = {
        "random_head_camera_dis": 0,
        "clean_background_rate": 1,
        "crazy_random_light_rate": 0,
    }
    drift = {
        key: randomization.get(key)
        for key in nonzero_randomizers
        if randomization.get(key) != expected_values[key]
    }
    if drift:
        raise RuntimeError(f"demo_clean randomization strength drift: {drift}")


def validate_hf_revision_marker(
    marker_path: str | Path,
    *,
    expected_revision: str = RMBENCH_HF_DATASET_REVISION,
) -> dict[str, Any]:
    """Validate the WARM-owned attestation for the pinned HF snapshot.

    Download/preparation tooling must write this marker outside the official
    Git checkout.  Merely configuring a revision string is not accepted as
    evidence that assets/data were obtained from that revision.
    """

    path = Path(marker_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Pinned RMBench Hugging Face revision marker not found: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid HF revision marker: {path}")
    if payload.get("repo_id") != RMBENCH_HF_REPOSITORY:
        raise RuntimeError(
            f"HF marker repo_id mismatch: expected {RMBENCH_HF_REPOSITORY!r}, "
            f"got {payload.get('repo_id')!r}"
        )
    if payload.get("revision") != expected_revision:
        raise RuntimeError(
            f"HF marker revision mismatch: expected {expected_revision}, "
            f"got {payload.get('revision')!r}"
        )
    return payload


def validate_policy_source(
    policy_source: str | Path,
    *,
    policy_name: str,
) -> Path:
    """Validate a WARM-owned policy package without touching RMBench.

    The parent directory is placed on ``PYTHONPATH`` by the evaluator wrapper,
    so the official evaluator can import ``policy_name`` while the public
    checkout remains byte-for-byte clean and receives no untracked symlink.
    """

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", policy_name):
        raise ValueError(f"Unsafe policy name: {policy_name!r}")
    source = Path(policy_source).expanduser().resolve()
    if source.name != policy_name:
        raise RuntimeError(
            f"Policy package directory must be named {policy_name!r}: {source}"
        )
    required = ("__init__.py", "deploy_policy.py", "deploy_policy.yml")
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"WARM RMBench policy source is incomplete ({missing}): {source}"
        )
    return source


def prepare_runtime_overlay(
    checkout_root: str | Path,
    runtime_root: str | Path,
) -> dict[str, Any]:
    """Create a WARM-owned, disposable view of the official checkout.

    RMBench's evaluator writes ``eval_result`` relative to its working
    directory.  Running it in the source checkout would therefore mutate a
    public repository even when Git push is disabled.  This overlay links
    read-only inputs from the validated checkout and owns all writable paths.
    It must be created fresh for every task run.
    """

    checkout = Path(checkout_root).expanduser().resolve()
    runtime = Path(runtime_root).expanduser().resolve()
    try:
        runtime.relative_to(checkout)
    except ValueError:
        pass
    else:
        raise RuntimeError("RMBench runtime overlay must be outside the checkout")
    if runtime.exists() and any(runtime.iterdir()):
        raise RuntimeError(f"RMBench runtime overlay is not empty: {runtime}")
    runtime.mkdir(parents=True, exist_ok=True)

    excluded = {".git", "eval_result", "policy"}
    linked: list[str] = []
    try:
        for child in sorted(checkout.iterdir(), key=lambda path: path.name):
            if child.name in excluded:
                continue
            target = runtime / child.name
            os.symlink(child, target, target_is_directory=child.is_dir())
            linked.append(child.name)
        (runtime / "eval_result").mkdir()
        # The official evaluator appends ./policy to sys.path.  Keep an empty,
        # WARM-owned directory there; our package is supplied via PYTHONPATH.
        (runtime / "policy").mkdir()
    except Exception:
        # Do not attempt recursive deletion here.  A failed overlay is
        # deliberately left as inspectable evidence and cannot be reused.
        raise RuntimeError(
            "Failed to create the read-only RMBench runtime overlay; the "
            "server filesystem must support symbolic links"
        )
    return {
        "checkout": str(checkout),
        "runtime": str(runtime),
        "linked_entries": linked,
        "writable_entries": ["eval_result", "policy"],
        "external_checkout_mutated": False,
    }


_SUCCESS_RE = re.compile(
    r"^\s*Success\s+Rate\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)
_REWARD_RE = re.compile(
    r"^\s*Reward\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)
_EPISODE_RE = re.compile(
    r"^episode_id=(\d+),\s*seed=(\d+),\s*result=(Success|Fail)\s*$"
)


@dataclass(frozen=True)
class RMBenchEpisodeResult:
    episode_id: int
    seed: int
    success: bool


@dataclass(frozen=True)
class RMBenchResult:
    task_name: str
    task_config: str
    episodes: int
    successes: int
    success_rate: float
    mean_reward: float
    episode_records_sha256: str
    official_result_file: str
    official_log_file: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_official_episode_records(
    log_file: str | Path,
    *,
    expected_episodes: int = RMBENCH_EPISODES_PER_TASK,
) -> tuple[RMBenchEpisodeResult, ...]:
    """Parse the exact ordered accepted episodes from one official log."""

    log_path = Path(log_file).expanduser().resolve()
    if not log_path.is_file():
        raise FileNotFoundError(f"Official RMBench episode log not found: {log_path}")
    records: list[RMBenchEpisodeResult] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = _EPISODE_RE.match(line.strip())
        if match:
            records.append(
                RMBenchEpisodeResult(
                    episode_id=int(match.group(1)),
                    seed=int(match.group(2)),
                    success=match.group(3) == "Success",
                )
            )
    if len(records) != int(expected_episodes):
        raise RuntimeError(
            f"Expected {expected_episodes} episode records in {log_path}, "
            f"found {len(records)}"
        )
    expected_ids = list(range(int(expected_episodes)))
    actual_ids = [record.episode_id for record in records]
    if actual_ids != expected_ids:
        raise RuntimeError("RMBench episode IDs are not the exact ordered 0..N-1 sequence")
    seeds = [record.seed for record in records]
    if any(right <= left for left, right in zip(seeds, seeds[1:])):
        raise RuntimeError(
            "RMBench accepted environment seeds must be strictly increasing and unique"
        )
    return tuple(records)


def validate_official_seed_namespace(
    log_file: str | Path,
    seed_protocol_file: str | Path,
    *,
    expected_episodes: int = RMBENCH_EPISODES_PER_TASK,
    expected_root_seed: int | None = None,
) -> dict[str, Any]:
    """Validate actual accepted seeds against an honest root-seed namespace.

    The official evaluator may skip unstable setup seeds before each accepted
    episode.  Consequently, a prebuilt protocol can bind only the ordinal and
    minimum candidate seed, not the actual accepted seed.  Actual accepted
    sequences are published after the run and compared across matrix cells.
    """

    protocol_path = Path(seed_protocol_file).expanduser().resolve()
    if not protocol_path.is_file() or protocol_path.suffix != ".npy":
        raise FileNotFoundError(
            f"RMBench seed namespace protocol not found: {protocol_path}"
        )
    try:
        protocol = np.load(protocol_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Invalid RMBench seed protocol: {protocol_path}") from exc
    expected_shape = (int(expected_episodes), 2)
    if protocol.shape != expected_shape or protocol.dtype != np.int64:
        raise RuntimeError(
            f"RMBench seed protocol must be int64 {expected_shape}, got "
            f"{protocol.dtype} {protocol.shape}"
        )
    ordinals = np.arange(int(expected_episodes), dtype=np.int64)
    if not np.array_equal(protocol[:, 0], ordinals):
        raise RuntimeError("RMBench seed protocol ordinals must be exact 0..N-1")
    lower_bounds = protocol[:, 1]
    if not np.array_equal(lower_bounds, lower_bounds[0] + ordinals):
        raise RuntimeError(
            "RMBench seed protocol lower bounds must be one root namespace sequence"
        )
    if expected_root_seed is not None:
        if (
            isinstance(expected_root_seed, bool)
            or not isinstance(expected_root_seed, int)
            or expected_root_seed < 0
        ):
            raise ValueError("expected_root_seed must be a non-negative integer")
        expected_start = 100_000 * (1 + expected_root_seed)
        if int(lower_bounds[0]) != expected_start:
            raise RuntimeError(
                "RMBench seed protocol namespace does not match the evaluator "
                f"root seed: expected {expected_start}, got {int(lower_bounds[0])}"
            )
    records = load_official_episode_records(
        log_file, expected_episodes=expected_episodes
    )
    actual = np.asarray([record.seed for record in records], dtype=np.int64)
    if np.any(actual < lower_bounds):
        raise RuntimeError(
            "RMBench actual accepted seed fell below its contracted namespace bound"
        )
    actual_digest = hashlib.sha256(actual.tobytes(order="C")).hexdigest()
    protocol_digest = hashlib.sha256(protocol.tobytes(order="C")).hexdigest()
    return {
        "not_accepted_seed_claim": True,
        "episodes": int(expected_episodes),
        "namespace_start": int(lower_bounds[0]),
        "seed_protocol_file": str(protocol_path),
        "seed_protocol_array_sha256": protocol_digest,
        "actual_accepted_seed_sha256": actual_digest,
        "actual_accepted_seed_min": int(actual[0]),
        "actual_accepted_seed_max": int(actual[-1]),
        "actual_accepted_seeds": actual,
    }


def derive_rmbench_policy_query_seed(
    root_seed: int,
    *,
    task_name: str,
    episode_index: int,
    frame_index: int,
    evaluation_namespace: str,
) -> int:
    """Derive the exact per-replan seed shared by FastWAM and WARM.

    This deliberately mirrors WARM's formal online ``QueryId`` namespace so
    the same-data FastWAM comparison does not repeatedly sample one fixed
    Gaussian source at every replan.  It also prevents a policy implementation
    from consuming process-global NumPy/Torch RNG state and perturbing the
    benchmark's environment/instruction stream.
    """

    if isinstance(root_seed, bool) or not isinstance(root_seed, int) or root_seed < 0:
        raise ValueError("root_seed must be a non-negative integer")
    namespace = str(evaluation_namespace)
    if not namespace or namespace.strip() != namespace or "\x00" in namespace:
        raise ValueError("evaluation_namespace must be normalized and non-empty")
    task = task_by_name(task_name)
    task_id = RMBENCH_TASKS.index(task)

    # Local imports keep the dependency-light protocol parser cheap for users
    # that do not instantiate an online policy.
    from fastwam.memory.candidate_cache import QueryId
    from fastwam.memory.manifest import sha256_canonical_json
    from fastwam.memory.online_retrieval import derive_online_query_seed

    namespace_sha256 = sha256_canonical_json(
        {"evaluation_namespace": namespace}
    )
    query_id = QueryId(
        dataset_id=f"warm-online/rmbench/rmbench/{namespace_sha256}",
        dataset_index=task_id,
        episode_index=episode_index,
        frame_index=frame_index,
    )
    return derive_online_query_seed(root_seed, query_id, namespace_sha256)


def parse_official_result(
    result_file: str | Path,
    *,
    task_name: str,
    expected_episodes: int = RMBENCH_EPISODES_PER_TASK,
    log_file: str | Path | None = None,
) -> RMBenchResult:
    """Parse and cross-check one official RMBench result and episode log."""

    task_by_name(task_name)
    result_path = Path(result_file).expanduser().resolve()
    if not result_path.is_file():
        raise FileNotFoundError(f"Official RMBench result not found: {result_path}")
    log_path = (
        Path(log_file).expanduser().resolve()
        if log_file is not None
        else result_path.with_name("eval_log.txt")
    )
    if not log_path.is_file():
        raise FileNotFoundError(f"Official RMBench episode log not found: {log_path}")

    success_values: list[float] = []
    reward_values: list[float] = []
    for line in result_path.read_text(encoding="utf-8").splitlines():
        success_match = _SUCCESS_RE.match(line)
        if success_match:
            success_values.append(float(success_match.group(1)))
        reward_match = _REWARD_RE.match(line)
        if reward_match:
            reward_values.append(float(reward_match.group(1)))
    if len(success_values) != 1 or len(reward_values) != 1:
        raise RuntimeError(
            f"Expected exactly one Success Rate and Reward in {result_path}; "
            f"found {len(success_values)} and {len(reward_values)}"
        )
    success_rate = success_values[0]
    mean_reward = reward_values[0]
    if not math.isfinite(success_rate) or not 0.0 <= success_rate <= 1.0:
        raise RuntimeError(f"Invalid success rate {success_rate!r} in {result_path}")
    if not math.isfinite(mean_reward):
        raise RuntimeError(f"Invalid reward {mean_reward!r} in {result_path}")

    records = load_official_episode_records(
        log_path, expected_episodes=expected_episodes
    )

    successes = sum(record.success for record in records)
    logged_rate = successes / float(expected_episodes)
    if not math.isclose(success_rate, logged_rate, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(
            f"Success rate mismatch: result={success_rate}, episode log={logged_rate}"
        )
    record_payload = [asdict(record) for record in records]
    record_sha = hashlib.sha256(_canonical_json_bytes(record_payload)).hexdigest()
    return RMBenchResult(
        task_name=task_name,
        task_config=RMBENCH_TASK_CONFIG,
        episodes=int(expected_episodes),
        successes=successes,
        success_rate=success_rate,
        mean_reward=mean_reward,
        episode_records_sha256=record_sha,
        official_result_file=str(result_path),
        official_log_file=str(log_path),
    )


def snapshot_result_files(base: str | Path) -> dict[Path, tuple[int, int]]:
    """Capture result mtime/size before one official evaluator invocation."""

    root = Path(base).expanduser().resolve()
    if not root.exists():
        return {}
    return {
        path.resolve(): (path.stat().st_mtime_ns, path.stat().st_size)
        for path in root.rglob("_result.txt")
        if path.is_file()
    }


def discover_new_result_file(
    base: str | Path,
    before: Mapping[Path, tuple[int, int]],
) -> Path:
    """Find exactly one result created or changed by an evaluator process."""

    root = Path(base).expanduser().resolve()
    changed: list[Path] = []
    for path in root.rglob("_result.txt") if root.exists() else ():
        resolved = path.resolve()
        current = (resolved.stat().st_mtime_ns, resolved.stat().st_size)
        if before.get(resolved) != current:
            changed.append(resolved)
    if len(changed) != 1:
        rendered = ", ".join(str(path) for path in sorted(changed)) or "<none>"
        raise RuntimeError(
            f"Expected exactly one new official RMBench result below {root}; "
            f"found {len(changed)}: {rendered}"
        )
    return changed[0]


def assert_exact_task_sequence(
    tasks: Sequence[str],
    *,
    suite: str,
) -> None:
    expected = [task.name for task in tasks_for_suite(suite)]
    actual = [str(task) for task in tasks]
    if actual != expected:
        raise RuntimeError(
            f"RMBench {suite} task sequence drift: expected {expected}, got {actual}"
        )


def write_manifest(path: str | Path, *, suite: str) -> Path:
    """Write a self-verifying WARM-owned protocol record."""

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    selected = tasks_for_suite(suite)
    payload = {
        **task_manifest(),
        "task_manifest_sha256": RMBENCH_TASK_MANIFEST_SHA256,
        "selected_suite": suite,
        "selected_tasks": [task.name for task in selected],
        "is_official_score": len(selected) == len(RMBENCH_TASKS),
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output
