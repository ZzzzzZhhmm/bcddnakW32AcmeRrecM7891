from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from fastwam.benchmarks.rmbench import (
    RMBENCH_CODE_REVISION,
    RMBENCH_EPISODES_PER_TASK,
    RMBENCH_HF_DATASET_REVISION,
    RMBENCH_HF_REPOSITORY,
    RMBENCH_PILOT_TASK_NAMES,
    RMBENCH_TASKS,
    RMBENCH_TASK_MANIFEST_SHA256,
    assert_exact_task_sequence,
    discover_new_result_file,
    derive_rmbench_policy_query_seed,
    parse_official_result,
    prepare_runtime_overlay,
    snapshot_result_files,
    task_manifest,
    tasks_for_suite,
    validate_hf_revision_marker,
    validate_official_seed_namespace,
    validate_policy_source,
    validate_protocol_pins,
    validate_read_only_checkout,
)


OFFICIAL_TASK_NAMES = [
    "observe_and_pickup",
    "rearrange_blocks",
    "put_back_block",
    "swap_blocks",
    "swap_T",
    "blocks_ranking_try",
    "press_button",
    "cover_blocks",
    "battery_try",
]


def test_exact_paper_task_and_pilot_manifests_are_stable() -> None:
    assert [task.name for task in RMBENCH_TASKS] == OFFICIAL_TASK_NAMES
    assert [task.name for task in tasks_for_suite("official9")] == OFFICIAL_TASK_NAMES
    assert [task.name for task in tasks_for_suite("pilot3")] == list(
        RMBENCH_PILOT_TASK_NAMES
    )
    assert task_manifest()["episodes_per_task"] == 100
    assert len(RMBENCH_TASK_MANIFEST_SHA256) == 64
    assert_exact_task_sequence(OFFICIAL_TASK_NAMES, suite="official9")
    with pytest.raises(RuntimeError, match="task sequence drift"):
        assert_exact_task_sequence(list(reversed(OFFICIAL_TASK_NAMES)), suite="official9")
    with pytest.raises(ValueError, match="official9.*pilot3"):
        tasks_for_suite("custom")


def test_protocol_pins_fail_closed() -> None:
    kwargs = {
        "code_revision": RMBENCH_CODE_REVISION,
        "hf_revision": RMBENCH_HF_DATASET_REVISION,
        "manifest_sha256": RMBENCH_TASK_MANIFEST_SHA256,
        "task_config": "demo_clean",
        "episodes_per_task": RMBENCH_EPISODES_PER_TASK,
    }
    validate_protocol_pins(**kwargs)
    kwargs["episodes_per_task"] = 99
    with pytest.raises(RuntimeError, match="episodes_per_task"):
        validate_protocol_pins(**kwargs)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _make_fake_checkout(root: Path) -> str:
    (root / "script").mkdir(parents=True)
    (root / "task_config").mkdir()
    (root / "policy").mkdir()
    (root / "LICENSE").write_text("MIT\n", encoding="utf-8")
    (root / "script" / "eval_policy.py").write_text("# fixture\n", encoding="utf-8")
    limits = {task.name: task.step_limit for task in RMBENCH_TASKS}
    (root / "task_config" / "_eval_step_limit.yml").write_text(
        yaml.safe_dump(limits), encoding="utf-8"
    )
    (root / "task_config" / "demo_clean.yml").write_text(
        yaml.safe_dump(
            {
                "domain_randomization": {
                    "random_background": False,
                    "cluttered_table": False,
                    "clean_background_rate": 1,
                    "crazy_random_light_rate": 0,
                    "random_head_camera_dis": 0,
                    "random_table_height": 0,
                    "random_light": False,
                },
                "embodiment": ["aloha-agilex"],
            }
        ),
        encoding="utf-8",
    )
    _git(root, "init")
    _git(root, "config", "user.email", "fixture@example.invalid")
    _git(root, "config", "user.name", "Fixture")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "fixture")
    _git(root, "remote", "add", "origin", "https://github.com/RoboTwin-Platform/RMBench.git")
    _git(root, "remote", "set-url", "--push", "origin", "DISABLED")
    return _git(root, "rev-parse", "HEAD")


def test_read_only_checkout_validation_accepts_local_drift(tmp_path: Path) -> None:
    head = _make_fake_checkout(tmp_path)
    attestation = validate_read_only_checkout(tmp_path, expected_revision=head)
    assert attestation["tracked_clean"] is True
    (tmp_path / "LICENSE").write_text("changed\n", encoding="utf-8")
    again = validate_read_only_checkout(tmp_path, expected_revision=head)
    assert again["root"] == str(tmp_path.resolve())


def test_hf_revision_marker_is_exact(tmp_path: Path) -> None:
    marker = tmp_path / "hf.json"
    marker.write_text(
        json.dumps(
            {
                "repo_id": RMBENCH_HF_REPOSITORY,
                "revision": RMBENCH_HF_DATASET_REVISION,
                "allow_patterns": ["embodiments/**", "objects/**", "data/*/demo_clean/**"],
            }
        ),
        encoding="utf-8",
    )
    assert validate_hf_revision_marker(marker)["revision"] == RMBENCH_HF_DATASET_REVISION
    marker.write_text(
        json.dumps({"repo_id": RMBENCH_HF_REPOSITORY, "revision": "main"}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="revision mismatch"):
        validate_hf_revision_marker(marker)


def test_policy_and_runtime_overlay_never_mutate_public_checkout(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "official"
    checkout.mkdir()
    _make_fake_checkout(checkout)
    policy = tmp_path / "warm_policy"
    policy.mkdir()
    for name in ("__init__.py", "deploy_policy.py", "deploy_policy.yml"):
        (policy / name).write_text("# fixture\n", encoding="utf-8")
    assert validate_policy_source(policy, policy_name="warm_policy") == policy.resolve()

    before = _git(checkout, "status", "--porcelain=v1", "--untracked-files=all")
    overlay = tmp_path / "runtime"
    try:
        attestation = prepare_runtime_overlay(checkout, overlay)
    except RuntimeError as exc:
        if "symbolic links" in str(exc):
            pytest.skip("local filesystem does not permit symbolic links")
        raise
    assert attestation["external_checkout_mutated"] is False
    assert (overlay / "script" / "eval_policy.py").is_file()
    assert (overlay / "eval_result").is_dir()
    assert (overlay / "policy").is_dir()
    assert _git(checkout, "status", "--porcelain=v1", "--untracked-files=all") == before


def _write_result_fixture(directory: Path, *, successes: int) -> tuple[Path, Path]:
    directory.mkdir(parents=True)
    result = directory / "_result.txt"
    result.write_text(
        f"Timestamp: fixture\n\nSuccess Rate: {successes / 100.0}\n\nReward: 2.5\n",
        encoding="utf-8",
    )
    log = directory / "eval_log.txt"
    log.write_text(
        "header\n"
        + "".join(
            f"episode_id={index}, seed={100000 + index}, "
            f"result={'Success' if index < successes else 'Fail'}\n"
            for index in range(100)
        ),
        encoding="utf-8",
    )
    return result, log


def test_result_parser_cross_checks_all_one_hundred_episode_records(tmp_path: Path) -> None:
    result, log = _write_result_fixture(tmp_path / "run", successes=37)
    parsed = parse_official_result(
        result,
        log_file=log,
        task_name="put_back_block",
    )
    assert parsed.episodes == 100
    assert parsed.successes == 37
    assert parsed.success_rate == 0.37
    assert len(parsed.episode_records_sha256) == 64

    result.write_text("Success Rate: 0.38\nReward: 2.5\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Success rate mismatch"):
        parse_official_result(result, log_file=log, task_name="put_back_block")


def test_seed_namespace_distinguishes_bounds_from_actual_accepted_seeds(
    tmp_path: Path,
) -> None:
    _, log = _write_result_fixture(tmp_path / "run", successes=37)
    protocol = np.stack(
        (
            np.arange(100, dtype=np.int64),
            100000 + np.arange(100, dtype=np.int64),
        ),
        axis=1,
    )
    path = tmp_path / "put_back_block.seed_protocol.npy"
    np.save(path, protocol, allow_pickle=False)
    attestation = validate_official_seed_namespace(
        log, path, expected_root_seed=0
    )
    assert attestation["not_accepted_seed_claim"] is True
    assert attestation["namespace_start"] == 100000
    assert attestation["actual_accepted_seeds"].shape == (100,)

    protocol[10, 1] += 2
    np.save(path, protocol, allow_pickle=False)
    with pytest.raises(RuntimeError, match="lower bounds"):
        validate_official_seed_namespace(log, path)

    protocol[:, 1] = 100000 + np.arange(100, dtype=np.int64)
    np.save(path, protocol, allow_pickle=False)
    with pytest.raises(RuntimeError, match="root seed"):
        validate_official_seed_namespace(log, path, expected_root_seed=17)


def test_fastwam_baseline_uses_the_warm_per_query_seed_namespace() -> None:
    from fastwam.memory.candidate_cache import QueryId
    from fastwam.memory.manifest import sha256_canonical_json
    from fastwam.memory.online_retrieval import derive_online_query_seed

    namespace = "warm-rmbench-full-v1"
    digest = sha256_canonical_json({"evaluation_namespace": namespace})
    query = QueryId(
        f"warm-online/rmbench/rmbench/{digest}",
        2,
        4,
        20,
    )
    assert derive_rmbench_policy_query_seed(
        17,
        task_name="put_back_block",
        episode_index=4,
        frame_index=20,
        evaluation_namespace=namespace,
    ) == derive_online_query_seed(17, query, digest)
    assert derive_rmbench_policy_query_seed(
        17,
        task_name="put_back_block",
        episode_index=4,
        frame_index=20,
        evaluation_namespace=namespace,
    ) != derive_rmbench_policy_query_seed(
        17,
        task_name="put_back_block",
        episode_index=4,
        frame_index=30,
        evaluation_namespace=namespace,
    )


def test_result_discovery_requires_exactly_one_changed_file(tmp_path: Path) -> None:
    base = tmp_path / "results"
    before = snapshot_result_files(base)
    first, _ = _write_result_fixture(base / "one", successes=1)
    assert discover_new_result_file(base, before) == first.resolve()

    before = snapshot_result_files(base)
    _write_result_fixture(base / "two", successes=2)
    _write_result_fixture(base / "three", successes=3)
    with pytest.raises(RuntimeError, match="found 2"):
        discover_new_result_file(base, before)


def test_sim_rmbench_config_carries_the_pinned_protocol() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "sim_rmbench.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    evaluation = cfg["EVALUATION"]
    assert evaluation["code_revision"] == RMBENCH_CODE_REVISION
    assert evaluation["hf_dataset_revision"] == RMBENCH_HF_DATASET_REVISION
    assert evaluation["task_manifest_sha256"] == RMBENCH_TASK_MANIFEST_SHA256
    assert evaluation["task_config"] == "demo_clean"
    assert evaluation["eval_num_episodes"] == 100
    assert evaluation["suite"] == "official9"
    assert cfg["MULTIRUN"]["max_tasks_per_gpu"] == 1
