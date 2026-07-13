from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from fastwam.memory.manifest import sha256_canonical_json, sha256_file
from fastwam.memory.online_retrieval import (
    derive_online_query_seed,
    make_online_query_id,
)
from fastwam.models.warm.online_contract import WarmOnlineRunContract
from fastwam.models.warm.online_pair_contract import (
    ALLOWED_CONFIG_DIFFERENCE_PATHS,
    WarmOnlinePairContract,
)
from scripts.verify_warm_online_pair_results import (
    OnlinePairResultsVerificationError,
    _derive_episode_simulator_seed,
    _shared_science_identity,
    main,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _online_contract(policy: str) -> WarmOnlineRunContract:
    suffix = "fixed" if policy == "fixed_context_top1" else "null"
    return WarmOnlineRunContract(
        training_run_contract_sha256=_digest("train-contract"),
        validation_run_contract_sha256=_digest("dev-contract"),
        m1_data_config_sha256=_digest("m1-data-config"),
        warm_checkpoint_sha256=_digest(f"checkpoint:{suffix}"),
        training_attestation_sha256=_digest(f"training-attestation:{suffix}"),
        shared_training_recipe_sha256=_digest("shared-training-recipe"),
        training_runtime_sha256=_digest("shared-training-runtime"),
        bank_manifest_sha256=_digest("bank-manifest"),
        bank_content_sha256=_digest("bank-content"),
        encoder_contract_sha256=_digest("encoder"),
        encoder_runtime_sha256=_digest("encoder-runtime"),
        camera_contract_sha256=_digest("camera"),
        dino_checkpoint_tree_sha256=_digest("dino"),
        dino_checkpoint_file_count=3,
        normalization_stats_sha256=_digest("stats"),
        action_space_contract_sha256=_digest("action-space"),
        catalog_sha256=_digest("catalog"),
        audit_sha256=_digest("audit"),
        resolved_eval_config_sha256=_digest(f"config:{suffix}"),
        vae_checkpoint_sha256=_digest("vae"),
        text_encoder_tree_sha256=_digest("text"),
        tokenizer_tree_sha256=_digest("tokenizer"),
        evaluation_namespace_sha256=_digest("eval-namespace"),
        task_suite="libero_goal",
        task_id=7,
        task_description="open the middle drawer",
        root_seed=17,
        initial_states_sha256=_digest("initial-states"),
        bddl_sha256=_digest("bddl"),
        retrieval_implementation="exact_cosine_frozen_dino_v1",
        top_k=2,
        source_policy=policy,
        memory_sigma=0.2,
        action_horizon=16,
        action_dim=7,
        git_commit="a" * 40,
        git_dirty=False,
    )


def _pair_contract(
    fixed: WarmOnlineRunContract, null: WarmOnlineRunContract
) -> WarmOnlinePairContract:
    return WarmOnlinePairContract(
        fixed_online_run_contract_sha256=fixed.sha256,
        gaussian_null_online_run_contract_sha256=null.sha256,
        fixed_resolved_eval_config_sha256=fixed.resolved_eval_config_sha256,
        gaussian_null_resolved_eval_config_sha256=(
            null.resolved_eval_config_sha256
        ),
        fixed_warm_checkpoint_sha256=fixed.warm_checkpoint_sha256,
        gaussian_null_warm_checkpoint_sha256=null.warm_checkpoint_sha256,
        fixed_training_attestation_sha256=fixed.training_attestation_sha256,
        gaussian_null_training_attestation_sha256=(
            null.training_attestation_sha256
        ),
        shared_training_recipe_sha256=fixed.shared_training_recipe_sha256,
        shared_training_runtime_sha256=fixed.training_runtime_sha256,
        parity_report_sha256=_digest("passing-parity-report"),
        shared_science_identity_sha256=sha256_canonical_json(
            _shared_science_identity(fixed)
        ),
        allowed_config_difference_paths=ALLOWED_CONFIG_DIFFERENCE_PATHS,
        observed_config_difference_paths=ALLOWED_CONFIG_DIFFERENCE_PATHS,
        git_commit="a" * 40,
        git_dirty=False,
    )


def _pair_identity(pair: WarmOnlinePairContract, side: str) -> dict[str, str]:
    return {
        "pair_contract_sha256": pair.sha256,
        "comparison_kind": pair.comparison_kind,
        "side": side,
    }


def _event_id(frame: int) -> dict[str, object]:
    return {
        "dataset_id": "libero_goal_no_noops_lerobot",
        "dataset_index": 2,
        "episode_index": 11,
        "start_frame": frame,
    }


def _replan(
    *,
    contract: WarmOnlineRunContract,
    pair: WarmOnlinePairContract,
    side: str,
    episode_index: int,
    replan_index: int,
    frame_index: int,
) -> dict[str, object]:
    query = make_online_query_id(contract, episode_index, frame_index)
    query_value = {
        "dataset_id": query.dataset_id,
        "dataset_index": query.dataset_index,
        "episode_index": query.episode_index,
        "frame_index": query.frame_index,
    }
    seed = derive_online_query_seed(
        contract.root_seed, query, contract.evaluation_namespace_sha256
    )
    prompt_sha = _digest("prompt")
    evidence_side = "shared-pre-policy" if replan_index == 0 else side
    proprio_sha = _digest(
        f"proprio:{evidence_side}:{episode_index}:{replan_index}"
    )
    model_input_sha = _digest(
        f"model-input:{evidence_side}:{episode_index}:{replan_index}"
    )
    common: dict[str, object] = {
        **_pair_identity(pair, side),
        "query_id": query_value,
        "absolute_sim_step": frame_index,
        "source_policy": contract.source_policy,
        "derived_seed": seed,
        "raw_camera_sha256": {
            "image": _digest(
                f"raw-main:{evidence_side}:{episode_index}:{replan_index}"
            ),
            "wrist_image": _digest(
                f"raw-wrist:{evidence_side}:{episode_index}:{replan_index}"
            ),
        },
        "processed_camera_sha256": {
            "image": _digest(
                f"processed-main:{evidence_side}:{episode_index}:{replan_index}"
            ),
            "wrist_image": _digest(
                f"processed-wrist:{evidence_side}:{episode_index}:{replan_index}"
            ),
        },
        "prompt_sha256": prompt_sha,
        "proprio_sha256": proprio_sha,
        "model_input_sha256": model_input_sha,
        "evaluator_latency_s": {
            "input_prepare_or_retrieval_s": 0.01,
            "model_inference_s": 0.02,
            "online_pipeline_s": 0.04,
        },
        "replan_index": replan_index,
    }
    source: dict[str, object]
    if side == "gaussian_null":
        common.update(
            {
                "bound_step_sha256": None,
                "context_key_sha256": None,
                "candidate_payload_sha256": None,
                "candidates": [],
            }
        )
        source = {
            "policy": "gaussian_null",
            "component": 0,
            "selected_rank": None,
            "selected_event_id": None,
            "memory_selected": False,
            "memory_sigma": 0.2,
            "derived_seed": seed,
        }
        common["model"] = {"retrieval": None, "source": source}
        return common

    step_sha = _digest(f"step:{episode_index}:{replan_index}")
    selected_event = _event_id(frame_index)
    candidates = [
        {
            "rank": 0,
            "valid": True,
            "bank_row": 19,
            "event_id": selected_event,
            "cosine_score": 0.75,
        },
        {
            "rank": 1,
            "valid": False,
            "bank_row": -1,
            "event_id": None,
            "cosine_score": 0.0,
        },
    ]
    common.update(
        {
            "bound_step_sha256": step_sha,
            "context_key_sha256": _digest(f"context:{episode_index}:{replan_index}"),
            "candidate_payload_sha256": _digest(
                f"payload:{episode_index}:{replan_index}"
            ),
            "candidates": candidates,
        }
    )
    retrieval = {
        "query_id": query_value,
        "online_contract_sha256": contract.sha256,
        "training_run_contract_sha256": contract.training_run_contract_sha256,
        "validation_run_contract_sha256": contract.validation_run_contract_sha256,
        "bank_manifest_sha256": contract.bank_manifest_sha256,
        "bank_content_sha256": contract.bank_content_sha256,
        "step_sha256": step_sha,
        "prompt_sha256": prompt_sha,
        "proprio_sha256": proprio_sha,
        "model_input_sha256": model_input_sha,
        "ranked_event_ids": [selected_event, None],
        "bank_rows": [19, -1],
        "cosine_scores": [0.75, 0.0],
        "candidate_valid_mask": [True, False],
        "latency_s": {
            "preprocess_s": 0.001,
            "dino_s": 0.002,
            "search_s": 0.003,
            "gather_s": 0.004,
            "total_s": 0.011,
        },
    }
    source = {
        "policy": "fixed_context_top1",
        "component": 1,
        "selected_rank": 0,
        "selected_event_id": selected_event,
        "memory_selected": True,
        "memory_sigma": 0.2,
        "derived_seed": seed,
    }
    common["model"] = {"retrieval": retrieval, "source": source}
    return common


def _result(
    *,
    contract: WarmOnlineRunContract,
    pair: WarmOnlinePairContract,
    pair_file_sha256: str,
    side: str,
    replan_counts: tuple[int, ...] = (2, 80),
) -> dict[str, object]:
    episodes = []
    success_indices = [0]
    failure_indices = [1]
    for episode_index, count in enumerate(replan_counts):
        success = episode_index == 0
        wait_steps = 5
        policy_max_steps = 400
        replan_steps = 5
        if success:
            last_replan_frame = wait_steps + replan_steps * (count - 1)
            policy_steps = last_replan_frame - wait_steps + 1
            environment_steps = wait_steps + policy_steps
            final_frame_index = environment_steps - 1
        else:
            policy_steps = policy_max_steps
            environment_steps = wait_steps + policy_steps
            final_frame_index = environment_steps
        episodes.append(
            {
                **_pair_identity(pair, side),
                "episode_index": episode_index,
                "success": success,
                "simulator_seed": _derive_episode_simulator_seed(
                    contract.root_seed,
                    contract.task_suite,
                    contract.task_id,
                    episode_index,
                ),
                "termination_reason": "success" if success else "max_steps",
                "final_frame_index": final_frame_index,
                "environment_step_count": environment_steps,
                "policy_action_step_count": policy_steps,
                "configured_wait_steps": wait_steps,
                "configured_policy_max_steps": policy_max_steps,
                "configured_replan_steps": replan_steps,
                "replan_count": count,
                "replans": [
                    _replan(
                        contract=contract,
                        pair=pair,
                        side=side,
                        episode_index=episode_index,
                        replan_index=index,
                        frame_index=wait_steps + replan_steps * index,
                    )
                    for index in range(count)
                ],
            }
        )
    header = {
        "schema": "warm.libero-online-evaluation-header",
        "version": 2,
        **_pair_identity(pair, side),
        "online_run_contract_sha256": contract.sha256,
        "training_run_contract_sha256": contract.training_run_contract_sha256,
        "validation_run_contract_sha256": contract.validation_run_contract_sha256,
        "source_policy": contract.source_policy,
        "runtime_attestation": {
            "git_commit": contract.git_commit,
            "git_dirty": False,
            "normalization_stats_loaded_sha256": (
                contract.normalization_stats_sha256
            ),
            "pair_contract_file_sha256": pair_file_sha256,
            "parity_report_file_sha256": pair.parity_report_sha256,
            "m1_data_config_sha256": contract.m1_data_config_sha256,
            "training_attestation_file_sha256": (
                contract.training_attestation_sha256
            ),
            "shared_training_recipe_sha256": (
                pair.shared_training_recipe_sha256
            ),
            "training_runtime_sha256": pair.shared_training_runtime_sha256,
            "model_loaded_checkpoint_sha256": contract.warm_checkpoint_sha256,
            "bddl_sha256_after_environment_load": contract.bddl_sha256,
        },
        "contract": contract.to_dict(),
        "pair_contract": pair.to_dict(),
    }
    return {
        "task_suite": contract.task_suite,
        "task_id": contract.task_id,
        "task_description": contract.task_description,
        "successes": 1,
        "total_episodes": 2,
        "gpu_id": 0,
        "success_episodes": success_indices,
        "failure_episodes": failure_indices,
        "start_time": "2026-07-13 10:00:00",
        "duration": 12.5,
        "warm_online_header": header,
        "warm_online_episodes": episodes,
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> dict[str, object]:
    fixed = _online_contract("fixed_context_top1")
    null = _online_contract("gaussian_null")
    pair = _pair_contract(fixed, null)
    pair_path = tmp_path / "contracts" / "pair.json"
    _write_json(pair_path, pair.to_dict())
    pair_file_sha = sha256_file(pair_path)
    fixed_path = tmp_path / "fixed" / "libero_goal" / "gpu0_task7_results.json"
    null_path = tmp_path / "null" / "libero_goal" / "gpu0_task7_results.json"
    fixed_value = _result(
        contract=fixed,
        pair=pair,
        pair_file_sha256=pair_file_sha,
        side="fixed",
        replan_counts=(2, 80),
    )
    null_value = _result(
        contract=null,
        pair=pair,
        pair_file_sha256=pair_file_sha,
        side="gaussian_null",
        replan_counts=(3, 80),
    )
    _write_json(fixed_path, fixed_value)
    _write_json(null_path, null_value)
    return {
        "fixed_contract": fixed,
        "null_contract": null,
        "pair": pair,
        "pair_path": pair_path,
        "fixed_path": fixed_path,
        "null_path": null_path,
        "fixed_value": fixed_value,
        "null_value": null_value,
    }


def _argv(fixture: dict[str, object], output: Path, *, directories: bool = False) -> list[str]:
    fixed_path = Path(fixture["fixed_path"])
    null_path = Path(fixture["null_path"])
    return [
        "--pair-contract",
        str(fixture["pair_path"]),
        "--fixed-results",
        str(fixed_path.parents[1] if directories else fixed_path),
        "--gaussian-null-results",
        str(null_path.parents[1] if directories else null_path),
        "--output",
        str(output),
    ]


def test_verifier_accepts_real_schema_directories_and_records_prefix_scope(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "reports" / "verified.json"
    assert main(_argv(fixture, output, directories=True)) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "verified"
    assert report["trajectory_equivalence_claim"] is False
    assert report["pairing_scope"] == (
        "first_pre_policy_inputs_and_common_query_id_seed_replan_prefix"
    )
    assert report["bound_step_verification_scope"] == (
        "opaque_emitter_digest_and_model_retrieval_equality_only"
    )
    assert report["task_episode_key_count"] == 2
    assert report["parity_report_sha256"] == fixture["pair"].parity_report_sha256
    assert report["fixed"]["task_artifact_identities"][0][
        "warm_checkpoint_sha256"
    ] == fixture["fixed_contract"].warm_checkpoint_sha256
    first = report["episodes"][0]
    assert first["fixed_replan_count"] == 2
    assert first["gaussian_null_replan_count"] == 3
    assert first["common_replan_prefix_count"] == 2
    assert first["replan_length_relation"] == "fixed_prefix_ended_first"
    assert first["trajectory_equivalence_claim"] is False


@pytest.mark.parametrize(
    ("side", "mutate", "match"),
    [
        (
            "null",
            lambda value: value["warm_online_episodes"][0]["replans"][0].update(
                {"bound_step_sha256": _digest("forbidden")}
            ),
            "forbidden Gaussian-null memory-read telemetry",
        ),
        (
            "fixed",
            lambda value: value["warm_online_episodes"][0]["replans"][0].update(
                {"candidate_payload_sha256": None}
            ),
            "candidate_payload_sha256",
        ),
        (
            "fixed",
            lambda value: value["warm_online_episodes"][0]["replans"][0][
                "model"
            ].update({"retrieval": None}),
            "model.retrieval must be an object",
        ),
    ],
)
def test_verifier_rejects_null_memory_reads_and_missing_fixed_capability_evidence(
    tmp_path: Path,
    side: str,
    mutate,
    match: str,
) -> None:
    fixture = _fixture(tmp_path)
    value = deepcopy(fixture[f"{side}_value"])
    mutate(value)
    _write_json(Path(fixture[f"{side}_path"]), value)
    output = tmp_path / "report.json"
    with pytest.raises(OnlinePairResultsVerificationError, match=match):
        main(_argv(fixture, output))
    assert not output.exists()


def test_verifier_rejects_off_schedule_replan_frame(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    null_value = deepcopy(fixture["null_value"])
    replan = null_value["warm_online_episodes"][0]["replans"][1]
    # Recompute all query-derived evidence so only the episode schedule check
    # catches the off-cadence replan.
    replan["query_id"]["frame_index"] = 11
    replan["absolute_sim_step"] = 11
    null_contract = fixture["null_contract"]
    query = make_online_query_id(null_contract, 0, 11)
    replan["derived_seed"] = derive_online_query_seed(
        null_contract.root_seed,
        query,
        null_contract.evaluation_namespace_sha256,
    )
    replan["model"]["source"]["derived_seed"] = replan["derived_seed"]
    _write_json(Path(fixture["null_path"]), null_value)
    output = tmp_path / "report.json"
    with pytest.raises(OnlinePairResultsVerificationError, match="replan frames"):
        main(_argv(fixture, output))
    assert not output.exists()


def test_verifier_rejects_replan_count_inconsistent_with_policy_steps(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    null_value = deepcopy(fixture["null_value"])
    null_value["warm_online_episodes"][0]["configured_replan_steps"] = 6
    _write_json(Path(fixture["null_path"]), null_value)
    output = tmp_path / "report.json"
    with pytest.raises(
        OnlinePairResultsVerificationError,
        match="replan_count does not match policy/replan steps",
    ):
        main(_argv(fixture, output))
    assert not output.exists()


def test_verifier_rejects_first_pre_policy_input_mismatch(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    null_value = deepcopy(fixture["null_value"])
    null_value["warm_online_episodes"][0]["replans"][0][
        "raw_camera_sha256"
    ]["image"] = _digest("different-initial-observation")
    _write_json(Path(fixture["null_path"]), null_value)
    output = tmp_path / "report.json"
    with pytest.raises(
        OnlinePairResultsVerificationError,
        match="first pre-policy observation evidence differs",
    ):
        main(_argv(fixture, output))
    assert not output.exists()


def test_verifier_rejects_only_one_side_reaching_policy_phase(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    null_value = deepcopy(fixture["null_value"])
    episode = null_value["warm_online_episodes"][0]
    episode["environment_step_count"] = 1
    episode["policy_action_step_count"] = 0
    episode["final_frame_index"] = 1
    episode["replan_count"] = 0
    episode["replans"] = []
    _write_json(Path(fixture["null_path"]), null_value)
    output = tmp_path / "report.json"
    with pytest.raises(
        OnlinePairResultsVerificationError,
        match="exactly one policy records a pre-policy replan",
    ):
        main(_argv(fixture, output))
    assert not output.exists()


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda candidates: candidates[0].update({"cosine_score": 1.01}),
            r"outside \[-1, 1\]",
        ),
        (
            lambda candidates: candidates[1].update(
                {
                    "valid": True,
                    "bank_row": 20,
                    "event_id": _event_id(20),
                    "cosine_score": 0.80,
                }
            ),
            "not non-increasing",
        ),
        (
            lambda candidates: candidates[1].update(
                {
                    "valid": True,
                    "bank_row": 19,
                    "event_id": _event_id(20),
                    "cosine_score": 0.50,
                }
            ),
            "duplicate bank row",
        ),
    ],
)
def test_verifier_rejects_noncanonical_candidate_ranking(
    tmp_path: Path,
    mutate,
    match: str,
) -> None:
    fixture = _fixture(tmp_path)
    fixed_value = deepcopy(fixture["fixed_value"])
    candidates = fixed_value["warm_online_episodes"][0]["replans"][0]["candidates"]
    mutate(candidates)
    _write_json(Path(fixture["fixed_path"]), fixed_value)
    output = tmp_path / "report.json"
    with pytest.raises(OnlinePairResultsVerificationError, match=match):
        main(_argv(fixture, output))
    assert not output.exists()


def test_verifier_rejects_incomplete_or_duplicate_episode_evidence(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    null_value = deepcopy(fixture["null_value"])
    null_value["warm_online_episodes"][1]["episode_index"] = 0
    _write_json(Path(fixture["null_path"]), null_value)
    output = tmp_path / "report.json"
    with pytest.raises(OnlinePairResultsVerificationError, match="success disagrees|duplicate"):
        main(_argv(fixture, output))
    assert not output.exists()


def test_verifier_rejects_episode_seed_and_counter_fabrication(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    null_value = deepcopy(fixture["null_value"])
    null_value["warm_online_episodes"][0]["environment_step_count"] += 1
    _write_json(Path(fixture["null_path"]), null_value)
    output = tmp_path / "report.json"
    with pytest.raises(
        OnlinePairResultsVerificationError,
        match="counters are inconsistent|step counts are inconsistent",
    ):
        main(_argv(fixture, output))
    assert not output.exists()


def test_verifier_rejects_cross_policy_episode_limit_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    null_value = deepcopy(fixture["null_value"])
    episode = null_value["warm_online_episodes"][0]
    episode["configured_wait_steps"] = 6
    episode["environment_step_count"] += 1
    episode["final_frame_index"] += 1
    for replan in episode["replans"]:
        replan["query_id"]["frame_index"] += 1
        replan["absolute_sim_step"] += 1
        null_contract = fixture["null_contract"]
        query = make_online_query_id(
            null_contract,
            episode["episode_index"],
            replan["query_id"]["frame_index"],
        )
        replan["derived_seed"] = derive_online_query_seed(
            null_contract.root_seed,
            query,
            null_contract.evaluation_namespace_sha256,
        )
        replan["model"]["source"]["derived_seed"] = replan["derived_seed"]
    _write_json(Path(fixture["null_path"]), null_value)
    output = tmp_path / "report.json"
    with pytest.raises(
        OnlinePairResultsVerificationError,
        match="configured episode limits",
    ):
        main(_argv(fixture, output))
    assert not output.exists()


def test_verifier_rejects_different_complete_task_episode_key_sets(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    null_value = deepcopy(fixture["null_value"])
    null_value["total_episodes"] = 1
    null_value["successes"] = 1
    null_value["success_episodes"] = [0]
    null_value["failure_episodes"] = []
    null_value["warm_online_episodes"] = null_value["warm_online_episodes"][:1]
    _write_json(Path(fixture["null_path"]), null_value)
    output = tmp_path / "report.json"
    with pytest.raises(OnlinePairResultsVerificationError, match="task/episode key sets"):
        main(_argv(fixture, output))
    assert not output.exists()


def test_verifier_rejects_policy_specific_header_artifact_mismatch(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixed_value = deepcopy(fixture["fixed_value"])
    fixed_value["warm_online_header"]["contract"][
        "resolved_eval_config_sha256"
    ] = _digest("wrong-config")
    _write_json(Path(fixture["fixed_path"]), fixed_value)
    output = tmp_path / "report.json"
    with pytest.raises(OnlinePairResultsVerificationError, match="identity mismatch"):
        main(_argv(fixture, output))
    assert not output.exists()


def test_verifier_rejects_duplicate_json_keys_without_writing_report(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixed_path = Path(fixture["fixed_path"])
    fixed_path.write_text('{"task_suite":"a","task_suite":"b"}\n', encoding="utf-8")
    output = tmp_path / "report.json"
    with pytest.raises(OnlinePairResultsVerificationError, match="duplicate key"):
        main(_argv(fixture, output))
    assert not output.exists()
