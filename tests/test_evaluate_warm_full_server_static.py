from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_warm_full_server.sh"


def test_formal_launcher_explicitly_enables_full_online_runtime() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    stable_output = source.index('"output_dir=${WARM_EVAL_ROOT}/unused_train_output"')
    enabled = source.index('"EVALUATION.warm_online.enabled=true"')
    mode = source.index('"EVALUATION.warm_online.mode=full_retrospection"')
    resolve = source.index("--cfg job --resolve")
    contract = source.index("python scripts/build_warm_online_contract.py")
    stability = source.index("resolved_config_stable sha256=")
    rollout = source.rindex('python experiments/libero/eval_libero_single.py "${HYDRA_OVERRIDES[@]}"')
    assert stable_output < enabled < mode < resolve < contract < stability < rollout


def test_formal_launcher_retains_clean_worktree_and_immutable_output_guards() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "formal evaluation requires a clean Git checkout" in source
    assert "immutable evaluation root already exists" in source
    assert 'task=libero_warm_online_2cam224_full' in source
    assert 'HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"' in source
    assert ".resolved_config.runtime-check.yaml" in source
    assert 'contract["resolved_eval_config_sha256"]' in source
