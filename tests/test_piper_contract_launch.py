from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "real"))
import train_piper_warm as piper_train  # noqa: E402


def _bindings(tmp_path: Path, base: Path) -> Path:
    processed = tmp_path / "processed"
    memory = processed / "memory"
    memory.mkdir(parents=True)
    (memory / "event_bank").mkdir()
    (memory / "train_candidates").mkdir()
    (memory / "dev_candidates").mkdir()
    payload = {
        "event_bank": str(memory / "event_bank"),
        "splits": {
            "train": {
                "candidates": str(memory / "train_candidates"),
                "query_corpus_sha256": "a" * 64,
            },
            "dev": {
                "candidates": str(memory / "dev_candidates"),
                "query_corpus_sha256": "b" * 64,
            },
        },
    }
    (memory / "training_bindings.json").write_text(json.dumps(payload), encoding="utf-8")
    base.write_bytes(b"ckpt")
    return processed


def test_build_contracts_non_main_does_not_spawn(tmp_path, monkeypatch) -> None:
    base = tmp_path / "step.pt"
    processed = _bindings(tmp_path, base)
    contract_dir = tmp_path / "contracts"
    contract_dir.mkdir()
    train = contract_dir / "train_source.json"
    dev = contract_dir / "dev_source.json"
    train.write_text("{}", encoding="utf-8")
    dev.write_text("{}", encoding="utf-8")
    (contract_dir / ".source_contracts.ready").write_text(
        str(base.resolve()) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(piper_train, "_is_launch_main_process", lambda: False)
    calls: list[object] = []
    monkeypatch.setattr(
        piper_train.subprocess,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    out = piper_train.build_contracts(
        processed=processed,
        base_checkpoint=base,
        contract_dir=contract_dir,
        overwrite=True,
    )
    assert out == (train, dev)
    assert calls == []


def test_build_contracts_main_strips_dist_env_and_drops_dead_lock(
    tmp_path, monkeypatch
) -> None:
    base = tmp_path / "step.pt"
    processed = _bindings(tmp_path, base)
    contract_dir = tmp_path / "contracts"
    contract_dir.mkdir()
    lock = contract_dir / ".train_source.json.warm-artifact.lock"
    lock.write_text(
        json.dumps(
            {
                "pid": 999999,
                "purpose": "publish WARM source-run contract",
                "schema": "warm.artifact-claim",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(piper_train, "_is_launch_main_process", lambda: True)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("MASTER_PORT", "29500")
    captured: list[dict] = []

    def fake_run(command, cwd=None, check=None, env=None):
        captured.append({"command": command, "env": env})
        output = Path(command[command.index("--output") + 1])
        output.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(piper_train.subprocess, "run", fake_run)
    piper_train.build_contracts(
        processed=processed,
        base_checkpoint=base,
        contract_dir=contract_dir,
        overwrite=True,
    )
    assert len(captured) == 2
    for item in captured:
        assert item["env"] is not None
        assert "RANK" not in item["env"]
        assert "MASTER_PORT" not in item["env"]
    assert not lock.is_file()
    ready = (contract_dir / ".source_contracts.ready").read_text(encoding="utf-8").strip()
    assert ready == str(base.resolve())


def test_build_contracts_distributed_reuses_prepared_files(
    tmp_path, monkeypatch
) -> None:
    base = tmp_path / "step.pt"
    processed = _bindings(tmp_path, base)
    contract_dir = tmp_path / "contracts"
    contract_dir.mkdir()
    train = contract_dir / "train_source.json"
    dev = contract_dir / "dev_source.json"
    train.write_text("{}", encoding="utf-8")
    dev.write_text("{}", encoding="utf-8")
    (contract_dir / ".source_contracts.ready").write_text(
        str(base.resolve()) + "\n", encoding="utf-8"
    )
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "0")
    calls: list[object] = []
    monkeypatch.setattr(
        piper_train.subprocess,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    out = piper_train.build_contracts(
        processed=processed,
        base_checkpoint=base,
        contract_dir=contract_dir,
        overwrite=True,
    )
    assert out == (train, dev)
    assert calls == []


def test_build_contracts_distributed_requires_prepared_files(
    tmp_path, monkeypatch
) -> None:
    base = tmp_path / "step.pt"
    processed = _bindings(tmp_path, base)
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "1")
    try:
        piper_train.build_contracts(
            processed=processed,
            base_checkpoint=base,
            contract_dir=tmp_path / "contracts",
            overwrite=True,
        )
    except FileNotFoundError as error:
        assert "single process" in str(error)
    else:
        raise AssertionError("expected missing prepared contracts")


def test_build_contracts_distributed_requires_ready_marker(
    tmp_path, monkeypatch
) -> None:
    base = tmp_path / "step.pt"
    processed = _bindings(tmp_path, base)
    contract_dir = tmp_path / "contracts"
    contract_dir.mkdir()
    (contract_dir / "train_source.json").write_text("{}", encoding="utf-8")
    (contract_dir / "dev_source.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "1")
    try:
        piper_train.build_contracts(
            processed=processed,
            base_checkpoint=base,
            contract_dir=contract_dir,
            overwrite=False,
            wait_timeout_s=0,
        )
    except RuntimeError as error:
        assert "timed out waiting for source-run contracts" in str(error)
    else:
        raise AssertionError("expected missing ready marker")


def test_prepare_contracts_rejects_world_size(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("WORLD_SIZE", "4")
    try:
        piper_train.main(["--prepare-contracts", "--config", str(tmp_path / "x.yaml")])
    except RuntimeError as error:
        assert "--prepare-contracts" in str(error)
        assert "WORLD_SIZE" in str(error)
    else:
        raise AssertionError("expected leftover WORLD_SIZE")


def test_train_script_does_not_import_runtime_at_module_level() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "real"
        / "train_piper_warm.py"
    ).read_text(encoding="utf-8")
    assert "from fastwam.runtime import run_training" in source
    module_header = source.split("def _resolve_against", 1)[0]
    assert "from fastwam.runtime import run_training" not in module_header
    assert "from fastwam.datasets.lerobot.robot_video_dataset import" not in source
    assert '"--resume"' in source
    assert "from fastwam.training_complete import latest_training_state" in source


def test_assert_fresh_output_dir_skips_non_main_when_config_exists(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "config.yaml").write_text("seed: 1\n", encoding="utf-8")
    monkeypatch.setattr(piper_train, "_is_launch_main_process", lambda: False)
    piper_train._assert_fresh_output_dir(output)


def test_assert_fresh_output_dir_main_refuses_failed_leftover(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "config.yaml").write_text("seed: 1\n", encoding="utf-8")
    monkeypatch.setattr(piper_train, "_is_launch_main_process", lambda: True)
    try:
        piper_train._assert_fresh_output_dir(output)
    except RuntimeError as error:
        assert "leftover failed run" in str(error)
    else:
        raise AssertionError("expected leftover failed run")
