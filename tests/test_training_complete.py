from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from fastwam.training_complete import (
    TRAINING_COMPLETE_SCHEMA,
    is_training_complete,
    latest_training_state,
    write_training_complete_marker,
)

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "real"))
import train_piper_warm as piper_train  # noqa: E402


def test_write_training_complete_marker_is_atomic_and_readable(tmp_path: Path) -> None:
    marker = write_training_complete_marker(
        tmp_path,
        step=400,
        weights_path="/tmp/weights/step_000400.pt",
        state_path="/tmp/state/step_000400",
        reason="max_steps reached",
    )
    assert marker.name == "training_complete.json"
    assert is_training_complete(tmp_path)
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["schema"] == TRAINING_COMPLETE_SCHEMA
    assert payload["step"] == 400
    assert payload["reason"] == "max_steps reached"
    leftovers = list(tmp_path.glob(".training_complete.json.*.tmp"))
    assert leftovers == []


def test_is_training_complete_rejects_missing_or_invalid(tmp_path: Path) -> None:
    assert is_training_complete(tmp_path) is False
    (tmp_path / "training_complete.json").write_text("{}\n", encoding="utf-8")
    assert is_training_complete(tmp_path) is False
    (tmp_path / "training_complete.json").write_text("not-json", encoding="utf-8")
    assert is_training_complete(tmp_path) is False


def test_latest_training_state_picks_highest_step(tmp_path: Path) -> None:
    state = tmp_path / "checkpoints" / "state"
    (state / "step_000040").mkdir(parents=True)
    (state / "step_000400").mkdir()
    (state / "step_000200").mkdir()
    (state / "not_a_step").mkdir()
    (state / "step_oops").write_text("x", encoding="utf-8")
    assert latest_training_state(tmp_path) == state / "step_000400"
    assert latest_training_state(tmp_path / "empty") is None


def test_resume_state_dir_accepts_run_dir_or_step_dir(tmp_path: Path) -> None:
    step = tmp_path / "checkpoints" / "state" / "step_000400"
    step.mkdir(parents=True)
    resolved = piper_train._resume_state_dir(OmegaConf.create({"resume": str(tmp_path)}))
    assert resolved == step
    resolved_step = piper_train._resume_state_dir(
        OmegaConf.create({"resume": str(step)})
    )
    assert resolved_step == step
    assert piper_train._resume_state_dir(OmegaConf.create({"resume": None})) is None


def test_apply_cli_overrides_sets_resume(tmp_path: Path) -> None:
    cfg = OmegaConf.create({"output_dir": str(tmp_path / "out"), "resume": None})
    args = SimpleNamespace(
        run_steps=None,
        num_epochs=None,
        output_dir=None,
        save_every=None,
        eval_every=None,
        num_workers=None,
        resume=tmp_path / "state" / "step_000400",
    )
    (args.resume).mkdir(parents=True)
    piper_train._apply_cli_overrides(cfg, args)
    assert Path(str(cfg.resume)) == args.resume.resolve()


def test_assert_fresh_output_dir_skipped_when_resume_set(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "config.yaml").write_text("seed: 1\n", encoding="utf-8")
    (output / "training_metrics.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(piper_train, "_is_launch_main_process", lambda: True)
    try:
        piper_train._assert_fresh_output_dir(output)
    except RuntimeError as error:
        assert "already has a WARM run" in str(error)
    else:
        raise AssertionError("expected leftover in-progress run")
    step = output / "checkpoints" / "state" / "step_000400"
    step.mkdir(parents=True)
    cfg = OmegaConf.create({"resume": str(step), "output_dir": str(output)})
    assert piper_train._resume_state_dir(cfg) == step


def test_acp_helpers_select_state_and_reject_false_success(tmp_path: Path) -> None:
    helper = Path(__file__).resolve().parents[1] / "scripts" / "real" / "_acp_log.sh"
    run_dir = tmp_path / "run_20260921_135445"
    step = run_dir / "checkpoints" / "state" / "step_000400"
    step.mkdir(parents=True)
    script = f"""
set -euo pipefail
source "{helper}"
found="$(piper_latest_training_state "{run_dir}")"
test "${{found}}" = "{step}"
if piper_acp_require_training_complete "{run_dir}" 0; then
  echo "expected missing complete marker to fail"
  exit 2
fi
piper_acp_require_training_complete "{run_dir}" 0 || test "$?" = "1"
if piper_acp_require_training_complete "{run_dir}" 137; then
  echo "expected nonzero launch rc to fail"
  exit 2
fi
touch "{run_dir}/training_complete.json"
piper_acp_require_training_complete "{run_dir}" 0
"""
    subprocess.run(["bash", "-c", script], check=True)
