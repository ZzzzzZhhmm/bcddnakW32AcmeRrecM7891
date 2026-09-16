"""CPU-only checks of the handoff boundary, including failure cases."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fastwam.real.episodes import audit_episode, read_json, read_records, EpisodeWriter, ContractError
from fastwam.real.synthetic import make_episode
from fastwam.real.shadow import replay


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "episode"
        make_episode(self.root)

    def rewrite_rows(self, name, rows):
        (self.root / name).write_text("".join(json.dumps(x)+"\n" for x in rows), encoding="utf-8")

    def mutate_observation(self, apply):
        rows = read_records(self.root / "observations.jsonl")
        apply(rows)
        self.rewrite_rows("observations.jsonl", rows)

    def check_rejected(self, expected):
        report = audit_episode(self.root, True)
        self.assertFalse(report["ok"])
        self.assertIn(expected, str(report["errors"]))

    def test_complete_horizon_and_synthetic_separation(self):
        self.assertFalse(audit_episode(self.root)["ok"])
        report = audit_episode(self.root, True)
        self.assertTrue(report["ok"])
        self.assertEqual(report["complete_h32_windows"], 5)

    def test_missing_terminal_observation(self):
        self.mutate_observation(lambda rows: rows.pop())
        self.check_rejected("N+1")

    def test_stale_camera(self):
        self.mutate_observation(lambda rows: rows[2]["cameras"]["wrist"].update(timestamp_ns=0))
        self.check_rejected("stale sensor")

    def test_reversed_time(self):
        self.mutate_observation(lambda rows: rows[1].update(timestamp_ns=rows[0]["timestamp_ns"]))
        self.check_rejected("time reversed")

    def test_nan_robot_state(self):
        self.mutate_observation(lambda rows: rows[0]["robot"].update(gripper_width_m=float("nan")))
        self.check_rejected("non-finite")

    def test_bad_quaternion(self):
        self.mutate_observation(lambda rows: rows[0]["robot"].update(tcp_quaternion_xyzw=[0.,0.,0.,0.]))
        self.check_rejected("unit norm")

    def test_path_traversal(self):
        self.mutate_observation(lambda rows: rows[0]["cameras"]["external"].update(path="../outside.png"))
        self.check_rejected("traversal")

    def test_wrong_command_timing(self):
        rows = read_records(self.root / "commands.jsonl")
        rows[0]["timestamp_ns"] = 0
        self.rewrite_rows("commands.jsonl", rows)
        self.check_rejected("outside observation interval")

    def test_rejected_command_cannot_be_positive_training(self):
        rows = read_records(self.root / "commands.jsonl")
        rows[0]["accepted"] = False
        self.rewrite_rows("commands.jsonl", rows)
        self.check_rejected("rejected command")

    def test_episode_no_overwrite(self):
        with self.assertRaises(FileExistsError):
            EpisodeWriter(self.root, read_json(self.root / "episode.json"))

    def test_mock_replay_marks_evidence_and_preserves_output(self):
        output = Path(self.temp.name) / "shadow.json"
        result = replay(self.root, output, mock=True)
        self.assertEqual(result["calls"], 9)
        self.assertEqual(result["qualification"], "format_only")
        self.assertFalse(result["robot_commands_sent_by_harness"])
        with self.assertRaises(ContractError):
            replay(self.root, output, mock=True)

    def real_descriptor(self):
        meta = read_json(self.root / "episode.json")
        meta["synthetic"] = False  # Test fixture only; never a real-data claim.
        (self.root / "episode.json").write_text(json.dumps(meta), encoding="utf-8")
        release = Path(self.temp.name) / "release.json"
        release.write_text(json.dumps(dict(schema="warm.real.shadow-release.v1", action_shape=[32,7],
                              action_space="real_profile_normalized", execution_prefix=4, release_id="test")))
        return release

    def test_teacher_forcing_exposes_only_past_transitions(self):
        release = self.real_descriptor()
        self.mutate_observation(lambda rows: rows[0].update(hidden_answer="must-not-leak"))
        test = self
        class Backend:
            last = 0
            def reset(self, context):
                test.assertNotIn("split", context)
                test.assertNotIn("outcome", context)
            def observe_executed(self, command, successor):
                test.assertEqual(command["seq"], self.last)
                self.last += 1
                test.assertEqual(successor["seq"], self.last)
            def predict(self, obs):
                test.assertEqual(obs["seq"], self.last)
                test.assertNotIn("hidden_answer", obs)
                return [[0.]*7 for _ in range(32)]
            def synchronize(self):
                pass
        class Module:
            @staticmethod
            def create(path):
                return Backend()
        with patch("fastwam.real.shadow.importlib.import_module", return_value=Module):
            result = replay(self.root, Path(self.temp.name)/"real.json", backend_spec="test:create", release=release)
        self.assertEqual(result["qualification"], "recorded_observation_inference_only")

    def test_invalid_prediction_rejected(self):
        release = self.real_descriptor()
        class Backend:
            def reset(self, context): pass
            def synchronize(self): pass
            def predict(self, obs): return [[float("inf")]*7 for _ in range(32)]
        class Module:
            @staticmethod
            def create(path): return Backend()
        with patch("fastwam.real.shadow.importlib.import_module", return_value=Module):
            with self.assertRaises(ContractError):
                replay(self.root, Path(self.temp.name)/"bad.json", backend_spec="test:create", release=release)


if __name__ == "__main__":
    unittest.main()
