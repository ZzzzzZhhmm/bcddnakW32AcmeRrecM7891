"""CPU integration tests with real HDF5/parquet/MP4 I/O, never robot evidence."""
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fastwam.datasets.robotwin2.source import RoboTwin2Adapter, decode_rgb
from fastwam.preprocessing.config import load_config
from fastwam.preprocessing.contracts import PreparationError, read_json, write_json
from fastwam.preprocessing.features import decode_video, precompute
from fastwam.preprocessing.memory import build_memory
from fastwam.preprocessing.prepare import prepare
from fastwam.real.preprocessing.piper import PiperTeleopAdapter, base_rotation_delta, quaternion_rotvec
from fastwam.real.synthetic import make_episode


REPO = Path(__file__).resolve().parents[1]


def config(tmp_path, kind="piper"):
    template = "configs/real/preprocess_piper.template.json" if kind == "piper" else "configs/preprocessing/robotwin2_legacy.template.json"
    cfg = read_json(REPO / template)
    cfg["test_only"] = True
    cfg["output"] = str(tmp_path / "output")
    cfg["source"]["root"] = str(tmp_path / "raw")
    cfg["source"]["manifest"] = str(tmp_path / "episodes.json")
    cfg["profile"]["action_horizon"] = 4
    cfg["profile"]["action_video_freq_ratio"] = 1
    if kind == "piper":
        cfg["source"].update(allow_synthetic=True, calibration_id="SYNTHETIC_NOT_CALIBRATED",
                             control_frame="synthetic_base", tcp_frame="synthetic_tcp")
    return cfg


def replace_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def real_episode(cfg, name, *, split="train", session=None, offset=0, steps=8):
    root = Path(cfg["source"]["root"]) / name
    make_episode(root, steps=steps)
    meta = read_json(root / "episode.json")
    meta.update(episode_id=name, split=split, session_id=session or name)
    replace_json(root / "episode.json", meta)
    # Distinct factual visuals and commands, not merely renamed episodes.
    for image in root.rglob("*.ppm"):
        array = np.asarray(Image.open(image), dtype=np.uint8).copy()
        array[..., 0] += offset
        Image.fromarray(array).save(image)
    commands = [json.loads(row) for row in (root / "commands.jsonl").read_text().splitlines()]
    for i, row in enumerate(commands):
        row["tcp_position_m"][0] += offset * 0.001 + i * 0.0001
    (root / "commands.jsonl").write_text("\n".join(json.dumps(row) for row in commands) + "\n")
    return {"id": name, "path": name, "split": split}


def manifest(cfg, entries):
    write_json(Path(cfg["source"]["manifest"]), {"schema": "warm.source-episodes.v1", "episodes": entries})


def jpeg(rgb, *, marked=False):
    output = BytesIO()
    Image.fromarray(rgb if marked else rgb[..., ::-1]).save(output, format="JPEG", quality=100, subsampling=0)
    raw = output.getvalue()
    if marked:
        raw = raw[:2] + b"\xff\xfe\x00\x0aXPL-RGB1" + raw[2:]
    return raw


def robotwin_episode(cfg, name, *, split="train", offset=0, native=False):
    import h5py
    root = Path(cfg["source"]["root"])
    root.mkdir(parents=True, exist_ok=True)
    qpos = (np.arange(10 * 14).reshape(10, 14) * 0.001 + offset).astype(np.float32)
    rgb = np.zeros((10, 8, 8, 3), dtype=np.uint8)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = int(30 + offset * 30), 60, 220
    rgb[:, :, :, 0] += np.arange(10, dtype=np.uint8)[:, None, None]
    with h5py.File(root / (name + ".hdf5"), "w") as h5:
        if not native:
            h5.create_dataset("joint_action/vector", data=qpos)
            for key in ("head_camera", "left_camera", "right_camera"):
                cells = [jpeg(frame) for frame in rgb]
                h5.create_dataset(f"observation/{key}/rgb", data=np.array(cells, dtype=f"S{max(map(len, cells))}"))
        else:
            h5.create_dataset("data_format_version", data="v1.0")
            h5.create_dataset("additional_info/frequency", data=15)
            h5.create_dataset("instructions", data=json.dumps(["Put the object back."]))
            fields = [("left_arm_joint_states", slice(0, 6)), ("left_ee_joint_states", slice(6, 7)),
                      ("right_arm_joint_states", slice(7, 13)), ("right_ee_joint_states", slice(13, 14))]
            for key, indices in fields:
                h5.create_dataset("state/" + key, data=qpos[:-1, indices])
                h5.create_dataset("action/" + key, data=qpos[1:, indices])
            for key in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
                cells = [jpeg(frame, marked=True) for frame in rgb[:-1]]
                h5.create_dataset(f"vision/{key}/colors", data=np.array(cells, dtype=f"S{max(map(len, cells))}"))
    write_json(root / (name + ".json"), {"seen": ["Put the object back."], "unseen": ["Restore it."]})
    return {"id": name, "task": "put_back", "split": split, "path": name + ".hdf5", "instructions": name + ".json"}


class FixtureEncoder:
    """Only injected through the test-only Python API; never offered by the CLI."""
    test_only = True
    contract = {"schema": "test.fixture.encoder", "official_complete": False}

    def encode(self, full):
        n = len(full.states)
        pixels = next(iter(full.images.values())).mean(axis=(1, 2, 3))
        cls = np.stack((pixels, pixels * 0 + 1, np.arange(n) * 0.01, pixels * pixels), axis=1).astype(np.float32)
        spatial = np.broadcast_to(cls[:, :, None], (n, 4, 768)).copy()
        return dict(model_actions=full.actions[:-1], proprio=full.states,
                    gripper=full.states[:, -1], dino_cls=cls, semantic_features=spatial,
                    vae_features=np.broadcast_to(pixels[:, None, None, None], (n, 2, 4, 8)).copy())


def test_real_end_to_end_train_only_and_immutable(tmp_path):
    cfg = config(tmp_path)
    manifest(cfg, [real_episode(cfg, "a", offset=1), real_episode(cfg, "b", offset=20),
                   real_episode(cfg, "dev", split="dev", offset=80), real_episode(cfg, "test", split="test", offset=120)])
    path = prepare(cfg)
    stats = read_json(path / "dataset_stats.json")
    assert stats["num_episodes"] == 2
    assert stats["num_transition"] == 16
    assert stats["action"]["default"]["global_max"][0] < 0.03
    features = precompute(cfg, _test_encoder=FixtureEncoder())
    assert not (features / "test_features.list").exists()
    memory = build_memory(cfg)
    result = read_json(memory / "COMPLETE.json")
    assert result["events"] > 0 and result["source_episodes"] == 2
    assert result["test_only"] and result["train_only_repertoire"]
    assert result["caches"]["train"]["empty_queries"] == 0
    bindings = read_json(memory / "training_bindings.json")
    assert bindings["proprio_dim"] == 7 and bindings["recollection_action_mode"] == "delta"
    assert Path(bindings["data_config"]).is_file()
    with pytest.raises(FileExistsError):
        prepare(cfg)


@pytest.mark.parametrize("native", [False, True])
def test_robotwin_end_to_end_and_no_double_shift(tmp_path, native):
    cfg = config(tmp_path, "robotwin")
    if native:
        cfg["source"].update(layout="xpolicylab_v1", image_encoding="xpolicylab_jpeg")
    entries = [robotwin_episode(cfg, "a", offset=0, native=native), robotwin_episode(cfg, "b", offset=1, native=native),
               robotwin_episode(cfg, "dev", split="dev", offset=2, native=native)]
    manifest(cfg, entries)
    raw = RoboTwin2Adapter(cfg["source"], cfg["profile"]).read(entries[0])
    assert len(raw.actions) == 9
    np.testing.assert_allclose(raw.actions[0, :2], [0.014, 0.015])
    assert raw.images["cam_high"][0, 0, 0, 2] > 210
    prepare(cfg)
    precompute(cfg, _test_encoder=FixtureEncoder())
    assert read_json(build_memory(cfg) / "COMPLETE.json")["source_episodes"] == 2


@pytest.mark.parametrize("marked", [False, True])
def test_robotwin_color_marker(marked):
    rgb = np.full((8, 8, 3), [240, 20, 60], dtype=np.uint8)
    decoded = decode_rgb(jpeg(rgb, marked=marked), "xpolicylab_jpeg")
    np.testing.assert_allclose(decoded[0, 0], rgb[0, 0], atol=3)
    if marked:
        with pytest.raises(PreparationError, match="marker"):
            decode_rgb(jpeg(rgb, marked=True), "legacy_opencv_rgb_jpeg")


def test_quaternion_delta_wrap_and_sign():
    theta = np.deg2rad(179)
    current = [0, 0, np.sin(theta / 2), np.cos(theta / 2)]
    target = [0, 0, -np.sin(theta / 2), np.cos(theta / 2)]
    np.testing.assert_allclose(base_rotation_delta(target, current), [0, 0, np.deg2rad(2)], atol=1e-7)
    np.testing.assert_allclose(quaternion_rotvec([1, 0, 0, 0]), quaternion_rotvec([-1, 0, 0, 0]))
    with pytest.raises(PreparationError):
        quaternion_rotvec([1, 1, 1, 1])


def test_real_action_uses_command_not_measured_successor(tmp_path):
    cfg = config(tmp_path)
    entry = real_episode(cfg, "a", offset=10)
    raw = PiperTeleopAdapter(cfg["source"], cfg["profile"]).read(entry)
    np.testing.assert_allclose(raw.states[1, :3] - raw.states[0, :3], 0)
    assert raw.actions[0, 0] == pytest.approx(0.01)
    assert raw.actions[0, 6] == pytest.approx(0.04)


def test_real_session_split_rejected(tmp_path):
    cfg = config(tmp_path)
    a = real_episode(cfg, "a", session="same")
    b = real_episode(cfg, "b", session="same", split="dev", offset=10)
    adapter = PiperTeleopAdapter(cfg["source"], cfg["profile"])
    adapter.read(a)
    with pytest.raises(PreparationError, match="session"):
        adapter.read(b)


@pytest.mark.parametrize("change", ["calibration", "time", "camera", "split", "synthetic", "outcome"])
def test_real_bad_contracts(tmp_path, change):
    cfg = config(tmp_path)
    entry = real_episode(cfg, "a")
    root = Path(cfg["source"]["root"]) / "a"
    if change in {"calibration", "camera", "split"}:
        meta = read_json(root / "episode.json")
        key, value = {"calibration": ("calibration_id", "another"), "camera": ("camera_order", ["wrist", "external"]),
                      "split": ("split", "test")}[change]
        meta[key] = value
        replace_json(root / "episode.json", meta)
    elif change == "time":
        cfg["profile"]["fps"] = 10
    elif change == "synthetic":
        cfg["source"]["allow_synthetic"] = False
    elif change == "outcome":
        cfg["source"]["include_outcomes"] = ["failure"]
    with pytest.raises(ValueError):
        PiperTeleopAdapter(cfg["source"], cfg["profile"]).read(entry)


def test_duplicate_raw_and_failed_stage_not_published(tmp_path):
    cfg = config(tmp_path)
    manifest(cfg, [real_episode(cfg, "a"), real_episode(cfg, "b")])
    with pytest.raises(PreparationError, match="Duplicate"):
        prepare(cfg)
    assert not (Path(cfg["output"]) / "prepared").exists()
    failed = list(Path(cfg["output"]).glob(".prepared.staging-*/FAILED.json"))
    assert len(failed) == 1


def test_modified_stats_rejected_before_gpu(tmp_path):
    cfg = config(tmp_path)
    manifest(cfg, [real_episode(cfg, "a", offset=1)])
    path = prepare(cfg)
    (path / "dataset_stats.json").write_text("{}")
    with pytest.raises(PreparationError, match="changed"):
        precompute(cfg, _test_encoder=FixtureEncoder())


def test_test_encoder_refused_in_production(tmp_path):
    cfg = config(tmp_path)
    cfg["test_only"] = False
    manifest(cfg, [real_episode(cfg, "a", offset=1)])
    prepare(cfg)
    with pytest.raises(PreparationError, match="test-only"):
        precompute(cfg, _test_encoder=FixtureEncoder())


@pytest.mark.parametrize("field,value", [("action_dim", 0), ("fps", True), ("action_horizon", 3),
                                        ("gripper_state_indices", [77]), ("delta_action_mask", [True]),
                                        ("camera_keys", ["../bad"])])
def test_invalid_config(tmp_path, field, value):
    cfg = config(tmp_path)
    cfg["profile"][field] = value
    path = tmp_path / "config.json"
    write_json(path, cfg)
    with pytest.raises(PreparationError):
        load_config(path)


def test_path_escape_and_no_guessing(tmp_path):
    cfg = config(tmp_path, "robotwin")
    with pytest.raises(PreparationError, match="inside"):
        RoboTwin2Adapter(cfg["source"], cfg["profile"]).read({"path": "../escape.hdf5"})
    cfg["source"]["layout"] = "auto"
    with pytest.raises(PreparationError, match="never guessed"):
        RoboTwin2Adapter(cfg["source"], cfg["profile"])


def test_video_wrong_time_axis(tmp_path):
    from fastwam.preprocessing.lerobot_writer import write_video
    path = tmp_path / "v.mp4"
    write_video(path, np.zeros((5, 8, 8, 3), dtype=np.uint8), 20)
    with pytest.raises(PreparationError, match="timestamps"):
        decode_video(path, np.arange(4) / 20, 1e-4)


def test_native_wrong_qpos_alignment(tmp_path):
    import h5py
    cfg = config(tmp_path, "robotwin")
    cfg["source"].update(layout="xpolicylab_v1", image_encoding="xpolicylab_jpeg")
    entry = robotwin_episode(cfg, "a", native=True)
    with h5py.File(Path(cfg["source"]["root"]) / "a.hdf5", "r+") as h5:
        h5["action/left_arm_joint_states"][0, 0] += 0.1
    with pytest.raises(PreparationError, match="alignment"):
        RoboTwin2Adapter(cfg["source"], cfg["profile"]).read(entry)


def test_real_shared_gripper_normalizer_uses_train_union(tmp_path):
    cfg = config(tmp_path)
    entry = real_episode(cfg, "a", offset=1)
    commands_path = Path(cfg["source"]["root"]) / "a/commands.jsonl"
    commands = [json.loads(row) for row in commands_path.read_text().splitlines()]
    for row in commands:
        row["gripper_width_m"] = 0.06
    commands_path.write_text("\n".join(json.dumps(row) for row in commands) + "\n")
    manifest(cfg, [entry])
    stats = read_json(prepare(cfg) / "dataset_stats.json")
    for name in ("action", "state"):
        assert stats[name]["default"]["global_min"][6] == pytest.approx(0.04)
        assert stats[name]["default"]["global_max"][6] == pytest.approx(0.06)


def test_libero_import_uses_existing_explicit_catalog(tmp_path):
    from fastwam.preprocessing.contracts import RawEpisode
    from fastwam.preprocessing.lerobot_writer import convert_episodes
    cfg = read_json(REPO / "configs/preprocessing/libero.template.json")
    cfg["test_only"] = True
    cfg["profile"].update(action_horizon=4, action_video_freq_ratio=1)
    cfg["output"] = str(tmp_path / "libero_output")
    source = tmp_path / "sources"
    source.mkdir()
    entries = [{"id": str(i), "path": f"{i}.json", "split": "train" if i < 2 else "dev"} for i in range(3)]
    for entry in entries:
        write_json(source / entry["path"], entry)
    class FixtureAdapter:
        root = source
        def read(self, entry):
            value = int(entry["id"])
            return RawEpisode(entry["id"], "test_task", "Pick the block.", entry["split"],
                np.full((8, 8), value * 0.1, np.float32), np.full((8, 7), value * 0.1, np.float32),
                {key: np.full((8, 8, 8, 3), value * 60, np.uint8) for key in cfg["profile"]["camera_keys"]},
                (source / entry["path"],), {"synthetic": True})
    dataset = tmp_path / "existing_lerobot"
    convert_episodes(FixtureAdapter(), entries, dataset, cfg["profile"], "libero_fixture")
    cfg["source"] = {"roots": [str(dataset)], "catalog": str(dataset / "meta/warm_episode_catalog.json")}
    prepared = prepare(cfg)
    assert not (prepared / "dataset").exists()
    assert read_json(prepared / "COMPLETE.json")["roots"] == [str(dataset)]
    precompute(cfg, _test_encoder=FixtureEncoder())
    assert read_json(build_memory(cfg) / "COMPLETE.json")["source_episodes"] == 2


def test_full_memory_rejects_missing_vae(tmp_path):
    cfg = config(tmp_path)
    manifest(cfg, [real_episode(cfg, "a", offset=1), real_episode(cfg, "b", offset=20)])
    prepare(cfg)
    class DinoOnlyFixture(FixtureEncoder):
        def encode(self, full):
            values = super().encode(full)
            values.pop("vae_features")
            return values
    precompute(cfg, _test_encoder=DinoOnlyFixture())
    with pytest.raises(PreparationError, match="VAE"):
        build_memory(cfg)
