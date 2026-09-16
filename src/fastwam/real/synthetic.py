"""Tiny PPM fixtures for CPU contract checks; never real evidence."""
from .episodes import EpisodeWriter


def make_episode(root, steps=36):
    meta = dict(schema="warm.real.episode.v1", episode_id="synthetic-001", session_id="synthetic-session",
                task_id="return_origin", instruction="Return the object to its original place.",
                split="train", synthetic=True, calibration_id="SYNTHETIC_NOT_CALIBRATED",
                control_frame="synthetic_base", tcp_frame="synthetic_tcp", clock_id="synthetic-clock",
                clock_domain="client_monotonic_ns", camera_order=["external", "wrist"],
                action_semantics="commanded_absolute_tcp_pose_and_gripper_width",
                nominal_action_hz=20, max_observation_gap_ns=100_000_000, max_sensor_skew_ns=20_000_000)
    writer = EpisodeWriter(root, meta)
    robot = dict(tcp_position_m=[0.2, 0.0, 0.2], tcp_quaternion_xyzw=[0., 0., 0., 1.],
                 gripper_width_m=0.04, joint_position_rad=[0.] * 6)

    def observation(i):
        stamp = 1_000_000_000 + i*50_000_000
        cameras = {}
        for view in meta["camera_order"]:
            relative = f"{view}/{i:06d}.ppm"
            path = writer.root / relative
            path.parent.mkdir(exist_ok=True)
            with path.open("xb") as handle:
                handle.write(b"P6\n4 4\n255\n" + bytes([i % 255, 50, 100])*16)
            cameras[view] = dict(path=relative, width=4, height=4, color_space="RGB", timestamp_ns=stamp)
        return dict(seq=i, timestamp_ns=stamp, cameras=cameras, robot=dict(robot, timestamp_ns=stamp))

    writer.start(observation(0))
    for i in range(steps):
        command = {k: robot[k] for k in ("tcp_position_m", "tcp_quaternion_xyzw", "gripper_width_m")}
        command.update(seq=i, observation_seq=i, timestamp_ns=1_000_000_000+i*50_000_000+1_000_000, accepted=True)
        writer.append_transition(command, observation(i+1))
    writer.finish("success", notes="SYNTHETIC fixture only; no physical task occurred.")
