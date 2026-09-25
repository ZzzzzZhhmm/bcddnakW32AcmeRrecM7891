"""Named Piper training profile; not a robot driver or motion qualification."""

PIPER_CONTROL_MODE = "base_delta_tcp_rotvec_plus_absolute_gripper_width_m"
PIPER_JOINT_CONTROL_MODE = "absolute_joint_target_rad_plus_absolute_gripper_width_m"
PIPER_EMBODIMENT = "piper_single_active_6dof"
PIPER_IMAGE_SIGNATURE = (("external", (3, 224, 224)), ("wrist", (3, 224, 224)))
