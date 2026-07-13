# Third-party notices

## FastWAM

Portions of this repository are derived from FastWAM, copyright 2026 The
FastWAM Authors, and are used under the MIT License included in `LICENSE`.

## RoboTwin

The imported `third_party/RoboTwin` subtree retains its own license and vendor
documentation. Its files must remain subject to the notices in that subtree.

## NVIDIA utility code (Apache-2.0)

`src/fastwam/utils/misc.py` contains NVIDIA-authored code, copyright
2025 NVIDIA CORPORATION & AFFILIATES, distributed under the Apache License
2.0. The applicable license text is included in `LICENSES/Apache-2.0`.

## Hugging Face LeRobot code (Apache-2.0)

The following vendored LeRobot-derived files are copyright 2024 The
HuggingFace Inc. team and are distributed under the Apache License 2.0:

- `src/fastwam/datasets/lerobot/constants.py`
- `src/fastwam/datasets/lerobot/lerobot/lerobot_dataset.py`
- `src/fastwam/datasets/lerobot/lerobot/datasets/compute_stats.py`
- `src/fastwam/datasets/lerobot/lerobot/datasets/utils.py`
- `src/fastwam/datasets/lerobot/lerobot/datasets/video_utils.py`

The applicable license text is included in `LICENSES/Apache-2.0`. Copyright
and license headers in those source files must be retained.

## Meta and PyTorch3D-derived rotation code (BSD-3-Clause)

`src/fastwam/datasets/lerobot/transforms/rotation.py` is copyright Meta
Platforms, Inc. and affiliates. The companion
`src/fastwam/datasets/lerobot/utils/rotation.py` states that it is adapted
from PyTorch3D. These BSD-style components are covered by the BSD 3-Clause
license included in `LICENSES/BSD-3-Clause`; their source notices must be
retained.

## Model weights and datasets

Model weights, LIBERO data, RoboTwin/RMBench assets, and offline feature
encoders are not relicensed by this repository. Download scripts and manifests
must record and preserve the terms of each external artifact. Large datasets,
weights, feature caches, memory banks, credentials, and experiment secrets must
never be committed.
