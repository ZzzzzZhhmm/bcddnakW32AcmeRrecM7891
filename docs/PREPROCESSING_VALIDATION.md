# Unified preprocessing validation — 2026-09-17

Scope: new dataset adapters, real teleoperation preprocessing, generated processor
configuration, existing factual memory composition, and the narrow Piper training
contract extension. No robot was connected or actuated.

## Passing checks

The following targeted run passed **154 tests**, with **1 Linux-only skip**:

```bash
python -m pytest \
  tests/test_unified_preprocessing.py tests/test_preprocessing_torch.py \
  tests/test_real_handoff.py tests/test_rmbench_conversion.py \
  tests/test_rmbench_conversion_validator.py tests/test_offline_pipeline.py \
  tests/test_feature_cache.py tests/test_episode_catalog.py \
  tests/test_lerobot_audit.py tests/test_full_episode_reader.py \
  tests/test_feature_precompute.py tests/test_event_bank.py \
  tests/test_candidate_cache.py tests/test_robotwin_processor_contract.py -q
```

The new tests exercise actual HDF5, JPEG color markers, parquet, H.264 MP4 and
exact timestamp decoding. They compose feature caches, event banks, candidate
caches and a causal recollection store. They also cover train-only statistics,
shared measured/commanded Piper gripper normalization, source/session isolation,
quaternion wrap/sign, duplicate content, tampered statistics, missing VAE,
short/invalid contracts and immutable output publication.

LIBERO import tests use an explicit existing LeRobot catalog; RoboTwin tests
cover both legacy and native XPolicyLab v1.0, including rejection of an incorrectly
shifted native action. Piper tests consume the existing real episode schema.
New tests use deliberately synthetic data and clearly marked fixture encoders.
No numerical result here is a robot success metric.

Actual FastWAMProcessor and image adapters were instantiated on CPU for Piper
and RoboTwin. Model-space action/state outputs were compared to the training
processor. Piper's strict 7D/7D training contract was accepted; a fake 8D Piper
state was rejected. Generated artifacts loaded into RuntimeCandidateResolver
and RetrospectiveFeatureStore; initial history was empty and unavailable future
targets were masked.

Environment: Windows, Python 3.12, NumPy 2.5.3, h5py 3.16.0, PyArrow 23.0.1,
PyAV 16.1.0, CPU Torch 2.14.0, torchvision 0.29.0. Test dependencies were installed
outside the repository; the existing GPU dependency pins were not modified.
One existing PyTorch warning comes from the read-only NumPy input in
`server_feature_encoders._default_spatial_resize`; this path is not mutated.

Also checked: Python compile, CLI help, dependency-light `plan`, raw/output/local
config Git ignores, and `git diff --check`.

## Existing failures reproduced on the pre-change implementation

An additional run of `test_runtime_candidate_dataset_adapter.py` and
`test_warm_retrospective_dataset.py` found **8 failures**. Replacing only the
modified `warm_candidates` module in the test process with its exact content
from base commit `f4b27108191f9e5eaeebc65cd44d38b8549a06b1` reproduced all eight
failures; no working-tree files were replaced.

- Six bound-fixture tests fail before reaching the candidate adapter because
  their LeRobot metadata omit the now-required normalized `video_path`.
- Two retrospective adapter tests fail on the existing equality requirement
  between candidate action width and start proprio width. LIBERO's nominal
  action/state widths are 7/8, so its full retrospective training path needs
  separate investigation before declaring a fresh full training run qualified.

These were not repaired or weakened in this preprocessing change. The new
Piper profile has 7D actions and 7D state; its actual processor and causal store
tests pass. The broader repository test suite is **not** claimed fully green.

## Not verified here

- Real partner raw logs, calibrated coordinate conventions and physical timing.
- Actual DINOv2/Wan VAE checkpoint encoding on the project's pinned CUDA stack.
- LIBERO checkpoint input-head migration from 8D Panda state to 7D Piper state.
- Full model fine-tuning, released inference backend, physical closed loop or
  emergency-stop/collision/watchdog implementation.

The partner must provide an initial raw sample and hardware/calibration intake.
The WARM team must complete GPU feature extraction and model-forward acceptance
before publishing a matching deployment bundle. Data-stage COMPLETE markers
are not hardware motion authorization.
