# Training transfer validation, 2026-11-10

Scope: a source handoff for a new FastWAM-based WAM without retrospection or memory.
No new GPU training, checkpoint inference, or robot execution was performed.

## Checks completed

- 54 tests passed across `test_real_training_transfer.py`,
  `test_unified_preprocessing.py`, `test_preprocessing_torch.py`,
  `test_libero_to_piper_migration.py`, and `test_real_handoff.py` on the isolated
  handoff worktree after integrating the latest upstream source.
- The 7 transfer tests passed again after adding an explicit Hydra composition
  check for the documented `train_new_tasks.local` configuration name.
- Joint preprocessing uses accepted six-joint target commands plus commanded
  gripper width, preserves TCP proprio, rejects missing/malformed/non-finite
  targets, computes train-only statistics, and publishes real parquet/video
  fixture artifacts. Optional historical memory compatibility is tested only
  to protect the existing WARM path; it is not part of the new project's route.
- Both new CLIs expose help successfully. The template `plan` resolves to the
  new joint profile without requiring DINO or memory assets. It intentionally
  does not claim that placeholder calibration values or raw files are verified.
- Config generation requires a complete prepared dataset, checks its file
  hashes, rejects synthetic releases, separates weight initialization from
  full-state resume, and does not load a model or silently choose a checkpoint.
- The ZIP builder enforces a source-file allowlist, excludes local configs,
  recorded data and checkpoints, verifies ZIP CRCs and every archived file hash,
  and includes a standalone recipient-side verification script.

One existing warning is emitted by the RoboTwin image adapter test about a
non-writable NumPy array passed to torch; that test passes and no change was
made to its implementation. An initial transfer-test expectation was corrected
because NaN JSON is already rejected earlier by the raw-record parser, which
is the desired rejection behavior.

## Environment and limits

Validation used Windows, Python 3.12.14 and torch 2.14.0+cpu from the local test
dependencies. This is an interface/CPU integration check, not validation of the
deployment GPU stack. The delivered project retains its pinned Linux training
dependencies, including torch 2.7.1+cu128, torchvision 0.22.1+cu128, and its
existing Accelerate/DeepSpeed versions.

The recipient must execute an actual GPU forward/backward, checkpoint strict
reload, and DEV model probe with their own model and assets. Actual SDK/CAN,
camera acquisition, action scheduling, limits, and the production inference
client/server are owned by the on-site team and are not included as qualified
implementations in this source package.

The original workspace's uncommitted paper record is preserved byte-for-byte.
Changes are made on the separate `handoff/wam-real-transfer-20261110` branch.
The bundle receipt records the actual committed source revision and ZIP hash;
no old experiment run is silently reclassified as a result of this handoff.
