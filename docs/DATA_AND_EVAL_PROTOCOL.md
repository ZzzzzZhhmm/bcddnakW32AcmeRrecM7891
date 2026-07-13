# WARM data and evaluation protocol

## LIBERO development data

Use the four official FastWAM LeRobot v2.1 archives:

```text
libero_spatial_no_noops_lerobot
libero_object_no_noops_lerobot
libero_goal_no_noops_lerobot
libero_10_no_noops_lerobot
```

They provide two RGB views, language, 8D state, 7D action, gripper state, and
episode boundaries. These are sufficient for v1 without human event, mask,
contact, depth, pose, or subtask annotations.

Event extraction operates on full episodes, never on overlapping 33-step
training windows. Visual teacher features may be sampled at the base video
stride while action/proprio/gripper signals remain at control rate.

## Split protocol

During development:

- stratify by task;
- use 45 demonstrations for train bank/training and 5 for dev per task;
- exclude the entire query episode from its candidate pool;
- fit normalization, whitening/PCA, thresholds, and any bank compression on
  train episodes only;
- freeze hyperparameters before the final all-50-demonstration retrain.

Every bank manifest records dataset and episode hashes, split, encoder revision,
normalization hash, action-space version, event extraction config, and software
commit. Evaluation uses a static closed bank and never writes rollout data back.

## Positive and negative data

FastWAM LIBERO demonstrations are treated as positive expert trajectories.
Unless an audited field proves otherwise, do not claim real failure memory.
Hard negatives are generated automatically from cross-episode candidates that
look similar but differ in action direction, gripper timing, task phase, or
observed effect.

## Evaluation order

### 1. LIBERO smoke and standard suites

First run a small `libero_10`/`libero_spatial` smoke. Final standard evaluation
uses all four suites and 50 rollouts per task. It measures baseline parity,
general manipulation ability, and whether memory harms Markov tasks; it is not
by itself proof of cross-task experience reuse.

### 2. LIBERO-PRO

Start with position/swap and task perturbations while retaining the standard
training-only bank. These settings expose visual-nearest-neighbor trajectory
copying and directly test whether consequence alignment selects the right
behavior under changed context.

### 3. RMBench

Use a separate RoboTwin-initialized WARM model; the LIBERO Franka checkpoint is
not compatible with the bimanual action space. Convert only the official nine
tasks from HDF5/RoboTwin format into a versioned LeRobot dataset. Pilot Put Back
Block, Rearrange Blocks, and Battery Try before all nine tasks/100 rollouts.

### 4. LIBERO-Plus

Treat as optional large-scale robustness evaluation after LIBERO-PRO and
RMBench. It has higher integration/evaluation cost and is not a memory-specific
benchmark.

## Leakage tests

Automated checks must fail the build when:

- a query and candidate share an episode id or content hash;
- an evaluation episode is in the training bank;
- a future semantic target enters Action DiT context;
- a generated future is written to episode or long-term memory;
- bank normalization/encoder versions disagree with the model checkpoint;
- an exact held-out task/object-goal pair appears in a composition-holdout bank.

