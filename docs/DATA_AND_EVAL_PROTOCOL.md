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

The production converter is `scripts/convert_rmbench_to_lerobot.py`. It has a
closed allow-list for the nine paper tasks and three closed data profiles:
`official50-dev45`, `scale200-dev190`, and `scale500-dev480`. It never writes
under the official RMBench checkout. The official50 profile requires the
official source; scaled profiles require a separately attested, automatically
collected source tree. The full score-oriented plan is in
`docs/RMBENCH_SOTA_TRAINING_EVALUATION_PLAN_ZH.md`. Pin both the official
dataset revision and the official RMBench code revision:

```bash
python scripts/convert_rmbench_to_lerobot.py \
  --source-root /server/external/RMBench \
  --output-root /server/data/rmbench_demo_clean_lerobot_v2_1 \
  --source-revision <exact-Hugging-Face-dataset-commit> \
  --source-dataset TianxingChen/RMBench \
  --rmbench-code-revision 57ee09cbc6267bc36ca0ac2d8d1c5c3b245c112c \
  --data-revision warm-rmbench-demo-clean-v1 \
  --dataset-id rmbench_demo_clean_v1 \
  --profile official50-dev45 \
  --split-seed 3407 \
  --workers 4
```

RMBench records `N` observed bimanual qpos states and JPEG observations, not an
independent command array. The converter therefore emits exactly `N-1`
factual transition rows:

```text
observation.state[t] = qpos[t]
action[t]            = qpos[t+1]
```

It strictly validates all `N` JPEG cells, including the terminal observation,
but does not fabricate a terminal action. Consequently,
`terminal_visual_excluded_from_training=true`: the one terminal observation
per episode is validated and hashed but is not published as a training row.
The manifest records its count plus a per-episode digest binding the terminal
float32 state and all three original terminal JPEG byte streams. The output
contains three ordered
cameras (`cam_high`, `cam_left_wrist`, `cam_right_wrist`) at the official
`LargeView` resolution of 240x320, native 14D qpos,
deterministic Parquet/MP4 metadata, a train-then-dev episode ordering, and
`meta/warm_episode_catalog.json`. Training and preprocessing must pass that
catalog with `episode_split=train`; development uses `episode_split=dev`.
Never use frame/window-level random splits.

The two notions of task are intentionally separate. Standard LeRobot
`task_index`/`tasks.jsonl` retain each episode's natural `seen` instruction, so
FastWAM receives the real language prompt. `episodes.jsonl.warm_task_identity`
stores the stable official source task name; WARM's episode-catalog scanner
uses only that field for the retrieval task vocabulary. Thus the retrieval key
is always DINO plus an exact nine-way task identity even when every episode has
a different instruction, and unseen evaluation wording cannot change its
dimension.

Publication is fail-closed: the destination must not exist, conversion happens
in a sibling staging tree, and the completed tree is renamed into place only
after every episode succeeds. `meta/rmbench_conversion_manifest.json` binds
every source HDF5/instruction SHA-256, source/data revisions, output
Parquet/MP4/meta SHA-256, exact split, and the terminal-observation policy.
Archive that manifest with all RMBench checkpoints and results.

The checked-in runner binds a deterministic 100-row seed namespace for each
task.  Its second column is the official lower bound
`100000 * (1 + root_seed) + ordinal`; it is not labelled as the post-filter
accepted-seed sequence.  Every rollout stores the actual 100 strictly
increasing accepted seeds emitted by the official evaluator.  The same-data
FastWAM baseline establishes the reference hash, and every formal WARM matrix
cell must reproduce that exact per-task accepted-seed sequence.

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
