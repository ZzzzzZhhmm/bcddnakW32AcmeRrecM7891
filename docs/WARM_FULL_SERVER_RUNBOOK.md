# Complete WARM: implementation and GPU-server runbook

This document is the executable path for the complete consequence-aligned
WARM model. It is intentionally separate from `M2_SOURCE_ONLY.md`: M2 tests a
fixed retrieved source, while complete WARM learns transition gist, action
utility reranking, bounded event adaptation, consequence consistency, and a
confidence gate before constructing the stochastic Action DiT source.

The end-to-end source path is implemented. As of 2026-07-19 it has completed a
one-update CUDA smoke and 40 optimizer updates across two four-H100 segments,
including checkpoint/resume-state publication. It has **not** yet completed full training
or simulator evaluation; these probes are execution evidence, not an empirical
model claim. See
[`WARM_FULL_ARCHITECTURE.md`](WARM_FULL_ARCHITECTURE.md) for the exact model,
causal-memory, gradient, and fallback contracts implemented by the code.
The post-probe interface and resume audit is recorded in
[`TRAINING_CHAIN_AUDIT_2026-07-19.md`](TRAINING_CHAIN_AUDIT_2026-07-19.md).

## What runs locally and what runs on the server

The authoritative source checkout on the workstation is `F:\WARM\code`.
Local work is limited to implementation, CPU/synthetic tests, compilation,
configuration resolution, Git, and manifest validation. The workstation does
not have the GPU memory or CUDA environment required to load or train the
FastWAM backbone. A local green test suite therefore never claims numerical
training or simulator success.

The Linux GPU server runs:

- frozen-DINO feature and factual Wan-VAE latent precomputation;
- H=32 event-bank and candidate-cache construction;
- checkpoint loading and forward/backward smoke tests;
- full WARM training;
- online retrieval parity plus LIBERO and official RMBench rollout evaluation.

Datasets, checkpoints, feature caches, event banks, rollout videos, and
credentials stay outside Git. Only source, configs, tests, and small reports
belong in the private repository.

## Fixed LIBERO contract

The production LIBERO recipe uses all four FastWAM LeRobot datasets, in this
exact order:

```text
libero_spatial_no_noops_lerobot
libero_object_no_noops_lerobot
libero_goal_no_noops_lerobot
libero_10_no_noops_lerobot
```

The data path is friendly to ordinary FastWAM demonstrations. It requires RGB,
action, proprioception, gripper state, task text, and episode boundaries. It
does not require manual event labels, masks, segmentation, depth, object pose,
contact labels, or subtask labels. Events and DINO state-change targets are
derived automatically from factual train/dev episodes. Wan-VAE latents are
also precomputed because offline causal working-memory replay must use the same
factual change evidence as online episode memory.

The formal action contract is:

```text
two cameras, horizontal 224 x 448 observation
33 factual observation frames and 32 actions per dataset sample
32-step action horizon
7 model-space action dimensions
8 proprioception dimensions
global sample stride = 1
```

`configs/model/warm.yaml` is intentionally closed to the canonical four-suite
catalog: its retrieval context width is `768 DINO CLS + 40 task one-hot = 808`.
A different catalog or DINO model requires a new explicit config and new
artifacts; silently reusing this one is a contract error.

Event actions remain in FastWAM's normalized seven-dimensional model action
space. This implementation does not perform geometric action
canonicalization. The event adapter's default bounded residual is measured in
normalized action units, not inferred dataset-standard-deviation units.

## 1. Clone the exact private commit

```bash
git clone --branch main --single-branch git@github.com:ZzzzzZhhmm/WARM.git
cd WARM
test -z "$(git status --porcelain)"
git rev-parse HEAD
export PYTHONPATH=src
```

Record that SHA. Do not train from an uncommitted directory or copy a mutable
worktree from the workstation.

For a manual-copy smoke/debug cycle, the ACP launcher accepts the explicit
`ALLOW_DIRTY_WARM_TRAINING=true` escape hatch. It bypasses the clean-worktree
gate and saves ordinary checkpoints, but deliberately omits formal
`.training.json` attestations. Such checkpoints are suitable for graph/OOM/
runtime debugging only and must not be used for formal comparisons or online
contract publication.

## 2. Define server inputs

```bash
export LIBERO_DATA_ROOT=/server/data/libero_mujoco3.3.2
export WARM_ARTIFACT_ROOT=/server/artifacts/warm_full_v1
export FASTWAM_BASE_CHECKPOINT=/server/checkpoints/fastwam/final.pt
export WARM_DINO_CHECKPOINT=/server/checkpoints/dinov2-base-pinned
export WARM_DINO_REVISION=<exact-40-character-Hugging-Face-commit>
export WARM_VAE_CHECKPOINT=/server/checkpoints/wan22/Wan2.2_VAE.safetensors
```

`WARM_ARTIFACT_ROOT` must be a new path. The builders publish immutable
artifacts and refuse overwrite. The DINO path must be a complete local snapshot
at the revision named by `WARM_DINO_REVISION`. `WARM_VAE_CHECKPOINT` must name
the single pinned Wan2.2 VAE checkpoint file, not a model directory.

## 3. Build factual features, H=32 bank, candidates, and contracts

Run:

```bash
bash scripts/prepare_warm_full_artifacts.sh
```

The script performs the complete chain in order:

1. episode-level train/dev catalog;
2. full parquet and ordered-camera audit;
3. train-only FastWAM normalization statistics;
4. frozen-DINOv2 factual features and factual Wan-VAE latent precomputation for
   train and dev;
5. H=32 hybrid event bank from train only;
6. independent H=32 dev oracle diagnostic;
7. stride-one train and dev top-32 candidate caches;
8. train and dev source-run contracts bound to the base checkpoint.

It publishes these runtime inputs:

```text
$WARM_ARTIFACT_ROOT/m1/features/train_features.list
$WARM_ARTIFACT_ROOT/m1/features/dev_features.list
$WARM_ARTIFACT_ROOT/m1/banks/hybrid_h32
$WARM_ARTIFACT_ROOT/m2/candidates/hybrid_h32_train_k32
$WARM_ARTIFACT_ROOT/m2/candidates/hybrid_h32_dev_k32
$WARM_ARTIFACT_ROOT/m2/contracts/hybrid_h32_train_source.json
$WARM_ARTIFACT_ROOT/m2/contracts/hybrid_h32_dev_source.json
```

The feature lists are the preferred full-WARM input. The precompute root
contains both splits, so passing that mixed root as one feature directory is
invalid. A directory input is allowed only when it contains exactly one
catalog-authorized split. Complete-WARM training rejects feature caches that
lack factual VAE features; rerun preprocessing with the pinned VAE rather than
fabricating or backfilling tensors.

Training episode inputs are produced by causal replay of the same factual
online state machine for every stride-one replan phase. For a query, initial
anchor, protected-latest observation, bounded event merges, and executed-action
summaries can depend only on preceding observations/actions--never on the
future semantic target or future suffix statistics.

Before allocating 5B-model training time, inspect:

```bash
python -m json.tool \
  "$WARM_ARTIFACT_ROOT/m1/oracle/hybrid_h32.json" | less
```

The top-32 oracle should improve materially over context top-1 and the previous
action prior. A failed oracle gate means retrieval coverage must be fixed; it
cannot be repaired by longer backbone training.

## 4. Resolve the full training configuration without loading a GPU model

The server launcher supplies all immutable paths. Resolve exactly what Hydra
will use before launch:

```bash
M1="$WARM_ARTIFACT_ROOT/m1"
M2="$WARM_ARTIFACT_ROOT/m2"

python scripts/train.py \
  task=libero_warm_2cam224_1e-4 \
  "model.run_contract_path=$M2/contracts/hybrid_h32_train_source.json" \
  "model.validation_run_contract_path=$M2/contracts/hybrid_h32_dev_source.json" \
  "model.base_checkpoint_path=$FASTWAM_BASE_CHECKPOINT" \
  "data.warm_candidates.train.bank_directory=$M1/banks/hybrid_h32" \
  "data.warm_candidates.train.candidate_directory=$M2/candidates/hybrid_h32_train_k32" \
  "data.warm_candidates.train.catalog_path=$M1/libero_catalog.json" \
  "data.warm_candidates.train.normalization_stats_path=$M1/train_stats/dataset_stats.json" \
  "data.warm_candidates.train.audit_report_path=$M1/libero_audit.json" \
  "data.warm_candidates.train.retrospective_feature_list=$M1/features/train_features.list" \
  "data.warm_candidates.val.bank_directory=$M1/banks/hybrid_h32" \
  "data.warm_candidates.val.candidate_directory=$M2/candidates/hybrid_h32_dev_k32" \
  "data.warm_candidates.val.catalog_path=$M1/libero_catalog.json" \
  "data.warm_candidates.val.normalization_stats_path=$M1/train_stats/dataset_stats.json" \
  "data.warm_candidates.val.audit_report_path=$M1/libero_audit.json" \
  "data.warm_candidates.val.retrospective_feature_list=$M1/features/dev_features.list" \
  --cfg job --resolve > /server/runs/warm_full_resolved.yaml
```

Check that the resolved target is
`fastwam.runtime.create_warm_retrospection`, the action horizon is 32, train
and dev query corpora differ, and no path is `null`. Also confirm that the
trainable scope contains the existing Action DiT/action expert, compact WARM
modules, and rank-16 Video DiT adapters at layers 9/19. The 5B Video DiT
backbone, VAE, and text encoder remain frozen. The full stage uses disjoint
action and video-only passes (`lambda_action=1`, `lambda_video=1`); the latter
never constructs Action DiT tokens. Future DINO features remain stop-gradient
auxiliary targets.

## 5. GPU smoke gate

Run the full model for one optimizer step before a long job:

```bash
export NPROC_PER_NODE=1
export RUN_ID=smoke-$(git rev-parse --short HEAD)
bash scripts/train_warm_full_server.sh \
  batch_size=1 num_workers=0 run_steps=1 \
  save_every=1 log_every=1 eval_every=0 \
  output_dir=/server/runs/warm_full_smoke/$RUN_ID
```

The smoke gate must verify, on CUDA:

- the exact base checkpoint loads before any training state;
- one factual Video DiT prefill produces both tapped token streams;
- the required-consequence gist is unchanged when candidate payloads change;
- candidate factual-effect validation cannot read the required gist;
- semantic bridge, required/predictive gist, reranker, event adapter,
  consequence selector, and gate all receive finite tensors;
- gate zero removes candidate source, candidate-aware predictive-gist delta,
  and action context, leaving the Gaussian no-long-memory path;
- forward, backward, optimizer step, checkpoint save, and checkpoint reload
  all succeed;
- Action DiT and WARM modules receive finite gradients while Video DiT, VAE,
  and text encoder remain frozen;
- peak allocated/reserved memory and step time are recorded.

Do not interpret one smoke step as evidence of task performance.

## 6. Full training

The current single-node recipes preserve effective global batch 128:

```text
4 x H100: ZeRO2, batch_size=8, gradient_accumulation_steps=4
8 x H100: ZeRO1, batch_size=8, gradient_accumulation_steps=2
```

Keep `max_steps=null`, `run_steps=null`, `num_epochs=10`,
`mot_checkpoint_mixed_attn=true`, and `save_every=2000` for the formal run.
The ACP wrapper validates this geometry and is intentionally single-node; use
the lower Accelerate launchers under a real scheduler for multi-node jobs.

```bash
export NPROC_PER_NODE=8
export RUN_ID=warm-full-v1-$(git rev-parse --short HEAD)
bash scripts/train_warm_full_server.sh \
  output_dir=/server/runs/warm_full/$RUN_ID \
  wandb.enabled=true \
  wandb.mode=offline \
  wandb.project=WARM \
  wandb.name=$RUN_ID
```

Extra Hydra overrides after the script name are forwarded unchanged. Do not
override artifact identities, action horizon, model target, or train/dev split.
Offline logging is the default privacy boundary. Network logging is not part
of the formal recipe; if it is deliberately enabled for a private workspace,
record that exception and keep credentials out of shell history and Git.
Every accepted checkpoint must have the trainer-produced `.training.json`
attestation beside it. A weights file without that sidecar is not a formal
WARM checkpoint.

Formal continuation uses the complete state directory, never a weights-only
file:

```bash
export RESUME=$RUN_DIR/checkpoints/state/step_NNNNNN
```

The step-matched weights file and `.training.json` must also exist under
`checkpoints/weights/`. Training-attestation v2 verifies the parent proof,
global step, optimizer/scheduler/batch/runtime facts, and a tree SHA-256 of the
entire Accelerate/DeepSpeed state before loading; it hashes the state again
after loading and records the parent checkpoint, parent attestation, resume
state, and resume step in every new sidecar. Existing v1 parents are accepted
only through this one-way verified upgrade. A changed world size, batch
geometry, source contract, runtime, or incomplete state fails closed. Resume
launches preserve the original `config.yaml` and publish an immutable
`config.resume.step_NNNNNN.<sha>.yaml` instead.

## 7. One-task full-WARM LIBERO evaluation

Complete WARM uses the online mode `full_retrospection`. Unlike the M2.1
fixed-vs-Gaussian diagnostic, it is a single learned checkpoint and does not
load an M2 pair contract. Its online contract still binds the checkpoint,
training attestation, bank, encoder, normalizer, exact task BDDL/initial states,
seed, and resolved evaluation config.

For ACP/CCI, prefer the reusable wrapper.  Its `prepare` action is run once on
the persistent AFS volume: it verifies the completed checkpoint, creates a
detached clean worktree at the checkpoint-attested Git commit, checks the
MuJoCo/LIBERO runtime, and snapshots exact task metadata and initial states for
all forty LIBERO tasks.  It performs no network operation.

Before the first `prepare`, install the online simulator once from CCI.  This
uses the existing WARM Python environment, pins MuJoCo 3.3.2 and official
LIBERO commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`, and places the public
LIBERO checkout outside the private WARM worktree.  Do not install LIBERO's
legacy `requirements.txt` into this environment because its old dependency
pins conflict with the trained WARM stack.

The training ACP wrapper does not use the interactive shell's `(base)` Python.
It activates the path-based environment at
`/mnt/afs/task3_2/L202500276_lwz/envs/warm` and unconditionally prepends that
environment's `bin` directory to `PATH`; the evaluation wrapper calls its
Python executable by absolute path.  Because this directory is on persistent
AFS, packages installed there survive new CCI/ACP containers that mount the
same workspace.  The selected private image supplies the compatible OS/CUDA
base, but does not change this interpreter selection.

For a network-free setup, upload these two public, pinned files to
`/mnt/afs/task3_2/L202500276_lwz/projects/WARM_external/bootstrap/libero_eval_v1/`:

```text
robosuite-1.4.0-py3-none-any.whl
LIBERO-8f1084e3132a.tar.gz
```

The setup verifies both SHA-256 digests, installs the wheel without dependency
resolution, and extracts the official LIBERO source.  If the files are absent,
it falls back to network installation while retaining a persistent pip cache
under `WARM_external/pip_cache`.  Archive extraction deliberately ignores
stored uid/gid/mode metadata because AFS root-squash rejects ownership changes;
failed extraction attempts clean only their verified job-local temporary path.
The official repository's double namespace layout is exposed through a pinned
`.pth` file in the persistent WARM environment and an explicit evaluation
`PYTHONPATH`; evaluation therefore does not depend on setuptools editable-hook
behavior.

The wrapper also exports `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`.  This is needed
because official LIBERO init-state files contain legacy NumPy objects while
PyTorch 2.6+ changed unspecified `torch.load` calls to `weights_only=True`.
This compatibility override is confined to the evaluation process: the LIBERO
source archive is pinned and SHA-verified, and WARM checkpoints are verified by
their training attestations before loading.

```bash
cd /mnt/afs/task3_2/L202500276_lwz/projects/WARM
bash scripts/setup_warm_libero_eval_env.sh
```

The setup is persistent on AFS and is not repeated by later ACP jobs.  The
evaluation wrapper also creates a non-interactive LIBERO configuration under
`WARM_evaluations/libero_config`, avoiding LIBERO's first-import prompt.

```bash
cd /mnt/afs/task3_2/L202500276_lwz/projects/WARM
EVAL_ACTION=prepare bash scripts/acp_warm_libero_eval.sh
```

Every later ACP task only selects one GPU/task/seed.  The wrapper reuses the
persistent setup and calls the same contract-bound evaluator documented below.
The online contract still rehashes all identity-bearing inputs on every formal
job; this integrity check is intentionally not cached.

```bash
cd /mnt/afs/task3_2/L202500276_lwz/projects/WARM
CUDA_VISIBLE_DEVICES=0 \
WARM_TASK_SUITE=libero_10 \
WARM_TASK_ID=0 \
WARM_ROOT_SEED=17 \
WARM_EVAL_LABEL=formal \
bash scripts/acp_warm_libero_eval.sh
```

Persistent state includes the environment, artifacts, exact-commit worktree,
and task snapshots.  Shell environment variables do not persist across ACP
containers; the wrapper owns all static defaults, so callers only provide the
four per-job values above.  A failed immutable job is retried with a new label
such as `WARM_EVAL_LABEL=retry1`, never by overwriting its output root.

The completed step-019100 checkpoint is bound to training commit
`c4763a975298de6f00939360551616af7902d57a`. That revision has one
evaluation-only defect which appears after a factual executed-action summary
has been committed: the stored repetition signature is compact (15 values for
LIBERO), while the next preview signature expands the gripper coordinates (21
values). The ACP wrapper recognizes both the exact commit and exact source-file
SHA-256, keeps the historical worktree clean, and enables only the
`action-summary-signature-v1` startup repair. The repair inserts zero-valued
terminal coordinates, so cosine similarity and normalized distance are
numerically unchanged; it only restores equal array widths.

Do not edit the detached checkpoint worktree manually. The wrapper verifies the
repair against the current committed file, writes
`evaluation_compatibility.json` into the immutable result root, and derives the
effective evaluation namespace as
`<base>-compat-<first-12-characters-of-patch-sha256>`. The online contract thus
binds the repaired evaluation identity, while the runtime attestation continues
to identify the unmodified training commit. Serial result validation rejects an
affected-checkpoint result that lacks this compatibility evidence.

Production v1 online coarse ANN intentionally runs the same contract-bound
frozen DINO encoder used to build the bank. The learned semantic bridge maps
the single-pass Video DiT world tokens into compact DINO-aligned tokens for
in-model world/consequence reasoning and factual episode memory; it is not the
online ANN query encoder in this implementation. Report the DINO retrieval
cost in policy latency.

The full online task fixes `EVALUATION.replan_steps=10`, exactly matching
`model.retrospection.episode_action_chunk_size=10`. The evaluator rejects a
different value so executed-action history cannot silently change distribution
between training and rollout.

Set one immutable task job:

```bash
export WARM_CHECKPOINT=/server/runs/warm_full/<run>/checkpoints/weights/step_<N>.pt
export WARM_TRAINING_ATTESTATION=${WARM_CHECKPOINT%.pt}.training.json
export WARM_VAE_CHECKPOINT=/server/checkpoints/wan22/Wan2.2_VAE.safetensors
export WARM_TEXT_ENCODER=/server/checkpoints/wan21/text_encoder
export WARM_TOKENIZER=/server/checkpoints/wan21/tokenizer

export WARM_TASK_SUITE=libero_10
export WARM_TASK_ID=0
export WARM_TASK_DESCRIPTION='put the red mug on the left plate'
export WARM_INITIAL_STATES=/server/libero/initial_states/libero_10_task_0.npy
export WARM_BDDL=/server/libero/bddl/libero_10/task_0.bddl
export WARM_ROOT_SEED=17
export WARM_EVAL_ROOT=/server/eval/warm_full/libero_10/task_0/seed_17

bash scripts/evaluate_warm_full_server.sh
```

`WARM_TASK_DESCRIPTION`, `WARM_INITIAL_STATES`, and `WARM_BDDL` must be the
exact values/files returned by the installed LIBERO task registry; the shown
description and paths illustrate the required shape, not a substitute for
registry lookup. The launcher resolves the Hydra config first, atomically
builds a single-checkpoint online contract, then executes the rollout with the
same override vector. It refuses an existing evaluation root.

Repeat with a distinct root for every task and seed. The first formal grid is:

```text
LIBERO-Spatial, Object, Goal, and LIBERO-10
all task ids
at least 3 deterministic root seeds
50 trials per task/seed for the final report
```

Report success rate, retrieval/gate activation, Gaussian fallback rate,
retrieval time, total policy latency, ODE steps, GPU memory, and checkpoint
identity. Compare against the exact FastWAM base checkpoint and the frozen M2
source-only result. Do not compare to a baseline trained with a different
normalizer, split, or action horizon.

## 8. Official RMBench data, training, and evaluation

RMBench uses a separate, pinned simulator checkout and the exact official nine
tasks. It does not reuse RoboTwin helper tasks as paper scores. The code pin is
`57ee09cbc6267bc36ca0ac2d8d1c5c3b245c112c`; the Hugging Face data pin is
`855e90e1213d150bf4889130e83398f107314681`. The external checkout's push URL
must be the literal `DISABLED`, while the WARM checkout must have exactly one
identical fetch/push remote pointing to the private `ZzzzzZhhmm/WARM` repository
through its canonical SSH or HTTPS URL. Set `WARM_EXPECTED_ORIGIN` when a
server uses one specific transport. Every formal launcher rejects a dirty
checkout or an existing output directory.

Define local, immutable inputs (all paths are examples):

```bash
export WARM_EXPECTED_ORIGIN=git@github.com:ZzzzzZhhmm/WARM.git
export WARM_CODE_REVISION=$(git rev-parse HEAD)
export RMBENCH_ROOT=/server/external/RMBench-official
git -C "$RMBENCH_ROOT" remote set-url --push origin DISABLED
export RMBENCH_SOURCE_ROOT=/server/datasets/rmbench_hf_snapshot
export RMBENCH_HF_REVISION_MARKER=/server/datasets/rmbench_hf_revision.json
export RMBENCH_LEROBOT_ROOT=/server/datasets/rmbench_lerobot_v1
export WARM_ARTIFACT_ROOT=/server/artifacts/warm_rmbench_v1
export FASTWAM_BASE_CHECKPOINT=/server/checkpoints/fastwam/final.pt
export WARM_DINO_CHECKPOINT=/server/checkpoints/dinov2-base-pinned
export WARM_DINO_REVISION=<exact-40-character-DINO-Hub-commit>
export WARM_VAE_CHECKPOINT=/server/checkpoints/wan22/Wan2.2_VAE.safetensors
```

The HF marker is JSON containing at least the exact `repo_id` and `revision`;
the official runner validates it instead of trusting a configured string.
Build the complete dataset/artifact chain once:

```bash
bash scripts/prepare_warm_rmbench_artifacts.sh
```

This performs the strict 450-episode conversion (50 demonstrations for each
of nine tasks), task-stratified whole-episode train/dev split, three-camera
byte audit, native 14D qpos/action statistics, DINO and Wan-VAE preprocessing,
H=32 event-bank construction, oracle diagnostic, stride-one candidate caches,
and independent train/dev source contracts. No masks, subtask labels, pose,
depth, or manual event annotation are introduced.

Precompute the shared text cache, then train both the complete model and the
same-data no-memory baseline.  The latter is not the released checkpoint used
unchanged: it starts from the same released FastWAM base weights and is adapted
on exactly the converted RMBench train split, processor, action statistics, and
action horizon used by WARM.

```bash
export RMBENCH_TEXT_CACHE=/server/artifacts/text/rmbench
mkdir -p "$RMBENCH_TEXT_CACHE"
python scripts/precompute_text_embeds.py \
  task=rmbench_warm_3cam384_1e-4 overwrite=false
export WARM_TRAIN_OUTPUT=/server/runs/warm_rmbench/full_v1
export FASTWAM_RMBENCH_TRAIN_OUTPUT=/server/runs/warm_rmbench/fastwam_same_data_v1
export NPROC_PER_NODE=8
export WANDB_MODE=offline
bash scripts/train_warm_rmbench_server.sh
bash scripts/train_fastwam_rmbench_server.sh
```

The text-embedding command consumes the same local, pinned Wan text encoder and
tokenizer configured for FastWAM; formal offline mode must find those snapshots
locally rather than downloading an unpinned model during the run.

For evaluation, select the WARM checkpoint and its trainer-published sidecar,
plus the independently trained same-data FastWAM checkpoint.  Build the closed
online bundle once; it contains every registered matrix cell and task as well
as the deterministic candidate-seed namespace.  The seed files are lower-bound
protocols, not a claim about seeds accepted after the official setup/expert
filters.

```bash
export WARM_CHECKPOINT=/server/runs/warm_rmbench/full_v1/checkpoints/weights/step_<N>.pt
export WARM_TRAINING_ATTESTATION=${WARM_CHECKPOINT%.pt}.training.json
export FASTWAM_RMBENCH_CHECKPOINT=/server/runs/warm_rmbench/fastwam_same_data_v1/checkpoints/weights/step_<N>.pt
export WARM_RMBENCH_ONLINE_CONTRACT=/server/artifacts/warm_rmbench_v1/online
export WARM_TEXT_ENCODER=/server/checkpoints/wan21/text_encoder
export WARM_TOKENIZER=/server/checkpoints/wan21/tokenizer
export WARM_RMBENCH_SUITE=official9
bash scripts/build_warm_rmbench_contract_bundle_server.sh

# Establish the accepted-seed reference with the same-data FastWAM baseline.
export FASTWAM_RMBENCH_EVAL_ROOT=/server/eval/warm_rmbench/fastwam_same_data_v1
bash scripts/evaluate_fastwam_rmbench_server.sh
export WARM_RMBENCH_ACCEPTED_SEED_REFERENCE="$FASTWAM_RMBENCH_EVAL_ROOT/summary.json"

# Optional single full-WARM run; the matrix below also evaluates this cell.
export WARM_EVAL_ROOT=/server/eval/warm_rmbench/full_warm
export WARM_NUM_GPUS=8
bash scripts/evaluate_warm_rmbench_server.sh
```

The contract variable names the bundle root, not one task file. Its closed
layout is:

```text
online/<experiment_id>/<task_name>.json
online/seeds/<task_name>.seed_protocol.npy
```

The standard launch uses experiment ID `full_warm`; the registered matrix uses
each checked-in cell ID. Every per-task contract binds that cell's experiment
controls, ODE count, resolved policy config, task module, checkpoint, and
shared deterministic seed namespace. Missing cells fail before simulator load.
Each completed task additionally stores and hashes the 100 seeds actually
accepted by the official evaluator. Formal WARM cells must match the baseline
reference sequence exactly; this is stricter and more truthful than treating
the pre-run lower bounds as accepted seeds.

The official score is exactly 100 rollouts for each of the nine registered
tasks. `pilot3` is only a smoke suite and is never reported as the official
nine-task score. Outputs are written to WARM-owned storage, never into the
external public checkout.

The checked-in matrix fixes context-only, source-only without consequence,
full WARM, four memory corruptions, and 2/4/8/10 ODE steps:

```bash
export WARM_EVAL_MATRIX_ROOT=/server/eval/warm_rmbench/matrix_v1
test -f "$WARM_RMBENCH_ACCEPTED_SEED_REFERENCE"
python scripts/run_warm_rmbench_matrix.py --plan-only
python scripts/run_warm_rmbench_matrix.py
```

Every matrix cell reruns the same official suite and seed with a new immutable
output root. The final `matrix_results.json` binds each manager summary by
SHA-256. To debug cheaply without changing the registered matrix, select a
cell and use the smoke suite:

```bash
python scripts/run_warm_rmbench_matrix.py \
  --suite pilot3 --experiment full_warm \
  --output-root /server/eval/warm_rmbench/pilot_full
```

`WANDB_MODE=offline`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and
`HF_DATASETS_OFFLINE=1` are launcher defaults. Any non-offline experiment
logging additionally requires the explicit `WARM_ALLOW_NETWORK_LOGGING=1`
exception.

## 9. Benchmark promotion

Use standard LIBERO first to prove non-regression and action-pattern reuse.
Then add memory-focused evaluation (RMBench or an equivalent reproducible
non-Markov suite) after its dataset/environment adapter is implemented and
contract-tested. LIBERO-PRO/Plus is valuable for robustness but is secondary to
the two primary claims:

1. consequence alignment rejects visually similar but effect-incompatible
   actions;
2. a verified historical event shortens the action transport path without
   sacrificing Gaussian fallback.

Required ablations are:

```text
FastWAM Gaussian baseline
visual/gist context only
action memory as context only
retrieved source without consequence alignment
complete WARM
random memory and hard effect-incompatible memory corruption
2/4/8/10 Action DiT integration steps
```

## 10. Local and server acceptance gates

Local workstation:

```powershell
Set-Location F:\WARM\code
python -m pytest -q
python -m compileall -q src scripts
git status --short
```

GPU server, from the exact clean private commit:

```bash
export WARM_REQUIRE_TORCH_TESTS=1
python -m pytest -q
python -m pytest -q tests --run-gpu   # if the server test plugin exposes it
python -m compileall -q src scripts experiments
nvidia-smi
git status --porcelain
```

If the repository does not expose a `--run-gpu` option, run its Torch-marked
tests explicitly and fail the gate if they are skipped. Preserve the resolved
config, package/CUDA versions, Git SHA, artifact manifests, smoke profile, and
training attestation with every reported result.

Local success establishes only code and contract completeness. The project is
experimentally complete only after the CUDA smoke gate, full training,
checkpoint reload, frozen-DINO online retrieval, causal factual-memory rollout,
and the stated LIBERO/ablation grid all finish from the recorded private SHA.
