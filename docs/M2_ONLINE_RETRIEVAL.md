# M2.1 contract-bound online retrieval

M2.1 tests one deliberately narrow claim in real LIBERO rollouts:

> At every replan, `fixed_context_top1` encodes the current simulator cameras
> with the same frozen-DINO recipe used by M1, searches the complete immutable
> train bank with exact cosine similarity, and uses the rank-zero normalized
> action as a stochastic Action DiT source. `gaussian_null` uses Gaussian
> noise and is prohibited from opening the bank or DINO artifacts.

This milestone does not claim consequence alignment, ANN equivalence, or
full-WARM performance. It also does not treat local CPU tests as experimental
evidence.

## Where each part runs

The checkout at `F:\WARM\code` is for source edits, configs, contract tests,
and other light Windows checks. Do not download large artifacts, train the
model, load DINO/Wan checkpoints, or run LIBERO there. Training, parity, and
rollouts run on a Linux CUDA server cloned from the private `main` branch:

```bash
git clone --branch main --single-branch git@github.com:ZzzzzZhhmm/WARM.git
cd WARM
test -z "$(git status --porcelain)"
export PYTHONPATH=src
git rev-parse HEAD
```

Record that commit. Every formal training attestation, online contract,
parity report, pair contract, and result must resolve to the same clean commit.

## Fixed/null scientific boundary

The comparison consists of two separately trained, policy-specific artifacts:

```text
fixed_context_top1:
  current raw cameras -> M1-equivalent processing -> frozen DINO
  -> full-bank stable exact cosine -> rank-zero model-space action
  -> mu + memory_sigma * epsilon

gaussian_null:
  current observation -> epsilon
  no event bank, DINO checkpoint, catalog, audit, or retriever read
```

The trainer writes an immutable sidecar next to each formal checkpoint:
`step_NNNNNN.pt` and `step_NNNNNN.training.json`. The attestation binds the
actual checkpoint bytes, policy, clean Git commit, shared training recipe,
train/dev source contracts, base checkpoint, optimizer/scheduler, batch and
world-size facts, precision, seed, and completed global step. It is required
even when live periodic validation is disabled; the independent DEV source
contract remains part of the formal data provenance.

The pair is not a same-checkpoint inference intervention. Its defensible claim
is a policy-specific trained-checkpoint comparison whose shared recipe and
distinct policy/checkpoint identities are machine checked. See
`docs/M2_ONLINE_PAIRING.md` for the exact boundary.

Both policies use the same root seed and evaluation namespace. A replan seed
is derived from that namespace and the exact `QueryId` (suite, task, episode,
absolute simulator frame). It is not derived from replan order. The result
verifier compares the common QueryId/seed prefix and retains real termination
reasons when policies produce trajectories of different lengths.

## Immutable server artifacts

The commands below use these canonical M1/M2 paths:

```bash
export WARM_M1=/srv/warm/artifacts/m1
export WARM_M2=/srv/warm/artifacts/m2
export FASTWAM_BASE=/srv/warm/artifacts/fastwam

export TRAIN_SOURCE="$WARM_M2/contracts/hybrid_h32_train_source.json"
export DEV_SOURCE="$WARM_M2/contracts/hybrid_h32_dev_source.json"
export EVENT_BANK="$WARM_M1/banks/hybrid_h32"
export DEV_CANDIDATES="$WARM_M2/candidates/hybrid_h32_dev_k32"
export M1_DATA_CONFIG="$WARM_M1/features/contracts/data_config.source.yaml"
export TRAIN_STATS="$WARM_M1/train_stats/dataset_stats.json"
export BASE_CHECKPOINT="$FASTWAM_BASE/checkpoints/weights/final.pt"

# Replace these with the exact trainer outputs selected for evaluation.
export WARM_FIXED_CHECKPOINT=/srv/warm/runs/fixed/checkpoints/weights/step_NNNNNN.pt
export WARM_NULL_CHECKPOINT=/srv/warm/runs/null/checkpoints/weights/step_NNNNNN.pt
export WARM_FIXED_ATTESTATION="${WARM_FIXED_CHECKPOINT%.pt}.training.json"
export WARM_NULL_ATTESTATION="${WARM_NULL_CHECKPOINT%.pt}.training.json"

export FIXED_CONFIG="$WARM_M2/configs/libero10-task0-fixed.resolved.yaml"
export NULL_CONFIG="$WARM_M2/configs/libero10-task0-null.resolved.yaml"
export FIXED_CONTRACT="$WARM_M2/contracts/libero10-task0-fixed.online.json"
export NULL_CONTRACT="$WARM_M2/contracts/libero10-task0-null.online.json"
export PARITY_REPORT="$WARM_M2/reports/libero10-task0.online-parity.json"
export PAIR_CONTRACT="$WARM_M2/contracts/libero10-task0.fixed-null-pair.json"

# Must exactly equal encoder_contract.compute.device, including any index.
export DINO_DEVICE=cuda
```

`DINO_CHECKPOINT`, `WAN_VAE_CHECKPOINT`, `WAN_TEXT_ENCODER`, `WAN_TOKENIZER`,
`EXACT_TASK_LANGUAGE`, `EXACT_INITIAL_STATES`, and `EXACT_BDDL` must point to
the exact server artifacts admitted by the relevant contracts. The initial
state file is the unmodified LIBERO task array, not a repetition expanded to
the trial count. Use `task.language` verbatim and the BDDL resolved from the
official LIBERO task object.

The catalog dataset order is fixed and all raw-data commands must preserve it:

```bash
export LIBERO_ROOT=/srv/warm/data/libero_mujoco3.3.2
LIBERO_DATASET_ROOTS=(
  "$LIBERO_ROOT/libero_spatial_no_noops_lerobot"
  "$LIBERO_ROOT/libero_object_no_noops_lerobot"
  "$LIBERO_ROOT/libero_goal_no_noops_lerobot"
  "$LIBERO_ROOT/libero_10_no_noops_lerobot"
)
```

## Formal execution order

The order is strict because later artifacts hash-bind earlier ones:

```text
1. resolve/review the fixed/null training configs, train, and obtain
   checkpoint attestations
2. resolve both evaluation configs with planned parity/pair paths
3. publish both online-run contracts
4. pass fixed-side online/offline DEV parity
5. publish the fixed/null pair contract, binding the parity report
6. run the actual fixed and null LIBERO rollouts
7. verify the two result sets as a pair
```

Do not build the pair before parity and do not run a formal rollout before the
pair exists.

## 1. Policy-specific training and attestations

Use the preflight resolution and launch recipe in `docs/M2_SOURCE_ONLY.md`
twice, changing only the allowed policy/output/logging fields. Save and review
each fully resolved training config before launch. Both runs must use the exact
train and DEV source contracts, base checkpoint, seed, optimizer, scheduler,
batch size, world size, precision, and target global step. A formal WARM
checkpoint is not complete without the trainer-produced `.training.json`
sidecar.

Before continuing:

```bash
test -f "$WARM_FIXED_CHECKPOINT"
test -f "$WARM_FIXED_ATTESTATION"
test -f "$WARM_NULL_CHECKPOINT"
test -f "$WARM_NULL_ATTESTATION"
test -z "$(git status --porcelain)"
```

Never hand-author, copy between policies, rename independently, or regenerate
an attestation after training.

## 2. Resolve the evaluation configs

Choose the parity and pair output paths before resolving either config. Hydra
`${now:...}` output paths are not suitable; bind explicit, distinct result
directories. The fixed example is:

```bash
python experiments/libero/eval_libero_single.py \
  task=libero_warm_online_2cam224 \
  "ckpt=$WARM_FIXED_CHECKPOINT" \
  model.source_policy=fixed_context_top1 \
  EVALUATION.task_suite_name=libero_10 \
  EVALUATION.task_id=0 \
  "EVALUATION.device=$DINO_DEVICE" \
  EVALUATION.output_dir="$WARM_M2/eval/libero10-task0-fixed" \
  "EVALUATION.dataset_stats_path=$TRAIN_STATS" \
  "EVALUATION.warm_online.contract_path=$FIXED_CONTRACT" \
  "EVALUATION.warm_online.pair_contract_path=$PAIR_CONTRACT" \
  "EVALUATION.warm_online.parity_report_path=$PARITY_REPORT" \
  "EVALUATION.warm_online.training_attestation_path=$WARM_FIXED_ATTESTATION" \
  "EVALUATION.warm_online.training_run_contract_path=$TRAIN_SOURCE" \
  "EVALUATION.warm_online.validation_run_contract_path=$DEV_SOURCE" \
  "EVALUATION.warm_online.base_checkpoint_path=$BASE_CHECKPOINT" \
  "EVALUATION.warm_online.bank_directory=$EVENT_BANK" \
  "EVALUATION.warm_online.normalizer_contract_path=$WARM_M1/features/contracts/normalizer_contract.json" \
  "EVALUATION.warm_online.encoder_contract_path=$WARM_M1/features/contracts/encoder_contract.json" \
  "EVALUATION.warm_online.camera_contract_path=$WARM_M1/features/contracts/camera_contract.json" \
  "EVALUATION.warm_online.m1_data_config_path=$M1_DATA_CONFIG" \
  "EVALUATION.warm_online.dino_checkpoint_path=$DINO_CHECKPOINT" \
  "EVALUATION.warm_online.catalog_path=$WARM_M1/libero_catalog.json" \
  "EVALUATION.warm_online.audit_report_path=$WARM_M1/libero_audit.json" \
  EVALUATION.warm_online.evaluation_namespace=libero-m2.1-formal-v1 \
  EVALUATION.warm_online.top_k=32 \
  seed=17 --cfg job --resolve > "$FIXED_CONFIG"
```

Resolve the null config with the same command after changing exactly:

```text
ckpt -> $WARM_NULL_CHECKPOINT
model.source_policy -> gaussian_null
EVALUATION.output_dir -> a distinct null directory
EVALUATION.warm_online.contract_path -> $NULL_CONTRACT
EVALUATION.warm_online.training_attestation_path -> $WARM_NULL_ATTESTATION
output redirection -> $NULL_CONFIG
```

The pair and parity paths, seed, namespace, simulator settings, M1 data recipe,
and every other scientific field remain identical.

## 3. Publish the online-run contracts

Build the fixed contract from its exact checkpoint, trainer attestation, and
resolved config:

```bash
python scripts/build_warm_online_contract.py \
  --training-run-contract "$TRAIN_SOURCE" \
  --validation-run-contract "$DEV_SOURCE" \
  --warm-checkpoint "$WARM_FIXED_CHECKPOINT" \
  --training-attestation "$WARM_FIXED_ATTESTATION" \
  --bank "$EVENT_BANK" \
  --normalizer-contract "$WARM_M1/features/contracts/normalizer_contract.json" \
  --encoder-contract "$WARM_M1/features/contracts/encoder_contract.json" \
  --camera-contract "$WARM_M1/features/contracts/camera_contract.json" \
  --data-config "$M1_DATA_CONFIG" \
  --dino-checkpoint "$DINO_CHECKPOINT" \
  --normalization-stats "$TRAIN_STATS" \
  --catalog "$WARM_M1/libero_catalog.json" \
  --audit-report "$WARM_M1/libero_audit.json" \
  --resolved-eval-config "$FIXED_CONFIG" \
  --vae-checkpoint "$WAN_VAE_CHECKPOINT" \
  --text-encoder "$WAN_TEXT_ENCODER" \
  --tokenizer "$WAN_TOKENIZER" \
  --evaluation-namespace libero-m2.1-formal-v1 \
  --task-suite libero_10 --task-id 0 \
  --task-description "$EXACT_TASK_LANGUAGE" \
  --initial-states "$EXACT_INITIAL_STATES" \
  --bddl "$EXACT_BDDL" \
  --root-seed 17 --top-k 32 \
  --source-policy fixed_context_top1 --memory-sigma 0.2 \
  --action-horizon 32 --action-dim 7 \
  --output "$FIXED_CONTRACT"
```

Repeat for null using `$WARM_NULL_CHECKPOINT`, `$WARM_NULL_ATTESTATION`,
`$NULL_CONFIG`, `$NULL_CONTRACT`, and `--source-policy gaussian_null`. The
null contract records comparison identities, but null evaluation is still
forbidden from opening bank/DINO data.

The builder rejects a dirty Git tree, mismatched checkpoint attestation,
processor recipe, DINO runtime, source contract, policy, seed, task, action
shape, artifact path, or resolved-config hash. It re-hashes inputs during
atomic publication.

## 4. Pass exact online/offline DEV parity

Parity runs only for the fixed side, on the Linux GPU/runtime recorded by the
M1 encoder contract. It decodes the audited raw DEV episodes for the bound
task and compares the production online retriever against the immutable M1
stride-one DEV feature/candidate artifacts.

```bash
python scripts/validate_warm_online_parity.py \
  --bank "$EVENT_BANK" \
  --dev-candidate-cache "$DEV_CANDIDATES" \
  --dev-feature-list "$WARM_M1/features/dev_features.list" \
  --training-run-contract "$TRAIN_SOURCE" \
  --validation-run-contract "$DEV_SOURCE" \
  --online-run-contract "$FIXED_CONTRACT" \
  --resolved-eval-config "$FIXED_CONFIG" \
  --data-config "$M1_DATA_CONFIG" \
  --normalizer-contract "$WARM_M1/features/contracts/normalizer_contract.json" \
  --encoder-contract "$WARM_M1/features/contracts/encoder_contract.json" \
  --camera-contract "$WARM_M1/features/contracts/camera_contract.json" \
  --normalization-stats "$TRAIN_STATS" \
  --catalog "$WARM_M1/libero_catalog.json" \
  --audit-report "$WARM_M1/libero_audit.json" \
  --dataset-root "${LIBERO_DATASET_ROOTS[0]}" \
  --dataset-root "${LIBERO_DATASET_ROOTS[1]}" \
  --dataset-root "${LIBERO_DATASET_ROOTS[2]}" \
  --dataset-root "${LIBERO_DATASET_ROOTS[3]}" \
  --dino-checkpoint "$DINO_CHECKPOINT" \
  --device "$DINO_DEVICE" \
  --output "$PARITY_REPORT"
```

Exact mode is the default. The gate checks context keys, validity masks, bank
rows, EventIds, action payloads, and full-bank stable cosine top-K. Only an
encoder contract that explicitly records CUDA `bfloat16` may use a documented
`--bfloat16-atol` up to `1e-3`; identities and action payloads remain exact.
The current DINO software/GPU runtime must match the encoder runtime recorded
by M1. Run one parity report for every task admitted to formal evaluation.

## 5. Publish the pair contract

After parity succeeds, publish the pair that binds the report:

```bash
python scripts/build_warm_online_pair_contract.py \
  --fixed-online-contract "$FIXED_CONTRACT" \
  --fixed-resolved-eval-config "$FIXED_CONFIG" \
  --fixed-training-attestation "$WARM_FIXED_ATTESTATION" \
  --gaussian-null-online-contract "$NULL_CONTRACT" \
  --gaussian-null-resolved-eval-config "$NULL_CONFIG" \
  --gaussian-null-training-attestation "$WARM_NULL_ATTESTATION" \
  --parity-report "$PARITY_REPORT" \
  --output "$PAIR_CONTRACT"
```

The explicit attestation paths must be the exact sidecars already bound by the
two online contracts and resolved configs; do not substitute a later
checkpoint sidecar. The pair builder verifies the shared training recipe and
all science-comparable online fields while requiring distinct
policy/checkpoint/config/output identities.

## 6. Run the actual fixed and null rollouts

Run the same Hydra override lists used to create `$FIXED_CONFIG` and
`$NULL_CONFIG`, now without `--cfg job --resolve`. The pair file and passing
parity report must already exist at the planned paths. For example:

```bash
# Reuse the exact fixed overrides from step 2.
python experiments/libero/eval_libero_single.py \
  task=libero_warm_online_2cam224 \
  "ckpt=$WARM_FIXED_CHECKPOINT" \
  model.source_policy=fixed_context_top1 \
  EVALUATION.task_suite_name=libero_10 EVALUATION.task_id=0 \
  "EVALUATION.device=$DINO_DEVICE" \
  EVALUATION.output_dir="$WARM_M2/eval/libero10-task0-fixed" \
  "EVALUATION.dataset_stats_path=$TRAIN_STATS" \
  "EVALUATION.warm_online.contract_path=$FIXED_CONTRACT" \
  "EVALUATION.warm_online.pair_contract_path=$PAIR_CONTRACT" \
  "EVALUATION.warm_online.parity_report_path=$PARITY_REPORT" \
  "EVALUATION.warm_online.training_attestation_path=$WARM_FIXED_ATTESTATION" \
  "EVALUATION.warm_online.training_run_contract_path=$TRAIN_SOURCE" \
  "EVALUATION.warm_online.validation_run_contract_path=$DEV_SOURCE" \
  "EVALUATION.warm_online.base_checkpoint_path=$BASE_CHECKPOINT" \
  "EVALUATION.warm_online.bank_directory=$EVENT_BANK" \
  "EVALUATION.warm_online.normalizer_contract_path=$WARM_M1/features/contracts/normalizer_contract.json" \
  "EVALUATION.warm_online.encoder_contract_path=$WARM_M1/features/contracts/encoder_contract.json" \
  "EVALUATION.warm_online.camera_contract_path=$WARM_M1/features/contracts/camera_contract.json" \
  "EVALUATION.warm_online.m1_data_config_path=$M1_DATA_CONFIG" \
  "EVALUATION.warm_online.dino_checkpoint_path=$DINO_CHECKPOINT" \
  "EVALUATION.warm_online.catalog_path=$WARM_M1/libero_catalog.json" \
  "EVALUATION.warm_online.audit_report_path=$WARM_M1/libero_audit.json" \
  EVALUATION.warm_online.evaluation_namespace=libero-m2.1-formal-v1 \
  EVALUATION.warm_online.top_k=32 seed=17
```

Run null with the exact null substitutions listed in step 2. Do not reuse an
output directory. Formal result publication is no-overwrite and atomic.

Every episode records its deterministic simulator seed, real termination
reason, final frame, environment/policy step counts, configured wait/max
steps, replan count, and per-replan QueryId/derived seed. Fixed replans also
record ranked EventIds/scores and immutable capability identities. Null
replans attest that no bank/DINO identity was read.

## 7. Verify paired results

Point each result argument at the task result file or a directory containing
only the intended formal result JSON files:

```bash
python scripts/verify_warm_online_pair_results.py \
  --pair-contract "$PAIR_CONTRACT" \
  --fixed-results "$WARM_M2/eval/libero10-task0-fixed" \
  --gaussian-null-results "$WARM_M2/eval/libero10-task0-null" \
  --output "$WARM_M2/reports/libero10-task0.fixed-null-verified.json"
```

The verifier revalidates strict schemas, task/episode partitions, contract and
runtime identities, deterministic simulator seeds, termination/count
consistency, null no-read evidence, fixed capability evidence, and the common
QueryId/seed prefix. It does not claim equal trajectories after the first
policy-dependent divergence.

## Server-only acceptance gates

Before publishing any success-rate table, all of the following must pass on
the server:

1. full tests with Torch-required tests enabled and no hidden skip;
2. checkpoint plus training-attestation verification for both policies;
3. real frozen-DINO and WARM checkpoint loading;
4. exact per-task online/offline DEV parity;
5. one fixed/null rollout smoke pair;
6. complete formal fixed/null rollouts and pair-result verification;
7. latency, peak GPU memory, gate/source telemetry, and multi-seed reporting.

No server experiment has been run merely because these commands and contracts
exist in the repository.
