# WARM training-chain audit (2026-07-19)

This note records the implementation audit performed after the first LIBERO
CUDA probes.  It separates software evidence from empirical evidence: the
complete graph has reached global step 40 across two 20-update four-H100
segments, but convergence and simulator success are still unmeasured.

## 1. Accepted end-to-end path

The production call chain is:

```text
acp_warm_libero.sh
  -> resolve task/artifact/batch/ZeRO overrides
  -> train_zero1.sh or train_zero2.sh
  -> accelerate launch
  -> scripts/train.py / Hydra
  -> fastwam.runtime.run_training
  -> immutable train/dev candidate datasets
  -> WarmRetrospectionFastWAM
  -> Wan22Trainer / DeepSpeed
  -> weights + full distributed state + training attestation
```

Every scalar that controls the optimizer loop is validated before the 6B model
is allocated.  Complete WARM additionally rejects a source policy other than
`fixed_context_top1` and rejects disabled mixed-attention checkpointing.  The
same validation runs again in the Trainer as defense in depth.

`acp_warm_libero.sh` is intentionally a **single-node** orchestrator.  Its
shared preflight and console log are not safe for two independent node shells.
The lower `train_zero1.sh` and `train_zero2.sh` launchers forward complete
Accelerate multi-node topology, but a scheduler must assign `MACHINE_RANK`,
`NUM_MACHINES`, `MASTER_ADDR`, and `MASTER_PORT`.

## 2. Canonical single-node batch contracts

For the current LIBERO corpus, ten epochs at effective global batch 128 resolve
to 19,100 optimizer updates.

| Hardware | ZeRO | Per-device batch | Accumulation | Effective global batch |
|---|---:|---:|---:|---:|
| 4 x H100 80 GB | 2 | 8 | 4 | 128 |
| 8 x H100 80 GB | 1 | 8 | 2 | 128 |

Formal training keeps:

```text
MAX_STEPS=null
RUN_STEPS=null
NUM_EPOCHS=10
MOT_CHECKPOINT_MIXED_ATTN=true
learning_rate=1e-4
lr_scheduler_type=cosine
weight_decay=1e-2
mixed_precision=bf16
max_grad_norm=1.0
SAVE_EVERY=2000
EVAL_EVERY=0
```

`RUN_STEPS` is only an invocation boundary for probes or wall-time segments; it
does not shorten the 19,100-step scheduler.  `MAX_STEPS` changes the scheduler
trajectory and must not be used for a smoke test.  Auto accumulation uses the
actual per-device batch and world size and prints the resolved global batch.

## 3. Data and causal-memory contract

The runtime binds all of the following by immutable manifests and SHA-256:

- train-only H=32 event bank;
- independent stride-one train and dev top-32 candidate corpora;
- episode catalog and train/dev membership;
- train-only action/proprio normalization statistics;
- DINO and Wan-VAE factual feature files;
- FastWAM base checkpoint and exact action/control schema.

Each sample contains 33 factual frames, 32 actions, two cameras, seven action
channels, and eight proprioception channels.  Candidate tail padding is
null-only, and explicit action/future masks remove padded supervision.  Runtime
loading checks tensor shapes, dtype, finiteness, query identity, episode
exclusion, and contract digests before a candidate reaches the model.

Episode working memory is replayed only from factual observations and actions
strictly preceding the query.  It contains the protected initial anchor,
bounded recent events, and executed-action summaries.  Current-query future
DINO/VAE features are targets only; they are never written into memory or
passed to Action DiT.

## 4. Model graph and gradient contract

The action pass sees the current observation only.  One Video DiT forward
provides layer-9/layer-19 world taps; a semantic bridge produces compact
DINO-aligned tokens.  The complete retrospective path then performs:

1. context reranking over factual candidates;
2. candidate-independent required-transition gist;
3. bounded event-action adaptation;
4. factual-effect/consequence consistency;
5. learned confidence gate;
6. stochastic source construction;
7. Action DiT flow refinement.

The source is

```text
x0 = g * (mu_memory + sigma_min * epsilon) + (1 - g) * epsilon
```

and the exact same distribution is used in training and inference.  Gate zero
is a continuous Gaussian FastWAM fallback.  Different action modes remain
separate exemplars rather than being averaged into one source.

The video co-training pass uses the 33-frame factual clip but constructs no
Action DiT tokens.  Therefore future video can supervise selected Video DiT
adapters without leaking into the action condition.  Future semantic targets
are stop-gradient auxiliary labels.

Trainable scope is explicit:

- full 1B Action DiT/action expert;
- semantic bridge, gist, reranker, event adapter, gate and conditioning heads;
- selected rank-16 Video DiT adapters;
- optional proprio bridge.

The 5B Video DiT base, VAE, text encoder, and DINO teacher remain frozen.
Zero-initialized bounded residual/context heads and gate bias `-2` preserve a
near-base starting point.

## 5. Optimizer and DeepSpeed ownership

Accelerate 1.12/DeepSpeed 0.18 own the DeepSpeed update inside the wrapped
backward call.  The Trainer does not invoke a second optimizer update.  At a
synchronized accumulation boundary it:

1. obtains and gathers the global gradient norm;
2. checks rank agreement for the BF16 overflow decision;
3. rejects non-finite updates;
4. audits critical WARM parameters after an accepted update;
5. advances the external scheduler exactly once;
6. clears gradients and increments `global_step` exactly once.

Non-DeepSpeed execution keeps the conventional optimizer-step then scheduler
step order.  Rank-invariant model initialization happens before distributed
preparation; runtime sampling is reseeded with a rank offset.  Dataset lengths,
overflow decisions, and critical-parameter finiteness are checked across all
ranks.

## 6. Epoch, sampling, and resume semantics

The resumable sampler creates one deterministic global permutation.  Accelerate
shards its batches across ranks.  `batch_in_epoch` therefore counts global
micro-batch positions; resume skips

```text
batch_in_epoch * per_device_batch * world_size
```

samples before sharding.  Epoch offset, optimizer, scheduler, RNG, sampler,
DeepSpeed partitions, model contract, and numerical-skip counter are restored.

Formal resume accepts the complete `checkpoints/state/step_NNNNNN` directory,
not a weights file.  It validates and hashes the whole state before load,
rehashes after load, verifies the step-matched weights attestation, and records
parent lineage in subsequent checkpoints.  World size, effective batch,
scheduler, source policy, artifacts, model config, and runtime changes fail
closed.  Resuming a state already at its target is idempotent and no longer
attempts to overwrite the same formal checkpoint.

## 7. Failure and shutdown behavior

- Invalid Hydra numerics/booleans fail before model allocation.
- Forward exceptions include rank, global step, epoch, batch offset and sample
  identity.
- Non-finite scalar loss is blocked before backward.
- Repeated BF16 gradient overflow aborts rather than silently corrupting all
  ranks.
- Trainer construction is cleanup-aware, so a late resume/attestation error
  still closes Accelerate/DeepSpeed.
- Normal completion and Python exceptions finish W&B and tear down NCCL once.
- Formal checkpoints never overwrite an existing weights/attestation pair.
- Quoted `HYDRA_EXTRA_ARGS` are parsed with `shlex`, without `eval`, globbing,
  or whitespace loss.

## 8. Evidence obtained

- 610 local tests pass; 12 are skipped only because the workstation lacks
  Torch/OmegaConf or symlink support.
- Shell syntax, Python compilation, configuration contracts, artifact readers,
  causal memory, source transport, attestation, and resume logic have CPU tests.
- All 11 downloaded historical resolved configs pass the current entry-point
  validator, including older configs that rely on the Trainer's default
  non-finite-skip limit.
- Server evidence includes a one-update end-to-end smoke and 40 four-H100
  optimizer updates across two resumable segments, with finite losses/gradient
  norms and a full checkpoint at step 40.

This evidence proves that the current chain can initialize, update and save; it
does not prove convergence or benchmark improvement.

## 9. Remaining operational risks

1. The workstation cannot execute CUDA/Torch tests.  A one-update clean-commit
   server smoke is still required after every training-path code change.
2. Full checkpoints contain large model and optimizer state.  Keep
   `SAVE_EVERY=2000`, monitor AFS quota before launch, and do not create a new
   final checkpoint every few wall-time steps unless that recovery point is
   intentional.
3. `EVAL_EVERY=0` is deliberate for training stability and cost.  Simulator
   rollout evaluation is a separate immutable job; training loss alone is not
   task success.
4. Formal evidence requires a clean private-repository commit and
   `ALLOW_DIRTY_WARM_TRAINING=false`.  Manually copied dirty probes remain
   unattested debugging artifacts.
5. The first clean post-audit run should remain a short `RUN_STEPS=1` probe.
   Only after its checkpoint/reload succeeds should `RUN_STEPS` be removed for
   the 19,100-step job.
