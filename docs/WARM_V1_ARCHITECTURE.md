# WARM v1 staged specification (historical)

> **Status:** This file records the earlier staged M1/M2 design and the safety
> constraints that motivated it. It is not the authoritative specification of
> the now-implemented complete model. Use
> [`WARM_FULL_ARCHITECTURE.md`](WARM_FULL_ARCHITECTURE.md) for the code-level
> architecture and
> [`WARM_FULL_SERVER_RUNBOOK.md`](WARM_FULL_SERVER_RUNBOOK.md) for the
> executable server workflow. Where this file proposes a learned online ANN
> bridge, hard null component, identity-only source adapter, FIFO-only working
> memory, or postponed full adapter, the full documents and current code
> supersede it. Safety requirements such as candidate-independent required
> consequence, factual-only episode writes, future isolation, and normalized
> model-space action terminology remain applicable.

## 1. Research claim

WARM is a retrieval-augmented Fast-WAM that uses a past factual
`pre-state -> action -> observed effect` event only when that observed effect is
compatible with the task-conditioned world transition currently required. The
accepted action becomes a source component for Action DiT flow refinement.

The safe claim is the complete mechanism, not any component in isolation:

1. predict the currently required consequence independently of long-term
   candidates;
2. compare it with the factual consequence stored by each candidate;
3. select a retrieved action source or an explicit Gaussian null source;
4. refine from that source with Action DiT.

WARM v1 does **not** claim a counterfactual dynamics model, cross-embodiment
action translation, semantic event labels, or a new foundation model trained
from scratch.

## 2. Corrections to the design draft

The following constraints are mandatory before implementation:

- The required-consequence representation may use current world tokens,
  instruction, proprioception, and real episode memory, but no long-term
  candidate payload. This prevents candidate self-confirmation.
- V1 compares a candidate's historically observed effect. It does not supervise
  every candidate action with the current sample's future and does not claim to
  know a counterfactual effect in the current scene.
- All long-term paths share the same null decision. A null selection removes
  long-term source and action context; no candidate-dependent gist remains.
- The bank key space is frozen. The online query bridge learns to map FastWAM
  world tokens into that fixed space. A trainable key encoder requires a full
  offline reindex and is not part of v1.
- Flow code follows FastWAM's convention: `sigma=1` is source and `sigma=0` is
  data. WARM supplies a conditional `source_action`; it does not introduce a
  second scheduler convention.
- Arm trajectories are continuous. Gripper state is discrete and uses
  zero-order hold/timing loss; it is never linearly interpolated as an arm pose.
- For LIBERO, the first action-space implementation is the existing normalized
  6D delta-EEF plus gripper representation. Dataset normalization is not called
  geometric canonicalization.

## 3. Training topology

The primary implementation is a two-stage adaptation, because retrieval must
happen before the action source is noised and passed to Action DiT.

### Stage 0: FastWAM base

Load the released FastWAM checkpoint that has already received video/action
co-training. Preserve the original joint loss implementation for baseline
reproduction. Freeze the VAE, text encoder, and Video DiT during WARM v1.

### Stage 1: action-only WARM adaptation

For each training sample:

1. encode only the current real frame;
2. run the existing action-only Video DiT prefill once;
3. retain layer-wise video KV cache and return the final first-frame world
   tokens;
4. build the query and load a precomputed leave-episode-out candidate pool;
5. predict the candidate-independent required consequence;
6. score factual candidate effects plus the null candidate;
7. sample/select a source component;
8. sample one action flow timestep and train Action DiT against the source-to-
   action velocity using the cached video K/V.

This path matches inference, does not expose future video tokens to action, and
does not require a second Video DiT pass. Future semantic features are training
targets only.

## 4. Event bank contract

V1 stores fixed-horizon chunks aligned with the base policy. Change points
choose useful query starts; they do not arbitrarily resample an entire event.
Uniform, change-point, and hybrid chunk banks must be compared before event
mining is elevated to a contribution.

Each entry contains:

```text
identity:
  event_id, dataset_id, dataset_revision
  episode_id, task_id, start_step, key_step, end_step
  split, embodiment_id, control_mode, action_space_version

fixed retrieval fields:
  context_key
  instruction_key
  start_proprio

factual world evidence:
  external_camera_pre_tokens
  external_camera_post_tokens
  effect_tokens
  effect_magnitude
  optional wrist retrieval tokens

action payload:
  model_space_action[H, D]
  optional physical_space_action[H, D] (post-M2/debug only)
  gripper_state[H]
  timing metadata and validity mask

provenance:
  encoder id/revision
  normalization-stat hash
  feature-schema version
  source episode content hash
```

The external fixed camera is the v1 source of consequence evidence. Wrist
features may help retrieve context but are not directly differenced to define
world effect because camera ego-motion dominates.

M2 consumes `model_space_action` directly because it is already produced by
the exact FastWAM normalizer.  It must not denormalize and renormalize that
payload.  A physical-space copy is not required by the source-only mechanism
and may be added later only as an explicitly versioned diagnostic payload.

The source of truth is sharded tensor payload plus tabular metadata and a
manifest. ANN indices and candidate caches are rebuildable derived artifacts.
For a LIBERO-scale pilot, exact normalized matrix search is preferred over a
new FAISS dependency. ANN becomes necessary only after profiling larger banks.

No prototype averaging, cluster statistics, or online long-term writes are in
v1. In particular, distinct action modes must never be averaged.

## 5. Episode working memory

The minimal online memory is deliberately simple:

```text
initial external-camera anchor
last 2-4 real observation features
last 2 executed action summaries
gripper transitions and retry count
```

It resets for every rollout and is updated only after a real observation. V1
uses FIFO retention; no learned subtask completion, slot merging, or generated
future is written into persistent memory.

## 6. World feature and fixed key space

FastWAM action-only inference already prefills the first-frame video branch and
caches its per-layer K/V. Extend that prefill API, backward compatibly, to also
return the final video tokens. Because `first_frame_causal` prevents first-frame
queries from seeing future frames, the corresponding training tokens are safe
from future leakage.

The offline bank uses a frozen patch-level visual teacher, frozen text features,
and fixed preprocessing. The online semantic bridge maps the final FastWAM
first-frame tokens into the same frozen key/consequence space.

During the oracle phase, online teacher features are allowed to test whether
the retrieval hypothesis is viable. The bridge is implemented only after the
oracle passes; its acceptance target is at least 95% of oracle Recall@K with at
most a two-point closed-loop success gap.

## 7. Independent required consequence

Define

```text
e_req = RequiredConsequenceEncoder(current_world, instruction,
                                   proprio, episode_memory)
```

No long-term candidate enters this encoder. Training uses the factual future of
the demonstrated action chunk as a stop-gradient semantic target. A soft
change-weighted pooling over external-camera patch features is preferred to raw
same-coordinate patch subtraction; both pre and post tokens remain available.

Candidate `i` supplies an observed factual consequence

```text
e_obs_i = ObservedEffectEncoder(pre_i, post_i, action_summary_i)
```

trained only on its own historical event. V1 does not predict what candidate
`i` would counterfactually cause in the current scene.

## 8. Candidate consequence scorer and null

One scorer replaces a chain of overlapping reranker/effect/gate modules:

```text
score_i = context_compatibility(current, pre_i)
        + consequence_direction(e_req, e_obs_i)
        - consequence_magnitude_mismatch(e_req, e_obs_i)
        + fixed support features
```

The logits include a learned Gaussian null component:

```text
pi = softmax([score_null, score_1, ..., score_K])
```

Training ranking targets are computed from fixed payload, never from a
candidate prediction that can game its own label:

```text
utility_i = - action_distance(stored_action_i, gt_action)
            - lambda * effect_distance(observed_effect_i, gt_effect)
```

Candidates are retrieved and hard negatives are cached offline. The scorer is
trained with a soft ranking target. Hard negatives deliberately share visual
context but differ in action direction, task phase, gripper timing, or observed
effect.

## 9. Retrieved source distribution

For a selected memory component `i`:

```text
source_i = mu_i + sigma_i * epsilon
```

For null:

```text
source_null = epsilon
```

Sampling a categorical component from `pi` defines a real mixture and keeps
discrete action modes separate. Deterministic top-1 is available for ablation,
but may claim only local stochasticity.

V1 first uses a deterministic action-space transform. On LIBERO delta-EEF this
is identity in physical delta coordinates followed by the exact FastWAM
normalizer. A learned bounded action adapter is postponed until source-only
evidence is positive.

With FastWAM's existing scheduler:

```text
noisy_action = add_noise(gt_action, source_action, timestep)
target       = training_target(gt_action, source_action, timestep)
```

Inference initializes `latents_action = source_action` and uses the unchanged
descending-sigma integration schedule.

## 10. Gradient routing

The main adaptation phase keeps mechanism identification explicit:

- action flow loss updates Action DiT LoRA/adapters;
- source component, candidate probabilities, and null choice are stop-gradient
  with respect to action flow loss;
- candidate scorer is trained by fixed utility ranking targets;
- required consequence is trained by future semantic alignment;
- any later bounded action adapter is trained by a separate arm trajectory loss
  plus gripper state/timing loss and residual regularization.

Only after all components pass their isolated tests may a low-learning-rate
joint fine-tuning ablation allow action loss into the source modules.

## 11. Inference

```text
current RGB
  -> VAE
  -> single-frame VideoDiT prefill
  -> final first-frame world tokens + video KV cache
  -> fixed-space query + episode memory
  -> retrieve K factual events
  -> required consequence (candidate independent)
  -> candidate consequence scorer + Gaussian null
  -> sample/select source component
  -> cached ActionDiT flow integration
  -> execute a short prefix
  -> update episode memory with real feedback
```

No complete future video, online segmentation, VLM planner, learned event
segmenter, or online long-term write is required.

## 12. Fallback terminology

Before base-model LoRA is enabled, selecting null with all residual outputs
zero-initialized can be tested for numerical parity with FastWAM. After Action
DiT adaptation, null is accurately described as the **Gaussian no-long-memory
path**, not an exact unchanged FastWAM policy. Both parity and non-regression
must be measured rather than asserted.
