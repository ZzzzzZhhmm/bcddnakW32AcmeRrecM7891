# WARM full architecture: implemented contract

## Status and scope

This document describes the complete WARM path implemented in this repository.
The preprocessing, training, checkpoint, causal working-memory, and online
LIBERO paths are present and covered by local CPU/synthetic contract tests.
Large-scale CUDA training, learned-checkpoint evaluation, simulator success,
latency, and non-regression are **not yet established**. Follow
[`WARM_FULL_SERVER_RUNBOOK.md`](WARM_FULL_SERVER_RUNBOOK.md) to produce that
evidence from an exact clean private commit.

WARM is not a generic history concatenation layer. Its implemented mechanism
is:

```text
current factual state + task + causal episode history
    -> candidate-independent required-consequence gist

frozen-DINO coarse ANN
    -> factual cross-episode state/action/effect candidates
    -> utility reranking + independently estimated candidate effects
    -> consequence-aligned best valid candidate
    -> continuous utility-supervised gate

selected normalized action + Gaussian noise
    -> gated stochastic Action DiT source

selected event + current state
    -> candidate-aware predictive gist and action context
    -> visible to Action DiT only through the same gate
```

The core safety invariant is stronger than candidate-independent ranking: when
the gate is zero, no selected-candidate source, predictive-gist delta, or action
context reaches Action DiT.

## 1. Fixed data and action contract

The canonical full configuration targets the four FastWAM LIBERO datasets with
two ordered cameras, 33 factual frames per sample, a 32-step action chunk,
seven normalized model-space action dimensions, and eight proprioception
dimensions. The fixed coarse context is:

```text
768-dimensional frozen DINOv2 CLS + 40-dimensional task one-hot = 808
```

The implementation uses ordinary FastWAM demonstrations and automatically
derives its evidence. Required raw fields are RGB, actions, proprioception,
gripper state, task identity/text, and episode boundaries. It requires no
manual event, subtask, mask, segmentation, depth, pose, or contact annotation.

Event actions are the exact FastWAM-normalized model-space chunks stored by the
immutable bank. The current implementation does **not** perform object-frame,
goal-frame, EEF-frame, or other geometric action canonicalization. The learned
adapter adds an elementwise bounded residual in normalized action units. Its
default scale is one normalized unit per action dimension; it must not be
described as a dataset-standard-deviation bound unless explicit scales are
actually supplied.

## 2. Immutable offline artifacts

Server preprocessing produces the following provenance-bound artifacts:

1. episode-level train/dev catalog and full data audit;
2. train-only FastWAM normalization statistics;
3. frozen-DINO context/spatial features and factual Wan-VAE latents for every
   authorized train/dev episode;
4. a train-only H=32 event bank containing context, normalized action, factual
   pre/effect evidence, all H+1 factual gripper states (so the final action's
   close/release transition is retained), derived timing, support, and
   provenance;
5. independent dev oracle diagnostics;
6. leave-episode-out stride-one top-32 train/dev candidate caches;
7. run contracts binding bank, cache, normalization, encoder, catalog, split,
   and base-checkpoint identity.

No feature, bank, or candidate artifact is rebuilt silently during training.
Long-term event writes are not part of formal v1 evaluation.

## 3. Causal episode working memory

Long-term memory is cross-episode and immutable. Episode working memory records
only the current rollout's factual observations and the action prefixes that
were actually sent to the environment.

Its bounded state contains:

```text
immutable initial factual anchor
bounded factual event endpoints with protected latest observation
bounded executed-action summaries
gripper transitions, change evidence, and repeated-attempt evidence
```

Every write requires a capability tied to the current episode generation.
Predicted futures and retrieved candidates can never be committed as factual
history. The online evaluator first obtains history strictly preceding a
replan, runs the policy, executes the configured prefix, and only then commits
the new real observation and that exact executed prefix.

The training adapter causally replays this same state machine for every possible
replan-phase residue. A query sees only preceding factual writes; event
thresholds, protected-latest retention, mass-weighted bounded merges, and
executed-action summaries follow the online policy. It does not mine a query's
working memory using statistics from the future episode suffix. The current
and future semantic target are kept separate from the historical inputs.

## 4. Coarse retrieval and the semantic bridge

The two semantic mechanisms have distinct roles and should not be conflated.

### 4.1 Production v1 coarse ANN

Training candidates are precomputed from frozen-DINO context keys. Online
LIBERO evaluation runs the same contract-bound frozen DINO encoder on the
factual current camera observation, appends the catalog task one-hot, and
retrieves the top-32 entries from the immutable bank. Encoder revision,
preprocessing, camera order, normalizer, and dimensions are checked by the
online contract.

Thus the current production path is **not DINO-free online**, and the learned
semantic bridge is not claimed as the ANN query encoder.

### 4.2 In-model world/semantic bridge

A single current-frame Video DiT prefill exposes aligned intermediate token
streams from two depths. A learned weighted projection and four-query
cross-attention bridge produce compact DINO-aligned semantic tokens. These
tokens support required-consequence reasoning, episode memory, and candidate
validation. Training aligns them with stop-gradient factual DINO teacher
tokens. There is no second Video DiT pass and no test-time future-video
generation.

## 5. Candidate-independent required consequence

The Retrospective Gist Adapter is first called with:

```text
current tapped world tokens
current bridged semantic tokens
causal episode tokens and executed-action summaries
instruction tokens
zero long-term event tensors and an all-false event mask
```

It returns eight required-gist tokens and a projected required-transition
embedding. Because all candidate tensors are masked, no candidate can
manufacture the criterion that later accepts itself. Training aligns this
required transition to the factual future semantic delta with direction and
log-magnitude losses. Future targets are stop-gradient and rows without a
valid future/action target are masked.

## 6. Event adaptation and independent factual effect validation

For every valid candidate, the Retrospective Event Adapter has separate roles.

### Action branch

The stored normalized action, compact close/open timing, current world,
required gist, proprioception, and text produce:

- an adapted action mean;
- an elementwise bounded normalized-space residual;
- four action-context tokens.

The zero-initialized residual makes the initial mean equal to the retrieved
action. The adapter is constrained to refine a retrieved exemplar, not replace
it with an unconstrained second policy.

### Factual effect branch

The consequence used to validate a candidate is estimated independently of
the required gist and the language-conditioned action hidden state. It uses
that candidate's stored action, stored factual effect delta, and the current
world summary. The factual stored delta remains the residual branch's neutral
initial prediction, so distinct candidates do not collapse to one common
effect at initialization.

This branch is supervised against each event's own factual observed effect.
On ordinary non-corrupted training rows, immutable utility weights may also
provide current-query alignment. Forced hard negatives retain factual
supervision and are not relabeled as if their action had caused the current
demonstration's future. WARM therefore does not claim a fully identified
counterfactual dynamics model.

## 7. Utility reranking and consequence alignment

The lightweight reranker consumes current/candidate contexts, action summary,
factual observed effect, timing, and valid mask. Its soft targets are computed
from immutable stored candidate actions and effects:

```text
utility_i = - masked_action_distance(stored_action_i, target_action)
            - lambda * effect_distance(observed_effect_i, target_effect)
```

The learned adapted action cannot move its own ranking label. Leave-episode-out
candidate caches and visually similar but action/effect-incompatible hard
negatives prevent self retrieval and train rejection behavior.

Candidate consistency compares the independently estimated candidate effect
with the candidate-independent required transition using both direction and
log-magnitude agreement. For each row, the highest-scoring valid candidate is
selected. Candidate-only logits are not compared to an uncalibrated constant
null logit.

## 8. Continuous null/fallback gate

Null/fallback is owned by one small utility-supervised sigmoid gate. It receives
the selected score, top-one/top-two margin, consequence consistency, support,
and action-deformation norm. Its target decreases continuously with selected
action and effect error. The final bias starts at `-2`, so training begins near
the Gaussian path.

If no valid candidate exists or the corruption curriculum forces null, the
memory mask forces the gate to exactly zero. Otherwise a low gate continuously
rejects an unhelpful candidate rather than relying on an uncalibrated hard
threshold.

## 9. Candidate-aware predictive gist and conditioning

Only after candidate selection is the gist adapter called a second time with
the selected event's factual pre/effect tokens. This produces a
candidate-aware predictive gist. Its contribution is:

```text
gist_for_action = required_gist
                + stopgrad(g) * (predictive_gist - required_gist)

action_context_for_action = stopgrad(g) * selected_action_context
```

The detached gate prevents the action objective from bypassing its explicit
utility supervision. More importantly, `g=0` makes Action DiT conditioning
candidate-independent, not merely its source tensor.

## 10. Consequence-aligned stochastic source transport

For selected adapted action mean `mu`, standard Gaussian noise `epsilon`, and
minimum local source noise `sigma_min=0.2`, WARM constructs:

```text
x0 = g * (mu + sigma_min * epsilon) + (1 - g) * epsilon
```

At `g=0`, `x0=epsilon`. At larger gate values, Action DiT starts nearer the
retrieved action while retaining stochasticity. Training uses this same source
distribution with FastWAM's flow convention, and inference initializes the
unchanged integration schedule from the same construction. Replacing the
source only at inference is not supported.

The Gaussian path means **no long-term-memory influence**. Once Action DiT has
been adapted, it is not automatically identical to the untouched FastWAM
checkpoint; non-regression must be measured.

## 11. Trainable scope and losses

The complete training scope is:

```text
trainable:
  existing Action DiT / action expert
  semantic bridge and world tap
  required/predictive gist adapter
  event action/effect/context adapter
  utility reranker
  continuous source gate
  episode-action projection and proprio bridge
  rank-16 bottleneck adapters at Video DiT layers 9 and 19

frozen:
  VAE
  text encoder
  remaining Video DiT world backbone parameters
  frozen DINO teacher/retrieval encoder
```

The configured full stage uses an action pass plus an isolated video-only pass
(`lambda_action=1`, `lambda_video=1`). The action pass reads only the factual
current frame. The video pass preserves FastWAM co-training but never
constructs Action DiT tokens, so demonstrated future RGB cannot leak into the
action velocity field. Stop-gradient factual DINO features additionally
supervise the bridge, required gist, and effect branches.

The total objective combines action flow, immutable-utility ranking, semantic
bridge alignment, required-gist direction/magnitude alignment, factual effect
alignment, utility-supervised gate BCE, and bounded action adaptation. Padding,
invalid candidates, missing future targets, and padded action timesteps are
masked. The corruption curriculum includes normal retrieval, no-memory rows,
and forced effect/action-incompatible candidates.

## 12. Training and inference sequence

### Training

1. Validate base checkpoint and immutable train/dev run contracts.
2. Load a factual current frame, target action, causal episode history, and
   leave-episode-out candidate payloads.
3. Run one Video DiT current-frame prefill (frozen backbone plus selected
   adapters) and obtain two token taps.
4. Build the candidate-independent required gist.
5. Adapt candidates, independently estimate effects, rerank, compare
   consequences, and select the best valid candidate.
6. Predict the continuous gate, construct the stochastic source, and gate all
   candidate-aware conditioning.
7. Train Action DiT from that source and optimize the masked auxiliary losses.
8. In a disjoint video-only pass, preserve FastWAM future-video co-training
   while updating only the selected Video DiT adapters; no Action DiT tokens
   are constructed in this pass.
9. Save WARM module state, Action DiT state, configuration identity, and the
   trainer attestation required by formal evaluation.

### Online inference

1. Obtain only factual history preceding the current replan.
2. Build the frozen-DINO current context and contract-bound top-32 candidates.
3. Run the complete model sequence above without future teachers.
4. Integrate Action DiT from the gated source and execute exactly the configured
   prefix (`10` actions in the canonical LIBERO full task).
5. Receive the new real observation and then commit it together with the exact
   executed prefix to episode memory.
6. Emit per-replan retrieval, selected event, gate, fallback, source, latency,
   and factual-memory telemetry.

## 13. Claims deliberately not made

Until server experiments establish them, this repository does not claim:

- benchmark improvement, real-time latency, or sample-efficiency gains;
- DINO-free online retrieval;
- geometric action canonicalization or cross-embodiment transfer;
- dataset-standard-deviation-scaled residual adaptation;
- counterfactual world simulation for arbitrary candidate actions;
- generated-future writes, learned subtask completion, or online long-term
  memory consolidation;
- exact FastWAM policy parity after Action DiT training.

The implemented scientific hypotheses and required ablations are listed in the
server runbook; only reproduced GPU/simulator evidence may promote them to
empirical claims.
