# WARM implementation roadmap

## Operating rules

- All development and Git history stay in the private `ZzzzzZhhmm/WARM`
  repository. Do not fork, push to, or open pull requests against FastWAM.
- Preserve required upstream and third-party license notices.
- Large data, weights, features, banks, rollout videos, and secrets are external
  artifacts referenced by versioned manifests.
- No full-model experiment begins until its cheaper oracle gate passes.

## M0 - private baseline and reproducibility

Deliverables:

- clean FastWAM import pinned to revision `45d8e145...`;
- root and third-party license notices;
- private origin only;
- environment lock for Python 3.10, PyTorch 2.7.1/CUDA 12.8;
- baseline LIBERO inference command and checkpoint/data manifests;
- CPU unit-test harness and one GPU smoke test.

Acceptance:

- released checkpoint loads;
- one LIBERO task runs end-to-end;
- standard FastWAM output is unchanged before WARM is enabled;
- checkpoint round-trip and seed reproducibility pass.

No-go: do not implement memory against a baseline that cannot be reproduced.

## M1 - data audit and oracle bank

Deliverables:

- full-episode LeRobot reader;
- byte-complete dataset audit for parquet plus ordered external/wrist MP4s;
- stratified 45-train/5-dev episode split per LIBERO task;
- immutable train-only FastWAM global-min/max artifact and provenance manifest;
- import-safe server precompute CLI for factual DINOv2 and optional Wan VAE
  features, with a fixed catalog-bound train+dev task vocabulary;
- uniform, change-point, and hybrid fixed-horizon event extractors;
- immutable bank schema and exact-search backend;
- leave-episode-out offline candidate cache;
- retrieval diagnostics dashboard.

Acceptance:

- zero episode overlap and zero exact parquet/camera/source/feature duplicate
  hash across splits;
- dev/test rows cannot affect normalization statistics or key vocabulary;
- every official feature collection exactly covers its requested catalog split;
- all chunks preserve FastWAM action horizon and normalization contract;
- top-32 oracle candidate action distance is at least 15-20% lower than plain
  context top-1 and recent-action prior;
- hybrid/event recall-per-byte must beat uniform before event mining is called a
  contribution.

No-go: if useful actions do not exist in the bank, redesign representation and
splits before any 5B training.

## M2 - source-only WARM

Deliverables:

- backward-compatible VideoDiT prefill API returning final first-frame tokens;
- action-only WARM training loss with existing KV cache and scheduler;
- fixed/oracle retrieval source sampler plus Gaussian null;
- Action DiT LoRA or explicitly scoped action-expert fine-tuning;
- source geometry metrics and Gaussian-parity tests.
- online observation-to-query retrieval bridge before any fixed-context
  closed-loop evaluation is claimed;
- an independently catalog-bound dev dataset/cache/run contract before periodic
  Trainer validation is enabled;
- immutable trainer-produced checkpoint attestations binding the actual
  policy-specific weights, shared training recipe, optimizer/runtime facts,
  train/dev source contracts, base checkpoint, seed/step, and clean Git commit;
- exact M1-processor and frozen-DINO runtime binding for online retrieval;
- a passing task-specific online/offline parity report, fixed/null pair
  contract, and post-rollout pair-result verification report.

Implementation note: M2 training consumes a distinct stride-one train-query
candidate cache. A closed source-run contract binds that cache, the event
bank, catalog/audit, action normalizer, and baseline checkpoint; the model and
Dataset resolver cross-check the same hashes before Trainer construction.
The M2.1 implementation now includes the online bridge and independent dev
contract path. A formal WARM training attestation always requires both train
and catalog-bound stride-one DEV source contracts, even when `eval_every=0` and
no periodic validation loop is scheduled. Local implementation tests are not
evidence of fixed-memory rollout success: on the server, training produces the
two policy-specific checkpoint attestations first; resolved evaluation configs
then bind those artifacts and the planned parity/pair paths; online contracts
are published; exact fixed-side DEV parity passes; the pair contract is
published; only then may actual fixed/null LIBERO rollouts and pair-result
verification run.

Compare the identical retrieved payload as context, residual, and source, plus
FastWAM and WarmPrior-style recent-action source.

Acceptance:

- source reduces expected source-to-GT displacement/flow curvature by about
  15% or more;
- fixed-context closed-loop acceptance is evaluated only after the online
  retrieval bridge is present and tested;
- closed-loop success improves at least about 3 percentage points over
  context-only and recent-action prior with positive three-seed confidence;
- ordinary Markov tasks regress no more than about 2 points.
- every reported fixed/null result is traceable through its training
  attestation, online contract, passing parity report, pair contract, and
  pair-result verification report.

No-go: if source is not better than context/residual, source transport cannot
remain the main contribution.

## M3 - consequence-aligned selection

Deliverables:

- independent RequiredConsequenceEncoder;
- factual ObservedEffectEncoder using external-camera pre/post evidence;
- one CandidateConsequenceScorer with null logit;
- fixed utility targets and cached hard negatives;
- action/effect shuffle sensitivity tests.

Acceptance:

- hard-negative pair accuracy at least about 70%;
- at least 10-point improvement over context-only candidate scoring;
- action/effect shuffle causes a substantial score drop;
- selected candidates have materially better GT action utility than context
  retrieval.

No-go: if consequence cannot reject visually similar, effect-incompatible
events, the central contribution is not supported.

## M4 - corruption, null, and safety

Deliverables:

- random, reversed-action, wrong-phase, wrong-effect, empty-bank, and unsupported
  action-space tests;
- calibrated null probabilities;
- failure telemetry: selected event id, scores, source component, bank hash.

Acceptance:

- helpful-vs-harmful AUROC about 0.80 or higher;
- corrupted-memory performance within about 2 points of the no-long-memory
  path;
- incompatible embodiment/control/action space always selects or forces null.

No-go: unreliable rejection blocks long-term source use in deployment.

## M5 - episode memory

Deliverables:

- per-rollout resettable initial anchor and short FIFO;
- executed action/gripper summary;
- training samples containing only preceding real events;
- LIBERO/RMBench wrappers with explicit reset/update lifecycle.

Acceptance:

- memory benchmark improves by about 5 points over source-only and simple frame
  stack;
- ordinary tasks do not regress more than about 2 points.

No-go: if raw frame stacking is equal or better, remove custom short-memory
logic from the paper.

## M6 - semantic bridge and scale

Deliverables:

- frozen online key space and VideoDiT-to-key bridge;
- sharded tensor payload/memmap I/O;
- optional ANN backend after exact-search profiling;
- 100k-entry latency and memory benchmark.

Acceptance:

- at least 95% of online-teacher Recall@K;
- closed-loop gap no more than about 2 points;
- WARM overhead below about 10% of end-to-end FastWAM inference latency.

## M7 - final evaluation

Main matrix:

1. FastWAM;
2. visual/action memory as context only;
3. retrieved source without consequence alignment;
4. full WARM;
5. full WARM under memory corruption/null.

Benchmarks:

- LIBERO four suites: standard ability and non-regression;
- LIBERO-PRO position/task perturbations: prevent trajectory-copy explanation;
- RMBench pilot: Put Back Block, Rearrange Blocks, Battery Try;
- RMBench all nine tasks only after the pilot passes;
- LIBERO-Plus is optional robustness work, not a first-paper blocker.

Report success, confidence intervals/seeds, flow displacement/curvature, ODE
steps, retrieval recall/utility, hard-negative accuracy, null AUROC, gate/null
rate, latency, peak GPU memory, bank I/O, and trainable parameters.

## Compute assumptions

The Windows workstation is a source-development machine only.  It is used for
editing, metadata/contract checks, and deterministic CPU unit tests; do not run
checkpoint feature encoding, simulator rollouts, or FastWAM 5B/6B training on
it.  Those jobs run from an exact private Git commit on the Linux CUDA server.

Plan the first full LIBERO memory-profile run on a Linux CUDA node. Until a
100-step peak-memory profile exists, budget the official conservative baseline:
one eight-GPU node. Adapter-only training may reduce this, but that is a measured
optimization rather than a promise.
