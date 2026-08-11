# RMBench pre-training qualification gate

Do not start shared or specialist WARM optimization merely because M1/M2 files
exist.  The derived RMBench artifacts must pass the no-training qualification
gate first:

```bash
PYTHONPATH=src python scripts/qualify_warm_rmbench_artifacts.py \
  --bank "$WARM_ARTIFACT_ROOT/m1/banks/hybrid_h32" \
  --train-feature-list "$WARM_ARTIFACT_ROOT/m1/features/train_features.list" \
  --train-candidate-cache "$WARM_ARTIFACT_ROOT/m2/candidates/hybrid_h32_train_k32" \
  --dev-feature-list "$WARM_ARTIFACT_ROOT/m1/features/dev_features.list" \
  --dev-candidate-cache "$WARM_ARTIFACT_ROOT/m2/candidates/hybrid_h32_dev_k32" \
  --output "$WARM_ARTIFACT_ROOT/m1/qualification/rmbench_h32.json" \
  --action-horizon 32 \
  --query-stride 4 \
  --max-event-stride 4 \
  --phase-tolerance 0.10 \
  --phase-recall-threshold 32=0.85 \
  --phase-recall-threshold 128=0.95 \
  --require-partial-action-queries
```

The command is fail-closed and publishes a report only after all checks pass.
It validates:

1. `normalized_phase`, `event_ordinal`, `successor_row`,
   `successor_event_start_frame`, and `action_valid_mask` for every event;
2. dense event coverage (no gap larger than the replanning stride), exact
   within-episode successor links, and the absence of synthetic/padded actions;
3. same-task, phase-compatible candidate recall independently for every task
   and every early/middle/late trajectory stratum, not only as a global average;
4. deterministic candidate-cache parity against a fresh exact search using the
   same factual cached context key and complete-episode leakage exclusions.
5. exact cache coverage of every catalog observation, including terminal
   partial-horizon action chunks. Such rows keep action retrieval/source/gate
   supervision while future-effect and gist losses are masked.

The parity check deliberately teacher-forces the factual context key.  It
proves the query-to-candidate boundary shared by training and online retrieval;
it does **not** prove rendered-image preprocessing parity.  Before a long run,
the server qualification sequence is therefore:

1. this artifact gate;
2. official expert-action replay in the simulator, with before/target/after
   joint telemetry and task predicates;
3. one-demo and five-demo overfit tests;
4. a 200–500 step smoke run with per-loss gradient norms and gate positive /
   negative examples;
5. only then shared WARM pretraining and specialist adaptation.

A missing temporal payload is not backward-compatible.  It means the old M1
bank and both M2 candidate caches must be rebuilt from the reusable per-frame
DINO/VAE feature caches.
