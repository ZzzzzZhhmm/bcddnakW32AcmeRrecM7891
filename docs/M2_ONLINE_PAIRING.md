# M2.1 fixed/null online pairing contract

M2.1 is a **policy-specific trained-checkpoint comparison**. The source policy
is part of training, so `fixed_context_top1` and `gaussian_null` must have
distinct checkpoint files and trainer-generated attestations. It must not be
reported as a same-weights inference intervention.

## What is proved

Each training attestation proves that a checkpoint was emitted by the formal
trainer at a clean Git commit and binds its:

- checkpoint SHA-256 and completed global step;
- source policy, train and DEV source contracts, and base checkpoint;
- fully resolved training config and policy-neutral shared recipe;
- seed, optimizer, scheduler, batch/accumulation/world size, effective batch,
  precision, and target step;
- repository commit and clean status.

Each online-run contract then binds that checkpoint/attestation to the exact
M1 bank and encoder provenance, resolved evaluation config, task/simulator
artifacts, seed namespace, and frozen model assets. The fixed-side parity
report proves that production online retrieval reproduces M1 DEV retrieval for
the bound task. The pair contract binds all of these identities before either
formal rollout.

The evidence does not prove identical optimization trajectories, identical
intermediate weights, or counterfactual same-weights behavior. Claims must be
limited to the contract-bound trained-policy comparison.

## Closed-world comparison rule

The two sides must agree on every science-comparable training and online fact,
including:

- shared training recipe, seed, step, optimizer/scheduler, batch/world size,
  precision, train/DEV source contracts, and base checkpoint;
- event-bank manifest/content, encoder runtime, camera/data-processor recipe,
  DINO, normalization/action-space, catalog, and audit identities;
- VAE, text encoder, and tokenizer identities;
- evaluation namespace, task suite/id/description, root seed, initial states,
  and BDDL;
- retrieval implementation, top-k, memory sigma, action horizon/dimension;
- passing parity-report identity and clean Git commit.

The policy-specific checkpoint and attestation identities must differ. At the
evaluation-config layer, only the closed allowlist may differ:

```text
/ckpt
/model/source_policy
/EVALUATION/warm_online/contract_path
/EVALUATION/warm_online/training_attestation_path
/EVALUATION/output_dir
```

The pair and parity paths are intentionally shared and may not differ:

```text
/EVALUATION/warm_online/pair_contract_path
/EVALUATION/warm_online/parity_report_path
```

A sampler, trial count, replan schedule, prompt, inference-step count, action
processing option, initial-state path, M1 data config, asset path, or any other
config difference is a hard failure. Textual aliases are expanded and
resolved; distinct strings pointing to the same checkpoint, attestation, or
output directory do not count as independent artifacts.

## Publication order

One task-specific pair is published in this order:

1. Resolve and review the fixed/null training configs, then train both with a
   shared formal recipe. Retain each `step_NNNNNN.pt` and adjacent
   `step_NNNNNN.training.json`.
2. Resolve fixed/null evaluation configs with the same planned parity and pair
   paths and distinct checkpoint/attestation/contract/output paths.
3. Build both online-run contracts with `--data-config` and
   `--training-attestation`.
4. Run `validate_warm_online_parity.py` against the fixed online contract and
   all four LIBERO dataset roots in catalog order.
5. Build the pair contract with the passing parity report.
6. Run the actual fixed and null rollouts using the exact resolved settings.
7. Verify the two result sets with
   `verify_warm_online_pair_results.py`.

The authoritative end-to-end commands and artifact names are in
`docs/M2_ONLINE_RETRIEVAL.md`.

## Pair build command

From the clean commit bound by training and both online contracts:

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

The builder validates strict contracts, canonical resolved-config hashes,
policy labels, planned paths, distinct policy artifacts, the passing parity
report, the shared science identity, and the clean repository identity. It
re-reads inputs under an exclusive artifact claim and publishes the pair JSON
atomically. The fixed/null attestation options must name the exact sidecars
already bound by the two online contracts and their resolved configs.

Run one pair contract per task-specific rollout job. Result aggregation must
retain the pair-contract SHA-256 and parity-report SHA-256; otherwise rows from
different comparisons could be silently mixed.

## Result pairing boundary

Both policies derive a simulator seed for every task/episode from the shared
root seed. Every replan derives a policy-independent noise seed from the same
evaluation namespace and `QueryId`. The result verifier requires exact task
and episode sets and compares QueryId/seed records only over the common replan
prefix.

Different actions can legitimately cause different observations, success
times, and trajectory lengths. Therefore:

```text
paired claim: same task/episode initialization and same random seed for each
              semantically matching replan in the common prefix

not claimed:  equal observations, equal trajectories, or equal suffix length
              after policy-dependent divergence
```

The verifier also checks real termination reasons and step/replan counts,
fixed retrieval capability evidence, and Gaussian-null no-bank/no-DINO-read
evidence. The verification report is itself atomically published outside both
input result directories.

## Same-weights null intervention

A future same-checkpoint intervention is a different experiment type. Do not
circumvent this schema with equal checkpoint hashes or a hand-edited policy:
that would contradict both the training attestation and online contract. Such
an experiment requires a separate runtime mechanism, contract schema, and
paper label.
