# Local and GPU-server workflow

WARM uses one private Git history but separates source development from
large-scale execution.

## Local workstation scope

The local workstation is the authoritative environment for:

- source and configuration changes;
- pure CPU unit tests and synthetic integration fixtures;
- event/bank schema, hashing, split, and leakage checks;
- CLI argument and manifest validation;
- Python compilation and static source inspection;
- private Git commits, branches, and review.

The current Windows checkout is `F:\WARM\code`. Keep implementation work on
the F: drive; do not create a second working copy under Documents for WARM.

Local checks do not claim that the 5B/6B model can be trained or evaluated on
this machine. GPU-only tests must be explicitly marked and are not silently
replaced by smaller-model evidence.

## GPU server scope

The Linux CUDA server is required for:

- installing the pinned FastWAM CUDA stack;
- loading Wan/FastWAM model weights;
- DINO and Wan VAE feature precomputation;
- 100-step peak-memory and throughput profiling;
- FastWAM baseline reproduction;
- WARM LoRA/adaptor training;
- LIBERO, LIBERO-PRO, RoboTwin, and RMBench rollouts;
- multi-seed metrics and final ablations.

Before a full run, execute a single-batch forward/backward smoke test, a
checkpoint round-trip, and a short simulator rollout. Resource assumptions are
only promoted to documented requirements after this profile.

## Transfer contract

Only code, small configs, tests, and content-hash manifests are transferred by
the private Git repository. The following stay in server-local or approved
artifact storage:

```text
datasets
base and trained checkpoints
DINO/VAE feature caches
event-bank payloads and indices
candidate caches
rollout videos
training/evaluation outputs
credentials
```

Every server run records:

```text
Git commit SHA
bank manifest SHA-256
episode catalog SHA-256
full parquet/camera audit SHA-256
train-only normalizer stats and manifest SHA-256
base checkpoint hash
resolved Hydra configuration
Python/CUDA/PyTorch/driver versions
random seeds and world size
```

Formal WARM training additionally publishes a no-overwrite
`step_NNNNNN.training.json` beside each `step_NNNNNN.pt`. Never treat weights
without that trainer-produced attestation as an admissible M2 online
checkpoint.

Results are accepted only when they can be traced back to this record.

The end-to-end commands for the learned consequence-aligned model are in
[`WARM_FULL_SERVER_RUNBOOK.md`](WARM_FULL_SERVER_RUNBOOK.md). The M1/M2
documents remain the isolated retrieval/source diagnostics and should not be
mistaken for the complete-model launch recipe.

The official RMBench path is likewise server-only and uses these closed
launchers from the clean private commit:

```text
scripts/prepare_warm_rmbench_artifacts.sh
scripts/train_warm_rmbench_server.sh
scripts/evaluate_warm_rmbench_server.sh
scripts/run_warm_rmbench_matrix.py
```

All four refuse mutable formal outputs. Their shared guard permits only the
configured WARM `origin`, verifies the pinned external checkout has push URL
`DISABLED`, and defaults experiment/model hubs to offline mode. Large converted
RMBench data and matrix results remain external artifacts, not Git content.

## Promotion sequence

1. Local CPU tests and compile checks pass.
2. Fast-forward the exact commit to the private `ZzzzzZhhmm/WARM` `main`
   branch.
3. Clone or pull that exact private `main` commit on the GPU server; never
   copy an uncommitted working tree.
4. Build or verify external artifacts against their manifests.
5. Run GPU smoke tests and record the resolved environment.
6. Run the current milestone's go/no-go experiment.
7. Return only small summaries/manifests to Git; keep large artifacts external.

For M2.1, step 6 expands to a strict artifact chain:

```text
policy-specific training checkpoint + attestation
  -> resolved fixed/null evaluation configs
  -> fixed/null online-run contracts
  -> fixed-side per-task online/offline parity report
  -> fixed/null pair contract
  -> actual fixed/null rollouts
  -> pair-result verification report
```

The online contracts require the exact M1 data config through `--data-config`
and each exact checkpoint sidecar through `--training-attestation`. The pair
cannot be published before parity, and formal rollouts cannot start before the
pair exists. See `docs/M2_ONLINE_RETRIEVAL.md` for commands.

After authenticating the server for the private repository, a fresh checkout
is intentionally simple:

```bash
git clone --branch main --single-branch git@github.com:ZzzzzZhhmm/WARM.git
cd WARM
git status --porcelain
git rev-parse HEAD
```

The status command must print nothing. Record the resulting commit SHA in
every artifact and experiment manifest.
