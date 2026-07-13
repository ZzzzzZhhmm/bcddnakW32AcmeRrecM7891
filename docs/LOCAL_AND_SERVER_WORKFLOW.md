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
dataset/normalizer revision
base checkpoint hash
resolved Hydra configuration
Python/CUDA/PyTorch/driver versions
random seeds and world size
```

Results are accepted only when they can be traced back to this record.

## Promotion sequence

1. Local CPU tests and compile checks pass.
2. Push the exact commit to `ZzzzzZhhmm/WARM`.
3. Pull that commit on the GPU server; never copy an uncommitted working tree.
4. Build or verify external artifacts against their manifests.
5. Run GPU smoke tests and record the resolved environment.
6. Run the current milestone's go/no-go experiment.
7. Return only small summaries/manifests to Git; keep large artifacts external.
