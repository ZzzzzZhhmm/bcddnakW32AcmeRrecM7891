# M2 source-only WARM

M2 isolates one question before consequence alignment is introduced:

> Does a leave-episode-out retrieved action work better as the stochastic
> source of Action DiT than the same model's Gaussian source?

M2 contains no learned consequence scorer, learned null gate, predictive gist,
action-context token, or action adapter.  This keeps the source claim directly
falsifiable.

## Local and server boundary

`WARM_REPO` denotes the checkout root (`F:\WARM\code` is the current Windows
example). Windows is used only for source, manifest/contract validation,
synthetic CPU tests, compilation, and private Git work. PyTorch/CUDA
installation, FastWAM weights, DINO/VAE encoding, training, and simulator
rollouts remain server-only and may use a different checkout path.

The local test suite intentionally skips Torch runtime tests when PyTorch is not
installed.  The GPU smoke gate must run them and treats any skip as failure.

## Data path

The train event bank and every candidate cache remain immutable artifacts. M1's
documented `hybrid_h16` bank is an H=16 diagnostic artifact; it is not compatible
with FastWAM's H=32 training action. Build a distinct H=32 bank from the same
audited train features and contracts:

```bash
export PYTHONPATH=src
: "${WARM_M1:?Set WARM_M1 to the immutable M1 artifact root}"
: "${WARM_M2:?Set WARM_M2 to the M2 artifact/run root}"
: "${FASTWAM_BASE:?Set FASTWAM_BASE to the bound FastWAM baseline root}"
mkdir -p "$WARM_M2"

python scripts/build_warm_event_bank.py \
  --feature-list "$WARM_M1/features/train_features.list" \
  --catalog "$WARM_M1/libero_catalog.json" \
  --audit-report "$WARM_M1/libero_audit.json" \
  --output "$WARM_M1/banks/hybrid_h32" \
  --summary "$WARM_M1/banks/hybrid_h32.summary.json" \
  --normalizer-contract "$WARM_M1/features/contracts/normalizer_contract.json" \
  --encoder-contract "$WARM_M1/features/contracts/encoder_contract.json" \
  --camera-contract "$WARM_M1/features/contracts/camera_contract.json" \
  --action-horizon 32 \
  --start-mode hybrid
```

The H=16 oracle result does not authorize H=32 training. Run an independent
dev-only oracle gate against the new bank before building a train cache:

```bash
python scripts/evaluate_warm_oracle.py \
  --bank "$WARM_M1/banks/hybrid_h32" \
  --feature-list "$WARM_M1/features/dev_features.list" \
  --catalog "$WARM_M1/libero_catalog.json" \
  --audit-report "$WARM_M1/libero_audit.json" \
  --output "$WARM_M1/oracle/hybrid_h32.json" \
  --query-stride 4 \
  --top-k 1,4,8,16,32 \
  --arm-loss mse
```

Only after that H=32 gate passes, build the separate stride-one train-query
cache used by M2 training:

```bash
python scripts/build_warm_candidate_cache.py \
  --bank "$WARM_M1/banks/hybrid_h32" \
  --catalog "$WARM_M1/libero_catalog.json" \
  --audit-report "$WARM_M1/libero_audit.json" \
  --feature-list "$WARM_M1/features/train_features.list" \
  --output "$WARM_M2/candidates/hybrid_h32_train_k32" \
  --query-split train \
  --query-stride 1 \
  --top-k 32 \
  --summary "$WARM_M2/candidates/hybrid_h32_train_k32.summary.json"
```

`query_stride=1` is mandatory. Retrieval excludes the complete query episode
by global episode identity, raw source-bundle hash, and feature-episode hash.
The dev oracle/candidate artifacts remain separate and are never reused for
training.

Bind that cache and the exact baseline FastWAM checkpoint into one portable
content contract before model construction:

```bash
python scripts/build_warm_source_run_contract.py \
  --bank "$WARM_M1/banks/hybrid_h32" \
  --candidate-cache "$WARM_M2/candidates/hybrid_h32_train_k32" \
  --base-checkpoint "$FASTWAM_BASE/checkpoints/weights/final.pt" \
  --output "$WARM_M2/contracts/hybrid_h32_train_source.json" \
  --query-split train \
  --expected-action-horizon 32 \
  --expected-action-dim 7
```

Build an independent stride-one dev cache and dev source contract from the
catalog-authorized dev episodes. It shares the immutable train bank and action
contracts, but its candidate manifest and query-corpus hashes must differ from
train; reusing the train cache is a hard error:

```bash
python scripts/build_warm_candidate_cache.py \
  --bank "$WARM_M1/banks/hybrid_h32" \
  --catalog "$WARM_M1/libero_catalog.json" \
  --audit-report "$WARM_M1/libero_audit.json" \
  --feature-list "$WARM_M1/features/dev_features.list" \
  --output "$WARM_M2/candidates/hybrid_h32_dev_k32" \
  --query-split dev \
  --query-stride 1 \
  --top-k 32 \
  --summary "$WARM_M2/candidates/hybrid_h32_dev_k32.summary.json"

python scripts/build_warm_source_run_contract.py \
  --bank "$WARM_M1/banks/hybrid_h32" \
  --candidate-cache "$WARM_M2/candidates/hybrid_h32_dev_k32" \
  --base-checkpoint "$FASTWAM_BASE/checkpoints/weights/final.pt" \
  --output "$WARM_M2/contracts/hybrid_h32_dev_source.json" \
  --query-split dev \
  --expected-action-horizon 32 \
  --expected-action-dim 7
```

The builder reopens the immutable artifacts, records their content hashes,
hashes the baseline checkpoint bytes, rechecks every input immediately before
an atomic publish, and refuses overwrite by default.

At runtime `RuntimeCandidateResolver` validates one exact bank/cache snapshot
and resolves:

```text
QueryId(dataset_id, dataset_index, episode_index, frame_index)
  -> bank_rows[K], mask[K], cosine_scores[K], EventId[K]
  -> model_space_action[K,H,D]
```

The `-1` padding sentinel is never passed to NumPy gather.  A missing full-
horizon non-padded query is an error; only an explicitly recognized padded
episode tail may become an all-null row.

`RuntimeCandidateDatasetAdapter` wraps the ordinary `RobotVideoDataset` after
Hydra has instantiated it. It enforces the catalog train split and stride one,
derives `QueryId` from the sample's returned provenance (never from DataLoader
position), and emits fixed-width tensors that the default collator can stack:

```text
warm_candidate_mu[K,H,D]
warm_candidate_mask[K]
warm_candidate_score[K]
warm_candidate_event_index[K]
warm_oracle_candidate_index[]
```

The oracle index is a deterministic model-space GT-action upper bound produced
by the dataset adapter. It is ignored by fixed/null policies and is forbidden
in deployment.

## Source policies

M2 implements exactly three policies:

```text
gaussian_null:
  component = 0
  source = epsilon

fixed_context_top1:
  component = cache rank 0 when valid, otherwise 0
  source = mu + memory_sigma * epsilon

oracle_action_top1:
  component = precomputed lowest-GT-action-distance candidate
  offline/training upper bound only; inference and rollout are forbidden
```

The Gaussian is drawn once through FastWAM's original RNG path.  Selection does
not consume that RNG.  The all-null branch returns the same Gaussian tensor and
does not inspect memory payloads.

## Scheduler convention

FastWAM uses sigma one as source and sigma zero as data:

```text
x_sigma = (1 - sigma) * action_gt + sigma * source_action
target  = source_action - action_gt
```

M2 calls the unchanged scheduler:

```python
noisy_action = scheduler.add_noise(action_gt, source_action, timestep)
target = scheduler.training_target(action_gt, source_action, timestep)
```

Inference initializes `latents_action = source_action` and retains the original
descending-sigma Euler integration.  Bank actions are already normalized
FastWAM model-space `[H,D]` tensors; no denormalization or action-mode averaging
is allowed.

## Model path

`WarmSourceFastWAM.training_loss` dispatches only to the current-frame path:

```text
video[:, :, 0:1]
  -> frozen VAE
  -> frozen Video DiT t=0 prefill once
  -> layer-wise video K/V + final safe world tokens
  -> Action DiT cached path
  -> action flow loss only
```

The old `MoT.prefill_video_cache() -> list[K/V]` API remains unchanged.  The new
`prefill_video_cache_with_tokens()` returns the same cache plus final world
tokens for later milestones.  Demonstrated future frames are not encoded by
the M2 loss.

Trainer model hooks freeze VAE, text encoder, and Video DiT, and train only the
Action Expert plus the optional proprio projection.  Base FastWAM models that
do not implement the hook retain their original trainable scope.

## Run and checkpoint contract

Every formal M2 run, including Gaussian null, requires a closed
`WarmSourceRunContract` containing:

```text
bank manifest/content hashes
candidate manifest/query-corpus hashes
catalog and full audit hashes
normalizer and action-space-contract hashes
base checkpoint hash
query split, raw sample stride, action horizon and dimension
```

`global_sample_stride` must equal one.  WARM checkpoints store the source
policy, `memory_sigma`, complete run contract, and its canonical SHA-256.  Load
fails on any mismatch.  Artifact identity is content-based, not path-based.
Before Trainer construction, the model also cross-checks the actual resolver
owned by the training dataset against every bank/cache/catalog/audit/action
field in the run contract. A contract for artifact A cannot be paired with a
DataLoader reading artifact B.

The baseline checkpoint is loaded only after its SHA-256 matches the contract.
The loader rejects any checkpoint already containing `warm_source` state, so a
previous WARM run cannot silently become the declared FastWAM baseline.

The current contract does not yet bind every frozen component or simulator
evaluation setting. In particular, the Wan VAE is admitted by the existing
`WAN22_MODEL_REGISTRY` structural hash check in the loader, but the VAE bytes,
text/tokenizer assets, resolved model configuration, and evaluation policy are
not all fields of `WarmSourceRunContract` v1. The launch procedure below saves
their available provenance, but a complete frozen-component/evaluation contract
remains a required hardening item before closed-loop claims.

## Server launch skeleton

The source task config intentionally contains null artifact paths and fails
closed until they are supplied. Keep one override array for both the Hydra
composition check and the actual launch so the reviewed config is exactly the
one that runs. The train dataset and the runtime candidate adapter must both
read the same train-only statistics file:

```bash
export RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
RUN_DIR="./runs/libero_warm_source_2cam224_1e-4/$RUN_ID"
TRAIN_CONTRACT="$WARM_M2/contracts/hybrid_h32_train_source.json"
DEV_CONTRACT="$WARM_M2/contracts/hybrid_h32_dev_source.json"

WARM_OVERRIDES=(
  "task=libero_warm_source_2cam224_1e-4"
  "model.source_policy=fixed_context_top1"
  "model.run_contract_path=$TRAIN_CONTRACT"
  "model.validation_run_contract_path=$DEV_CONTRACT"
  "model.base_checkpoint_path=$FASTWAM_BASE/checkpoints/weights/final.pt"
  "data.warm_candidates.train.bank_directory=$WARM_M1/banks/hybrid_h32"
  "data.warm_candidates.train.candidate_directory=$WARM_M2/candidates/hybrid_h32_train_k32"
  "data.warm_candidates.train.catalog_path=$WARM_M1/libero_catalog.json"
  "data.warm_candidates.train.audit_report_path=$WARM_M1/libero_audit.json"
  "data.warm_candidates.train.normalization_stats_path=$WARM_M1/train_stats/dataset_stats.json"
  "data.warm_candidates.val.bank_directory=$WARM_M1/banks/hybrid_h32"
  "data.warm_candidates.val.candidate_directory=$WARM_M2/candidates/hybrid_h32_dev_k32"
  "data.warm_candidates.val.catalog_path=$WARM_M1/libero_catalog.json"
  "data.warm_candidates.val.audit_report_path=$WARM_M1/libero_audit.json"
  "data.warm_candidates.val.normalization_stats_path=$WARM_M1/train_stats/dataset_stats.json"
  "data.train.pretrained_norm_stats=$WARM_M1/train_stats/dataset_stats.json"
  "data.val.pretrained_norm_stats=$WARM_M1/train_stats/dataset_stats.json"
)

mkdir -p "$RUN_DIR"

test -z "$(git status --porcelain)" || {
  echo "Refusing a formal experiment from a dirty worktree" >&2
  exit 1
}

# Re-hash the exact Parquet and ordered camera-video bytes used by the
# training config.  Metadata hashes alone cannot detect a replaced episode.
python scripts/audit_warm_lerobot.py \
  --catalog "$WARM_M1/libero_catalog.json" \
  --dataset-root ./data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot \
  --dataset-root ./data/libero_mujoco3.3.2/libero_object_no_noops_lerobot \
  --dataset-root ./data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot \
  --dataset-root ./data/libero_mujoco3.3.2/libero_10_no_noops_lerobot \
  --output "$RUN_DIR/libero_audit.recomputed.json"
python - \
  "$WARM_M1/libero_audit.json" \
  "$RUN_DIR/libero_audit.recomputed.json" <<'PY'
import json
from pathlib import Path
import sys

expected = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
observed = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if observed.get("report_sha256") != expected.get("report_sha256"):
    raise SystemExit(
        "Current LIBERO source bytes do not match the contract-bound audit"
    )
PY

# Keep the verified dataset/artifact trees read-only for the duration of the
# run; the audit is a start-of-run proof, not a lock against later mutation.

python scripts/train.py "output_dir=$RUN_DIR" "${WARM_OVERRIDES[@]}" \
  --cfg job --resolve \
  > "$RUN_DIR/resolved_config.preflight.yaml"
git rev-parse HEAD > "$RUN_DIR/git_commit.txt"

python - "$TRAIN_CONTRACT" "$RUN_DIR/run_contract_fingerprint.json" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

source = Path(sys.argv[1])
payload_bytes = source.read_bytes()
payload = json.loads(payload_bytes)
canonical = json.dumps(
    payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
).encode("utf-8")
record = {
    "contract_file_sha256": hashlib.sha256(payload_bytes).hexdigest(),
    "contract_canonical_sha256": hashlib.sha256(canonical).hexdigest(),
}
Path(sys.argv[2]).write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

# Point this at the exact Wan2.2 VAE file selected by the resolved config.
: "${WAN_VAE_CHECKPOINT:?Set WAN_VAE_CHECKPOINT to the resolved Wan VAE file}"
python - "$WAN_VAE_CHECKPOINT" "$RUN_DIR/vae_loader_fingerprint.json" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

from fastwam.models.wan22.helpers.io import hash_model_file
from fastwam.models.wan22.helpers.loader import WAN22_MODEL_REGISTRY

path = Path(sys.argv[1]).expanduser().resolve()
loader_hash = hash_model_file(path)
registered = next(
    (
        item
        for item in WAN22_MODEL_REGISTRY
        if item["model_name"] == "wan_video_vae"
        and item["model_hash"] == loader_hash
    ),
    None,
)
if registered is None:
    raise SystemExit(
        f"Wan VAE is not admitted by WAN22_MODEL_REGISTRY: {loader_hash}"
    )
record = {
    "resolved_path": str(path),
    "loader_structural_hash": loader_hash,
    "registry_model_name": registered["model_name"],
    "registry_model_hash": registered["model_hash"],
    "state_dict_converter": getattr(
        registered.get("state_dict_converter"), "__name__", None
    ),
}
digest = hashlib.sha256()
with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
record["checkpoint_sha256"] = digest.hexdigest()
registry_record = {
    key: record[key]
    for key in (
        "registry_model_name",
        "registry_model_hash",
        "state_dict_converter",
    )
}
canonical = json.dumps(
    registry_record, sort_keys=True, separators=(",", ":"), ensure_ascii=True
).encode("utf-8")
record["registry_entry_sha256"] = hashlib.sha256(canonical).hexdigest()
Path(sys.argv[2]).write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

bash scripts/train_zero1.sh 8 "${WARM_OVERRIDES[@]}"
```

Formal WARM training publishes every weights checkpoint as an immutable pair:

```text
$RUN_DIR/checkpoints/weights/step_NNNNNN.pt
$RUN_DIR/checkpoints/weights/step_NNNNNN.training.json
```

The adjacent trainer-produced attestation binds the actual checkpoint bytes,
policy, shared training recipe, optimizer/scheduler and batch/world-size
facts, precision, seed and global step, train/DEV source contracts, base
checkpoint, and clean Git commit. Publication is no-overwrite; do not
hand-author, copy, rename independently, or regenerate the sidecar. M2.1
online contract construction requires the selected sidecar through
`--training-attestation`.

Do not continue if Hydra cannot fully resolve the configuration or if the two
normalization paths differ. The adapter additionally hashes the file and checks
it against the bank's action-space contract at runtime. Preserve the runtime
`config.yaml` alongside the preflight config, Git commit, canonical/file
run-contract hashes, and VAE loader-registry fingerprint in every experiment
directory.

Run `gaussian_null`, `fixed_context_top1`, and the training-only
`oracle_action_top1` from the same baseline contract and candidate payload.
This keeps initialization, data order, Gaussian draws, and artifact identity
paired; only source selection changes.

The portable task deliberately leaves `eval_every=0`; after all overrides
above are present, a formal launch may set it positive. Regardless of whether
periodic validation is scheduled, formal training attestation requires the
independent `DEV_CONTRACT` alongside the train contract. If validation is
enabled, runtime also validates the DEV dataset, requires
`warm_query_split=dev`, and rejects train/dev candidate or query-corpus
identity reuse before validation starts.

## Required GPU smoke gate

Install the server dependencies and first run the mandatory Torch tests without
allowing the local no-Torch skips:

```bash
export WARM_REQUIRE_TORCH_TESTS=1
python -m pytest -q \
  tests/test_runtime_candidate_dataset_adapter.py \
  tests/test_source_transport.py \
  tests/test_warm_source_model_torch.py
python -m pytest -q
python -m compileall -q src scripts tests
```

Before formal training, the server must then pass:

1. all tests, with `tests/test_source_transport.py` executed rather than
   skipped, including `tests/test_warm_source_model_torch.py` under
   `WARM_REQUIRE_TORCH_TESTS=1`;
2. disabled/null source parity with base FastWAM under identical seeds;
3. old/new prefill K/V equality and final-token shape checks;
4. one source-only forward/backward proving Video DiT has no gradients and
   Action Expert does;
5. checkpoint round-trip with strict run-contract binding;
6. future-frame perturbation invariance of the source-only loss;
7. Gaussian/null action-only inference smoke from the bound baseline checkpoint;
8. fixed-context action-only forward/backward smoke using the offline
   candidate payload and exact train dataset adapter.

M2.1 now provides a contract-bound online frozen-DINO retrieval bridge for
LIBERO. Closed-loop success becomes reportable only after the server passes the
additional gates in `docs/M2_ONLINE_RETRIEVAL.md`: fixed/null QueryId and seed
parity, online-vs-offline dev feature/top-K parity, strict checkpoint binding,
and one real CUDA LIBERO rollout with complete per-replan telemetry. Until
those gates pass, the local implementation makes no closed-loop performance
claim.
