# M1 offline event-bank pipeline

This milestone proves that useful cross-episode action chunks exist before any
FastWAM parameter is trained. All large arrays remain server-local. Git stores
only the code and small JSON summaries/manifests.

## Artifact chain

```text
LeRobot episodes + split catalog
  -> parquet + ordered-camera byte audit
  -> train-only FastWAM normalization artifact
  -> GPU feature precompute (one immutable cache per episode)
  -> train-only fixed-horizon event bank
  -> dev leave-episode-out candidate cache
  -> oracle action-distance report
```

Every boundary checks the exact catalog, train-only action normalizer and its
manifest, encoder bundle, camera layout, source-episode content, encoded
feature content, dtype, shape, and SHA-256. A query is excluded by global
episode identity, raw-source hash, and identity-independent feature-content
hash.

## State/action alignment

The canonical feature-cache time axis is:

```text
model_actions:      [T, H_action]
proprio:            [T+1, H_state]
observed gripper:   [T+1]
context keys:       [T+1, D_key]
semantic features:  [T+1, ..., D_sem]
optional VAE:       [T+1, ...]
```

For a LeRobot episode with one action recorded at each observed frame, use the
first `N-1` actions and all `N` factual observations. This supplies a real
post-state for every retained action. WARM never pads or resamples an event
chunk.

## 1. Catalog and audit

Run on the GPU server after downloading the four FastWAM LIBERO archives:

```bash
export WARM_M1=/server/artifacts/warm/m1
mkdir -p "$WARM_M1"

python scripts/build_warm_episode_catalog.py \
  --dataset-root data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot \
  --dataset-root data/libero_mujoco3.3.2/libero_object_no_noops_lerobot \
  --dataset-root data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot \
  --dataset-root data/libero_mujoco3.3.2/libero_10_no_noops_lerobot \
  --dev-per-task 5 --seed 20260713 \
  --output "$WARM_M1/libero_catalog.json"

python scripts/audit_warm_lerobot.py \
  --catalog "$WARM_M1/libero_catalog.json" \
  --dataset-root data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot \
  --dataset-root data/libero_mujoco3.3.2/libero_object_no_noops_lerobot \
  --dataset-root data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot \
  --dataset-root data/libero_mujoco3.3.2/libero_10_no_noops_lerobot \
  --hash-episode-tables \
  --output "$WARM_M1/libero_audit.json"
```

The production audit hashes every parquet plus both ordered camera MP4s. It
exits non-zero when a table, a camera file, an ordered camera bundle, or a full
source bundle is duplicated across splits. Feature-level content hashes provide
a second exact duplicate check after visual encoding. `--metadata-only` exists
only for inspection and cannot authorize a production cache.

## 2. Train-only normalization statistics

Do not use a dataset-level statistics JSON whose episode membership is
unknown. Compute FastWAM-compatible global min/max from the catalog-authorized
train episodes only:

```bash
python scripts/compute_warm_train_stats.py \
  --catalog "$WARM_M1/libero_catalog.json" \
  --audit-report "$WARM_M1/libero_audit.json" \
  --dataset-root data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot \
  --dataset-root data/libero_mujoco3.3.2/libero_object_no_noops_lerobot \
  --dataset-root data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot \
  --dataset-root data/libero_mujoco3.3.2/libero_10_no_noops_lerobot \
  --data-config configs/data/libero_2cam.yaml \
  --output "$WARM_M1/train_stats"
```

The immutable output contains exactly `dataset_stats.json` and
`train_stats_manifest.json`. The manifest binds the catalog, full audit, data
config, exact sorted train source bundles, counts, dimensions, recipe, and Git
commit. Statistics use all raw rows, including the final recorded action, to
match FastWAM's global non-stepwise min/max recipe. Dev/test rows can never
influence normalization.

## 3. Feature caches

`save_episode_feature_cache` is the source-of-truth cache writer. Each cache
contains only numeric arrays (`allow_pickle=False`) and an adjacent version-3
manifest. `load_episode_feature_cache` returns a `LoadedFeatureCache`; its
arrays are read-only and remain bound to the verified manifest.

Use a complete local DINOv2 snapshot pinned to its exact 40-character Hub
commit. Validate the full plan without importing CUDA dependencies:

```bash
python scripts/precompute_warm_features.py \
  --data-config configs/data/libero_2cam.yaml \
  --catalog "$WARM_M1/libero_catalog.json" \
  --audit-report "$WARM_M1/libero_audit.json" \
  --dataset-root data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot \
  --dataset-root data/libero_mujoco3.3.2/libero_object_no_noops_lerobot \
  --dataset-root data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot \
  --dataset-root data/libero_mujoco3.3.2/libero_10_no_noops_lerobot \
  --dataset-stats "$WARM_M1/train_stats/dataset_stats.json" \
  --dataset-stats-manifest "$WARM_M1/train_stats/train_stats_manifest.json" \
  --dino-checkpoint /server/checkpoints/dinov2-base-pinned \
  --dino-revision "$DINO_COMMIT_SHA" \
  --output "$WARM_M1/features" \
  --plan-only
```

Remove `--plan-only` on the CUDA server to publish the immutable train+dev
cache. DINO runs on the external camera; both cameras pass through FastWAM's
exact validation image transforms and horizontal layout. The default M1 oracle
does not require Wan VAE features. Add `--include-vae --vae-checkpoint ...`
only for an explicitly VAE-dependent ablation.

The entrypoint emits the three immutable contract JSON files used below and
exact copies of the data config, train stats, and train-stats manifest:

```text
normalizer_contract.json
encoder_contract.json
camera_contract.json
```

Their raw file SHA-256 values are stored in every episode cache. Changing even
JSON whitespace therefore creates a new artifact contract rather than silently
reusing stale features. Independently computed train-only and dev-only caches
use one catalog-bound train+dev task vocabulary, so their key dimensions remain
compatible while test instructions remain invisible.

The normalizer contract is also the closed-world action-space contract.  For
LIBERO v1 it binds the seven model channels, arm/gripper partition, model-space
gripper threshold, control mode, embodiment, normalization recipe, and the raw
SHA-256 of the exact FastWAM dataset-statistics file.  Oracle tools read these
semantics from the bank; CLI dimension flags are consistency assertions only.

## 4. Build the train event bank

Put train feature-cache paths in `train_features.list`, one path per line:

```bash
python scripts/build_warm_event_bank.py \
  --feature-list "$WARM_M1/features/train_features.list" \
  --catalog "$WARM_M1/libero_catalog.json" \
  --audit-report "$WARM_M1/libero_audit.json" \
  --output "$WARM_M1/banks/hybrid_h16" \
  --summary "$WARM_M1/banks/hybrid_h16.summary.json" \
  --normalizer-contract "$WARM_M1/features/contracts/normalizer_contract.json" \
  --encoder-contract "$WARM_M1/features/contracts/encoder_contract.json" \
  --camera-contract "$WARM_M1/features/contracts/camera_contract.json" \
  --action-horizon 16 \
  --start-mode hybrid
```

The builder does not trust a cache's self-reported split.  Every episode id,
split, raw table hash, and `N observation -> N-1 action` length is re-proved
against the immutable catalog and its fully hashed audit report.  Only a
catalog-authorized `train` collection can publish a bank.

The builder refuses a dirty Git worktree by default and records the exact Git
commit in the bank provenance. `--allow-dirty` exists only for synthetic tests
and debugging, not accepted experiment artifacts.

Bank directories are immutable snapshots: the production CLI has no overwrite
mode, refuses a pre-existing output path, reloads the saved bank, and validates
its full WARM payload contract before publishing the external summary.

Build `uniform`, `event`, and `hybrid` banks with otherwise identical
contracts. Do not call event mining a contribution unless its
recall/utility-per-byte beats uniform sampling.

## 5. Cache dev candidates

```bash
python scripts/build_warm_candidate_cache.py \
  --bank "$WARM_M1/banks/hybrid_h16" \
  --feature-list "$WARM_M1/features/dev_features.list" \
  --catalog "$WARM_M1/libero_catalog.json" \
  --audit-report "$WARM_M1/libero_audit.json" \
  --output "$WARM_M1/candidates/hybrid_h16_dev" \
  --summary "$WARM_M1/candidates/hybrid_h16_dev.summary.json" \
  --query-stride 4 \
  --top-k 32
```

The candidate cache accepts only catalog-authorized `dev` queries.  Its own
manifest records the exact event-bank manifest/content hashes, query-corpus
hash, catalog/audit binding, retrieval implementation, horizon, stride, top-K,
and all three episode-exclusion rules. Loading it against a different bank,
recipe, or regenerated query corpus fails before training.

## 6. Evaluate the oracle gate

LIBERO model-space actions have arm dimensions `0..5` and gripper command
dimension `6`:

```bash
python scripts/evaluate_warm_oracle.py \
  --bank "$WARM_M1/banks/hybrid_h16" \
  --feature-list "$WARM_M1/features/dev_features.list" \
  --catalog "$WARM_M1/libero_catalog.json" \
  --audit-report "$WARM_M1/libero_audit.json" \
  --output "$WARM_M1/oracle/hybrid_h16.json" \
  --query-stride 4 \
  --top-k 1,4,8,16,32 \
  --arm-loss mse
```

`--arm-dims`, `--gripper-dims`, and `--gripper-threshold` may be supplied as
defensive assertions, but cannot override the bank's immutable action-space
contract.  Oracle evaluation is deliberately dev-only; final test rollouts are
not a hyperparameter-selection surface.

Proceed to source-only WARM only when top-32 oracle action distance is roughly
15--20% lower than both context top-1 and the previous-chunk action prior.
Otherwise change the fixed key/effect representation or data split before
spending 5B-model GPU time.

## Local verification

The Windows workstation runs only deterministic CPU checks:

```powershell
Set-Location F:\WARM\code
python -m pytest -q
python -m compileall -q src scripts
```

Actual feature encoding, FastWAM checkpoint loading, training, and simulator
rollouts happen only after pulling the exact private Git commit on the Linux
CUDA server.
