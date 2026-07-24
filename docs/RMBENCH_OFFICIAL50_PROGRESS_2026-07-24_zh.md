# WARM RMBench official50 阶段性进展说明（截至 2026-07-24）

本文档面向方案与工程负责人，对照 `docs/RMBENCH_SOTA_TRAINING_EVALUATION_PLAN_ZH.md`
的 Gate 与数据 profile，记录 **official50-dev45 bootstrap** 在 CCI/ACP 共享
AFS 上已完成的工作、产物路径、验收数据与代码版本。大型数据与 artifact 不入
Git；Git 仅保存代码、配置与本说明。

服务器 WARM 仓库与 GitHub 私有库 `ZzzzzZhhmm/WARM` 的 `main` 分支一致时，
HEAD 应为文末「代码版本」中的 commit。

---

## 1. 目标与执行依据

| 文档 | 用途 |
|---|---|
| `docs/RMBENCH_SOTA_TRAINING_EVALUATION_PLAN_ZH.md` | SOTA 路线：profile、shared/specialist、Gate A-D、4 卡训练与 ACP 评测 |
| `docs/WARM_FULL_SERVER_RUNBOOK.md` | 第 8 节 official RMBench 严格路线（与 SOTA 并行参考） |
| `configs/rmbench/sota_v1.json` | 机器可读 task/profile 注册表 |

当前阶段目标：**official50-dev45** 快速打通（方案文档 3.1 / Gate A），而非
scale200 主力成绩。scale200 采集与 M1/M2 尚未开始。

固定协议 pin（与仓库一致）：

| 项 | 值 |
|---|---|
| RMBench 代码 revision | `57ee09cbc6267bc36ca0ac2d8d1c5c3b245c112c` |
| Hugging Face 数据 revision | `855e90e1213d150bf4889130e83398f107314681` |
| 数据 profile | `official50-dev45`（九任务各 50 demo，train 45 / dev 5，seed 3407） |
| WARM 训练根 seed | `3407` |

---

## 2. 里程碑进度（对照 SOTA 方案 Gate）

| Gate / 阶段 | 状态 | 说明 |
|---|---|---|
| 外部依赖（RMBench checkout + HF 快照） | **完成** | 见第 4 节路径 |
| Gate A：数据与转换 | **完成** | 450 episode LeRobot，验证通过 |
| Gate B：M1/M2 | **完成** | feature/bank/oracle/cache/contract 已发布 |
| Gate C：训练（10-step 探针） | **完成（单卡 CCI）** | shared smoke 10 step；方案建议 4 卡探针，ACP 可选复跑 |
| Gate C：训练（shared 30k） | **未开始** | 需 ACP 4×H100 |
| 9× specialist 30k | **未开始** | 需 ACP |
| Gate D：仿真评测 | **未开始** | 需 RMBENCH_ROOT + 一任务一卡 ACP |
| scale200 采集 + M1/M2 | **未开始** | 无 CCI MuJoCo 采集条件 |

---

## 3. 产物与质量数据

### 3.1 产物根目录

```text
WARM_ARTIFACT_ROOT=/mnt/afs/task3_2/L202500276_lwz/projects/WARM_artifacts/rmbench_official50_v1
```

不可覆盖；若重建须换新根目录。

### 3.2 数据与外部快照（AFS，非 Git）

| 用途 | 路径 |
|---|---|
| HF 官方数据快照 | `/mnt/afs/task3_2/L202500276_lwz/datasets/rmbench_official50_hf_snapshot`（约 39 GB） |
| HF revision marker | `/mnt/afs/task3_2/L202500276_lwz/datasets/rmbench_hf_revision.json` |
| RMBench 只读 checkout | `/mnt/afs/task3_2/L202500276_lwz/external/RMBench-official` @ `57ee09c...` |
| LeRobot 转换结果 | `/mnt/afs/task3_2/L202500276_lwz/datasets/rmbench_official50_lerobot`（约 1.6 GB） |
| 文本 T5 embedding cache | `/mnt/afs/task3_2/L202500276_lwz/projects/WARM_artifacts/text/rmbench_official50_v1`（9 个 task prompt） |

### 3.3 M1

| 产物 | 相对路径 | 关键数据 |
|---|---|---|
| catalog | `m1/rmbench_catalog.json` | 450 episodes，九任务 official50 profile |
| audit | `m1/rmbench_audit.json` | 三相机 MP4 + parquet 审计，train/dev 无重叠 |
| conversion manifest | `m1/rmbench_conversion_manifest.json` | 绑定 HF/code revision |
| train stats | `m1/train_stats/` | 405 train episodes，249966 frames |
| DINO+VAE features | `m1/features/`（约 5.6 GB） | 450/450 episode |
| event bank H=32 | `m1/banks/hybrid_h32/` | **46783 events**，405 source train episodes |
| oracle 报告 | `m1/oracle/hybrid_h32.json` | dev 查询 stride 4，6423 queries，coverage 100% |

**Oracle（dev，相对 context top-1 / recent baseline 的平均相对改善）**：

| top-K | coverage | vs context | vs recent |
|---:|---:|---:|---:|
| 1 | 100% | 0% | 66.8% |
| 4 | 100% | 42.7% | 81.4% |
| 8 | 100% | 55.5% | 85.8% |
| 16 | 100% | 66.0% | 89.2% |
| 32 | 100% | **73.6%** | **91.9%** |

top-32 相对 context 改善远高于「检索是否值得做」的粗门槛（与 LIBERO 门类似
的 15–20% 量级对比），**Gate B oracle 门通过**。

### 3.4 M2

| 产物 | 关键数据 |
|---|---|
| `m2/candidates/hybrid_h32_train_k32/` | 405 feature caches，237006 queries，7584192 candidates，空查询 0 |
| `m2/candidates/hybrid_h32_dev_k32/` | 45 feature caches，25494 queries，815808 candidates，空查询 0 |
| `m2/contracts/hybrid_h32_train_source.json` | train source-run contract |
| `m2/contracts/hybrid_h32_dev_source.json` | dev source-run contract |

FastWAM base（contract 绑定）：  
`checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt`

### 3.5 训练冒烟（Gate C 探针）

| 项 | 值 |
|---|---|
| 阶段 | `WARM_RMBENCH_STAGE=shared`，`WARM_RMBENCH_DATA_PROFILE=official50-dev45` |
| 步数 | `WARM_MAX_STEPS=10` |
| 硬件 | CCI 单卡 H100 80GB |
| global batch | 128（per_device 8 × grad_accum 16 × world 1） |
| Run 目录 | `/mnt/afs/task3_2/L202500276_lwz/projects/WARM/runs/rmbench_official50_shared_smoke10_20260724_094845` |
| 权重 | `.../checkpoints/weights/step_000010.pt`（约 12 GB） |
| Attestation | `.../checkpoints/weights/step_000010.training.json`（git `0d380fb...`，bf16 ZeRO-1） |

日志（AFS，非 Git）：  
`/mnt/afs/task3_2/L202500276_lwz/datasets/rmbench_official50_prepare.log`、  
`/mnt/afs/task3_2/L202500276_lwz/datasets/rmbench_cci_post_m1m2.log`

---

## 4. 测试与注册表校验

在 `warm` 环境、`PYTHONPATH=src:.` 下执行：

```bash
python scripts/plan_warm_rmbench_sota.py --data-profile official50-dev45 \
  --validate-manifest "$WARM_ARTIFACT_ROOT/m1/rmbench_conversion_manifest.json"
pytest tests/test_rmbench_sota.py tests/test_rmbench_server_scripts_static.py \
  tests/test_rmbench_conversion_validator.py -q
```

截至 2026-07-24：**11 passed**。conversion 全量字节哈希验证在 prepare 链中已做；
正式训练前 `train_warm_rmbench_server.sh` 使用 `--skip-artifact-byte-hashes` 快速
复核 manifest/catalog identity。

---

## 5. 代码版本与 RMBench 相关提交

| Commit | 说明 |
|---|---|
| `8836071` | Add score-oriented RMBench WARM pipeline（SOTA 脚本/注册表/文档） |
| `0d380fb` | 修复 conversion validator 与 `robotwin_qpos_action_contract` 关键字调用 |

**当前 `main` HEAD（文档编写时）**：`0d380fb09014f91bff62650fb41151f766fc71a7`

---

## 6. CCI 环境说明（不入库，供运维参考）

以下仅在 AFS 上，用于 CCI 容器重启后构建；ACP 正式任务应使用干净 checkout +
`train_warm_rmbench_server.sh`，不依赖这些 wrapper。

| 项 | 路径或做法 |
|---|---|
| 容器 git/SSH 自愈 | `bash scripts/cci_bootstrap.sh` |
| AFS 上 `renameat2` 失败 | `datasets/warm_afs_site/` 对 LeRobot 发布做 fallback |
| CCI `python`  wrapper | `datasets/bin/warm_python`（保证 `warm` 环境与 `-c` 参数） |
| 一键 post-M1M2 | `datasets/cci_rmbench_post_m1m2.sh`（text cache + smoke） |

已知坑：formal 脚本要求 **clean git**；`train_stats` / feature 发布亦校验工作树。
CCI 上曾 stash 局部修改或先 commit 再跑 formal 链。

---

## 7. 下一步（ACP 4×H100）

1. **shared 正式训练**：`NPROC_PER_NODE=4`，`PER_DEVICE_BATCH_SIZE=8`，
   `TARGET_GLOBAL_BATCH_SIZE=128`（grad_accum 自动为 4），去掉 `WARM_MAX_STEPS`，
   新 `WARM_TRAIN_OUTPUT`。详见 SOTA 文档第 6.1 节。
2. **九个 specialist**：同上 4 卡配置，逐任务 `WARM_RMBENCH_STAGE=specialist`。
3. **scale200**：simulator 自动采集 → 新 `WARM_ARTIFACT_ROOT` → 再训练（SOTA 默认 profile）。
4. **评测**：dev 选 checkpoint → `build_warm_rmbench_sota_task_contract_server.sh` →
   ACP 单卡 `evaluate_warm_rmbench_task_server.sh`（每任务 100 episode）。

RMBench **无** `acp_warm_libero.sh` 式统一 wrapper；ACP 任务命令为设置环境变量后
执行 `bash scripts/train_warm_rmbench_server.sh`（或评测脚本）。

---

## 8. 与 LIBERO 产物的并列关系

| 项目 | 产物根 |
|---|---|
| LIBERO official50 等价链 | `.../WARM_artifacts/libero_v1`（已完成，见 `docs/STATUS_REPORT_2026-07-16_zh.md`） |
| RMBench official50 bootstrap | `.../WARM_artifacts/rmbench_official50_v1`（本文档） |

两者独立；不可混用 checkpoint 或 contract。
