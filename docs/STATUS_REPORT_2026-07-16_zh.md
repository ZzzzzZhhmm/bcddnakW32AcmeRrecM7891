# WARM LIBERO 阶段性工作总结与情况说明（截至 2026-07-16）

本文档面向总项目工程师，完整记录 LIBERO 方向从环境准备到 M1/M2 不可变产物链
建成、训练冒烟门调试的全部工作：做了什么、依据什么协议做的、完成质量如何
（附关键数据）、代码改了哪里、踩过哪些环境坑，以及下一步的 ACP 训练安排。

服务器代码与 GitHub 私有仓库 `ZzzzzZhhmm/WARM` 的 `main` 分支保持一致，
本文档末尾附提交对照表。

---

## 1. 目标与执行依据

目标：按 `docs/IMPLEMENTATION_ROADMAP.md` 的里程碑推进 WARM 的 LIBERO
训练实验。执行依据以仓库内文档为准：

| 文档 | 用途 |
|---|---|
| `docs/IMPLEMENTATION_ROADMAP.md` | 里程碑 M0-M7 的交付物与验收门 |
| `docs/WARM_FULL_SERVER_RUNBOOK.md` | 服务器执行手册（本次严格按其第 1-6 节推进） |
| `docs/M1_OFFLINE_PIPELINE.md` | M1 产物链协议 |
| `docs/M2_SOURCE_ONLY.md` / `docs/M2_ONLINE_RETRIEVAL.md` | M2 训练与 fixed/null 对照证据要求 |
| `docs/DATA_AND_EVAL_PROTOCOL.md` | 数据切分与评测协议 |

核心原则（roadmap 强制）：更便宜的 oracle 门不通过，不允许启动任何 5B
级全模型训练；所有正式产物要求干净 git 工作树并绑定 SHA-256 身份。

## 2. 里程碑进度总览

| 里程碑 | 状态 | 说明 |
|---|---|---|
| M0 基线与可复现 | 完成 | FastWAM 基线权重、环境、许可证均就位（此前已有） |
| M1 数据审计与 oracle 事件库 | **完成，验收门通过** | 见第 3 节数据 |
| M2 产物（候选缓存 + contract） | **完成** | 见第 3 节数据 |
| M2 训练（source-only） | **冒烟门通过** | 首次冒烟暴露一个 attestation bug，修复后单卡 1 步全流程通过 |
| M3-M7 | 未开始 | 完整 WARM 的模块代码已实现，等训练验证 |

对应 runbook 章节：第 1-3 节（克隆、输入定义、特征/事件库/候选/contract
构建）全部完成；第 4 节（Hydra 全量解析）通过；第 5 节（GPU 冒烟门）
**通过**；下一步是第 6 节（正式训练，ACP 多卡）。

## 3. 产物链详情与质量数据

所有产物位于 `WARM_ARTIFACT_ROOT=/mnt/afs/task3_2/L202500276_lwz/projects/WARM_artifacts/libero_v1`，
不可覆盖（重建需换新根目录）。数据集为四个 LIBERO 套件的 LeRobot 格式
（`data/libero_mujoco3.3.2/*_no_noops_lerobot`）。

### 3.1 M1（全部完成）

| 产物 | 位置（相对产物根） | 关键数据 |
|---|---|---|
| episode catalog | `m1/libero_catalog.json` | 1712 个 episode，每任务 45 train / 5 dev 分层切分，seed 20260713 |
| 字节级审计 | `m1/libero_audit.json` | parquet + 双相机 MP4 全量哈希，train/dev 零重叠零重复 |
| train-only 归一化统计 | `m1/train_stats/` | 仅 train 行参与，dev 不可影响统计 |
| DINO+VAE 特征 | `m1/features/`（5.7 GB） | 1712/1712 episode 编码完成；DINOv2-base 固定 revision `f9e44c81...1415`，Wan2.2 VAE |
| H=32 事件库 | `m1/banks/hybrid_h32/`（4.7 GB） | **172,293 个事件**，hybrid 起点抽取，含动作/效果负载与内容哈希 |
| oracle 门报告 | `m1/oracle/hybrid_h32.json` | 6,960 个 dev 查询（stride 4），见下表 |

**oracle 门结果（M1 验收的决定性指标）**——要求 top-32 oracle 动作距离比
context top-1 低至少 15-20%，实测：

| top-K | oracle 动作距离 | 相对 context top-1 改善 |
|---|---|---|
| 1 | 0.1760 | 0%（基准） |
| 4 | 0.1017 | 42.2% |
| 8 | 0.0753 | 57.2% |
| 16 | 0.0538 | 69.5% |
| 32 | 0.0391 | **77.8%** |

覆盖率 100%（每个查询在泄漏排除后仍有候选）。**77.8% 远超 15-20% 门槛，
门通过**：事件库中确实存在大量高价值历史动作，检索表示有效，按协议可以
进入训练阶段。

### 3.2 M2（全部完成）

| 产物 | 关键数据 |
|---|---|
| train 候选缓存 `m2/candidates/hybrid_h32_train_k32/` | stride-1，**195,972 个查询、6,271,104 条候选**，空查询 0 |
| dev 候选缓存 `m2/candidates/hybrid_h32_dev_k32/` | stride-1，26,957 个查询、862,624 条候选，空查询 0 |
| train contract `m2/contracts/hybrid_h32_train_source.json` | SHA-256 `804db751...3248` |
| dev contract `m2/contracts/hybrid_h32_dev_source.json` | SHA-256 `98100db6...0544` |

两份 contract 均绑定事件库 manifest/content 哈希、catalog/audit 哈希、
归一化统计哈希与 FastWAM 基线 checkpoint 哈希（`1000437c...9579`），
训练器在构造前会交叉核验这些哈希，防止产物错配。

检索实现为精确余弦 + 整 episode 泄漏排除（身份 + source/feature 内容
哈希三重排除），与 oracle 评估共用同一 `EventBank.search` 代码路径。

## 4. 代码改动清单（按提交顺序）

以下提交均已推送到 GitHub `main`，服务器与远端一致。

| 提交 | 内容 | 动机与验证 |
|---|---|---|
| `867f316` | 新增 `scripts/acp_warm_libero.sh` | ACP/服务器统一入口，五种 RUN_KIND（download_dino / prepare_artifacts / prepare_m2 / oracle_check / train），内含消融参数说明与 TODO 高亮 |
| `641448e` | DINO 下载走可配置 HF 镜像 | 集群无法直连 huggingface.co，走 hf-mirror.com（commit SHA 与官方一致） |
| `59d0038` | `.gitignore` 增加 `tmp/` | ACP 日志目录不入库 |
| `05ea308` | 放宽 DINO `config.image_size` 与推理尺寸的强等校验 | dinov2-base 预训练 518x518，WARM 用 224x224；DINOv2 位置编码支持动态插值，原校验过严导致特征预计算中断 |
| `99240c7` | 相机像素契约加 1e-5 舍入容差并 clip 回 [0,1] | 抗锯齿双线性缩放对饱和像素产生 1.000000238 级浮点越界，原严格校验误报 |
| `7ac66c1` | ACP 容器内注入任务级 git safe.directory | root 容器 + 用户属主仓库触发 git dubious ownership 拒绝 |
| `13bdc5a` | 新增 RUN_KIND=prepare_m2（断点续跑） | 容器重启导致整链脚本无法中途恢复；已完成产物自动跳过，半成品报错提示删除 |
| `32f5ba9` | prepare_m2 扩展为可补跑 oracle | 第一次 oracle 跑了 8 小时被容器重启杀掉（无中间落盘），需要可恢复的重算入口 |
| `ec5f110` | **向量化 `EventBank.search` 排除掩码 + 缓存 float64 键矩阵** | 见下方专项说明 |
| `2cc98f6` | 训练 attestation 剔除 Accelerate 注入的 `steps_per_print=inf` | 见第 6 节冒烟门问题 |

### 4.1 检索性能优化专项（`ec5f110`，对结果零影响）

问题：原 `EventBank.search` 每次查询用 Python 循环遍历全部 172,293 个
事件构造泄漏排除掩码（每事件两次 `np.array_equal` 哈希比较），并且每次
查询都把整个 172,293x768 键矩阵重新拷贝为 float64（约 1GB 分配）。实测
单次检索 1.08 秒；oracle 要对 5 个 top-K 各做一遍全量检索（6,960x5 =
34,800 次），实跑 10.5 小时；train 候选缓存 195,972 次检索按此速度需要
2.5-3 天，在容器几小时一重启的环境中不可能完成。

修复：排除掩码改为整体向量化 numpy 布尔运算（episode 身份用惰性缓存的
int64 列数组比较，哈希排除用 `(hash_rows == digest).all(axis=1)`）；
float64 键矩阵与范数加载后缓存一次。排序逻辑（stable argsort）一字未动。

验证（三层）：
1. 111 个相关单测全部通过；
2. 在真实 172,293 事件库上抽样 17 个带完整三重排除的查询，新旧实现的
   32 个候选**索引、事件 ID、分数逐位相同**；
3. 单次检索 1084ms -> 88ms（12.3 倍），train 缓存实际总耗时约 5.3 小时
   完成（原估 2.5-3 天）。

## 5. 基础设施问题记录（对后续 debug 重要）

1. **CCI 容器每隔数小时整体重启**（非单纯网络断连）。证据：tmux 服务进程
   被杀、`~/.ssh` 与 `/tmp` 被清空、所有 nohup 进程消失，而宿主机 uptime
   40+ 天。影响与对策：
   - tmux/nohup 只能防 SSH 断连，防不住容器重启；
   - 所有长任务必须"可断点续跑 + 产物只写 AFS"；`prepare_m2` 即按此设计；
   - 第一次 oracle 评估（8 小时、无中间落盘）因此报废一次；
   - **正式训练必须放 ACP 独立作业，且用 `SAVE_EVERY` + `RESUME` 防 walltime**。
2. 网络：`huggingface.co` 不可达，走 `hf-mirror.com`；GitHub 443/22 均不
   通，走 `ssh.github.com:443`（SSH key 已在 AFS 留有备份，容器重启后可恢复）。
3. git：容器以 root 运行、仓库属主为提交用户，需 safe.directory 例外
   （ACP 脚本已内置自动注入）。

## 6. 当前状态：训练冒烟门（runbook 第 5 节）

按 runbook 要求，正式训练前先单卡跑 1 个优化步。首次冒烟（2026-07-16
凌晨）结果：

- **通过的部分**：Hydra 全量 preflight 解析成功（21 项 WARM 产物覆盖注入
  正确）；5B 模型组件加载、基线 checkpoint 覆盖、DeepSpeed ZeRO-2 优化器
  初始化全部正常（单卡显存峰值约 20.1 GB）；
- **失败点**：Trainer 构造期生成训练 attestation 时抛
  `TrainingAttestationError: resolved training config must contain only
  finite canonical JSON values`。

根因（已定位并修复，`2cc98f6`）：Accelerate 1.12 的 `DeepSpeedPlugin`
无条件向 `deepspeed_config` 注入 `steps_per_print = float("inf")`（源码
`accelerate/utils/dataclasses.py:1290`，用途仅为关闭 DeepSpeed stdout
日志），而 WARM 的 attestation 要求配置可序列化为规范 JSON（禁止
inf/nan）。修复为在捕获运行时配置时剔除该注入值（它是日志节奏而非数值
训练事实，不影响可复现性），attestation 相关 21 个单测通过。

**修复后冒烟门已通过**（2026-07-16 凌晨，退出码 0）：单卡完成 1 个完整
优化步——基线 checkpoint 加载、contract 哈希核验、前向/反向/优化器更新、
checkpoint 保存（`runs/libero_warm_source_2cam224_1e-4/20260715_181530_*/
checkpoints/weights/step_000001.pt`）以及配套的 `.training.json`
attestation 全部成功；单卡显存峰值约 20.1 GB（batch_size=1，ZeRO-2）。
runbook 第 5 节的冒烟清单（checkpoint 加载、有限梯度、存取 checkpoint
等）已满足，可以提交正式 ACP 训练。

另注：第二次冒烟曾因 "formal WARM checkpoint publication requires a
clean Git worktree" 失败——训练器在发布 checkpoint 时强制要求干净工作树
（当时本报告文档尚未提交所致），这是设计行为；ACP 提交前确保所有改动已
提交即可。

## 7. 下一步计划与 ACP 启动命令

实验顺序（依据 roadmap M2 的正式证据要求）：

1. **M2 source-only 对照对**：消融 (b) gaussian_null 与 (c)
   fixed_context_top1，同一 recipe 各训一个带 `.training.json` attestation
   的 checkpoint（M2 验收要求成对呈现）；
2. **完整 WARM**：消融 (e)，验证后果对齐重排 + 连续 gate + 语义桥；
3. FastWAM 基线 (a) 不需重训，发布 checkpoint 即对照。

ACP 任务 command（8 卡；脚本会自动做 preflight、选 ZeRO stage、注入产物
覆盖并核验哈希）：

```bash
# (c) M2 source-only, fixed source
cd /mnt/afs/task3_2/L202500276_lwz/projects/WARM && \
RUN_KIND=train TASK_NAME=libero_warm_source_2cam224_1e-4 SOURCE_POLICY=fixed_context_top1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 \
bash scripts/acp_warm_libero.sh

# (b) M2 source-only, Gaussian null：仅改 SOURCE_POLICY=gaussian_null

# (e) 完整 WARM
cd /mnt/afs/task3_2/L202500276_lwz/projects/WARM && \
RUN_KIND=train TASK_NAME=libero_warm_2cam224_1e-4 SOURCE_POLICY=fixed_context_top1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 \
bash scripts/acp_warm_libero.sh
```

ACP 注意事项：

- 任务有 walltime 上限时加 `MAX_STEPS=<步数> SAVE_EVERY=<间隔>`，下一段用
  `RESUME=runs/<task>/<run_id>/checkpoints/state/step_NNNNNN` 续跑；
- 卡数不足 8 时改 `NPROC_PER_NODE`（脚本自动切 ZeRO-2），可用
  `PER_DEVICE_BATCH_SIZE=8 GRADIENT_ACCUMULATION_STEPS=auto` 维持全局
  batch 128（`TARGET_GLOBAL_BATCH_SIZE` 可调）；
- 训练要求干净 git 工作树（脚本与仓库内部双重强制），改代码后先提交；
- 完整 WARM（含 33 帧 video pass）显存高于 M2，OOM 时参考脚本头部的
  显存组合建议（`MOT_CHECKPOINT_MIXED_ATTN=true` 等）。

## 8. 风险与排查提示（供后续 debug 参考）

- 产物链不可覆盖：`prepare_*` 检测到已存在产物会拒绝执行，重建需换新的
  `WARM_ARTIFACT_ROOT`；候选缓存的完成标志是目录内 `candidate_manifest.json`
  （有目录无 manifest 即半成品，需手动删除后续跑）；
- 训练器构造前会核验 contract 中全部 SHA-256，任何产物路径/内容错配都会
  在启动早期显式报错——这是设计行为，不要绕过；
- attestation 对配置的规范 JSON 要求较严，升级 accelerate/deepspeed 版本
  时注意其向配置注入的非 JSON 值（本次 `steps_per_print=inf` 即一例）；
- 依赖版本敏感点：accelerate 1.12.0、Python 3.10、PyTorch 2.7.x/CUDA 12.8
  （见 BASELINE.md 与环境锁定）；
- oracle 报告 50 MB 含全部 per-query 明细，可用于进一步的检索质量分析。

## 附：本阶段提交对照（服务器与 GitHub main 一致）

```
2cc98f6 Exclude Accelerate's injected infinite steps_per_print from training attestation
ec5f110 Vectorize event-bank search exclusion mask and cache the float64 key matrix
32f5ba9 Extend prepare_m2 to rebuild the missing oracle report before M2 caches
13bdc5a Add resumable prepare_m2 run kind for finishing M2 caches and contracts after M1
7ac66c1 Inject job-scoped git safe.directory for root ACP containers
99240c7 Tolerate float rounding overshoot from antialiased resize in camera pixel contract
05ea308 Allow DINO snapshots whose pretraining resolution differs from WARM inference size
59d0038 Ignore ACP job log directory
641448e Route DINO snapshot download through configurable HF mirror endpoint
867f316 Add ACP entrypoint for WARM LIBERO artifact preparation and training
```
