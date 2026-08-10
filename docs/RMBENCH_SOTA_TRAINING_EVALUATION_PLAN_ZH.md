# WARM × RMBench 高成功率训练与评测实施方案

更新日期：2026-07-22
目标：在不修改 RMBench 九项任务定义、成功判定与每任务 100 次正式 rollout 的前提下，以最终成功率为首要目标，完成可复现、可审计的 WARM 训练与评测链路。

## 1. 最终工程决策

本路线不是把 LIBERO checkpoint 直接拿到 RMBench 上测试。RMBench 使用三相机、双臂 14D qpos 控制和专门的记忆任务，因此需要独立的数据转换、M1/M2 构建、完整 WARM 训练和在线评测。

最终采用三层模型策略：

1. `shared`：九任务联合训练，是统一 WARM foundation checkpoint，也是 specialist 失败时的回退模型。
2. `specialist`：每个任务从同一 FastWAM base 独立训练，只读取该任务的 train/dev episode；用于冲击最高任务成功率。
3. checkpoint selection：只用 dev split 选择 shared/specialist checkpoint 和运行参数；正式 100-episode 测试集不参与调参。

specialist 暂不从 shared WARM checkpoint 热启动。原因不是效果判断，而是当前正式训练 attestation 只完整绑定 FastWAM base → WARM 的初始化关系；在没有新增“WARM checkpoint fork provenance”之前，静默热启动会使 provenance 失真。待主链稳定后可单独实现并消融。

固定根 seed 为 `3407`。保留现有 RMBench 三相机协议，不为了对齐某一竞品减少相机或改变任务。

## 2. 已完成的代码能力

### 2.1 严格数据 profile

机器可读注册表位于 `configs/rmbench/sota_v1.json`，profile 定义位于 `src/fastwam/benchmarks/rmbench_sota.py`：

| profile | 每任务总 episode | train | dev | 用途 |
|---|---:|---:|---:|---|
| `official50-dev45` | 50 | 45 | 5 | 官方数据快速打通和基线 |
| `scale200-dev190` | 200 | 190 | 10 | 第一主力训练集，当前 SOTA 注册表默认 |
| `scale500-dev480` | 500 | 480 | 20 | 第二阶段扩量，重点用于 M(n) |

转换器会验证九个任务的精确 episode 数、源数据身份、固定 seed 切分和 manifest。任何数量不符、任务缺失或 profile 冒充都会在构建前失败。

`official50-dev45` 只能接受固定官方数据源 `TianxingChen/RMBench`。`scale200`/`scale500` 必须是服务器上用固定 simulator/task config 自动采集并记录 provenance 的数据源，不能把同一 episode 复制多份凑数量。

### 2.2 自动 event 与 task/event-balanced sampling

M1 沿用完整 WARM 的无人工标注处理：DINO/VAE factual features → action-grounded change points → state-action-effect event bank → train/dev candidate cache。无需手工 event、subtask、mask、pose 或 contact 标签。

训练 sampler 为 `rmbench_task_event_balanced`：

- 每个 task 的总采样概率严格归一为相同，避免长轨迹任务主导 minibatch；
- 每个 task 内，最近一个 action-summary chunk 中出现 factual event 的 query 权重乘 `1.5`；
- replacement sampling 保持每个 epoch 的更新数不变；
- `seed + epoch + resume offset` 决定采样，支持确定性续训。

### 2.3 真正的 task specialist

新增 `episode_task_allowlist` 后，specialist 会从底层 LeRobot episode catalog 真正过滤训练和验证 episode，而不是只给共享数据换一个任务名。候选 cache、QueryId、catalog 和 normalization contract 仍严格校验。

### 2.4 task-specific 在线记忆和控制

每个任务 profile 同时定义：

- `train_steps`；
- M(1)/M(n) 的 recent factual event 容量；
- executed-action summary 容量；
- `replan_steps`；
- Action DiT ODE steps；
- retrieval top-K。

这些值已经贯通：训练数据分块 → checkpoint retrospection config → 在线 episode memory → official policy runtime projection → online contract → evaluator。不会再出现“训练按 10 步摘要、评测按 4 步重规划”或“配置写 20 个 event、线上仍硬编码 6 个”的情况。

## 3. 数据策略

### 3.1 先跑官方 50 条 bootstrap

目的不是最终成绩，而是完成以下 gate：

- 九任务转换和三相机视频完整；
- M1 feature/event/bank 与 M2 top-32 cache 完成；
- oracle top-K 报告正常；
- shared 训练能稳定运行；
- 至少一个 M(1) 和一个 M(n) 任务能跑完 5–10 episode dev rollout。

官方 50 条应保留真实 dev5。不要把 50 条全部放入 train 后再用 test 选 checkpoint；那会失去可靠的模型选择依据。

### 3.2 自动扩展到 200 条/任务

这是第一主力方案。服务器需要在固定 RMBench commit、`demo_clean` task config、三相机 observation 和 14D qpos action 下，用 simulator expert 自动采集到每任务 200 条成功 episode。建议同时记录失败尝试到独立 negative archive，但只把成功、完整、可重放的 episode 放入 positive source bank。

每条自动采集 episode 必须通过：

1. task success；
2. 相机帧数、qpos/action 长度和时间对齐；
3. 无 NaN/Inf；
4. 初始状态和随机 seed 可追溯；
5. episode 内容 hash 去重；
6. task name 与官方九任务闭集一致。

当前仓库已经完成转换、验证和下游 artifact 链路；实际 simulator collection 仍必须在服务器的固定 RMBench checkout 中执行，因为本地没有 MuJoCo/GPU 环境。采集完成后，把 source directory 交给 `prepare_warm_rmbench_artifacts.sh`，不需要人工标注。

### 3.3 何时扩展到 500 条/任务

仅在 `scale200` 的 dev 曲线显示数据仍是主要瓶颈时扩展。优先顺序：

1. `blocks_ranking_try`、`press_button`、`cover_blocks`、`battery_try`；
2. `swap_blocks`、`swap_T`；
3. 其余 M(1)。

如果失败来自在线 retrieval/gate 而不是 imitation coverage，盲目扩到 500 不会解决问题，应先看 telemetry 中 gate、selected event、effect consistency 和 fallback 比例。

## 4. 服务器一次性准备

服务器或持久镜像中需要：

- WARM 私有仓库当前 `main`；
- 可运行训练的 `warm` Python 环境（PyTorch/CUDA、Transformers、Datasets、OpenCV 等）；
- 固定只读 RMBench checkout：`57ee09cbc6267bc36ca0ac2d8d1c5c3b245c112c`；
- 官方数据 revision marker：`855e90e1213d150bf4889130e83398f107314681`；
- FastWAM RoboTwin/RMBench-compatible base checkpoint；
- DINOv2-base checkpoint及其固定 revision；
- Wan VAE checkpoint；
- Wan text encoder 与 tokenizer；
- 足够的持久存储用于 source、LeRobot、features、bank、candidate cache、runs 和 eval evidence。

建议固定目录：

```bash
export PROJECT=/mnt/afs/.../projects/WARM
export RMBENCH_ROOT=/mnt/afs/.../external/RMBench-official
export RMBENCH_SOURCE_ROOT=/mnt/afs/.../datasets/rmbench_scale200_source
export RMBENCH_SOURCE_REVISION=<自动采集source snapshot的40位Git/manifest revision>
export RMBENCH_LEROBOT_ROOT=/mnt/afs/.../datasets/rmbench_scale200_lerobot
export RMBENCH_HF_REVISION_MARKER=/mnt/afs/.../datasets/rmbench_hf_revision.json
export WARM_ARTIFACT_ROOT=/mnt/afs/.../artifacts/warm_rmbench_scale200_v1
export FASTWAM_BASE_CHECKPOINT=/mnt/afs/.../checkpoints/fastwam_rmbench_base.pt
export WARM_DINO_CHECKPOINT=/mnt/afs/.../checkpoints/dinov2-base
export WARM_DINO_REVISION=<固定40位commit>
export WARM_VAE_CHECKPOINT=/mnt/afs/.../checkpoints/Wan2.2_VAE.safetensors
export WARM_TEXT_ENCODER=/mnt/afs/.../checkpoints/wan/text_encoder
export WARM_TOKENIZER=/mnt/afs/.../checkpoints/wan/tokenizer
export RMBENCH_TEXT_CACHE=/mnt/afs/.../artifacts/text/rmbench
```

模型、数据和大型 artifact 不提交 GitHub；GitHub 只保存代码、配置和说明。

## 5. Artifact 构建

### 5.1 官方 bootstrap

```bash
cd "$PROJECT"
export WARM_RMBENCH_DATA_PROFILE=official50-dev45
export RMBENCH_SOURCE_DATASET=TianxingChen/RMBench
bash scripts/prepare_warm_rmbench_artifacts.sh
```

### 5.2 scale200 主力数据

完成自动采集后使用全新输出目录：

```bash
cd "$PROJECT"
export WARM_RMBENCH_DATA_PROFILE=scale200-dev190
export RMBENCH_SOURCE_DATASET=warm-rmbench-auto-scale200-v1
export RMBENCH_SOURCE_REVISION=<该自动采集snapshot的40位revision>
export RMBENCH_DATASET_ID=rmbench_scale200_v1
bash scripts/prepare_warm_rmbench_artifacts.sh
```

该脚本会顺序完成转换、完整 audit、train stats、DINO/VAE features、event bank、dev oracle、train/dev K=32 candidate cache 和两个 source contract。所有输出采用 no-overwrite；失败后应检查并删除明确的不完整新目录，不能覆盖已发布 artifact。

文本 embedding cache 单独运行一次：

```bash
python scripts/precompute_text_embeds.py \
  task=rmbench_warm_3cam384_1e-4 \
  data.train.dataset_dirs="[$RMBENCH_LEROBOT_ROOT]" \
  data.val.dataset_dirs="[$RMBENCH_LEROBOT_ROOT]" \
  data.train.text_embedding_cache_dir="$RMBENCH_TEXT_CACHE" \
  data.val.text_embedding_cache_dir="$RMBENCH_TEXT_CACHE" \
  overwrite=false
```

## 6. 训练顺序

### 6.1 shared checkpoint

注册表默认 `scale200-dev190`、30k optimizer steps、seed 3407，并为 shared 模型使用 20 个 recent events、8 个 action summaries 和 10-step replan。4×H100 足够完成完整 WARM；用 ZeRO-1、每卡 batch 8、梯度累积 4 可得到 global batch 128。8×H100 时梯度累积 2。

```bash
export WARM_RMBENCH_STAGE=shared
export WARM_RMBENCH_DATA_PROFILE=scale200-dev190
export WARM_TRAIN_OUTPUT=/mnt/afs/.../runs/rmbench/shared-scale200-v1
export NPROC_PER_NODE=4
export PER_DEVICE_BATCH_SIZE=8
export GRADIENT_ACCUMULATION_STEPS=4
export TARGET_GLOBAL_BATCH_SIZE=128
export ZERO_STAGE=1
bash scripts/train_warm_rmbench_server.sh
```

先以 `WARM_MAX_STEPS=10` 做数值探针；探针必须使用新输出目录。正式训练去掉该变量。训练脚本默认拒绝覆盖、固定 seed、固定 data/model contract，并在 sampler 中保存 resume offset。

### 6.2 九个 specialist

每个 specialist 从同一个 FastWAM base 独立启动，并自动读取任务 profile。例如：

```bash
export WARM_RMBENCH_STAGE=specialist
export WARM_RMBENCH_SPECIALIST_TASK=blocks_ranking_try
export WARM_RMBENCH_DATA_PROFILE=scale200-dev190
export WARM_TRAIN_OUTPUT=/mnt/afs/.../runs/rmbench/specialists/blocks_ranking_try-v1
export NPROC_PER_NODE=4
export PER_DEVICE_BATCH_SIZE=8
export GRADIENT_ACCUMULATION_STEPS=4
export TARGET_GLOBAL_BATCH_SIZE=128
export ZERO_STAGE=1
bash scripts/train_warm_rmbench_server.sh
```

official50 ACP 正式任务优先使用封装入口，它会统一环境、四卡 batch contract、
Hydra preflight、console log 和严格续训：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
WARM_RMBENCH_SPECIALIST_TASK=blocks_ranking_try \
bash scripts/acp_warm_rmbench_specialist.sh
```

这里没有把 shared checkpoint 静默当作 specialist 初始化权重；当前 specialist
仍从相同 FastWAM base 独立训练，因此可以与 shared 并行启动。若后续实现
shared→specialist warm-start，必须先给 training attestation 增加独立的 fork
provenance，不能复用普通 full-state resume。

只需替换任务名和输出目录。合法任务顺序为：

```text
observe_and_pickup
rearrange_blocks
put_back_block
swap_blocks
swap_T
blocks_ranking_try
press_button
cover_blocks
battery_try
```

不在第一轮加入分组学习率。当前 optimizer attestation 绑定单一 LR；未扩展 attestation 就添加 parameter groups 会让记录与实际优化器不一致。先保证严格可复现，再把 grouped LR 作为独立代码变更和消融。

## 7. checkpoint 选择

每个 run 至少保留：早期、中期、最后 checkpoint 及 training attestation。选择次序：

1. checkpoint 能通过加载、contract 和 10-step rollout smoke；
2. dev success rate；
3. 同 success 下选择错误 memory fallback 更可靠者；
4. 再比较 ODE latency、gate activation 和 source-to-GT distance。

不要默认最后一步一定最佳。每个任务先用 dev10 做 2–3 个候选 checkpoint；正式 100-episode official rollout 只跑冻结后的获胜配置。

## 8. specialist contract 与单卡正式评测

每个 specialist checkpoint 都需要自己的 contract bundle。先设置 checkpoint、attestation、输出目录和任务：

```bash
export WARM_RMBENCH_TASK=blocks_ranking_try
export WARM_CHECKPOINT=/mnt/afs/.../step_014000.pt
export WARM_TRAINING_ATTESTATION=/mnt/afs/.../step_014000.training.json
export WARM_RMBENCH_ONLINE_CONTRACT=/mnt/afs/.../contracts/blocks_ranking_try-step014000
bash scripts/build_warm_rmbench_sota_task_contract_server.sh
```

wrapper 会自动选择 seed-3407 matrix、任务所需 replan/ODE/top-K/memory capacity，并只构建这一任务的 contract。

随后一张 80GB H100 运行一个任务：

```bash
export CUDA_VISIBLE_DEVICES=0
export WARM_RMBENCH_TASK=blocks_ranking_try
export WARM_EVAL_ROOT=/mnt/afs/.../eval/blocks_ranking_try-step014000-s3407
bash scripts/evaluate_warm_rmbench_task_server.sh
```

ACP 本身负责进程生命周期，不需要 tmux，也不在启动命令中执行 `git pull`。一个 ACP 占一张卡；有 4 张 H100 时可并行四个不同任务，每个任务使用独立 checkpoint、contract root 和 eval root。

### 8.1 一次性持久化仿真环境

正式 evaluator 不只需要 WARM 训练环境，还需要 SAPIEN、MPLib 和
CuRobo。RMBench 官方源码虽然顶层导入 Open3D，但当前正式
`demo_clean.yml` 协议显式关闭 depth、pointcloud 和 segmentation，实际 rollout
不会调用 Open3D。WARM 因而使用 fail-closed 的轻量 import guard 代替约 400 MB
的 Open3D wheel；如果协议意外开启上述输出，preflight 会直接拒绝运行。
RMBench 官方 `script/requirements.txt` 固定
`torch==2.4.1`，不能直接安装到 checkpoint 所绑定的 WARM
`torch==2.7.1+cu128` 环境，否则模型 runtime 会漂移。

因此只在同步新代码后的 CCI 中执行一次：

```bash
cd /mnt/afs/task3_2/L202500276_lwz/projects/WARM

CUDA_VISIBLE_DEVICES=0 \
WARM_RMBENCH_TASK=blocks_ranking_try \
bash scripts/bootstrap_warm_rmbench_eval_env.sh
```

脚本在持久化 AFS 中创建
`/mnt/afs/task3_2/L202500276_lwz/envs/warm-rmbench-eval`。它继承原 WARM
环境，精确安装官方 simulator 版本和 CuRobo v0.7.8 对应 commit，应用官方
SAPIEN/MPLib 补丁，然后验证完整 import closure、指定任务模块、单张 H100 和
无头 ray-tracing renderer。成功标志为 `RMBENCH_EVAL_ENV_READY`。

下载产物保存在
`WARM_external/wheelhouse/rmbench-eval-v2`，pip cache 保存在
`WARM_external/pip-cache/rmbench-eval-v2`，均位于持久化 AFS。镜像可通过
`WARM_RMBENCH_PYPI_MIRROR` 覆盖；默认使用阿里云 PyPI 镜像。镜像不保留的
SAPIEN 3.0.0b1 使用 8 路可续传分段下载并核对官方 SHA-256。`yourdfpy`
使用最小依赖安装，不解析其未被 CuRobo 使用的 `trimesh[easy]` 扩展。

该步骤不是每个 ACP 都运行；环境保存在 AFS，后续所有 specialist 共用。ACP
wrapper 会在创建 contract 或 immutable eval root 前重新执行只读 preflight；
环境缺包、版本漂移、CuRobo provenance 不符或 renderer 不可用都会立即报出，
不会再跑到正式 rollout 中途才发现。

### 8.2 ACP 正式评测

对于 checkpoint 训练完成后 `main` 已继续更新的常见情况，ACP 正式评测应优先使用封装入口：

```bash
cd /mnt/afs/task3_2/L202500276_lwz/projects/WARM

CUDA_VISIBLE_DEVICES=0 \
WARM_RMBENCH_TASK=blocks_ranking_try \
WARM_EVAL_LABEL=formal100-s3407-v4 \
bash scripts/acp_warm_rmbench_specialist_eval.sh
```

该入口从 checkpoint 的 canonical training attestation 读取训练 commit，在
`WARM_evaluations/code/<training-commit>` 创建或复用 detached worktree，并在该
只读 worktree 中构建任务 contract 和运行官方 100-episode evaluator。它不会
切换主 checkout、不会执行 `pull/fetch`，也不会放宽 contract 对
checkpoint/evaluator commit 一致性的要求。不同任务可各占一个单卡 ACP 并行运行；
若重跑同一任务，必须提供新的 `WARM_EVAL_LABEL`。

已训练的 `f77c633...` specialist 所绑定的历史 contract builder 有一个局部变量
遗漏：encoder contract 已完成校验，但其 `compute_device` 返回值未被
`_build_contract` 保存。ACP wrapper 仅在训练 commit 和两份历史 builder 的
SHA-256 同时精确匹配时，使用
`scripts/evaluation_compat/rmbench_f77_contract_v1/` 注入这一个已校验返回值；
历史 worktree 保持 clean，兼容 runner 的哈希进入 evaluation namespace 和日志。
未知 commit 或未知源文件哈希一律拒绝兼容，不会把新模型代码混入旧 checkpoint。

## 9. 正式报告

九任务每项 100 episodes，报告：

- 每任务 success rate；
- M(1) 平均、M(n) 平均、九任务宏平均；
- accepted seed 文件与官方 stdout；
- checkpoint/contract/artifact hashes；
- 每任务平均 rollout 时间、Action DiT ODE 步数；
- gate activation、fallback、retrieval rank、consequence consistency；
- shared 与 specialist 的差值。

论文主表使用冻结的最佳 specialist；shared 作为单模型对照。消融至少包括：FastWAM base、context-only、source-only/no-consequence、完整 WARM，以及正确/错误 memory corruption。

## 10. 阶段 gate

### Gate A：数据

- 九任务数量与 profile 完全一致；
- train/dev 无 episode hash 重叠；
- 三相机、14D state/action、长度与 NaN 检查通过。

### Gate B：M1/M2

- feature lists、bank、K32 cache 和 train/dev contract 全部发布；
- dev oracle top-K 明显优于 random；
- task/event strata 非空。

### Gate C：训练

- 10-step 4卡探针通过；
- loss/gate/source 数值有限；
- save/resume 后 sampler 顺序与 optimizer step 连续；
- checkpoint attestation 与实际配置一致。

### Gate D：评测

- M(1)+M(n) 各一个 5–10 episode pilot；
- online memory 容量、replan、ODE、top-K 与 contract 一致；
- telemetry 每个 replan 都能对应 factual query 和 selected/fallback decision；
- 再启动九任务正式 100-episode 测评。

## 11. 尚需在服务器完成的工作

1. 确认现有 FastWAM base 是否确实适配三相机 RoboTwin/RMBench 14D profile；否则先训练同数据 FastWAM baseline。
2. 用官方 checkout 的 expert collector 自动生成 scale200；本地不具备 simulator 条件。
3. 构建 scale200 M1/M2 并检查 oracle/strata 报告。
4. 4卡依次完成 shared 和九个 specialist；按 dev 选择 checkpoint。
5. 为获胜 checkpoint 构建 task contract，在 ACP 中一任务一卡并行正式评测。
6. 只有 scale200 明确数据不足时，再升级 M(n) 到 scale500。

这条链路的核心原则是：提高分数的自由度放在训练数据量、task specialist、真实 episode memory 容量、重规划频率和 ODE 预算；任务定义、成功判定、100-episode 正式协议和测试隔离保持不变。
