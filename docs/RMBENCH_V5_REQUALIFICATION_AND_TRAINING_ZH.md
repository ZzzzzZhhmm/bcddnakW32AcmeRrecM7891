# RMBench V5 重新资格验证与训练方案

## 1. 结论

旧的 `blocks` 三轮 0% 不能直接归因于 WARM 的方法思想。旧链路至少包含几类在不重新训练时就可判定的基础问题：事件 bank 稀疏、动作后缀被错误当成事件推进、在线线程约束把步长为 4 的合法 successor 拒绝、双夹爪 timing 混合、短期记忆缺少类型与时间顺序，以及官方执行器可能捕获 TOPP 异常后静默跳过手臂动作。

V5 的目标不是再盲目做第四次完整训练，而是先证明：

1. 数据、事件链、候选 cache 和在线检索在同一个严格契约内；
2. 所有 WARM 子模块在短训练中真实获得梯度；
3. source gate 同时见到有用与应拒绝的 memory，并能无泄漏地退回 Gaussian source；
4. 官方执行器实际移动了被命令的左右臂，而不只是 Python 调用返回成功；
5. 共享模型合格后，九个 specialist 只从完整共享 WARM 权重分叉。

任一门禁失败，都不应开始长训练。

## 2. V5 中已经落地的关键修复

### 2.1 数据和事件

- RMBench 仍使用事实转移 `state[t] -> action[t]=qpos[t+1]`，终止观测不伪造动作。
- H32 event 以 4 步 stride 密集构建；每行保存 `normalized_phase`、`event_ordinal`、`successor_row`、`successor_event_start_frame` 和完整 `action_valid_mask`。
- bank 中每个动作必须是事实完整 horizon，不允许用 padding 伪装成 source。
- 训练语言增强只从官方 `seen` wording 中按 seed 确定性采样；`unseen` wording 仅记录并默认禁止进入训练。

### 2.2 长期 action memory

- 检索增加显式 successor lane；连续 replan 可以从同一历史 episode 的下一个真实 event 继续，而不是反复取同一个 prefix。
- 在线 phase 只拒绝不前进的 event；任何严格晚于当前 cursor 的合法 event 可参与排序。
- 动作 source 永远使用检索 event 的完整事实 H32，不再在推理侧把 action suffix 左移并复制末帧。
- candidate phase/ordinal 进入模型，使 reranker、event adapter 和 consequence gate 可区分视觉相似但阶段不同的事件。

### 2.3 短期视觉/动作记忆

- working memory 明确区分 initial anchor、factual recent event、executed-action summary 和 latest event。
- token 同时带 role 和 relative-age，保留时间因果顺序。
- 只在官方环境返回真实 observation 后写入；预测 future 不写入永久记忆。

### 2.4 训练与评测边界

- 共享阶段训练 Action DiT、WARM adapters、proprio bridge 和选定 video adapters。
- specialist 从完整且有 attestation 的共享 WARM checkpoint 做 weights-only fork；optimizer、scheduler、sampler 和 step 全部重新开始。
- specialist 冻结共享 Action DiT，只适配紧凑 WARM/proprio/video adapter，降低 45 条示范导致的灾难性遗忘。
- 训练每个日志步写出 `training_metrics.jsonl`，包括各 WARM 模块梯度范数、各辅助 loss、gate 正负样本覆盖和 source 泄漏。
- 正式 eval 会重新计算 M1/M2 qualification，而不是只检查同名 JSON 是否存在。
- 官方 qpos executor 在每次 action 后核对 before/target/after；左右臂存在显著命令但事实完全不动时立即失败，暴露被官方 `take_action` 静默吞掉的 TOPP 失败。

## 3. 旧产物的复用与必须重建项

可以复用：

- 官方原始示范、MP4、HDF5/Parquet 原始事实数据；
- Fast-WAM base checkpoint、DINO、VAE、T5 和 tokenizer；
- 与原始 episode 严格绑定的逐帧视觉特征，只要 feature contract 和转换 manifest 校验仍通过。

必须用 V5 代码重新产生：

- 转换数据的 `meta/warm_instruction_variants.jsonl`；
- M1 H32 temporal event bank；
- train/dev M2 candidate caches；
- M2 必须覆盖每个 catalog 真实观测帧；终局 partial-action 样本继续监督
  action retrieval/source/gate，仅在缺少 H32 future teacher 时屏蔽 gist/effect loss；
- source contracts 和 qualification report；
- 共享 WARM checkpoint；
- 所有 specialist checkpoint。

V3/V4 bank、candidate、checkpoint 与 V5 schema 不兼容，不得续训或评测。

## 4. 服务器门禁顺序

### Gate A：重新构建并验证 M1/M2

`prepare_warm_rmbench_artifacts.sh` 已在构建末尾自动运行严格 qualification。合格报告必须位于：

```text
$WARM_ARTIFACT_ROOT/m1/qualification/rmbench_h32.json
```

它验证密集 successor 链、early/middle/late phase recall，以及 train/dev cache 对事实 query 的 exact-search parity。训练和 eval 启动器都会用 `--verify-existing` 重新计算，任何人工拷贝、部分重建或字节变化都会阻止 GPU 启动。

### Gate B：共享 300-step smoke

首次建议用已经准备好的 `official50-dev45` 验证正确性，避免把 scale 扩大与代码正确性混在一起：

```bash
cd /mnt/afs/task3_2/L202500276_lwz/projects/WARM

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WARM_RMBENCH_DATA_PROFILE=official50-dev45 \
WARM_RUN_STEPS=300 \
WARM_TRAIN_OUTPUT=/mnt/afs/task3_2/L202500276_lwz/projects/WARM/runs/rmbench_official50_shared/shared-official50-dev45-s3407-v5-smoke300 \
bash scripts/acp_warm_rmbench_shared.sh
```

然后运行训练资格检查：

```bash
cd /mnt/afs/task3_2/L202500276_lwz/projects/WARM

PYTHONPATH=src /mnt/afs/task3_2/L202500276_lwz/envs/warm/bin/python \
scripts/qualify_warm_rmbench_training_smoke.py \
  --metrics runs/rmbench_official50_shared/shared-official50-dev45-s3407-v5-smoke300/training_metrics.jsonl \
  --stage shared \
  --output runs/rmbench_official50_shared/shared-official50-dev45-s3407-v5-smoke300/smoke_qualification.json
```

只有输出 `TRAINING_SMOKE_QUALIFIED` 才能进入下一阶段。它要求 Action DiT 和全部 WARM 分组在至少 80% 的记录中有非零梯度、所有 WARM loss 存在、gate 正负样本均出现、正常 source 确实启用且 forced-negative source 无泄漏。

### Gate C：小数据过拟合和官方执行审计

在完整共享训练前，使用固定的一条和五条 demonstration 做短过拟合，确认 action loss 能显著下降。随后在官方 simulator 中回放 expert qpos，并保存每步左右臂的 before/target/after、task predicate 与视频。任何显著 target 但事实关节不动的情况都必须先修 simulator/planner，不允许靠增加训练步数掩盖。

### Gate D：共享正式训练

正确性门禁通过后，为了最终成功率，主实验应采用 `scale200-dev190`，并使用独立且匹配的 dataset、text cache 与 artifact root。不要只改 profile 名称而继续指向 official50 产物。

```bash
cd /mnt/afs/task3_2/L202500276_lwz/projects/WARM

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WARM_RMBENCH_DATA_PROFILE=scale200-dev190 \
RMBENCH_LEROBOT_ROOT=/mnt/afs/task3_2/L202500276_lwz/datasets/rmbench_scale200_lerobot \
RMBENCH_TEXT_CACHE=/mnt/afs/task3_2/L202500276_lwz/projects/WARM_artifacts/text/rmbench_scale200_v1 \
WARM_ARTIFACT_ROOT=/mnt/afs/task3_2/L202500276_lwz/projects/WARM_artifacts/rmbench_scale200_v1 \
WARM_SHARED_RUN_ROOT=/mnt/afs/task3_2/L202500276_lwz/projects/WARM/runs/rmbench_scale200_shared \
WARM_SPECIALIST_RUN_ROOT=/mnt/afs/task3_2/L202500276_lwz/projects/WARM/runs/rmbench_scale200_specialists \
bash scripts/acp_warm_rmbench_shared.sh
```

默认共享轨迹为 30,000 optimizer steps、global batch 128、四张 80G H100、root seed 3407、task/event/progress balanced sampler。若 ACP 时间受限，用 `WARM_RUN_STEPS` 分段，并用同一 output 和最新 `WARM_RESUME_STATE` 做正式 full-state resume；不得重新从 base 启动。

### Gate E：九个 specialist

共享 checkpoint 完成并通过 held-out loss/rollout probe 后，才启动 specialist。例如：

```bash
cd /mnt/afs/task3_2/L202500276_lwz/projects/WARM

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WARM_RMBENCH_STAGE=specialist \
WARM_RMBENCH_SPECIALIST_TASK=blocks_ranking_try \
WARM_RMBENCH_DATA_PROFILE=scale200-dev190 \
RMBENCH_LEROBOT_ROOT=/mnt/afs/task3_2/L202500276_lwz/datasets/rmbench_scale200_lerobot \
RMBENCH_TEXT_CACHE=/mnt/afs/task3_2/L202500276_lwz/projects/WARM_artifacts/text/rmbench_scale200_v1 \
WARM_ARTIFACT_ROOT=/mnt/afs/task3_2/L202500276_lwz/projects/WARM_artifacts/rmbench_scale200_v1 \
WARM_SHARED_RUN_ROOT=/mnt/afs/task3_2/L202500276_lwz/projects/WARM/runs/rmbench_scale200_shared \
WARM_SPECIALIST_RUN_ROOT=/mnt/afs/task3_2/L202500276_lwz/projects/WARM/runs/rmbench_scale200_specialists \
bash scripts/acp_warm_rmbench_specialist.sh
```

启动器会要求共享 `step_030000.pt`、对应 `.training.json` 和 parent `config.yaml`，生成不可变 fork manifest，并以新 optimizer/scheduler 从 specialist step 0 开始。若共享 checkpoint 不存在，默认直接失败。`WARM_RMBENCH_ALLOW_DIRECT_BASE_SPECIALIST=true` 仅供显式 debug，不能作为正式结果。

## 5. 如何解释下一轮结果

- artifact qualification 失败：数据/事件/检索接口问题，禁止训练。
- smoke gradient qualification 失败：模块未接入 loss、gate 样本不足或 source 泄漏，属于实现/训练链问题。
- expert qpos audit 失败：simulator/executor 问题，不是模型成功率问题。
- 小样本无法过拟合：监督、normalization、action/control 接口或模型可训练范围仍有错误。
- 上述全部通过但正式 rollout 仍低：才进入方法层分析，优先看 phase-conditioned retrieval recall、gate calibration、source deformation、短期 memory utilization 和 closed-loop replan error，而不是继续无依据增加步数。

这套顺序把“无需训练即可发现的问题”与“必须通过训练/评测才能观察的问题”分开。下一次长训练的前提是所有前置证据合格，而不是代码能够启动。
