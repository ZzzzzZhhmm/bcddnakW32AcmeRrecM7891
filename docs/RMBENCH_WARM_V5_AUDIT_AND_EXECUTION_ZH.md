# RMBench WARM V5：根因审计、实现修复与训练准入

## 1. 结论

此前 `blocks_ranking_try` 多轮训练后持续零成功率，不能被解释为“评测次数少”。
现有日志中，策略能够完整加载 checkpoint、持续输出有限动作并运行到 episode
上限；与此同时，memory 候选几乎总能被选中，source gate 也经常放行。因此首先要
解决的是训练与在线记忆语义、事件相位和训练阶段继承关系，而不是继续增加相同配置
的训练步数。

本轮静态审计确认的主要问题为：

1. **specialist 没有继承 task-shared WARM。** 旧流程让每个只有 45 条训练演示的
   specialist 从通用 RoboTwin Fast-WAM 起步，同时学习 RMBench 域适配、Action DiT
   和全部 WARM 机制，训练问题欠约束。
2. **稀疏 event bank 无法覆盖每 4 步重规划的任务相位。** 视觉相近但相位不同的
   候选大量进入 top-32，检索可以“相似”却不一定是下一段可执行动作。
3. **跨 demonstration 比较 raw frame index。** 不同演示长度和执行速度不同，原始
   帧号不能表示可比较的任务进度；这会形成错误的回退/前进约束。
4. **训练和在线 short memory 不在同一特征域。** 离线 replay 使用 frozen DINO
   2x2 token，而在线曾写入 semantic-bridge token，导致工作记忆分布漂移。
5. **左右 gripper timing 被一个标量合并。** 完整 14D action 仍在，但显式 timing
   metadata 丢失了左右手的独立时序。
6. **训练采样被长演示和长 approach 阶段支配。** frame-weighted sampling 不能保证
   每个 episode 和早/中/晚任务相位获得接近的训练质量。
7. **DEV loss 原先覆盖不足。** 单个随机窗口不能判断 shared 或 specialist 是否真正
   学到所有任务/相位，且 shared 的字典序 strata 选择可能只覆盖一个任务。

V5 对这些问题采用 fail-closed 修复。它不是对旧 V1-V4 checkpoint 的续训版本。

## 2. V5 数据与 memory 语义

### 2.1 Dense factual event bank

event bank 使用以下 factual start 的并集：

- change-point / gripper event；
- 每 4 帧一个均匀 anchor（与 online replan stride 对齐）；
- 最后一个完整 32-step action window 的 tail-aligned start。

每个 entry 新增并校验：

- `normalized_phase`；
- `event_ordinal`；
- `successor_row`；
- `successor_event_start_frame`；
- `action_valid_mask`。

successor 必须位于同一 factual episode、ordinal 严格加一、start frame 严格增加。
不存在完整 32-step action 的 tail 不得伪造、padding 或作为 source。

### 2.2 Phase-aware causal source lease

在线接受一个 event source 后：

- 同一 demonstration 的相同或回退 event 在一个完整 horizon 内 source-ineligible；
- 明确 successor 作为下一条 factual state-action-effect 候选；
- 不再通过人工移动 action suffix 制造不存在的“后续动作”；
- 不同 demonstration 的 raw frame index 完全不比较，只由当前 context、归一化 phase、
  learned utility 与 consequence 决定。

### 2.3 Same-domain working memory

在线 retrieval 已经对当前真实 RGB 计算 frozen DINO 2x2 spatial tokens。V5 将这组
不可变 factual token 写入 episode memory；semantic bridge 仍用于 action inference，
但不再冒充 DINO 历史。这样训练 replay 与在线 short memory 都是：

```text
initial factual DINO anchor
+ bounded factual event/replan DINO tokens
+ actually executed action summaries
```

initial/event/latest/action 使用独立 role id，并加入相对 age。生成 future、预测状态和
未执行 action 均不得写入 short memory。

### 2.4 Independent bimanual timing

bank 中的 scalar gripper signal 继续只负责“任一 gripper 是否变化”的 event mining。
Event Adapter 的 timing 不再读取这个合并标量，而是从 candidate 14D action 与 factual
start proprio 恢复：

```text
left:  close_phase, close_valid, open_phase, open_valid
right: close_phase, close_valid, open_phase, open_valid
```

因此 RMBench `timing_dim=8`；LIBERO 单 gripper 仍为 4。invalid candidate 必须精确
zero-pad。

## 3. 训练分布与阶段

### 3.1 Hierarchical sampler

训练权重按以下层级归一化：

```text
task -> episode -> occupied progress bin (16 bins) -> frame
```

initial/final horizon 和 recent-event frame 只在所属 bin 内得到有限 event boost，不改变
整个 task、episode 或 phase 的总质量。这避免最长少数演示主导梯度。

### 3.2 Shared WARM first

正式训练顺序固定为：

1. 用九个任务全部 train split 训练 `shared-s3407-v5`；
2. 保存完整、attested WARM weights；
3. specialist 通过 training-fork manifest **weights-only** 初始化；
4. specialist 使用全新 optimizer、scheduler、sampler、epoch 和 global step；
5. specialist 冻结 shared Action DiT，只调整 compact WARM、proprio 与 selected video
   adapters，降低 45-demo 灾难性漂移。

specialist 默认禁止直接从通用 Fast-WAM base 开始。研究性 direct-base ablation 必须
显式设置 opt-in，且不能作为正式结果。

### 3.3 DEV evaluation

每 500 steps 评估 32 个 deterministic DEV windows。选择器按
task -> episode -> progress bin round-robin，无放回；shared 阶段首先覆盖不同 task，
specialist 阶段覆盖不同 episode/phase。记录总 loss 以及每个 WARM 子损失。

训练日志还记录下列 pre-clip gradient group norm：

- Action DiT（shared 阶段）；
- semantic bridge；
- retrospective gist；
- event adapter；
- reranker；
- source gate；
- episode memory encoder；
- video adapters。

关键分支长期为零梯度属于训练失败，不允许仅因总 loss 下降而继续长跑。

## 4. 无训练 artifact qualification

`scripts/qualify_warm_rmbench_artifacts.py` 在 GPU 启动前验证：

1. bank temporal payload 和完整 action mask；
2. 每个 episode 的 dense stride 与 successor chain；
3. 每个 task 的 early/middle/late phase recall（K=32/128）；
4. train/dev candidate cache 与 fresh teacher-forced search 的精确 parity；
5. candidate cache 完整覆盖每个 catalog 真实观测帧，终局 partial-action 行不得
   静默退化成 null memory；
6. train/dev feature split 与 cache split 不串用。

artifact 构建时生成 qualification report；每次正式训练又用 `--verify-existing` 从当前
bytes 重算并要求结果完全相同。旧 report、被重建过的 bank/cache 或不同阈值都会在
分配 GPU 前失败。

## 5. 训练前必须通过的 simulator/data gates

artifact qualification 证明检索数据结构正确，但不证明 simulator action boundary。
长训练前必须依次通过：

### G0：official expert action replay

- 直接回放 official qpos action；
- 至少一个成功 episode；
- 记录 before/action/after joint、左右 gripper 和 success predicate；
- 若失败，停止模型训练并修复 simulator/action 接口。

### G1：normalization replay

- official action -> normalize -> denormalize -> simulator；
- 与 G0 的动作逐元素一致（容差由 action contract 决定）；
- 若 G0 成功而 G1 失败，修复 dataset stats 或 action normalizer。

### G2：converted demonstration replay

- 只读取转换后的 LeRobot state/action/camera；
- 证明 state[t]、action[t]、terminal step 和 camera order 没有 off-by-one；
- 失败时不得归因于 WARM。

### G3：one-demo overfit

- 关闭 memory corruption，固定一条演示；
- teacher-forced action error 必须明显趋近零；
- 进行短闭环 rollout，至少复现动作方向、双 gripper 时序和显著 progress；
- 无法 overfit 表示 model/data plumbing 有错误。

### G4：five-demo smoke（200-500 steps）

- 所有 loss、weights、gradients finite；
- normal row 上 reranker/gist/event/source gate 梯度非零；
- gate target 同时存在 positive 与 negative；
- checkpoint save/load、fork manifest 和 10-step continuation 通过。

### G5：Fast-WAM same-data baseline

用相同 converted data、camera、normalizer 和 Action DiT 训练/评估无 WARM baseline。

- baseline 也为零：优先检查域适配、数据量、action boundary；
- baseline 非零而 full WARM 为零：问题才可定位到 retrieval/source/memory；
- 未完成该对照前，不允许把零成功简单归因于“方法能力不足”。

## 6. 正式诊断矩阵

每个 checkpoint 先在固定小集合上运行：

| 模式 | 目的 |
|---|---|
| Gaussian / no memory | 同数据 Fast-WAM 能否基本闭环 |
| context-only | short/long memory context 是否有益 |
| source-only-no-consequence | retrieved action source 本身是否可执行 |
| full WARM | consequence + gate 是否进一步改善 |
| oracle candidate | coarse candidate 集合中是否存在正确 action |
| wrong/reversed/phase-shift corruption | gate 能否拒绝错误 memory |

必须同时报告：success、episode progress、candidate phase/rank、gate、source acceptance、
selected event identity、successor、action norm、双 gripper timing、重复 source 比例。
只有“全部失败”而没有这些变量不足以支持修改模型。

## 7. 兼容性与重新构建要求

以下旧产物不可用于 V5 正式实验：

- sparse `hybrid_h32` bank；
- 由旧 bank 建立的 train/dev candidate caches 和 source contracts；
- V1-V4 specialist checkpoint/optimizer state；
- 直接从 generic Fast-WAM 训练的 specialist 作为 V5 parent。

可复用的是通过原始数据审计的 converted dataset 和逐帧 DINO/VAE feature cache。
应从这些 feature cache 重新构建 dense bank、candidate caches、contracts 与 qualification
report，然后训练 shared V5，再 fork specialists。

## 8. 发布判断

代码层面的检查只能证明接口闭合、训练/推理一致和已知根因被消除，不能预先保证
RMBench 成功率。只有 G0-G5 全部通过且至少一个 shared/specialist 小规模 rollout
产生非零成功，才允许启动九个 specialist 的完整训练。
