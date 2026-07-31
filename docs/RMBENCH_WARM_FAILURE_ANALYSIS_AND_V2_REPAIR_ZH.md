# RMBench WARM 失败归因与 V2 修复方案

## 1. 结论

`blocks_ranking_try` 的现有结果不能解释为“eval 次数太少”。已检查的 17 个
不同 episode 全部运行到 3500-step 上限且全部失败；与此同时，完整 WARM
checkpoint、长期 bank、在线短期 memory 和 action source 路径均被实际加载。
因此当前首要问题是策略机制与训练监督，而不是再重复更多相同配置的 rollout。

旧 checkpoint 的关键在线统计为：memory source 平均 gate 约 0.919，几乎每次
replan 都采用长期动作；但 top-1/top-2 候选差仅约 6.8e-4，连续两次选择同一
候选的比例约 13.2%，事件相位相关性约 0.056。失败 episode 中相似动作重复可达
数百次，gate 没有随真实失败反馈下降。该组合表明模型不是“没有使用 memory”，
而是在高歧义、低连续性的条件下过度使用 memory。

## 2. 根因

### 2.1 没有可用的在线 null source

旧 gate 使用 absolute reranker score。候选 softmax logit 可以整体平移，因此
absolute score 没有概率含义。即使 32 个候选几乎等价，旧路径仍选一个候选并
以高 gate 改写 Action DiT source。

### 2.2 长期动作事件缺少时序线程

每次 replan 独立选择视觉相似事件，无法优先延续同一 demonstration 中相邻、
单调前进的动作片段。对 `blocks_ranking_try` 这类 M(n) 任务，这会造成动作模式
在不同 episode/phase 之间频繁跳转。

### 2.3 短期 memory 没有参与最终 retrieval query

短期 visual/action memory 虽进入 predictive gist，但 learned reranker 仍主要看
当前 context。观察相似而历史不同的阶段容易得到同一组排序。

### 2.4 consequence predictor 存在 identity shortcut

旧实现为 `predicted_effect = stored_effect + zero_initialized_residual`，训练 target
又是同一个 `stored_effect`。它无需学习“当前世界 + 候选动作会造成什么”，所以
consequence consistency 不能可靠拒绝不合适动作。

### 2.5 没有闭环停滞回退

Episode memory 已正确保存真实 executed action summary 和真实视觉 observation，
但 source gate 没有使用 repeated-attempt evidence。错误 action memory 可以被连续
执行直到 episode 超时。

## 3. V2 代码修复

### 3.1 Shift-invariant source acceptance

候选选择新增：selected probability、top-1/top-2 probability margin、normalized
entropy。learned gate 不再读取 absolute logit。source 额外经过可解释的质量层：

```text
quality = sqrt(probability_quality * entropy_quality)
          * exp(-stagnation_decay * stagnation)
effective_gate = learned_gate * quality
```

推理时只有 effective gate、entropy、candidate probability 和 stagnation 同时通过
阈值才使用 memory component；否则 component=0，精确回退到 Gaussian source。

### 3.2 Phase-aware retrieval

Initial anchor、recent factual visual events 和 executed-action summaries 先压缩为
episode representation，再形成 learned query delta 进入 top-32 utility reranker。
ANN index 不变，因此不需要重建 M1/M2；但同一当前画面在不同历史阶段可以得到
不同候选排序。

### 3.3 Temporal event thread

在线状态记住上一次真正被 source 接受的 EventId。相同 source episode 内、沿
start_frame 单调前进且接近期望位移的候选获得正 prior；回退片段和跨 episode
切换受到惩罚。连续三次 null 后释放线程。每次 simulator episode reset 都清空
线程，绝不跨测试 episode 污染。

### 3.4 Factual current-state action-to-effect supervision

共享 effect predictor 现在输入 action temporal summary、stored effect prior 和当前
world tokens，直接输出 predicted effect，不再把 stored effect 加到答案上。训练时
同时使用两种可识别监督：

1. 当前 demonstration 的 factual
   `(current world, GT action, zero prior) -> future semantic delta`；
2. 仅对 action/effect utility 高的候选，用 soft utility 权重监督在线实际使用的
   candidate-effect 路径。

第一项禁止网络只复制 stored effect，第二项确保 effect-prior projection 确实被训练，
而不是在推理时接收从未见过的非零输入。无需 counterfactual label、人工 event 标签
或额外 future-video rollout。

### 3.5 Hard-negative rejection

RMBench curriculum 调整为 45% normal、10% drop、10% forced-null、35% factual
hard negative。hard-negative row 的 gate target 被明确置零，不能再因 action/effect
数值尺度较小而得到高 helpfulness target。utility/gate temperature 从 1.0 收紧到
0.25，提高 visually-similar-but-action-incompatible 样本的区分度。

### 3.6 Factual stagnation

短期 memory 中已存在的 repeated flag 被转成 recency-weighted stagnation score。
它同时进入 learned gate 和确定性 source attenuation；达到 0.75 时强制 null。
该信号只来自真实执行动作，不使用生成 future，不会产生自回归 memory 污染。

### 3.7 诊断证据

每个 replan 额外记录：learned/effective gate、source quality、candidate probability、
probability margin、entropy、stagnation、thread prior、episode-query delta norm、相机
数值统计和 model/environment action 数值统计。SHA-256 证据仍保留，可同时排查
黑帧、归一化漂移、动作饱和和错误 source。

## 4. Checkpoint 兼容性

WARM retrospection checkpoint schema 从 v3 升为 v4。effect head、gate 输入和
phase-aware query 都改变了参数结构，因此：

- 不允许从旧 v3 `state/step_*` resume；
- 不允许用旧 v3 weight 做 V2 正式 eval；
- M1 event bank、M2 top-32 candidate cache、DINO/VAE feature 和数据集 stats 可复用；
- 新 V2 specialist 从相同 FastWAM base checkpoint 重新训练。

## 5. 训练与评测 Gate

### Gate 0：基础链路

先在同一个 official simulator/runtime 上运行 FastWAM no-memory baseline，并检查
新增 camera/action stats。若 baseline 也是 0/10，优先修复 base checkpoint、相机
顺序或 action normalization；不要把问题归因给 memory。

### Gate 1：V2 smoke

只训练一个 specialist 500 steps，要求：loss 全部 finite、checkpoint v4、hard
negative gate target 可下降、dev factual-effect cosine 明显高于随机。该阶段只验证
数值与监督闭环。

### Gate 2：短训在线 pilot

在 2k、4k、6k checkpoint 分别跑同一组固定 10 个 development seeds，并同时跑：

1. no-memory / Gaussian fallback；
2. context-only；
3. source-only/no-consequence；
4. full WARM。

若 full WARM 仍为 0/10，或 memory acceptance 持续高于 90%，停止训练并检查证据，
不继续消耗到 14k。若 context-only 有效但 source 退化，说明 source calibration 或
action canonicalization 仍有问题；若所有分支均失败，则优先检查 base policy/data。

### Gate 3：official50 诊断通过后扩大数据

45-train/5-dev 适合打通链路，不适合作为冲击最终 SOTA 的数据上限。通过 Gate 2
后，使用 official RMBench collector 自动生成 scale200-dev190，并重新构建 M1/M2。
所有 event、semantic delta、timing 和 short-memory training trace 仍自动产生，无需
人工标注或分割。M(n) specialist 优先扩大不同 object order、initial state 和
recovery perturbation 的覆盖，而不是只重复同一 45 条轨迹更多 epoch。

### Gate 4：正式训练和 checkpoint 选择

scale200 specialist 使用 global batch 128、4xH100、ZeRO-1。每 1000 steps 保存；
checkpoint 只用 development pilot 选择。最终冻结 checkpoint 后再运行官方 100
episode protocol，不能用最终 test seeds 反复调参。

## 6. 建议监控阈值

以下是早停诊断阈值，不是论文结果约束：

- memory acceptance 不应长期接近 0% 或 100%；
- uniform/高 entropy candidate row 应绝大多数回退 Gaussian；
- hard-negative learned gate 应显著低于 normal row；
- 同一 accepted event thread 的连续率应明显高于旧版 13.2%；
- repeated-attempt count 上升时 effective gate 必须下降；
- factual effect validation 必须随训练改善，不能维持 identity baseline；
- 10-episode pilot 仍为 0 时不得直接扩展到另外七个 specialist。

这套修复保持 WARM 的核心：长期 `state-action-effect` memory 仍生成 Action DiT
source，短期真实 visual memory 仍预测任务所需 transition；变化在于 memory 只有
在“候选明确、后果匹配、阶段连续、闭环仍有进展”时才被允许主导动作。

## 7. Evidence 流式诊断

新脚本不会把数百 MB JSONL 一次性载入内存，可以在 eval 尚未结束时直接运行：

```bash
python scripts/analyze_warm_rmbench_evidence.py \
  --evidence /path/to/warm_online_evidence.jsonl \
  --output /path/to/evidence_analysis.json
```

它汇总 success、candidate selection、真正采用 memory source 的比例、候选熵、停滞、
同一示范轨迹的单调延续率，以及 camera/action 数值证据覆盖率。对旧 v1 evidence 的
实测结果是 16 个完整 episode、0 success、14514 次 replan、memory-source acceptance
100%、平均 gate 0.91885、top-1/top-2 cosine gap 0.000682、最大重复尝试 853；这进一步
排除了“仅仅是 eval 数量太少”的解释。
