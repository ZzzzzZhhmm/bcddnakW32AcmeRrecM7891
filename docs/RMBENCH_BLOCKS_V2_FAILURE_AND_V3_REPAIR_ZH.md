# RMBench `blocks_ranking_try` V2 失败分析与 V3 修复

## 1. 结论

V2 的 0 成功率不是由“只评测了十几次”造成的，也不能通过继续把同一
checkpoint 多跑几十次解决。8 个完整 episode 均执行到官方上限并失败；更关键的
是，7179 次 replan 中虽然 100% 找到了候选事件，但只有 17 次真正采用了 memory
source，接受率仅 `0.2368%`。因此 V2 实际上几乎一直退化为 Gaussian Fast-WAM，
没有在评测中实现 WARM 的核心 action-source transport。

这次失败主要是两类实现问题叠加，而不是一个单独超参数：

1. RMBench 的 14D 动作是 absolute qpos target，却被短期记忆按 delta action 累加；
2. source gate 在训练和推理中被候选概率、候选索引熵和错误停滞信号连续相乘，
   导致一个看似正常的 learned gate 最终被压到接近零。

V3 修复这两个根因，并保证 hard negative、短期视觉记忆和在线因果时序一致。
V2 checkpoint 的 retrospection schema 是 v4；V3 升为 v5，必须从相同 Fast-WAM
base 重新训练，禁止 resume V2 optimizer/state。

## 2. V2 证据

对 `warm_online_evidence.jsonl` 的流式诊断得到：

| 指标 | V2 观测值 | 含义 |
|---|---:|---|
| 完成 episode | 8 | 已足以排除偶然单次失败 |
| 成功 | 0 | 当前配置失效 |
| replan | 7179 | 在线闭环确实持续运行 |
| candidate selection | 100% | bank 和 ANN 不是空的 |
| memory source acceptance | 0.2368% | action memory 几乎从未进入 source |
| learned gate mean | 0.4031 | gate 网络本身并非全零 |
| effective gate mean | 0.000813 | 校准层把 learned gate 压没了 |
| source quality mean | 0.00891 | 旧质量乘积发生系统性塌缩 |
| stagnation mean / p50 | 0.9778 / 0.9995 | 几乎全程被误判为停滞 |
| 最大 repeated attempt | 753 | 绝对 qpos 被错误解释为重复 delta |
| episode event write rate | 3.36% | 0.08 阈值不适合三相机 RMBench 特征尺度 |
| camera/action evidence coverage | 100% / 100% | 不是黑帧或动作证据缺失 |

训练末期同样出现 source starvation：`warm_learned_gate_mean≈0.36`，但
`warm_gate_mean≈0.0086`，`warm_source_quality_mean≈0.0107`。因此 Action DiT 在
14000 步训练中也几乎没有见过真正的 memory-conditioned source。末期
`loss_action` 已降到约 `3e-4`，只能说明它拟合了 demonstration action；不能说明
WARM source 路径被训练。继续增加相同训练步数不会修复这一机制断路。

## 3. 根因与代码修复

### 3.1 Absolute-qpos action summary

RMBench 每步下发的是左右臂目标关节位置及 gripper target，不是关节增量。旧实现
计算：

```text
mean = mean(absolute qpos target)
displacement = sum(absolute qpos target)
```

连续两个正常控制 chunk 的绝对关节配置天然高度相似，因此 cosine repetition 很快
饱和。V3 按 action-space contract 自动选择 summary mode：

```text
relative_target[t] = normalized_action_target[t] - normalized_start_proprio
mean_displacement = mean(relative_target)
final_displacement = relative_target[-1]
```

gripper channel 仍保留原始 terminal command 和开闭时序，不参与 qpos 相减。
LIBERO 等 delta-action 数据继续使用原来的 sum 语义。

### 3.2 真实进展约束重复尝试

相似动作本身不是失败：抓取前的慢速接近通常会连续产生相似关节目标。V3 只有在
以下条件同时成立时才把 action summary 写成 `repeated=True`：

```text
action signature 相似
AND normalized signature distance 足够小
AND factual world-token change 与 VAE change 都没有超过静止阈值
```

也就是说，停滞只来自“相似动作 + 真实观测无进展”，不会再由动作连续性单独触发。

### 3.3 离线训练与在线预览的因果对齐

在线当前 replan 时，最新 executed prefix 已知，但当前观测的 bridge feature 要等
本次 model forward 返回后才能写入 episode memory。因此该 prefix 是未配对的
preview，不能提前判断停滞。

V3 的离线 retrospective replay 使用完全相同语义：最新 chunk 保留 repetition
similarity 作为诊断，但 `repeated=False`；只有更早、已经具有 factual post-action
observation 的 chunk 才能建立 stagnation。这样训练不再偷用在线不可见的即时结果。

### 3.4 Source exposure 与 deployment rejection 解耦

旧实现为：

```text
effective_gate = learned_gate
               * candidate_probability_quality
               * candidate_entropy_quality
               * exp(-3 * stagnation)
```

候选索引熵并不等价于动作模态熵：32 个高度相似的成功 demonstration 可能导致高
index entropy，但它们提供的是同一个有效动作模式。再叠加错误的 stagnation，V2
训练 source exposure 被压到不足 1%。

V3 采用：

- 训练：normal row 直接用 learned gate 混合 source，使 Action DiT 确实学习
  memory-conditioned transport；候选概率/熵不再作为连续乘数；
- 推理：candidate probability、entropy、factual stagnation 作为 hard eligibility；
  learned gate 决定是否接受，已接受 source 再按真实停滞平滑衰减；
- 无候选、候选不合格或 gate 不足时，仍精确回退到 Gaussian source。

### 3.5 Hard negative 不能进入 source

V2 的 hard-negative row 虽然 gate target 为零，但在 gate 尚未学会拒绝的训练早期
仍可进入 Action DiT source。V3 在构造 flow source 之前显式把
`forced_rejection_mask` 行置为零；它们只训练 reranker/gate 拒绝，不污染 action
source。

curriculum 调整为：

```text
normal 60% / drop 10% / null 10% / hard negative 20%
```

45-demo pilot 中优先保证正常 source exposure；20% hard negative 足以学习拒绝，
无需用 35% 样本持续压制 source。

### 3.6 RMBench 短期视觉事件尺度

V2 change score 中位量级约 `0.02`，而 floor 固定为 `0.08`，事件写入率只有
3.36%。V3 在 action-space contract 为 RoboTwin/RMBench 时使用：

```text
change_threshold_floor = 0.015
change_threshold_ceiling = 0.25
```

前三次更新后仍由 median/MAD 自适应阈值控制，并由容量 20 的 bounded memory
做相邻冗余合并。因此它不会无限保存每一帧，但可以保留真实接触、移动、释放和
顺序变化。写入源仍只是真实 observation，不写入生成 future。

### 3.7 Temporal thread 降为 tie-breaker

V2 thread continuity 已从早期版本的约 13% 提高到约 82%，说明线程机制有效；但
`0.75` 的 prior 可能在首次选择后压过 learned utility。V3 将 thread score 降至
`0.20`、switch penalty 降至 `0.05`。它只解决近似候选的时序平局，不充当高层
planner，也不会强迫模型继续错误 demonstration。

## 4. 数据与训练策略

### 4.1 official50 的正确定位

当前 specialist 只有 45 个 train episode、5 个 dev episode。整个九任务统计中的
405 train episode 只用于统一 normalization/bank；并不等价于该任务有 405 条监督
轨迹。45 条数据足以验证 WARM 机制是否工作，但不是冲击最终 SOTA 的数据上限。

V3 首先复用现有 official50 M1/M2，因为现有 evidence 已证明：

- bank/candidate cache 可读且每次都有候选；
- 候选来自正确任务 demonstration，并大致随阶段向后移动；
- 本次修复不改变长期 event payload 或 ANN key schema。

因此 official50 V3 不需要为修代码重复构建 M1/M2。只有扩充到 scale200 后才必须
对新数据重新运行转换、audit、feature、bank、candidate cache 与 oracle report。

### 4.2 V3 机制 pilot，禁止直接再跑满 14k

从 Fast-WAM base 新建 `*-s3407-v3`，依次保留 500、2000、4000、6000 step
checkpoint。每个 checkpoint 在固定 dev seeds 上至少运行：

1. `no_memory`；
2. `context_only`；
3. `source_only_no_consequence`；
4. `full`。

进入下一阶段前要求：

- 所有 loss/gradient finite；
- normal-row training effective gate 不再接近零；
- hard-negative source rate 精确为零；
- online memory acceptance 不长期低于 5% 或高于 95%；
- stagnation p50 不再接近 1；
- event write rate 不再低于 5%；
- `source_rms_reduction` 相对 Gaussian 为正并随训练改善；
- 至少一个 memory-enabled 分支在固定 pilot seed 上产生成功，或明显优于
  no-memory 的 task progress。

如果四个分支全为 0，先验证相同 simulator 中的 Fast-WAM base 与 imitation-only
checkpoint；如果 context 有进展但 source 无进展，再检查 action canonicalization
和 source geometry；不能继续盲目增加到 14k。

### 4.3 scale200 自动扩充

机制 pilot 通过后，用固定官方 simulator/task config 自动收集每任务约 200 条成功
episode，不做人工 event、mask 或 subtask 标注。对 `blocks_ranking_try` 优先扩大：

- 不同 block 初始位置与排序；
- 不同抓取顺序和双臂分工；
- 接触失败后的恢复轨迹；
- 同一视觉状态下不同历史进度；
- gripper close/release timing 的有效变化。

只把成功、完整、可重放的 episode 放入 positive source bank；失败 rollout 放入
独立 negative archive，用于 hard-negative/拒绝训练，不能直接成为 source。

scale200 使用 task/event-balanced sampler，event boost 为 `2.0`，global batch
128、4×H100、ZeRO-1。建议以 20k/25k/30k checkpoint 的固定 dev rollout 选模，
而不是仅看末步 action loss。每次 checkpoint 选择都同时报告 success、source
acceptance、stagnation、event write、source geometry 和 gate calibration。

## 5. 新诊断与防回归

`scripts/analyze_warm_rmbench_evidence.py` 现在额外输出：

- gate/source/stagnation 的 p50/p90/p99；
- factual update 与 episode event write rate；
- 实际执行 action 数；
- 自动 failure signals：source starvation、source always-on、stagnation saturation、
  event-write starvation。

V2 文件会同时触发：

```text
memory_source_starvation
factual_stagnation_saturated
episode_event_write_starvation
```

V3 回归测试覆盖 absolute-target summary、真实进展清除重复标记、未配对 preview
不提前判停滞、训练 source exposure 不被 deployment heuristic 清零、hard negative
强制 Gaussian fallback，以及 checkpoint v5 兼容边界。

## 6. 后续若仍失败，需要归档的最小文件

每次 pilot 不需要上传整个 run，只需归档：

```text
训练：
  console.log
  resolved_config.preflight.yaml
  config.yaml
  dataset_stats.json
  checkpoint 对应 *.training.json

评测：
  eval_log.txt
  official_stdout.log
  warm_online_evidence.jsonl
  online_contract.json
  至少 1 个完整失败视频和 1 个成功视频（若有）

长期 memory 诊断：
  m1/oracle/hybrid_h32.json
  m1/rmbench_catalog.json
  m1/rmbench_audit.json
```

前三组足以判断训练、在线 source 和控制闭环；最后一组只在需要进一步分析
`blocks_ranking_try` 的 per-task candidate recall/action oracle 上限时提供。
