# RMBench `blocks_ranking_try` V3 失败分析与 V4 修复

## 1. 结论

`blocks6000-dev5090-v3-full` 不是因为评测次数少而暂时没有成功。下载结果包含 7 个已开始、6 个完整结束的 episode；6 个完整 episode 全部运行到官方 3500-action 上限，成功数为 0。第 7 个 episode 是归档时尚未结束，不影响对机制失效的判断。

V3 已经修复 V2 的 source starvation：在 5373 次 replan 中，87.83% 真正采用了 memory source，learned gate 均值为 0.508，effective gate 均值为 0.457。因此当前失败不能解释为“WARM 没有使用 memory”。相反，主要故障是 **memory event phase lock**：模型大量重复使用同一训练事件作为 action source，但没有推进到后续任务阶段。

V4 的修复目标不是继续提高 gate，而是建立一个严格的因果不变量：

> 一个检索 action event 只能在一个 action horizon 内作为 source；超过该 horizon 后，同一 phase（包括其他 demonstration 中的 phase-aligned exemplar）只能保留为 context，不能再次改变 action-flow source。模型必须检索到更晚事件，或者精确回退到 Gaussian Fast-WAM。

## 2. 本次证据

对 87.75 MB `warm_online_evidence.jsonl` 的逐行分析结果如下。

| 指标 | V3 观测值 | 解释 |
|---|---:|---|
| 已完成 episode | 6 | 每个均运行到 3500-action 上限 |
| 成功 episode | 0 | 不是单次随机失败 |
| replan | 5373 | 在线闭环持续运行 |
| 实际执行动作 | 21464 | replan=4 的 FIFO 执行链正常 |
| candidate selection | 100% | bank、ANN 和 payload 可用 |
| memory source acceptance | 87.83% | V2 的 source starvation 已修复 |
| learned / effective gate mean | 0.508 / 0.457 | memory 对 action source 有强影响 |
| source quality mean | 0.878 | 不是 gate 全部拒绝 |
| event write rate | 6.78% | 短期事实记忆在写入，不是空 memory |
| stagnation mean | 0 | 旧 repeated-attempt 检测未发现循环 |
| selected probability mean | 0.223 | reranker 不够确定 |
| normalized entropy mean / p90 | 0.852 / 0.946 | 候选仍高度多义 |
| top1-top2 DINO gap mean | 0.000677 | 粗检索近似候选非常密集 |
| dominant event | episode 257, frame 587 | 被 source 采用 2613 次，占 55.37% |
| exact consecutive repetition | 67.19% | 相邻 replans 常为完全同一 event |
| phase-locked transition | 80.90% | event frame 几乎没有向后续阶段推进 |
| longest phase lock | 83 replans | 单一 phase 连续影响约 332 个实际动作 |

诊断脚本现在会对该证据同时报告：

```text
dominant_memory_event_collapse
memory_event_phase_lock
```

## 3. 已排除的底层适配问题

### 3.1 代码版本

本地 `main`、`origin/main`、V3 training attestation 均为：

```text
4f3678353f891c94eb3538639cfc1e4a135b3683
```

因此结果不是由服务器使用了另一份未同步代码造成的。

### 3.2 相机接口

官方 evaluator 提供 `head_camera`、`left_camera`、`right_camera`。WARM 严格映射为训练侧 `cam_high`、`cam_left_wrist`、`cam_right_wrist`，顺序与 official50 转换器一致。证据中的三路原始图像和模型输入均为有限值且随环境变化；OIDN `invalid handle` 是 SAPIEN 渲染器的降噪告警，并未生成空帧或 NaN，不是 0 成功率的原因。

### 3.3 14D qpos/action 接口

官方 observation 和 action 都采用：

```text
[left_arm_6, left_gripper, right_arm_6, right_gripper]
```

转换器用 `state[t] = qpos[t]`、`action[t] = qpos[t+1]`，evaluator 的 `take_action(qpos)` 按完全相同的左右臂索引拆分。normalizer 的 action round-trip 在每次 replan 都进行数值校验。不存在 arm swap、gripper index 错位或 action reverse。

V4 额外记录 factual qpos、normalized proprio、即将执行的 4 步 qpos delta、左右臂 delta 和 gripper target，以便下一次无需视频也能判定策略是在运动、抖动还是保持错误姿态。

### 3.4 Horizon 与队列

V3 resolved config 同时满足：

```text
action_horizon = 32
episode_action_chunk_size = 4
retrospective_action_summary_chunk_size = 4
eval replan_steps = 4
```

`RecedingHorizonQueue` 按 FIFO 执行前 4 个 action target，不会逆序。证据中的 5373 replans 对应 21464 个已执行动作，也与 4-step receding horizon 一致。

### 3.5 数据 partition

specialist 训练只使用 `blocks_ranking_try` 的 45 条 train demonstration，5 条 dev demonstration 不进入训练监督。长期 bank 虽为九任务共享物理存储，但 777D context key 含固定的 task one-hot；当前候选 episode 范围与 blocks partition 一致，未发现跨任务 action source 泄漏。

## 4. 根因

V3 的 temporal-thread 实现每次接受候选后都执行：

```text
thread_event = selected_event
thread_query_frame = current_query_frame
```

即使下一次仍选择完全同一个 event，也会重置 query anchor。于是 expected event frame 永远只比当前 event 前进 4 步；同一最近邻始终获得正 thread prior，永远不会被判定为已经消耗完毕。跨 demonstration 的相同阶段还可以通过切换 episode id 继续绕过旧 thread 约束。

这与本次证据完全一致：source 已被高频使用，但 55.37% 都来自 episode 257/frame 587，且 phase lock 可持续数百个真实动作。更多训练步数不会自动修复这个确定性的在线状态机错误；继续评测同一 V3 checkpoint 也只会重复失败。

V3 的 repeated-attempt 信号没有形成第二道保护。它要求 action signature 相似且 world/VAE change 都低于阈值；模拟器中的轻微视觉变化足以不断清零计数，所以本次 `maximum_repeated_attempt_count=0`。V4 不再依赖这个启发式信号来保证 event source 的有界使用。

## 5. V4 修复

### 5.1 持久 phase anchor

首次接受 event 时记录：

```text
anchor_event_start
anchor_query_frame
```

后续选到同一或 phase-aligned event 时不更新 anchor。只有候选的 event frame 真正超过当前 high-water mark 才推进 anchor。该规则跨 demonstration 生效，不能通过切换 rollout id 绕过。

### 5.2 Action-horizon source lease

同一 event 在后续 replan 再次被选中时，V4 不再重新使用 action chunk 的第 0 步，而是按已经执行的真实环境动作数移动 cursor。对于 absolute-qpos source，新的 suffix 同时以 cursor 前一个历史 target 作为 canonical start 重新对齐当前 qpos；close/open timing 也同步平移。这样 4-step receding horizon 会依次消费 `[0:4]`、`[4:8]`、…，而不是永远重复 `[0:4]`。

当：

```text
current_query_frame - anchor_query_frame >= action_horizon
```

且候选没有推进到更晚 phase 时，该候选被标记为 `thread_source_eligible=false`。它仍可作为 context token 帮助 Action DiT 理解历史，但：

```text
effective source probability = 0
source quality = 0
accepted source mask = false
```

因此 action flow 精确回退到 Gaussian source。即使 reranker 选择 null component，也不会在三次空检索后清除已经耗尽的 phase high-water mark；只有真正的前向 event 或新的环境 episode 可以重置它。

### 5.3 审计字段

每次 replan 新增：

```text
thread_source_eligible
thread_phase_elapsed_actions
factual_qpos_stats
normalized_proprio_stats
executed_prefix_qpos_delta_stats
executed_prefix_qpos_step_delta_stats
executed_prefix_arm_delta_stats
executed_prefix_gripper_targets
```

证据分析新增 dominant-event、exact repetition、phase-lock rate 和 longest phase-lock 检测。这样下一次可以直接区分：检索锁死、source starvation、动作无位移、gripper 错误或真实任务阶段推进失败。

## 6. 训练与评测决策

V4 checkpoint schema 升为 v6，输出目录改为 `*-s3407-v4`。禁止把 V2/V3 optimizer state resume 到 V4；从相同 Fast-WAM base 新建训练，确保训练和 formal evaluation 使用同一 commit 和完整 attestation。

当前 official50 只含 45 条 blocks train demonstration，适合验证机制，不足以保证最终 SOTA。执行顺序应是：

1. 10-step、4×H100 smoke，验证 loss、gradient、checkpoint 和新 telemetry contract；
2. 1000-step checkpoint 做 3 个固定 seed 的小规模 `no_memory/context_only/full` 诊断；
3. full 分支必须不再触发 phase-lock，并至少表现出有效 task progress，才继续 6000/14000；
4. 机制通过后扩充自动成功轨迹到约 200 条/任务，重建 conversion、M1/M2 bank/candidates，再做 SOTA 训练；
5. 正式成功率必须在 fixed seed protocol 下完成足够 episode，不用个别成功视频替代统计。

下一次最小归档集：

```text
训练：console.log、resolved_config、config.yaml、dataset_stats、training.json
评测：official_stdout.log、warm_online_evidence.jsonl、online_contract.json
视频：至少 1 个完整失败 rollout；出现成功后再保留 1 个成功 rollout
```

尤其需要一个完整失败 MP4。现有数值证据足以确定 phase lock，但视频才能进一步判断 Gaussian fallback 后是否已经能完成 button press、block grasp/release 和顺序交换；如果仍失败，下一步应针对 imitation/action geometry，而不是再次盲调 memory gate。
