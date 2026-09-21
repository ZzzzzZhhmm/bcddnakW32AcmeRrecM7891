# 剩余实验数据与下一步执行指令

2026-09-22更新：W02-A、W06与N00梯度诊断均已完成，不重跑。N00两个真实TRAIN batch梯度有限且非零；FP32 master保留了小更新，无证据要求改模型或降低阈值。新单卡工作为W08记录DEV观测的完整在线链路profiling；四卡续训已完成CPU预检，见[ACP命令与自动汇总](ACP_RESUME_ZH.md)。所有已获数值见[实验记录](EXPERIMENT_RECORD_ZH.md)。

## 尚缺什么

| 优先级 / 工作包 | 缺少的数据 | 影响论文位置 | 当前放行条件 |
|---|---|---|---|
| P0 / W00 主表与消融身份 | LIBERO 1974/2000、RMBench 755/900对应的逐episode结果、checkpoint/config；六行消融独立训练或推理干预身份 | 表3/7/9 | 已恢复的历史LIBERO为1936/2000，不能冒充1974/2000；RMBench成熟兼容模型未定位 |
| P0 / W01 后果预测与排序 | 同一恢复状态下逐候选H32真实端点、预测/历史/query-only三参照、失败与missing、独立适用性标签 | 表19/20、p.9 | 正式模型、renderer、完整state恢复backend、1-query pilot都须通过 |
| P0 / W02 学会拒绝与边界 | 自然gate与独立正负标签；eligible FA/TA分子分母；unknown；all-inapplicable闭环结果；正式模型零门控复验 | 表23/24、p.9 | 已通过的smoke forced-null不能替代这些数据；复用W01分支 |
| P0 / W03 配对差值 | 同reset/seed的方法逐episode成对结果、任务/episode聚类信息 | 表12 | 只有单方法边际汇总无法恢复配对CI |
| P0 / W04 干扰覆盖 | 原五轮donor协议、逐次替换/eligible记录、失败/回退及成对结果 | 表10、图3、表14 | 找不到原协议则明确建立新cohort，不能补造旧覆盖率 |
| P1 / W05 同信息对照 | 一个预算匹配history-aware retrieval–residual对照的训练身份与评测结果 | 表21 | P0模型/评测通路通过后再选一个，避免扩张 |
| P1 / W06 source贡献 | 成熟模型的自然非零g前缀；必要时独立训练scale-control及公平SR | 表22 | 当前50前缀全g=0，局部诊断没有有效激活覆盖 |
| P1 / W07 历史作用 | 语义定义的critical/noncritical记录、只改requirement read的配对误差/choice及episode区间 | 表18 | 真实事件语义标签、示范未来或独立choice标准 |
| P1 / W08 性能与失败 | 正常在线链路端到端median/p95、显存、bank/token规模；可追溯失败案例 | 表25、C.5–C.6 | 离线DiT耗时不能替代端到端测量；需要可运行在线通路 |

## 推荐执行顺序

1. **训练更新诊断已通过，转向兼容checkpoint续训。** 原四卡完整状态已找回并核验，使用原recipe从300续至15000作为首段，保留总计划30000步；由用户提交四卡ACP。单卡同时补W08完整在线链路的记录前缀profiling，不再重复原50前缀source或forced-null实验。
2. 同时继续只读寻找98.7%和83.9%所对应的正式运行产物，核对旧schema与当前机制差异。找到模型优先补评估；找不到则明确新实验范围，不能用smoke补原主表。
3. 在具备NVIDIA graphics/Vulkan挂载的环境修复/验证renderer，完成实际RMBench状态恢复backend，再跑1 query×2 candidates的H32恢复与端点pilot。没有这个闭环不能生成W01/W02正式ACP。
4. 根据pilot的实测每branch耗时冻结两任务样本预算，单卡先做W01并复用W02标签。多卡只拆独立episode shard，不改实验口径。W03/W04仅在有配对原始记录/协议后推进。

四卡首段预计16–20小时，已绑定原weights、完整state和不变recipe，现场仍需四卡runtime与恢复检查。步骤3的恢复backend尚待实现和验证，因此W01/W02正式rollout仍未放行。完整300→30k约30小时计算，另需初始化/保存/验证及后续评测，至少两个24小时ACP。

## 查看本轮单卡profiling（不申请ACP，不重复启动）

在CCI执行；任务自带完整日志和一小时硬时限：

```bash
set -euo pipefail
RUN=/mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations/nonreal_online_profile_20260922
tail -n 30 "$RUN/profile200.launcher.log"
```

预声明Press Button五个DEV episodes，每个40个连续replan，共200测量query；另20预热query排除。真实在线检索、历史和推理；观测和已执行动作来自记录轨迹，模型输出不执行，不产生任务SR。contract准备与模型加载受共享存储影响，预计30–45分钟；只有wrapper complete/exit0和200条记录验收通过才记为完成。

## 下一步可直接派发的工作指令

> 完成W08记录前缀profiling并保存真实成本指标；用户执行四卡续训首段后，验收checkpoint与原始训练指标，再补新checkpoint上尚缺的自然gate/source数据。继续完善RMBench renderer和完整状态恢复，完成1-query×2-candidate的H32 pilot后再派发W01/W02。每项数值、失败和原始产物及时写入EXPERIMENT_RECORD_ZH.md。不要重跑已完成的N00/N02/N06 smoke诊断，不重训六种消融。
