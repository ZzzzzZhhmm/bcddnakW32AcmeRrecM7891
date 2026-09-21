# 剩余实验数据与下一步执行指令

2026-09-21 12:55 UTC核验：W02-A与W06工程诊断已完成，当前没有正在运行的GPU实验。所有已获数值见[实验记录](EXPERIMENT_RECORD_ZH.md)。

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

1. **先做训练更新诊断与正式模型身份恢复。** CPU诊断已确认alpha=0.11914低于0.15，v_det全真且非空；gate输出权重已有变化，不能判为完全未训练。单卡检查真实训练batch上gate loss/梯度、optimizer参数注册、一步前后参数变化、BF16与FP32更新差异及保存/加载一致性。目标是识别训练或数值问题，不以降低阈值通过实验。先限定1–4个batch，不启动长训练。
2. 同时继续只读寻找98.7%和83.9%所对应的正式运行产物，核对旧schema与当前机制差异。找到模型优先补评估；找不到则明确新实验范围，不能用smoke补原主表。
3. 在具备NVIDIA graphics/Vulkan挂载的环境修复/验证renderer，完成实际RMBench状态恢复backend，再跑1 query×2 candidates的H32恢复与端点pilot。没有这个闭环不能生成W01/W02正式ACP。
4. 根据pilot的实测每branch耗时冻结两任务样本预算，单卡先做W01并复用W02标签。多卡只拆独立episode shard，不改实验口径。W03/W04仅在有配对原始记录/协议后推进。

步骤1的时间需要完整训练forward/backward实测；步骤3的恢复backend尚待实现和验证，不能给出已可运行的正式ACP。已有4H100训练速度粗估约30小时才能把shared从300续到30k，另需初始化/保存/验证、至少两个24小时ACP及后续评测；这个估计不是批准开跑的恢复命令。

## 可直接复核的CPU命令（已验证，不申请ACP）

在CCI执行，输出使用新时间戳并保留控制台日志：

```bash
set -euo pipefail
PY=/mnt/afs/task3_2/L202500276_lwz/envs/warm-rmbench-eval/bin/python
BASE=/mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations
TOOLS="$BASE/evidence_archive_20260921/diagnostics"
OUT="$TOOLS/gate_activation_$(date -u +%Y%m%dT%H%M%SZ).json"
"$PY" "$TOOLS/analyze_nonreal_gate_activation.py" \
  --run-root "$BASE/nonreal_source_r2_20260921" --output "$OUT" \
  2>&1 | tee "$OUT.console.log"
```

该命令只复核并汇总已完成的50前缀，不产生新的任务成功率，也不修改checkpoint。预计数秒到一分钟。

## 下一步可直接派发的工作指令

> 优先处理P0前置阻塞：在CCI单张H100上完成真实训练batch的gate梯度、optimizer更新、BF16数值与checkpoint保存/恢复诊断，限1–4个batch；保留原checkpoint，不改推理阈值。继续核对正式模型和论文主表原始结果的身份。随后完善RMBench renderer与完整状态恢复，完成1-query×2-candidate的H32 pilot。每项数值、失败和原始产物及时写入EXPERIMENT_RECORD_ZH.md。完成上述验收后才生成正式W01/W02 ACP命令和基于实测的耗时预算；不要重新运行已经完成且全g=0的smoke source对比，也不要直接重训六种消融。
