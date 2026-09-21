# WARM 非真机实验：72 小时执行计划

本计划取代交接文档的五天排期。资源：1–4 张 H100 80GB；每个 ACP 最长 24 小时。本轮只推进 P0 和无需新训练的必要工具。ACP 由用户在网页手动提交。

最终工程验收：服务器 H100 上通过 **124 tests，0 failures，0 skips**；1 条已有测试夹具的 readonly NumPy 警告。实际使用下文 ACP wrapper 执行，退出码 0，运行前后源码哈希一致。日志目录：`/mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations/nonreal72h_jobs/20260921T111938Z-50bc29b9`。另已通过两任务、四个合成 query 的候选/gate CLI 全流程；合成数据仅验证软件，不构成实验结果。

交付索引：`HANDOFF_VALIDATION.json` 是代码验收与未放行条件；`ARTIFACT_AUDIT.json` 是 11 份 checkpoint 元数据及六行消融 U 身份清单；`configs/nonreal/TASK_BOARD.json` 是三天任务状态。源文件哈希是运行代码身份，不是 checkpoint 训练时的 Git commit。

## 当前结论（2026-09-21 实测）

**先解除模型身份与渲染阻塞，再消耗 ACP 配额。当前没有已放行的正式 rollout 命令。**

已通过 SSH 检查 `/mnt/afs/task3_2/L202500276_lwz/projects/WARM` 及其 runs、WARM_artifacts、WARM_evaluations、WARM_training、tmp/temp。没有执行 Git 操作，也没有启动训练或正式评测。

| 发现 | 对实验的影响 |
|---|---|
| 当前节点可见 1 张空闲 H100 80GB；持久环境 Torch 2.7.1+cu128、SAPIEN 3.0.0b1 可导入 | 可运行真实 Torch 代码验证；不等于可运行仿真 |
| 各训练目录最新 checkpoint 共打开检查 11 份 | 文件名中的 v1/v5 不是模型 schema；必须读权重元数据 |
| 当前 schema v8 只找到 shared 300-step smoke | 可验证工程路径；不能作为成熟策略或论文主表的 checkpoint |
| PressButton/CoverBlocks/Battery 12k：schema v3；BlocksRanking 14k：v3/v4/v5；v6 仅 1k | 不能修改版本号或跳过加载检查后作为 v8 证据 |
| 未找到 PutBackBlock 的正式 specialist，也未找到主表/六行消融/五轮干扰对应的逐 episode RMBench 结果 | 原论文汇总值继续标成“已报告、未复核”；不能生成假逐 episode 记录 |
| 实际 renderer preflight：`failed to find a rendering device`；容器未发现 NVIDIA Vulkan/GL 图形驱动库 | 当前容器尚不能启动 W01/W02-B/C rollout；需要具有 graphics/Vulkan 驱动挂载的运行环境并重新实测 |
| 原代码 Torch 回归 16 failed / 23 passed | 主要是旧测试夹具/字段/门控预期；本轮修正夹具，不为通过测试修改现有训练机制 |

原始核验文件在服务器 `WARM_evaluations/nonreal_72h_20260921/`：`inventory.json`、`checkpoints.jsonl`、`baseline_tests.log`、`probe_tests.log`、`simulator_preflight.log`。inventory 中的 `.pt` 也包含 text embedding，不能把初始 82 个条目都当成策略权重。目录扫描只能证明此次范围内的发现，不能证明其他服务器没有产物。

本地提供的 PDF SHA256 为 `b1f5d91da26dc5fef3a9560826f3a3329946cd0c79a7f2425dda6a2b3a8b5931`，与交接文档写的 `fa79304…` 不同。本计划使用交接文档的实验缺口定义；回填页码/表格前必须确认最终 PDF 版本。

## 已实现的 P0 代码与边界

| 工作包 | 本轮实现 | 仍需什么才能形成论文证据 |
|---|---|---|
| W00 | `scripts/nonreal_inventory.py`；任务板与 manifest 模板；只读服务器发现 | 正式 checkpoint 与原论文各行的身份关联；旧原始记录 |
| W01 | `research/branches.py`：每个候选 H32 无 replanning；记录 planned/actual commands、提前终止、异常；六类恢复指纹；失败终止后续分支；真实模型导出 adapted μ、历史/预测 effect、requirement | **实际 RMBench backend 尚未接入/验收**；需要完整状态/控制器/任务计数/RNG/观测/策略状态恢复。接口及 toy tests 不算仿真恢复通过 |
| W02-A | runtime-only `force_null` 同时关闭 source 和 candidate conditioning；保留 token 数；固定随机数测试候选替换不变性 | 目前验证的是真实 source/conditioning 图和 tiny experts；尚未完成正式 checkpoint 的 50-prefix、完整 Action DiT 输出比较 |
| W02-B | 导出自然选择的 α/g/κ/ζ、v_det、实际阈值；按 sealed proposal 的 selected rank 关联独立标签；FA/TA、unknown/empty/hard-veto/mean g 与 episode bootstrap | 自然/干扰 cohort 的独立适用性标签；eligible-negative 覆盖 |
| W02-C | 分支引擎保留空/不完整证据，不将未知标为无效 | empty 与全候选 all-inapplicable 闭环评估均未完成；不能拿强制 null 代替 |
| W03 | MSE、κ、成对排序、tie=0.5、top-tie applicability、query→episode→task macro；配对 success contrast/Γ；拒绝缺配对与多 checkpoint 混用 | 真实逐 episode / candidate records |
| W04 | 按 cohort/protocol/task/condition 统计实际替换率与 gate 前 eligible 比例及 query 分布 | 原 donor pool/规则/seed/odd-K 等协议和逐 query 记录；未实现新的 donor 替换策略，不冒充旧实验 |

额外提供 `gaussian` / `scale_only` 局部 source 干预，但没有排新训练。No-comparison 只去掉排序和 gate 输入中的 κ，保留 effect predictor、适配、reranker 的历史 effect 输入和两条 memory 路径。原 `source_only_no_consequence` 语义不变，不能把它当 no-comparison。

所有新增 controls 都是 **I：固定 checkpoint 的推理干预**，不进入 checkpoint，不改变 training loss；不开启时不改变 source/conditioning 行为。W00 中“无双库/仅 recollection/仅 repertoire”的独立训练开关未实现，六行训练消融身份仍为 U，不能用当前 controls 补写成 T。

集成检查另修复两处真实接口错误：

| 设计预期 | 原实现 | 修复与影响 |
|---|---|---|
| RMBench 的 bank scalar timing 是 4D，模型从左右夹爪动作生成 8D timing | 在生成前即按模型 8D 拒绝 bank 的 4D 输入 | 先验证 bank 4D，再生成并验证模型 8D；修复 v8 在线 context 构建失败，不改变模型权重 |
| LIBERO action=7、proprio=8；不能把双手指状态猜成单夹爪动作通道 | 离线 timing 无条件要求 action/proprio 等宽 | 维度不同时，仅明确单夹爪契约使用已有 observed-gripper timeline，与在线路径一致；多夹爪仍要求对齐且拒绝猜测 |

单独在 H100 上通过了 fp32/bf16 的双门归零 source 图检查，容差取同输入重复调用误差；仍不等价于正式 checkpoint 的完整 DiT / 仿真验证。

## Claim → 实验 → 证据

| Claim / hypothesis | 实验 | Metric / 判据 | 放行条件 |
|---|---|---|---|
| 预测 effect 比历史 effect 更能反映当前动作后果 | 同一 held-out query 的所有合法候选实际执行 H32 | effect MSE；对 observed κ 有差异的 pairs 排序；episode-clustered paired difference | 预测先封存；恢复通过；完整端点才计算 Eobs；三参考共同支持集 |
| gate 学会拒绝不适用候选 | 自然选择候选的独立适用性标签 | eligible FA/TA、分子分母、unknown 覆盖、CI | 不混入强制 null；v_det 不包含 α 阈值 |
| 双路关闭后没有候选信息泄漏 | 固定 prefix、Gaussian、solver、mask、token 数；先测 identical-call 容差再替换候选 | 完整 DiT 后、后处理前 max action error | 容差在替换前冻结；不据此声称 learned rejection |
| 事件干扰优势来自比较机制 | normal/donor × Full/no-comparison 同 reset/layout 配对 | Γ = no-comparison drop − Full drop；task macro CI | 原 donor 规则找回或明确作为新 cohort；训练/推理干预分开 |

统计单位是固定任务内的 source episode/reset block。同一 reset 的重复评估放入同一 cluster；不把候选数或 query 数当独立样本。零分母为 null；不根据 CI 包含 0 宣称 non-inferiority。程序输出比例为 0–1，写表百分比时再乘 100。

## 三天安排与停止扩张条件

| 时间 | 主任务 | 可交付物 / 决策 |
|---|---|---|
| T0–4h | W00：找回正式 checkpoint、原记录；修复或更换有图形驱动的容器 | checkpoint/config/bank/normalizer 身份表；renderer 通过日志。找不回就停止“复现原表”的承诺，不用 smoke 替代 |
| T4–12h | 在现有调试节点完成真实 backend 恢复、1 query×2 candidates、异常/提前成功自检；再做每任务 5-prefix pilot | 原始 probe、branch 记录、恢复误差、单 branch / episode 耗时。**这些通过之前不排正式 ACP** |
| T12–36h | ACP-A：W01 自然候选；ACP-B（有第2张GPU）：W02 独立标签/边界；若需要同一分支结果则直接复用 | 首批正式且完整的证据；每 job ≤23h，预先固定 query 列表，不按结果停采 |
| T24–48h | 有原协议/模型时才用 ACP-C/D 补 donor telemetry 或配对评估；否则继续自然机制样本 | 每个 cohort 有独立 manifest。不要默认重跑 7000 rollouts |
| T48–60h | 只完成已有任务、补缺失文件、盲审标签；不启动新训练/新方法 | evidence coverage、missing/exclusion 清单 |
| T60–72h | CPU 统计、失败案例、论文回填和 claim 收缩 | 可追踪结果、均值/CI、明确不能支撑的 claim |

1 GPU 时按 A→B 顺序，优先两个任务的 W01；4 GPU 时每个独立 rollout worker 1 GPU，固定不重叠 episode 清单，合并前检查重复 ID。GPU 数不是样本独立性的定义。

独立标签复核和统计使用 CPU/人工流程，不为它们单开 GPU 任务；ACP-B 只有实际候选分支或边界闭环需要 GPU。第二张 GPU 优先补 W01 的另一任务或预先划分的 episode shard，先确保两任务覆盖，再考虑 donor 扩展。

正式规模以 pilot ETA 一次性冻结：`预算秒 / 每查询(K×branch秒+恢复+推理秒)`，预留至少 20% IO/恢复余量，且保留最后 12 小时统计。交接建议 40 source episodes/task、最多 3 prefixes 不是必须硬凑的数量；缩减后报告实际覆盖。K 使用 checkpoint/bank 实际值，不能默认 8。

如果确无成熟 v8 checkpoint：目前不能在 72h 内承诺补齐全部 claim。已找到 smoke 末尾速度约 0.274 optimizer steps/s，但这不是新 GPU 数/新训练配方的可靠 ETA；30k 步约 30h 只是同吞吐粗估，之后还有 specialist 和评估。优先找回论文 checkpoint；不要默认从头训练六种变体。任何缩短训练的版本都须重新标预算、同预算对照，不能回填原表。

## ACP 命令与日志

源代码已放入独立发布目录，避免覆盖服务器原 checkout；没有使用 worktree 或 Git。验证命令（调试节点执行，不必浪费一次 ACP 排队）：

```bash
WARM_REQUIRE_CUDA_PROBE=1 bash /mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations/nonreal_72h_20260921/release/scripts/acp_nonreal72h.sh
```

同一命令可粘贴到 ACP 做环境迁移后的核验；它只运行工程回归，不产生论文实验结果。真实实验 spec 必须在 checkpoint、backend、renderer 和 1-query pilot 均通过后生成并改为 `readiness=ready`。目前任务板中的 W01/W02 rollout 是 blocked，**不要提交占用 GPU**。

后续提交统一用以下形式（先在调试节点加 `--dry-run` 验证，再粘贴不含该参数的命令）：

```bash
bash /ABSOLUTE_RELEASE/scripts/acp_nonreal72h.sh /ABSOLUTE_FROZEN_JOB.json --dry-run
bash /ABSOLUTE_RELEASE/scripts/acp_nonreal72h.sh /ABSOLUTE_FROZEN_JOB.json
```

`nonreal_job.py` 接收 argv 数组，不执行 shell 拼接。spec 中写入输入 SHA256、qualification 文件 SHA256、明确 claim/evidence_type 和 ≤82800 秒时限。旧 ACP specialist launcher 会操作 worktree，故本计划不直接调用它。

每次运行独立目录包括 `job.json`、完整文件级 `source_manifest.json`、`run_manifest.json`、合并 stdout/stderr 的 `console.log`、每分钟 heartbeat 和最终 exit 的 `events.jsonl`。保留退出码；超时、信号、日志失败、运行中源文件变化均不标为 complete。SIGKILL/节点消失无法写最终 exit，应依据缺失 exit 认定 incomplete。校验不通过写 `preflight.jsonl`，不启动实验子进程。未知 Git commit 保持 null，不用 source hash 冒充训练时 commit。

统计命令（有真实输入后，无需 GPU）：

```bash
PYTHON=/mnt/afs/task3_2/L202500276_lwz/envs/warm-rmbench-eval/bin/python
CODE=/mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations/nonreal_72h_20260921/release
"$PYTHON" "$CODE/scripts/assemble_nonreal_candidates.py" --index /ABSOLUTE/frozen_queries.jsonl --branches /ABSOLUTE/candidate_branches.jsonl --labels /ABSOLUTE/independent_labels.jsonl --kind candidates --output /ABSOLUTE/candidate_queries.jsonl
"$PYTHON" "$CODE/scripts/assemble_nonreal_candidates.py" --index /ABSOLUTE/frozen_queries.jsonl --branches /ABSOLUTE/candidate_branches.jsonl --labels /ABSOLUTE/independent_labels.jsonl --kind gates --output /ABSOLUTE/natural_gates.jsonl
"$PYTHON" "$CODE/scripts/analyze_nonreal_evidence.py" --kind candidates --input /ABSOLUTE/candidate_queries.jsonl --config "$CODE/configs/nonreal/candidate_metrics.json" --output /ABSOLUTE/new_candidate_metrics.json
"$PYTHON" "$CODE/scripts/analyze_nonreal_evidence.py" --kind gates --input /ABSOLUTE/natural_gates.jsonl --config "$CODE/configs/nonreal/candidate_metrics.json" --output /ABSOLUTE/new_gate_metrics.json
"$PYTHON" "$CODE/scripts/analyze_nonreal_evidence.py" --kind episodes --input /ABSOLUTE/episode_outcomes.jsonl --config "$CODE/configs/nonreal/paired_metrics.json" --output /ABSOLUTE/new_paired_metrics.json
```

`candidate_metrics.json` 的 tolerance 是待冻结的新建议，不是恢复出来的旧论文参数。输出以独占创建方式写入，已有结果不可静默覆盖。`--kind gates` 使用自然 gate JSONL；`--kind donors` 使用 donor coverage JSONL。

`frozen_queries.jsonl` 每行需有 `task, episode_id, query_id, cohort, probe_path, probe_metadata_sha256`；候选 ID 使用封存数组的 `rank-0, rank-1, …`。分支必须由 `execute_branches(..., proposal_sha256=metadata_sha256)` 记录。标签单独保存 `query_id, candidate_id, applicable(0/1/null), label_provenance`。组装器拒绝跨 proposal 的分支、重复/悬空 ID 和未确认恢复的结果；未尝试/未完成端点不会补零。自然 gate 汇总拒绝 forced-null、no-comparison 等推理干预；all-inapplicable 标记只说明该 query，不证明整条闭环都满足边界条件。

模型采集开关为 official policy overrides 的 `warm_research_probe_config`（绝对 JSON 路径）、`warm_research_probe_output`（源码外的新目录），并要求 `warm_experiment_id` 以 `nonreal-` 开头。这是诊断扩展，不能直接把输出混入旧 formal result。Gaussian/scale-only 的因果比较必须从相同 prefix 恢复；三个独立闭环 trajectory 不保证保持同一 realized g。

## 下一步必须解除的两项外部阻塞

1. 提供或找到论文使用的正式 v8 checkpoint / 对应源码及解析配置；若论文实际使用的是旧 schema，先明确对应方法版本，不能把旧权重套上新机制。
2. 在目标 ACP 镜像/容器中提供 NVIDIA graphics/Vulkan 驱动挂载，并让真实 renderer preflight 通过。当前没有安装/覆盖主机驱动，也没有更改系统 NVIDIA runtime。

随后继续完成 W01 的实际 backend 和 pilot；在它们通过之前，这份交付属于“代码与证据基础设施已验证，正式测量未放行”，不是剩余实验已完成。
