# Full / 同信息池化：最终3000原生比较冻结设计

17:18:44UTC、两角色最终3000结果产生前，已在AFS冻结`MATCHED_NATIVE_FINAL3000_EVAL_DESIGN_V1/DESIGN.json`，SHA3484ad1477a868607a916638c631c80df25501db59dd82edbf6651c768238e33。3项设计/部署差异文件及manifest380b7d45…已在AFS与本机`tmp/rebuttal/execution/20261111_iter10`核验。状态为design frozen、not dispatched，资源时钟未开始；不能将该文档当作已运行或已有性能数据。

沿用旧primary56b6ec9a…事前指定的Rearrange Blocks/Battery Try任务、各首50个expert-admitted seeds、340800000起始/2000候选上限及相同unseen指令协议。新比较的两角色是Full与独立训练PooledMemoryPrior，均固定seed3407/final3000，100配对/200策略回合。旧Full/null实验和缺失技能协议保持原身份，不用新角色结果补填旧pending，不用旧sharedv8或中途100/200权重替代新final3000。3408/3409训练尚未完成，不报告训练种子不确定性。

| 项目 | 冻结要求 |
| --- | --- |
| 原生回合终止 | RB700、BT1000，或official executor success；不截短成另一个cap |
| 共同推理 | H32/prefix4/NFE10/K32，同bank、encoder、normalizer与因果信息访问 |
| 模型和部署 | 明确Full/Pooled不同factory/schema，共同eval source87199cc6…；Full训练4efb…、pool训练48ce…身份保留 |
| 初始配对 | fresh reset的RGB、官方关节API、逻辑场景和指令精确相同；case奇偶对调执行先后 |
| 随机性 | 共同base seed和role-independent task/case namespace；核验初始Gaussian。轨迹分叉后不宣称后续当前观测/history相同 |
| 统计 | 完整100对后的逐任务配对成功差与两固定任务宏平均；gain/harm精确边际及Bonferroni界，不把K候选或训练seed伪作独立样本 |
| 未知与失败 | policy失效/中断/pending独立报告，保留原seed和失败attempt，不填0、不按结果换seed；未齐全时仅报complete-case描述 |

共同部署边界使用已核验的原processor affine backward解码，不夹断生成的motor command；已实际执行的prefix经原forward映射进入history。Full训练快照到共同eval部署的两份完整规范化diff已经保全，变更包含执行prefix审计及其telemetry；不把它说成所有部署源码bytes相同。此前94项模型/memory/dataset静态核对与pool100真实10/10原生DEV资格保留；这些不是final3000效能结果。

正式准入仍需两final3000完整状态/逐字节checkpoint收据、明确typed contracts、Full实际原生资格、固定50-reset roster、各worker可写模拟器/planner/临时路径隔离及真实并发确定性检查。原native worker全局lease不得被准备阶段绕过。AFS须留足原训练state/CAS与有界新轨迹，不删历史原件。每回合保留初末图、全部H32生成/已执行动作状态、telemetry/源与输入hash和预定query证据，避免无界重复大tensor占满AFS。

实现门槛另行明确：现有冻结`run_rebuttal_rmbench_pairs.py`只处理Full/candidate-null与单一checkpoint，不能直接充当新Full/Pooled比较入口。下一项独立工程工作是实现并验证明确双checkpoint/双factory、真正Pooled模式的原生collector与有界续跑收据。须在TEST前明确基础设施异常、可归因的策略无效输出和原生任务终止的分类；原因未知的失败不得补0，已观测的策略失败也不能通过换seed隐去。冻结设计本身不代表这个运行程序已完成。

资源设计为最多6台单H100、10个task×10-case固定分片，每分片首次准入起最多12小时，总准入上限120GPU小时，统一phase最多36小时且不晚于Nov12 18:00UTC。标准优先；只有实际GPU配额拒绝后才闲时。排队、失败和重试都计入原分片/phase时钟，不因恢复重置。按既有吞吐估计约70–90有用GPU小时只是排期估计，不是测得完工保证。当前没有新增ACP评测任务或GPU时间消耗。

该比较支持BB-W2、HH-W1/3的同信息机制价值问题，仍是shared-stage受控补充，不是具名ReCAP/MemoryWAM复现或原稿逐任务specialist准确率复现，也不等同另一个missing-skill的2pp非劣协议。
