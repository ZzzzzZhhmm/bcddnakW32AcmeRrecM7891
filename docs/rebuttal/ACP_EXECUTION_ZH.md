# WARM rebuttal：CCI开发与ACP运行

作者于2026-11-09确认可申请ACP算力，标准资源优先，标准GPU配额不足时再使用闲时资源。CCI用于修改、测试和冻结代码，也可运行适合单卡且小于4小时的任务；ACP只运行准备好的快照。自动填写的默认配置见`configs/nonreal/rebuttal_acp_defaults.json`。

- 私有镜像：`ccr-zhicheng-02/lwz_wam:lwz-workspace-20261029145800`。
- AFS：`afs-p3-01e`，子目录`/`挂载到`/mnt/afs`，以当前用户权限读写。
- 工作空间：`p3-spatial-intelligence`，上海一区`cn-sh-01z`。

首批ACP入口`run_rebuttal_acp_pairs.py`为两个独立worker各分配一张平台可见GPU，分别运行Full和candidate-null的前10个Rearrange Blocks官方回合。使用已绑定的作者v8-step3000模型、原core、原bank/data recipe与固定v3主计划；H32、prefix4、NFE10、K32、官方700步和自然事实历史不变。仅当独立初始图像、qpos、逻辑任务状态、指令和完整角色身份一致后才合并。资源并行度不会变成采样预算改变。

作业启动时记录实际GPU、驱动、Python/Torch以及控制器/core/配置身份。旧图形客户端只适配580.95.05驱动；不同驱动必须留下preflight失败，另行在CCI准备并验证客户端后重提，不能混装或跳过核验。每个批次最多6小时，失败、未完成回合和退出码保留。输出、运行日志与结果均在AFS的Git外目录；作业内不拉取代码、不探测Git远程、不安装依赖。

本轮已完成的Battery Try正式case0为Full0/1、null0/1，两个角色各执行1000步。这只是一个固定完整pair，不替换投稿表格，也不足以建立策略收益。正式机制source0因第一条事实轨迹不匹配停止；原分支与失败保留，在CCI修复后重新资格验证。匹配Full和conditioned-prior各到step100，Full完整state续训和AFS无损重建通过，但3000更新及SR比较仍未完成。

首个闲时2卡作业已创建，UID `9515df9ea9354f32acc2cda0ae1a4f75`；但Worker事件明确为`FailedScheduling / insufficient nodeSpec quota`，涉及CPU和内存，未运行策略。较紧凑的单卡8核128GiB规格可使用两个独立进程共享这一张分配GPU，各自模型/检索器与事实历史独立；之前CCI已实际容纳两角色约57GiB。新执行布局必须显式冻结为`shared_single_gpu`，最多12小时，不能把共享卡结果用作延迟基准。该布局保持原前10对、官方上限和采样配置，仅改变资源分配。

机制采集另设`--factual-replay-only`工程验证：与原Full source一致，在模型构造后重新reset，不在事实轨迹中pack/unpack；逐条比较重新生成的命令及before/after qpos，不以原动作驱动策略。只运行完整原回合，不截短；即使通过，也不计为新增独立SR或反事实机制证据。普通机制模式在分支之前就检查事实首条命令，避免在已偏离源轨迹后继续采集。异常结束仍记录实际已完成query数量，原失败记录不覆盖。

新闭环使用`qualify_rebuttal_reset_roster.py`按原expert准入顺序预先封存reset roster，不依据策略成功与否选seed。`run_rebuttal_resumable_campaign.py`在同一分配GPU上逐角色执行固定case，完整角色须通过模型/core/协议/runtime、700/1000官方步数、native轨迹与初始图像哈希核验，再单独压缩封存；重启后仅补缺失角色。未提交的JSONL尾部不作为标签，完整角色重复会停止并要求显式核销，不按成绩挑选。子进程继承campaign文件锁，防止控制器中断后与残留worker重复运行；原始开始时间和12小时总预算不会随容器重启归零。每角色最多三次尝试，代码错误须诊断后修复，不无限盲重试。

自动导出`verified_pairs.json`、`paired_counts.csv`和`verified_counts.tex`，完整与pending分开；`plot_rebuttal_campaign.py`从已核验记录生成原始结果矩阵SVG/PNG，待完成项保持灰色，不补零。原case0与CCI进行中的case1/2可以作为明确的外部来源，ACP仅负责不重叠的固定case3–9，合并前逐项核对reset及来源。上述是执行层变化，科学worker仍绑定已验证的原core，K32/H32/NFE10/prefix4、checkpoint和任务上限不变。

ACP显示“运行中/完成”仅代表平台状态。科学验收须读取原始命令、checkpoint/config/source、完整回合、失败计数、配对身份与保存哈希。控制台创建失败或网络响应不确定时先核对实际作业，避免重复占用算力。用户其他项目的作业保持原样。
