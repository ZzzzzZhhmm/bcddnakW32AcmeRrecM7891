# 固定最终模型的非空family-holdout原生开发试验

本试验接续原P0方案登记的开发种子3407/3408，使用新的固定final3000 Full权重b0669047…，回答同一模型在等规模经验干预下的自然门值和Full/显式候选撤回行为。它是独立开发试验，不替代原正式missing-skill/2pp NI方案，也不进入RB/BT固定100对Full/Pooled主分析。

## 冻结范围

- RB原生700步、H32/prefix4/NFE10/K32/recent20/summary8，root3407。两角色同一Full工厂、schema及checkpoint，保留各自真实因果history；candidate-null只显式撤回候选的source和conditioning影响。自然Full不改变门阈值或机制。
- 两个已生成、不可变的原bank搜索域：clean_size_matched和target_skill_family_holdout，各19,540个原事件；完整bank89,680及原EventID保持。family-holdout移除六类block任务，不意味着所有剩余事件都不适用。
- 开发物理种子严格为3407、3408，不选择正式340800000…种子或按表现替换。两个独立资源段各最多3h/单H100，总至多6GPUh；每段仅一个固定种子、两条件×两角色。Stage A先3407，Stage B预登记3408，不因Stage A的得分选择是否运行。若固定专家准入拒绝，保留拒绝，无替换。
- Stage A按clean Full/null、holdout Full/null；Stage B条件和角色顺序反转。第一准入时间包含reset资格、模型加载、排队/失败，窗口不得重置。每角色上限2400s，未完成或存储/运行异常保持unknown，不补0或称原生完成。
- 沿用冻结H861004fa…的实际native执行函数与a97e308…私有模拟器，逐命令记录API、真实物理关节和TOPP观察；新的pilot schema与正式collector验证器分离，原正式validator不改变。

## 原件及判断边界

每个角色保存完整生成H32、逐命令物理轨迹、初始/最终观测、实际类型/权重/合同、每次query的Gaussian及输入hash、自然门值和有效source gate。候选撤回须实际g/source_gate为0，source精确为同一Gaussian；配对只要求初始状态/输入/噪声一致，后续各自因果history允许分化。

自然Full在预定q0/q8保存全部候选动作和元数据供后续独立完整池覆盖测量；未到达的q8单列，不换查询。当前family移除和正门值均不能直接变成“技能真的缺失”或false activation标签。独立coverage未知时保持null；单个/两个开发seed的相同成败也不能建立等效或NI，不给出显著性或总体成功率优势。

既有三个旧shared-stage初始gate诊断保留。本试验使用新的固定final3000 checkpoint，不覆盖旧结果，不因试验结果更换正式权重、模型、阈值、样本或指标。只有真实完成且原件全核验的开发观察才可进入后续材料，必须标明开发范围。

## 实现与保护

`src/fastwam/research/missing_skill_pilot.py`约束固定开发身份、窗口、请求顺序及配对；`scripts/run_rebuttal_missing_skill_pilot.py`复用冻结模型/原生执行，显式记录研究干预，避免把null误记为production mode未改。19项CPU协议检查已在CCI通过，尚不是policy或模拟成功。

第一阶段计划使用当前闲置的rb-40已认证私有runtime和独立cache，必须另登记外部保留标记；在已知退出、全部原件封存及显式释放前，正式rb-40不得准入。CCI-local flock不能独自证明跨节点排他；正式准备器需读取该保留标记。其他独立worker不改。

每角色有界原件和可用空间检查只用于工程保护；若触发则保留unknown和原窗口。当前AFS空间持续变化，准入前核对实际余量；不删除任何原始数据、checkpoint、CAS或其他团队成果。代码先在CCI实现验证，再按英文commit/push并离线同步；运行程序不访问Git远端。所有运行根、冻结计划/快照、心跳和退出/归档收据保存在Git外。具体是否已启动以最新current_iteration及真实进程为准。
