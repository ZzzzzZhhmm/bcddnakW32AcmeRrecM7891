# 原 BT 固定状态后果与修复证据

本协议在新增独立 branch truth 前声明。原模型为作者确认的 `battery_try-s3407-v1`，固定 checkpoint12000（SHA27c137d6…）及原 f77 source ff32342d…；原稿 DEV 选点仍未知，不能据此替换原稿准确率。事实来源是已冻结 cohort case0/physical seed340800000、root3407 的完整 BT 官方回合，原计划9d479fa7…、roster612cb5d1…和全50前注册b1be58f9…保持。全50原BT性能评估未由本协议自动启动。

独立 schema `warm.original-BT-fixed-cohort-fresh-pool.v1` 与旧 BRT14k/seed0 的 schema 分开。主 query 8/48/96 来自事前注册，不依据结果增加、替换或筛选。每个状态的 selected rank 来自该次原始模型输出；不能借用 BRT 的28/0/10。22:45:53UTC CPU只读核验发现 q8已事前封存、selected rank4、32个有效候选、gate0.8984375；这不是适用性、预测准确率、修复或成功率结果。

完整事实回合须达到原生1000步或官方成功终止，并且初态、首次H32、前四条API/实测物理与此前资格精确对应。每条 trace 的序号和原生命令计数连续。未到达主 query 或其原执行前缀不完整，明确记为 unavailable，不换点。事实前缀若包含 TOPP 拒绝，须另有两遍完整源绑定审计；原始错误不能直接作为忽略 guard 的理由。

每个 branch 都新建独立 CPU PhysX 场景，按同一已录事实命令重放到该状态，核对所有 API 前后值、实测关节 qpos/qvel、每四步原三策略RGB、任务指令和最终完整 parent/RNG；不用物理 unpack。完整源的 actor/task/RGB/物理全部隐藏状态不因这些检查被默认为完备捕获。原 source 的候选、预测、原H32生成、RNG和future-truth seal 先于新物理执行，不能用评测真值重新拟合预测投影。

顺序先做 Full–Full–原选 adapted proposal–Full 的完整 H32 AABA，重复 Full 端点/原 DINO 特征须精确相同，raw要实测不同物理端点。资格失败保留原件，不能缩H32、换rank或放宽条件。通过后固定执行全部原有效候选，最后重复 Full。`--qualify-only`与后续pool接续共用同一冻结计划和原绝对截止，已封存分支只读复用，不重跑。单状态拟使用至多2小时的一张H100；任何更长运行交ACP，先标准，实际配额不足才闲时。

后果真值沿用原 f77 `_mean_effect`：四个原冻结DINO delta token的均匀均值，不混用新shared模型的加权/前态投影。只对完整H32端点在同一支持集报告 centered MSE及constant/historical参照；无目标效应观测时不填写不可识别的absolute constant误差。TOPP/无效执行的后果及标签均保持unknown；提前成功与完整H32分开记录。

独立物理标签沿用已声明的 `warm.rmbench.physical-next-interaction.v1`，只衡量下一物理交互是否完成：抓取抬升、槽位放置、既有里程碑退化、物体跌落，或两电池都在槽内时到达此前事实历史未测试的极性组合。事实前缀逐步收集已有组合；隐藏正确组合不参与标签判定，更不给模型。几何边界2mm内保持unknown。读取标签前后检验 parent/endpoint 未被改动；标签计算不读取模型分数、gate、候选排名或role。

所选 raw 与 Full 对比在AABA完成后即单独保存：`raw=0,Full=1`为该状态的repair witness，`raw=1,Full=0`为harm，任一unknown则配对结论unknown。三个query都来自同一回合；32个候选不是32个独立任务试验，不据此估总体SR、独立训练不确定性或泛化收益。新增标签也不替换原稿19.0%所在cohort。

此入口尚未提供同状态强制零影响与Full的因果撤回比较。原 f77没有新shared代际的 `configure_research_probe` 接口，不能直接套用新接口或绕过 consumed BoundOnlineStep。后续只能通过原模型支持的控制和真实工厂绑定构造、精确核验原Full/同上下文随机性后另做撤回；保持该缺口显式开放。

22:44:35UTC实现相关54项检查通过（pytest11.63秒/总15.96秒），stdout fed2a9ad…；原BRT seal/result字段保留。检查属于实现验证，0新增独立物理branch。全部原件、实际运行收据和结果保存在Git外，数值到达后立即更新实验台账和回复—论文映射。

## 后续执行与资源修订（保留以上前注册原文）

23:00:36UTC在任何新独立branch truth前，三个主query的原selected rank冻结为4/3/20，全部32有效候选固定；登记ORIGINAL_BT_PRIMARY_RANKS_PREBRANCH_V1。完整native1000事实源已完成且任务未成功，2020原件a1a42358…及本机逐项核验保全。

q8已于23:14:29完成四分支H32 AABA；三个Full终点及原DINO效应精确相同，raw不同。原提案电池1终点z0.0278908424m，Full0.7598833442m；两者交互完成标签都是0，不计完成型repair。58原件b7e7359c…及本机独立高度/状态/像素证明完整。q8全部候选池于23:16:30接续，仍沿用原01:04:40.464620UTC截止；不能因尚未完成延长、换rank或重采。

23:55:30UTC针对**尚未首次准入**的q48与q96，根据q8已完成分支的attempt时间间隔、各查询固定前缀长度192/384和H32/36分支开销，声明独立资源修订BT_Q48_Q96_PREADMISSION_RESOURCE_AMENDMENT_V1：q48最多21600秒，q96最多39600秒，总上限连同已准入q8为19GPUh、最多1H100。原6GPUh计划作为历史保留；原q8的7200秒和绝对截止不变，其它旧训练/实验窗口均不变。预算计算不读取结果或效应预测，不更改样本、rank、模型、H32或主要指标。长任务在ACP运行；新计划首次准入时固定绝对截止，排队计入窗口，标准资源优先，实际GPU quota拒绝后才闲时。此资源修订不表示任务已提交或任何新实验成功。

## q96首次实际执行（保留原科学设计）

Nov11 20:16:01UTC按上述原修订首次冻结q96窗口，固定截止Nov12 07:16:01UTC，之后的排队与任何已知故障恢复均计入此窗口。q48已知终态与全量归档核销后才准入。20:23:14标准资源创建成功，唯一UID536a352f050b4142890aa99fb88aa23b；20:24:19实际科学runner启动，H10080GB及完整encoder runtime与原合同一致。

plan8e3ddd84…、原controllerf7cf0583…和原checkpoint27c137d6…绑定q96/rank20、全部32原候选、完整H32。该阶段独立于shared3000的100对控制研究，不替换其权重或样本，也不借其资源时钟延长期限。新root `original_BT_cohort_case0_q96_fresh_pool_v1`，不得与旧BRT q96混用。至20:29仍为第一Full的父状态重建，无完整端点；AABA、全池效应、repair/harm及unknown须逐一独立验收后才能用于回复。
