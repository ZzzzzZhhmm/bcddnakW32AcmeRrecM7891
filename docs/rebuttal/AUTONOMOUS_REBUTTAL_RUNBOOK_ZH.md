# WARM rebuttal：持续执行与11月13日稳定版验收

内部验收时间：**2026-11-13 23:00 Asia/Shanghai**，早于作者要求的11月14日。正式讨论截止按作者邮件为11月19日19:59北京时间。每小时在本对话推进；本机和桌面应用须保持运行。该定时任务覆盖完整研究与写作流程，不是只轮询某个GPU任务。

## 恢复与并发规则

先读仓库AGENTS.md、实验台账末尾和`tmp/rebuttal/planning/current_iteration.json`。后者指向当前回复、revision、证据索引及下一步。旧聊天中的running状态不是当前运行证明。先检查真实进程、心跳、退出收据、结果完整性和身份；未知提交不重提，已有固定样本不按成绩替换。

CCI别名`slai_debug_workspace_1`；工作目录`/mnt/afs/task3_2/L202500276_lwz/projects/WARM`。实验根目录`/mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations`。当前20261109运行材料在`rebuttal_20261109_acp`。CCI每4h重启，短段工作必须有持久产物；长任务交ACP。使用`configs/nonreal/rebuttal_acp_defaults.json`确认的私有镜像与AFS；先标准，实际GPU配额不足再闲时。不要把排队写成GPU活跃。

每次提交先拉取检查上游和共享文件重叠，保护真机团队并行工作。当前36个无关文档删除不属于本任务，不暂存、不恢复。英文提交主题和分点；提交后推送并核实。执行任务只用冻结快照，训练/eval/ACP不得运行Git远端操作或因工作树脏而失败。原始数据、权重、媒体和运行输出在Git外保全；更新实验台账中的数量、未知、失败、身份、哈希、位置和可支持的论文结论。

## 三条联动主线

1. 实验：优先后果预测的候选区分、提案修复与候选撤回的直接对照；随后同信息匹配训练、非空缺失技能、额外任务干扰、效率与真机补强。原v3科学计划已冻结；计数或预算变更必须单独记录理由与新身份，不静默改样本或统计目标。
2. 回复：维护20项weakness/question的逐项映射。共性证据统一补，逐审稿人正面回答。公开草稿使用原稿事实与已核验新证据；进行中部分在内部release gate中明确保留，不能先写成已完成。
3. 修订：以33页最终投稿及其已核对的完整LaTeX工程为底稿，保留核心科学问题和贡献。额外一页主文用于最关键机制与协议解释。数字、术语、对照定义、证据范围、表图位置须与回复一致。全部PDF由完整工程编译，检查匿名性、引用、主文10页上限和可读性。

## 当前需要接续的事实与阻塞

- RB固定case0/1/2各Full/null700步，共3对，均0/0；BT原完整case0也0/0。它们是新固定seed cohort，不能替代原稿任务均值，更不能声称新增收益。
- 原始投稿表12–18的结果按作者说明是真实结果，保留为原稿报告证据；旧synthetic注释是作者确认的残留。新复现实测与原稿结果使用不同证据ID，不伪装为原始日志恢复。
- 双遍fresh-scene native replay v2已exit0：RB源case0的全部700命令、六个固定query、物理状态与独立qpos/qvel跨两遍一致。随后Git `fabb74b`的fresh-scene分支入口已在同一source0/q8通过真实A-A-B-A：四个父状态精确一致，三次refined端点一致，raw端点在真实物理bytes上不同，共128条完整分支命令。此单query的raw/refined独立局部标签均0，candidate-null尚未执行，不能称为修复优势。下一步扩展固定source/query及K32后果读取，并从同一原始Full事实上下文生成null。不要继续使用会重编码事实父状态的旧pack/unpack入口获取正式机制数值。
- A-A-B-A是工程资格检查；正式cohort遇到raw/refined完全相同、未到达query、执行拒绝或几何歧义均须保留原身份，不因资格或成败选择替换query。新query的父状态与重复端点应单独核验，不能将q8通过推广为全部阶段已经通过。
- 首个完整fresh K32 source0/q8：28完整H32、4无效未知；predicted centered MSE0.077411高于query-constant0.052806，centered explained variance为-0.465948。这是一个真实待解决的直接证据问题，不能称新增预测优势。四个已生成父状态DINO的dtype/shape/bytes哈希已核对原在线telemetry全部相同；继续固定cohort、查动作/时间语义及原投稿模型对应关系，不按此query调参、换样或重缩放truth。
- 同一Full事实null动作封存现用`controller_bc13308/code`的v5，64项原生检查（含真实frozen/slotted BoundOnlineStep）和6个真实generation/replan工件契约preflight通过；查`RB_SOURCE0_NULLSEAL_V5_LAUNCH.json`及worker（PID2177770，19:31:11 UTC准入，5400秒上限）。v1布局失败、v2 NPY哈希契约停止、v3计时hash阻塞、v4 slotted对象兼容失败保留。实际v3/v4 q0完整H32与原source精确匹配；比较原绑定只在独立视图代入原5项实测耗时，其他绑定内容、实际capability和输入保持。NPY文件SHA与`manifest.sha256_array`不同。不要重复未知的v5，不将preflight当作null效果。source0 K32于19:27已知wall-time终止后19:31安全续跑，原12h窗口不变；source1/2首段约19:44到期，核实退出与lease后按同样冻结入口续跑。
- 官方`joint_action.vector`臂部是drive target，夹爪项也是controller API值。保持官方policy输入；物理运动证据使用独立articulation qpos/qvel和图像。旧命名native_qpos不自动等于实际关节传感。
- Full/control100→200训练及逐字节可重建state档案均已完成，不能再次启动同一段。续训operator v2误把运行中的部分summary当成失败，v3修复后原截止窗口保留，control另有预算修订。先核实两份200收据、本机CAS传输和空间准入，再续向固定选择点3000；100/200工程段不是TEST结论。
- 当前conditioned-prior是适配对照，不是官方ReCAP/WarmPrior复现。其信息输入忠实性、训练对齐和最终性能须明确，不以概念区别代替受控对照。两组200的401项本机压缩CAS备份已全部核验；其余固定checkpoint3000训练仍需空间/GPU准入。若ACP e持续等待，检查其他允许的闲时GPU资源是否适合独立匹配训练，保留两角色相同预算、硬件/配置修订记录及case3–9原所有权，不能重复提交该cohort。
- 当前AFS剩余容量约87GiB（2026-11-09 18:22 UTC观察，仅为当时值）。新训练段前重新检查空间与完整state保全，避免100-update重复大状态无界累积；分层须先验证完整重建和持久副本，不删除唯一checkpoint、数据或未知产物。CCI重启窗口仍应以实际控制台/容器信息核实，不从宿主PID1或旧hostname推算。
- ACP `warm-rb-resume-20261109-e`已创建且拥有固定case3–9；核实其真实状态及AFS心跳后再分配，不能与CCI重复。标准资源已明确成员GPU配额为0；闲时仍可能抢占。
- 作者最新更新（北京时间11月10日）：暂时找不到真机逐次结果与录像，先推进非真机工作。因此将新增真机实验与旧录像审计移出当前关键路径，不重复索取素材；原有协议`REAL_ROBOT_REBUTTAL_PROTOCOL_ZH.tex`保留为延期方案。论文原真机结果与已核实协议继续作为原稿报告证据，历史依赖与检索难度分别解释，原四任务20次不等于每个桌面外观20次。

## 日期与验收

- 11月10日：三位审稿人20项映射、已有证据可答首版、v3完整PDF与变更单；修复同状态分支资格，启动最关键直接实验。
- 11月11–12日：优先取得关键机制与公平对照，及时写入回复及正文；非空缺失技能、额外仿真任务干扰和位移泛化按固定协议推进。真机新增取证延期，不将其作为稳定版前置条件。
- 11月13日：核验全部重要论述的证据、逐项回复覆盖、命题假设、图表/数字/术语一致性和完整PDF。23:00前生成稳定候选包；尚未关闭的决定性问题须明确列出，不能用格式完工冒充实证完工。
- 11月14日后：保留稳定基线，按新增实测和真实讨论反馈有界更新。未经作者最终指示不发布OpenReview评论、不上传正式投稿、不发他人消息。

每轮至少推进一项可独立完成的实质工作。等待GPU时写回复、修订、核对原证据、整理图表或开发下一实验。每晚做跨材料一致性检查。完整cohort/同状态直接对照未完成时，保持内部release gate为blocked，不编造结果，也不轻率推翻原论文的核心贡献。

## 最新恢复锚点（2026-11-09 21:40 UTC）

- source0/q8物理null与post-null Full已exit0，各32步，共64分支命令，另64共同前缀命令；重复Full端点精确。Raw/Full/null局部标签全部0，单query资格不等于修复收益。80文件档案`FIRST_PHYSICAL_NULL_Q8_V1_ARCHIVE.json`已本机核验。
- v5原700命令事实重现和六个同历史null动作封存已exit0，92文件完整档案已本机核验。无需再启动这一source的同一封存任务。
- 旧CCI三路进程在连接重置后消失且lease空，旧exit保持未知。21:19以同Eb source恢复；source2后来在setup发生一次明确exit1的planner RPC连接重置，原件保留后21:27重试。当前PIDs2198110/2198111/2199487，21:33均运行，q48完整候选14/8/7；持续12h起点不变，旧partial保持未知。按`FRESH_POOL_RESET_RECOVERY_V1.json`、`FRESH_POOL_SOURCE2_RPC_RETRY_V1.json`和新runner真实收据核查，不找旧PID，不重复未知任务。
- ACP e真实发生DINO runtime契约失败，0正式完整role，已停止。新f UID5be8c6689d44435a8ef2c0b9497ec820已经创建，持有原case3–9、campaign/plan d8512a23…和原2026-11-10 08:31:04 UTC截止；不可再建重复接续。使用`controller_6b803c2/code`及encoder资格计划0df7b0f…，实际runtime必须保存、只允许platform单独记录，四份原事实DINO须精确bytes一致。最终CCI四份检查已exit0、16项原生通过；ACP实际尚0GPU等待，不能声称其兼容资格已经通过。
- Full200→500已完成父状态/冻结代码准入；标准a为真实WAIT_QUOTA后已停止，闲时b UID9c6000a4ce734a7bac717b2053197a1e持有唯一该段，21:34 cluster01仍0GPU等待，即使UI写启动中。使用`controller_d3abf15/code`和plan7d359a7d…，不要重复创建。当前control200经审计仅为动作先验消融，仍缺真正消费检索视觉/效应的同信息最终对照；新结构须从共同初始化独立训练，不静默继承control200预算。
- 已更新三位回复中信息匹配的定义、15个comment block与20项矩阵。PDF/完整LaTeX源包本轮不变，保留35页/10页主文验证；没有把单query无收益或中间训练写成新增机制优势。
- writing research-patch3已在AFS核验44项、SHA517ab30d…；当前live状态以`tmp/rebuttal/planning/current_iteration.json`为准。下一步还要直接核查模型/数据/动作时间语义，解决前三q8不及query-constant的实证问题，不调truth尺度或按结果换样。
