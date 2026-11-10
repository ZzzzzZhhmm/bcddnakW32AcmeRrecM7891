# 同信息池化对照：预训练冻结协议

**11月10日15:03 UTC状态更新：** 以下首次资格段落保留其历史时点。独立all-K对照现已完成1–100连续有限更新、真实typed checkpoint及完整state，并通过逐字节可重建归档；它已不处于0-update阶段。父100权重SHA cbdfcef2432f2e2b99ddfc22a95dc16450df43979c094c85e3f7787faecfe5ec，raw TAR SHA f132c9a5b5f917e9eb1a2a0c093811d0c1634c8b1211dadd43631f999ae4bb54。新100→3000计划fe595b17…使用原f9/source48ce6ba8…、完整父状态、共同配方与同master7526a475…截止；标准ACP c852e8e2e91f4d9ba9d42cfa0682bbb2已通过父状态核验并实际进入初始化/恢复。此时不能称已产生101+更新，须读真实metrics。此前f31fd1c5…因收据路径类型错误在创建训练输出前失败，已核对ACP离线原始堆栈；该失败与修复记录保留。原Full标准任务95e49547…持续运行，禁止重复提交。最新进度读canonical current_iteration与实验台账末尾。

本对照回答BB-W2、HH-W1、HH-W3：在共同骨干、回合历史和检索信息下，一个简单的记忆条件策略是否已经足够。它是新研究对照，不是ReCAP、WarmPrior或MemoryWAM的官方复现。原Full/control200原件与身份保留，control200仍是直接动作先验消融；新结构从共同released base独立初始化，已训练步数从0计。

## 输入与机制

| 项目 | 原Full | 旧conditioned-prior | 新pooled-memory-prior |
| --- | --- | --- | --- |
| 当前视觉、机器人状态、指令和因果回合历史 | 共同接口 | 共同接口 | 共同接口 |
| 检索动作 | K个动作供适配/选择 | 所选动作直接打包 | K个完整动作均进入池化 |
| 检索视觉前态和历史效应 | 使用 | 未用于conditioning | K个记录均进入池化 |
| 检索context、时间和阶段 | 使用 | 主要用于共同检索/规则 | 显式进入池化；当前query/state参与查询 |
| 提供给Action DiT的条件token | 8 gist + 4 action-context | 8 history-gist + 4 action | 8 history-gist + 4 pooled-memory |
| 动作均值、后果和效用机制 | 学习适配、预测、排序与门控 | 确定性起态对齐、coarse prior、常量0.25 | 确定性起态对齐、coarse prior、常量0.25 |

新对照将每个有效事件的完整H32×14动作、4×768视觉前态与历史效应、777维context、8维时间、4维阶段特征和4个空间mask位拼接，经一次线性投影/GELU/LayerNorm得到一个事件token。4个查询结合共同history gist均值、当前query和proprio，经单层cross-attention读取全部K个事件，输出4个memory token。该通用池化会学习内容权重；它不输出候选后果、效用分数、动作残差或置信度，也不据此选择动作source。

无效候选和无效空间token在投影前屏蔽；无有效候选时仅开放零key以防全mask softmax产生NaN，最终memory token保持精确零。动作先验采用共同确定性起态对齐和首个有效coarse候选，保留共同训练时的强制候选/拒绝及线程有效性规则。信息完整性不等于参数数目完全一致；须报告实际参数量和实测成本，不声称等参数或同延迟。

## 不允许改变的训练与选择条件

- common released base、seed3407、训练数据/bank/query约束、样本顺序、corruption schedule和优化器数学配方不变；结构独立schema禁止把旧control200或Full state静默载入新对照。
- 单卡micro-batch8、accumulation16、有效batch128；原30,000-update scheduler与固定3,000-update最终选择点不变。新增池化参数在独立RNG作用域中初始化，不推进共同训练RNG流。
- action/video-flow、bridge/gist训练项保留；删除机制对应的retrieval/effect/gate/adaptation损失权重为0。相同输入访问不要求保留被删除机制的损失。
- H32、执行prefix4、NFE10、K32及原固定评测cohort保持；不得按结果另选checkpoint或样本。先获得两组完整最终checkpoint再作匹配性能结论。
- 长训练交ACP。原Full200→500的b及后续失败/关闭记录保留；当前两角色3000训练以MATCHED_FINAL3000_RESOURCE_PHASE_20261110_ZH.md及canonical状态为准。新尝试需独立空间与GPU准入，不得与活动任务争用同一输出或删除唯一checkpoint。工程资格、训练更新数、最终TEST和收益证据分别登记。

## 资格与尚未关闭的门槛

原生45项检查已通过：每个未被选为coarse prior的有效候选的动作、视觉前态、历史效应、context、时间和阶段均可影响池化结果并得到梯度；padding与空候选保持有限/零；未来teacher变化不改变conditioning；12-token接口、共同模块初始化与RNG、独立checkpoint schema及原Full路径检查通过。首轮曾因新增schema字段未从父类校验视图移除而失败，原件保留，修复后复验通过。

真实训练样本index32与已封存在线query的BF16模块前向/反向已完成，runner明确exit0；两份实际输入均为32个有效候选，输出均为1×32×4×512，所有参数梯度有限。池化模块有5,767,168个参数。该预检中的history gist明确使用零占位，在线proprio也明确为接口占位；它不能称为完整采样器资格或匹配训练。历史gist的因果路由由独立模型检查覆盖。原始v1预检因共同dataset尝试把统计副本写入未配置的工作目录而失败，未执行优化器或policy；修复只给预检配置Git外工作目录，未改变归一化统计或原科学模型。

原始检查/失败日志、实际记录资格、源码和冻结配置共35项已双端保全：`same_information_control_and_time_audit_v1.tar.gz`，SHA `c7dddd91d7239cc84fdf5a1e5b699570d1dbdad87a556420f73cda37cba94596`。首次前向和并发运行时的单次模块耗时不作为部署延迟benchmark。首次资格时冻结seed3407配置SHA `d5a7bd8c76aee178991e593439124aec72077e1734eb831cf2dfda55883846c3`，当时训练更新数为0、尚未提交新训练任务；当前实际训练状态见文件头与canonical记录。

模块资格本身不能关闭审稿人的性能质疑。其后已经按冻结源码和配置完成独立100更新及持久state归档；固定3000训练及最终匹配评测仍未完成，不能提前写性能差异。

## 最终在线比较的独立准入门槛

完整训练及checkpoint schema检查不等于在线策略装载或采样器资格。等待训练时应先用已保全pool100作工程资格，不用于挑选TEST checkpoint：通过明确的研究配置调用create_pooled_memory_prior，保留训练时retrospection配置（含移除机制的四项loss=0）、严格typed checkpoint装载、源合同、原银行与normalizer/encoder验证。不能把pool权重交给Full构造器，也不能把Full模型临时关模块称作已训练all-K对照。

必须检验实际原生采样路径的H32×14有限输出、12-token条件接口、真实因果历史与全部K事件输入；采用现有固定DEV记录核对相机/状态/指令、检索bank rows、共同采样seed及当前模型输入身份。先完成这一资格，再按既有冻结任务/seed/绝对预算执行最终3000两角色匹配评测。工程资格不执行TEST挑点，不用单次前向速度代替部署profile；只有双方完整3000、完整轨迹及正式终态/未知样本核销才能计算性能差异。

正式回合的native horizon由实际任务配置记录。原700-command有界诊断仅解释其原固定scope，不自动等于官方回合上限，不能将其0成功推广为完整原生任务失败；新比较须明确终止条件与双方共同的评测协议。此前负结果和未知均继续保留。

最终提交`f9f1090`的Linux Git快照source为`48ce6ba809788fc1c2c55b5303cc9617e2852050dd2bd8e104dc8192ee98e358`。预检快照其余464个文件仅CRLF/LF不同，全部源码经行尾规范化后相同，六个本轮实现/检查文件原始bytes相同；最终Git快照又独立通过45项原生检查（19.72秒）。原预检source5347088c…保留原身份，不混写成同一字节源码。

已确认当前author recipe为shared-stage，而原稿RMBench使用逐任务specialist。新计划是共同shared阶段的研究对照，不称原稿逐任务协议的完全匹配复现，详见`CHECKPOINT_SCOPE_AUDIT_20261110_ZH.md`。旧Full b与固定v8任务的原窗口、模型身份和终态均保留；当前新phase及实际活动任务以文件头/canonical为准，不按结果改模型。
