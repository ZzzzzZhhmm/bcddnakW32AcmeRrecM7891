# 原BRT后果执行接续注意（2026-11-10）

这是一份实现入口核对，不是候选后果或修复结果。原模型f77/source ff32342d…与参考工程H=controller_eb70b41/source7a…分开加载；不要将v8结构/目标迁入原模型。原Full事实来源plan79a70972…固定seed0、600条采集命令、probe8/48/96。首观测、首生成和首四物理命令已实际精确匹配资格。这个600是本补充来源的冻结采集上限，不是经恢复确认的原稿DEV选点或原benchmark SR协议；当前原生console显示step_lim3500，不能将600处未成功标为整个原benchmark失败或据此改动固定采集上限。

## Reader与分支入口

- 原f77 archive实际存在experiments/rmbench/warm_policy/policy_core.py；当前H的FrozenFactualEffectReader使用这个官方观测映射。原f77没有research/factual_effect_reader.py，所以需要明确从H引入research辅助模块，同时保证fastwam.memory、models和policy_core仍来自已验证的原C。不要因ImportError把全局模型源码换成当前v8。
- Reader的bind_reader_files检查encoder/camera/M1/stats原hash；from_source_runtime还核对实际DINO runtime和checkpoint tree。原BRT runtime_args、warm_online_contract_path已在native_static_preflight内保存；实际DINO runtime2f3e5d29…与合同相同，无需数值合同豁免。当前文件入口src/fastwam/research/factual_effect_reader.py，不是猜测的其他文件名。
- 原_mean_effect(delta_tokens)与frozen M1/DINO负责独立物理端点的768维descriptor；不要用当前query future、历史效果或事后拟合作为未执行候选的真值。主centered MSE的query-constant在候选内中心化后为零；未保存的query requirement不能凭空填值。历史Ei须按原telemetry的bank row/event ID从精确原bank读取。
- 当前event hook封存适配动作、predicted effects、valid mask；原model_output含Full动作与telemetry。infer_kwargs的递归CPU快照对未支持的非Tensor类型只记录type，不能直接宣称完整可反序列化。补抓r_t/同上下文null时需从原bank、事实输入、history/RNG重建，并核验原input hash和重复Full输出。
- native_replay.native_parent/OfficialQposGuard/fresh_pool可作为fresh-scene分支工程入口，q8已通过精确A-A-B-A，q48/q96仍须各自认证。RMBenchQposBackend的pack/unpack白名单只有RB/BT，不能直接移除白名单冒充BRT已认证。用真正fresh reset及原实际命令重放，核查源观测/命令、所有fresh父状态/RNG/DINO、重复Full端点和真实关节状态；不改task或做物理unpack。
- rmbench_outcomes.observe_task_progress/label_next_interaction当前只定义RB/BT，BRT会拒绝。未建立独立BRT适用性rubric前，标签必须unknown；不能把该异常静默当0或把DINO预测误差当成功/修复标签。候选后果的直接MSE证据与任务适用性证据分开。

第一步固定q8、模型原选中raw与Full H32的A-A-B-A。Raw rank由已封存模型选择决定，不能看分支结果后调换。合格后按原valid mask/rank顺序执行全部K32，再固定q48/96；任何未到达/中断/无效保持未知。branch source在输出读取前封存，实际新资源段单独冻结，不重置原RB、Full500、one-replan或600采集窗口。


## 11月10日11:26 UTC补充核对与新执行协议

静态预检发现原f77模型不存在当前shared-stage的_spatial_effect_summary；原文件228行_mean_effect(tokens)直接对四token取均值，candidate observed_effect与demonstration target都走此函数。原四audited run的retrospection_model.py字节一致，因此后续原模型GT/历史参照必须使用原均匀delta均值，不能混用0.05 pre-state加变化能量权重的另一版本描述。第一次静态import失败保留，改用原函数后，namespace、原processor/M1/stats文件hash及DINO runtime2f3e5d29…均通过；尚未加载teacher或执行branch。

新增collector固定原BRT/sourcef77/step14000/seed0/q8/frame32，不按结果换点。先顺序Full、Full、model-selected raw rank28、Full，要求四父状态/RNG/原DINO特征精确、三个Full物理端点/效果精确、Raw物理端点确实不同；每次从新scene重放真实32命令，无物理unpack，不动现有BT/RB白名单。限定端点H32，source Full前四原生API向量及下一replan图像也必须匹配。额外RNG协议明确为每个物理parent恢复封存的source pre-forward RNG，不宣称源forward后的未保存RNG已经恢复。

A-A-B-A通过后，按原valid rank顺序执行其余31候选，复用已完整封存的rank28为同一候选而非重复样本，末尾再Full anchor。共36 branches、32 distinct candidates。所有动作、预测和历史Ei在新truth前封存；历史Ei按原event ID/bank row从完整hash核实的原bank取得。GT是原frozen DINO变化的_mean_effect。centered比较使用同一完整H32支持；任意query-constant中心化后为0，但未保存实际r_t，不报告伪造的query-only absolute MSE。BRT applicability继续unknown，early/invalid/incomplete不补0、不补效果。

新单卡资源段至多2小时，先在CCI实现/CLI检查并同步提交，再从唯一不可变文件运行；独立新plan/绝对截止，不重置先前窗口。每branch保存前后物理关节坐标与速度、native parent、DINO端点和独立终态收据；重复Full不增加独立样本数。进程失联先核销原runner/lease；已sealed终态分支只读复用，partial/失败attempt保留，续跑须原固定输入和原绝对截止。


## 12:51 UTC原q8终态与后继

原q8完整29/32、3 TOPP unknown，预测centered0.0525203/constant0.0181195/historical0.0524970，不支持该状态预测优势；原件ec533960…全保留。不能用已有AABA资格替代科学结果，也不能因不利换掉后续预封存q48/q96。Generic入口严格使用三个原probe及rank28/0/10，新增qualify-only允许CCI先做四个native分支，长pool由ACP在同plan/seal及绝对截止下接续。每个新query父状态须独立重放验证，语法/CLI通过不等于新AABA或收益。原q8文件/计划/截止/模型不变，真机延期。


## q48已审计的事实失败与后续执行边界

新两遍审计确认：source192尝试可精确重现，index101的原生TOPP拒绝也确定性复现，全部源API/每4步RGB和跨遍实际物理一致。只对事实历史使用独立hash清单允许复现这一已记录失败；不移除/重试该命令，不将它标为有效。Collector读取两份完整pass、源trace/query/frame及final native parent，任何新异常或缺失的预期异常立即失败；H32反事实候选仍用原严格guard，invalid/early保持unknown。无清单的q8路径不变。新q48修订保留原失败、同source/rank0和21:01:03UTC截止，ACP必须先完成原AABA检查再全K32。该工程修复不提供预测或修复优势。
