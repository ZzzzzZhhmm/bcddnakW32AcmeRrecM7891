# rkWY：依据 Bbqv 示例的逐问题回复优化

本轮只优化rkWY工作回复、补充条件性理论片段并登记论文同步；不恢复实验/自动任务，不修改Bbqv/Hh4E正文、冻结首轮包或v40 PDF，不发布评论。

## 工作材料与恢复入口

工作目录：`tmp/rebuttal/responses/20261114_rkWY_question_aligned/`。

- `Reviewer_rkWY.md`：1,967词，W1/W2/Q1/Q2四条中性问题标题及单句粗体结论、六张独立可读表；SHA256 `3f68612790f2110ab857633275cd7d2bcf07cf8419d9b4c69691a25af9c925af`。
- `AUTHOR_CHECKS_ZH.md`：位移解释、统计分母、原件保留、条件性推论与实际未关闭请求，单独面向作者。
- `QUESTION_STRATEGY_AND_REVISION_MAP.json`：四项原文、标题/首句、二十个既有LaTeX定位，以及新增理论片段和同步动作。
- `NUMERIC_RECONCILIATION_V1.json`、`CONTENT_VERIFICATION_V1.json`：五张定量表21行与来源比对，另一个适配角色表单独核对；保持来源链可追溯。
- `CONDITIONAL_EQUIVALENCE_REVIEW_V1.json`：推论条件、论证步骤和不包含的经验主张。
- `archive_inputs/`：原review、原首轮rkWY、既有论文来源及新理论片段的字节副本和SHA；不是新GPU轨迹。

当前指针为 `tmp/rebuttal/planning/writing_current_rkWY.json`。`working_response_index_20261114.json`现指向三份优化工作回复，全部20项原意见绑定、16组回复、27表；跨文本覆盖及数值来源检查见 `CROSS_RESPONSE_WORKING_CONSISTENCY_20261114.json`。这不是新官方提交包或论文全量验收。

## 针对性论证调整

1. W1解释task identity与episode progress的不同作用，以真实M(n)提升和原任务分解支持核心贡献；明确specialist与已测known-task范围，不把原结果写成统一未见任务策略。
2. W2用原20-trial结果和R2P/SOR历史依赖需求解释本域内考察维度，明示tabletop范围，不把任务复杂性升级为跨域/跨具身证据。
3. Q1区分control-start对齐、视觉条件残差、后果比较与Action DiT细化；保持兼容start-reference条件，不能将joint/TCP同维度当同语义。残差界不产生厘米位移保证；完整事件干预回答经验相关性，demo/event/DEV计数单列。
4. Q2先给双路径严格zero-gate撤回，再给原报告selected-proposal门控中位数和三种子常量控制。空池、混合误导候选、整个bank缺技能三种情况保持区分；中位数0.00不写成所有无用候选都被关闭。

## 条件性轨迹推论

新增可移入附录的源片段 `CANDIDATE_NULL_EPISODE_EQUIVALENCE_20261114.tex`。在同一参数、初始状态、每步严格zero gate、matched token/mask/native/solver/execution/stopping接口、耦合随机性，以及candidate-independent事实状态更新条件下，由现有单步withdrawal性质归纳得到与candidate-null版本的路径等价。时序和任何wall-clock停止条件也必须匹配。

该片段是条件性分析，不是对自然skill-absence检测、现实运行时所有条件或独立训练retrieval-free baseline的效能认证；当前正文/PDF未集成或重新编译该片段。论文同步队列保留前两位并增加rkWY四组，按作者授权以最终修订视角撰写，再逐项落实。

原namespace不利/unknown完整文本与v40附录继续保留，未改成理想低gate值，也不将旧报告门控中位数当作同一固定态的补跑替换。没有新SR、构造值或自动任务恢复；`all_scientific_gaps_closed=false`。
