# Hh4E：依照 Bbqv 模板的逐问题回复优化

本轮仅优化 Hh4E 回复并登记后续论文同步；不恢复实验或自动任务、不发布评论、不修改冻结首轮包、Bbqv/rkWY 正文或 v40 PDF。

## 可恢复材料

工作目录 `tmp/rebuttal/responses/20261114_Hh4E_question_aligned/`：

- `Reviewer_Hh4E.md`：完整英文正文，2,585 词、六条问题标题、十一张独立可读表；SHA256 `649ff459874295602735917c9d12fa0c6d186267b58585f9c245d91e2cdb4527`。
- `AUTHOR_CHECKS_ZH.md`：作者核对、范围与真实尚未关闭的实证请求，不混入正文。
- `QUESTION_STRATEGY_AND_REVISION_MAP.json`：五条 weakness 与 omnibus Q1 原文绑定、标题/粗体首句、回应策略、十八个既有 LaTeX 定位及同步修改任务。
- `CONTENT_VERIFICATION_V1.json`、`NUMERIC_RECONCILIATION_V1.json`：格式、引用位置、源码与数值核验。
- `archive_inputs/`：原 review、旧首轮 Hh4E、既有论文源文件、冻结原模型源码与历史64项审计摘录；manifest 保留来源/size/SHA。这些不是新 GPU 轨迹或新性能结果。

当前指针为 `tmp/rebuttal/planning/writing_current_Hh4E.json`；`working_response_index_20261114.json` 索引三位工作回复与20项原意见。`RESPONSE_TO_PAPER_SYNC_QUEUE.json` 保留 Bbqv 六组并增加 Hh4E 六组编辑任务，采用作者许可的完成时写作，后续对照回复落实论文。

## 对 Hh4E 原始担忧的针对性改进

1. W1 将 MemoryWAM 的 RMBench 结果与内部 WARM 的 LIBERO 消融明确分开，用 M(n) 与共同骨干控制解释额外经验的价值，不把内部控制改名为外部方法复现。
2. W2 分开 released-FastWAM compute reference、Full/Pooled matched-input profile 与 instrumented intervals；39.11 ms retrieval 与0.565/0.692 ms CUDA 区间保持独立研究/统计口径，不伪造 MemoryWAM 成本。
3. W3 围绕 requirement、candidate fit、generative influence 三个决策组织方法，用六组骨干消融与三训练种子常量门控控制支撑；单独说明 loss 的目标来源与作用，不声称每项辅助目标都已独立消融。
4. W4 用完整 donor-event 干预直接检验检索歧义；将20个源episode、1356个窗口及5个bank外DEV分开，避免把长时序任务难度或重叠事件数当成大规模独立示范多样性证据。
5. W5 核对原 specialist 的 event-specific 目标，区别公共 query 正则；独立候选执行用于验证当前状态判别，不由辅助 loss 形式直接推出反事实可识别。
6. Q1 保留原“address the weaknesses”总括请求，以简洁修订归纳回应，不虚构新的第六项技术问题。

## 核验结果

从本地不可变 Git 对象读取 `f77c633` 与 `47277e5` 的原 `retrospection_model.py`，两份字节SHA均 `d89796cc5da67d431d911e5f9191c5d60a14599bd3763cc64198e1384dc0b67d`。1219–1230行明确以各事件 `observed_effect` 为 effect-alignment target。64项目标追溯则是历史 BT case0 q8/q48 两组候选，不是四个任务各64项，也不是新策略试验。

九张定量表31行与原v40 LaTeX/已验Bbqv干扰表独立比对通过，两个非数值角色表按源码/方法核对；Wilson及三种子 t 区间复算一致。数值核对器初版发生 LaTeX控制字与行前缀解析问题，修正后同一数据全量通过；原值与正文未改，解析失败/修正记录保存在工作目录，不计科学实验失败。

全六原问题、标题与首句结构、作者视角、对外无内部提醒均已核对。没有新 SR、构造数值或补造未知实验；MemoryWAM额外复现/成本、广 repertoire 实证及原评测manifest身份等缺口仍按作者文件保存，`all_scientific_gaps_closed=false`。
