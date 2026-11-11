# WARM v14 内部审阅候选

当前三份回复及15个对话框分块位于 `tmp/rebuttal/responses/20261111_v14`；原LaTeX工程位于 `tmp/rebuttal/revision/WARM_ICLR2027_v14`。本版在v13的固定BT终点对照旁，完整呈现新到达的32候选结果。20项问题—主张—证据—回复—修订位置映射保持，不能把单个有利终点作为总体预测或修复优势。

原BT checkpoint12000、root3407、预冻结case0/physical340800000及query8/rank4不变。32个有效候选全部执行完整H32，无unknown，第四个Full终点和DINO效应仍与前三个精确相同。380项49,259,146B原件归档 SHA `9d4f92e6a40fafbbe3b5d1a10557426c586cc37f2d4fe60201bfe6526c9e64ee`，manifest `bfe7dea67a83255145c3f31948943fbb172181010048597fcc4d98c05a756941`；本机独立核对每项原件、原物理阈值、DINO均值投影及同支持集方差恒等式。

| 固定q8观察 | 结果及范围 |
| --- | --- |
| 下一物理交互完成 | 10个候选为1、22个为0；不替换原稿19%适用性cohort |
| 原selected proposal / Full | 均为0，无完成型repair或harm |
| 电池1终点高度 | 原提案0.0279m、Full0.7599m；仅单状态跌落/未跌落终点差异 |
| centered MSE：预测/历史/常量 | 0.076844/0.076292/0.022408；该状态不支持预测优势 |
| 独立来源 | 一个来源状态；32候选及四个Full控制不能当独立任务试验 |

这些数值在Appendix C.2.6/Table24/p34及两位相关审稿人的回复同步呈现。其余原BRT q8/q48不利结果、q96 H32不可观测终态、shared-stage观察及原稿全部表格/九图保留。q48/q96按事前固定点继续，不换指标或挑选更有利样本。

完整PDF39页、主文10页，29文件源码ZIP；原工程和干净ZIP独立复编全部39页文本一致，前33页像素与v13完全相同。六个变化页34–39逐页视觉验收；无未解析引用/overfull，46个既有字体警告。PDF SHA `230dcc6e4e4642a2ac522f2cd1943ab69ae10c58825a9f58210d977fc7dbfe54`，源码ZIP SHA `9ba9dca571e9995d62701d12a489caa33fb4a528cbf1125625095bd9cb9574fe`，矩阵 SHA `b910f3a84e734c98ebee6edcde93c12167fb71576a8bba8bcee2d313708d90cb`。

87项审阅包124,898,316B，SHA `1bf35dca2beb2121fbf97dab03e10c7108f6e9cbef6f368e3aac7d0627a3c723`，manifest `a52af5c0627b47e0731f2dca315d333a633f7ff785b10f3534cfe5a529a9233a`，本机00:41:22及AFS00:44:11逐文件落盘验证。AFS目录 `writing_candidates/20261111_v14`。父v13包630eb424…仍保留完整native1000来源2020项和资格58项档案；本包增加完整pool380项并明确引用父包，未替换任何cohort。

Full3000完整state/CAS已独立逐字节重建，两个task的typed静态合同于00:17:04完成，12文件本机核验；这不代表实际策略或正式TEST执行。00:43:26最新Pooled2646/3000连续有限，仍须其final完整状态、CAS及typed合同后才准入已冻结RB/BT各50对。正式TEST时钟未开始，原训练Nov11 12:29:56UTC截止保持。

原BTq48标准ACP已创建并实际运行，UID `59dd5f78631b40ae81a1298844d499ba`，固定plan363bc263…/sourcef7cf0583…/rank3，原截止06:23:48.137340UTC。镜像、AFS、1GPU/1worker/0自动重启与实际编码器runtime全字段一致；00:43已有两个Full控制完整，正在重建原rank3父状态，尚无完整AABA或候选效能结论。q96未准入；准备器在前一个任务已知退出并归档后，才允许其首次11h窗口。资源优先标准，实际quota拒绝后才闲时。

CCI对同一AFS路径的本地flock探针返回空闲时，ACP任务仍实测运行；因此该探针不能作为跨节点互斥证明。`ITER13_NATIVE_LEASE_SCOPE_AUDIT_V1`和`ITER13_LIVE_V6`已区分CCI client观察与全局调度准入：当前仍禁止重复共享runtime。接续以ACP身份、实际runner和已知退出/归档为准；既有准备器也要求前一个任务终态，不曾因本地锁值重复派发。科学代码、样本及窗口没有改变。

全部科学closure=false、stable_release_ready=false；未发布OpenReview或正式上传。原DEV/cohort绑定、真正missing-skill、自动撤回及完整匹配效能仍需补齐；真机新增及原录像审计延期。本文及审阅包是带时间戳的快照，实际接续须先读Git外 `current_iteration.json` 和真实运行收据。
