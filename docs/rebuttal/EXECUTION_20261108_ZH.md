# WARM rebuttal 首轮执行

三条线并行：固定模型实验、逐审稿人回复、原 LaTeX 工程修订。完整计划与材料位于 `tmp/rebuttal/`，原始数组、模型与PDF不进入Git。

## 实际已完成

- 本地、GitHub与CCI起始代码均为 `5eb321d`；CCI工作树干净。本地原有文档删除保持不动。
- 原LaTeX工程编译为33页；72 dpi下，第2–33页与投稿PDF逐像素一致，第1页仅差OpenReview附加提示。原工程保存在 `tmp/rebuttal/revision/baseline_source_20261108/`。
- 修订v1位于 `tmp/rebuttal/revision/WARM_ICLR2027/`，编译件位于 `build_revision/`；正文结束第10页，总长34页。只加入既有证据、监督和范围澄清；未插入新效能实验数值。
- 小规模step3000/v8源诊断已完成，见实验账本第19节；不是最终投稿模型复测或正式成功率。
- 新增恢复率统计代码，区分修复、损害、null增量和自然零门分层；独立标签及模拟器恢复仍须由采集器验证。

## 回复和论文修订

逐审稿人草稿与20项覆盖状态放在 `tmp/rebuttal/responses/20261108/`。实证缺口单独标记；未完成实验不得写作已完成，也不把工作稿直接发布。

`MANUSCRIPT_REVISION_V1.patch` 是基于已归档原工程的纯文本修订差异，保留源文件和具体修改以供版本审查；大体积图件和编译输出继续留在Git之外。应用前应先在副本中核对基线。

## 实验下一步

先完成step3000固定模型的自然门控与full/scale-only/Gaussian共同噪声诊断，再依据可用任务模型、完整状态恢复和独立标签冻结正式恢复cohort。当前诊断不代替近邻方法的匹配训练控制。

`configs/nonreal/rebuttal_recovery_protocol.json` 的状态明确为采集未就绪；`scripts/summarize_rebuttal_recovery.py` 只汇总已采集的独立分支标签，不能从回合成功推导修复率。

CCI同卡存在其他项目作业时，不终止其进程；耗时记录标记资源竞争，正式成本测量须另取可控窗口。运行从不可变源码快照执行，训练/评测不拉取Git；主分支同步不修改活动快照。

## 提交与同步

每次提交前拉取并检查upstream，只暂存本轮明确修改的源文件和小型说明；英文subject与bullet正文。提交后立即推送当前upstream并核对远端包含该commit，再同步CCI主checkout。保护并行修改，禁止force-push、reset或clean。
