# WARM rebuttal 首轮执行

三条线并行：固定模型实验、逐审稿人回复、原 LaTeX 工程修订。完整计划与材料位于 `tmp/rebuttal/`，原始数组、模型与PDF不进入Git。

## 实际已完成

- 本地、GitHub与CCI起始代码均为 `5eb321d`；CCI工作树干净。本地原有文档删除保持不动。
- 原LaTeX工程编译为33页；72 dpi下，第2–33页与投稿PDF逐像素一致，第1页仅差OpenReview附加提示。原工程保存在 `tmp/rebuttal/revision/baseline_source_20261108/`。
- 修订v1位于 `tmp/rebuttal/revision/WARM_ICLR2027/`；交付件为 `WARM_ICLR2027_revision_v1.pdf` 和同名 `_source.zip`。正文结束第10页，总长34页，已渲染检查全部页。只加入既有证据、监督和范围澄清；未插入新效能实验数值。
- step3000/v8共同噪声Full / scale-only / Gaussian诊断已完成并从原数组独立复核：10个来源episode、50个query、自然门控50/50非零。固定方差移除source均值后输出RMS差非零；只支持生成敏感性，不证明生成质量或闭环收益。DEV4 pilot与DEV50重叠，不能相加。
- SAPIEN渲染缺失组件已用精确版本提取库和进程专用环境修复；原runtime checker及实际128×128相机读回通过。未修改系统驱动或重启共享容器；完整任务与状态恢复资格仍待验证。
- 完整任务首轮在cuRobo规划warmup出现CUDA illegal instruction，尚未完成任务重置。失败原件已归档；私有CUDA12.8扩展重编译正在排查，原运行环境保留。不能把渲染通过等同于任务资格通过。
- 新增恢复率统计代码，区分修复、损害、null增量和自然零门分层；独立标签及模拟器恢复仍须由采集器验证。

## 回复和论文修订

逐审稿人草稿与20项覆盖状态放在 `tmp/rebuttal/responses/20261108/`。实证缺口单独标记；未完成实验不得写作已完成，也不把工作稿直接发布。

`MANUSCRIPT_REVISION_V1.patch` 是基于已归档原工程的纯文本修订差异，保留源文件和具体修改以供版本审查；大体积图件和编译输出继续留在Git之外。应用前应先在副本中核对基线。

## 实验下一步

已完成step3000固定模型的共同噪声诊断。下一步依据可用任务模型、完整状态恢复和独立标签冻结正式恢复cohort，并区分最终投稿模型与当前可用checkpoint。当前诊断不代替近邻方法的匹配训练控制，也不用于填充raw/refined/null修复率。

`configs/nonreal/rebuttal_recovery_protocol.json` 的状态明确为采集未就绪；`scripts/summarize_rebuttal_recovery.py` 只汇总已采集的独立分支标签，不能从回合成功推导修复率。

CCI同卡存在其他项目作业时，不终止其进程；耗时记录标记资源竞争，正式成本测量须另取可控窗口。运行从不可变源码快照执行，训练/评测不拉取Git；主分支同步不修改活动快照。

用户确认CCI每四小时可能自动重启、通常断连5–10分钟。安排有界作业并留归档余量；恢复连接后先核对PID、运行manifest和原始输出，不盲目重启或把断连推断为作业失败。

## 图形环境的可重复使用

库包来自NVIDIA官方580.95.05版本，SHA-256及实际资格日志见实验账本19.5。只提取、不安装，不替换系统驱动。参考来源：[官方包目录](https://download.nvidia.com/XFree86/Linux-x86_64/580.95.05/)、[extract-only说明](https://download.nvidia.com/XFree86/Linux-x86_64/580.95.05/README/installdriver.html)。

当前容器可在每个独立评测进程启动前source `WARM_evaluations/rebuttal_20261108/graphics_5809505/graphics_env.sh`。容器重建或驱动版本变化后必须重新核对版本及运行资格，不能默认旧配置仍有效。

`setup_rebuttal_graphics_env.py`只生成配置；其单测使用假库文件，仅验证配置保护逻辑，不能代表真实GPU资格。真实资格由独立日志和相机数组证明。完整任务重置、状态恢复、controller与计数器以及独立标签均需另验。

## 提交与同步

每次提交前拉取并检查upstream，只暂存本轮明确修改的源文件和小型说明；英文subject与bullet正文。提交后立即推送当前upstream并核对远端包含该commit，再同步CCI主checkout。保护并行修改，禁止force-push、reset或clean。
