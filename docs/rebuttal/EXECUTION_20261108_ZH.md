# WARM rebuttal 首轮执行

三条线并行：固定模型实验、逐审稿人回复、原 LaTeX 工程修订。完整计划与材料位于 `tmp/rebuttal/`，原始数组、模型与PDF不进入Git。

## 实际已完成

- 本地、GitHub与CCI起始代码均为 `5eb321d`；CCI工作树干净。本地原有文档删除保持不动。
- 原LaTeX工程编译为33页；72 dpi下，第2–33页与投稿PDF逐像素一致，第1页仅差OpenReview附加提示。原工程保存在 `tmp/rebuttal/revision/baseline_source_20261108/`。
- 修订v1位于 `tmp/rebuttal/revision/WARM_ICLR2027/`；交付件为 `WARM_ICLR2027_revision_v1.pdf` 和同名 `_source.zip`。正文结束第10页，总长34页，已渲染检查全部页。只加入既有证据、监督和范围澄清；未插入新效能实验数值。
- step3000/v8共同噪声Full / scale-only / Gaussian诊断已完成并从原数组独立复核：10个来源episode、50个query、自然门控50/50非零。固定方差移除source均值后输出RMS差非零；只支持生成敏感性，不证明生成质量或闭环收益。DEV4 pilot与DEV50重叠，不能相加。
- SAPIEN渲染缺失组件已用精确版本提取库和进程专用环境修复；原runtime checker及实际128×128相机读回通过。未修改系统驱动或重启共享容器。
- 完整任务首轮在cuRobo规划warmup出现CUDA illegal instruction，失败原件已归档。以相同固定源码、相同编译选项构建私有CUDA12.8扩展后，Rearrange Blocks和Battery Try均在seed3407通过官方任务重置、4路相机和H32关节保持。原运行环境保留，未关闭CUDA graph或绕过规划器。两任务合计64条保持命令，没有调用WARM策略；完整状态恢复仍待验收。
- 新增恢复率统计代码，区分修复、损害、null增量和自然零门分层；独立标签及模拟器恢复仍须由采集器验证。

## 回复和论文修订

逐审稿人草稿与20项覆盖状态放在 `tmp/rebuttal/responses/20261108/`。实证缺口单独标记；未完成实验不得写作已完成，也不把工作稿直接发布。

`MANUSCRIPT_REVISION_V1.patch` 是基于已归档原工程的纯文本修订差异，保留源文件和具体修改以供版本审查；大体积图件和编译输出继续留在Git之外。应用前应先在副本中核对基线。

## 实验下一步

作者现已明确选择v8-step3000作为本轮RMBench补实验模型；权重、训练配置和attestation身份见实验账本20.4及`tmp/rebuttal/execution/20261108/MODEL_SELECTION.json`。下一步在该固定模型下接入真实策略记忆、完成阶段资格与独立标签定义，再冻结正式恢复cohort。当前共同噪声诊断不代替近邻方法的匹配训练控制，也不用于填充raw/refined/null修复率。

`configs/nonreal/rebuttal_recovery_protocol.json` 的状态明确为采集未就绪；`scripts/summarize_rebuttal_recovery.py` 只汇总已采集的独立分支标签，不能从回合成功推导修复率。

CCI同卡存在其他项目作业时，不终止其进程；耗时记录标记资源竞争，正式成本测量须另取可控窗口。运行从不可变源码快照执行，训练/评测不拉取Git；主分支同步不修改活动快照。

用户确认CCI每四小时可能自动重启、通常断连5–10分钟。安排有界作业并留归档余量；恢复连接后先核对PID、运行manifest和原始输出，不盲目重启或把断连推断为作业失败。

## 图形环境的可重复使用

库包来自NVIDIA官方580.95.05版本，SHA-256及实际资格日志见实验账本19.5。只提取、不安装，不替换系统驱动。参考来源：[官方包目录](https://download.nvidia.com/XFree86/Linux-x86_64/580.95.05/)、[extract-only说明](https://download.nvidia.com/XFree86/Linux-x86_64/580.95.05/README/installdriver.html)。

当前容器可在每个独立评测进程启动前source `WARM_evaluations/rebuttal_20261108/graphics_5809505/graphics_env.sh`。容器重建或驱动版本变化后必须重新核对版本及运行资格，不能默认旧配置仍有效。

`setup_rebuttal_graphics_env.py`只生成配置；其单测使用假库文件，仅验证配置保护逻辑，不能代表真实GPU资格。真实资格由独立日志和相机数组证明。两个拟用任务已单独完成重置/相机/保持命令检查；状态恢复、controller与计数器以及独立标签仍需另验。

当前任务复验额外使用进程内 `PYTHONPATH=WARM_evaluations/rebuttal_20261108/curobo_cuda128_source/src`，不覆盖原cuRobo安装。重编译的五个扩展、未改动源码及官方编译器包身份已归档于账本19.7；实际评测启动器必须同时记录该overlay与图形环境。完整状态恢复需要覆盖物理量以外的控制器、阶段/反馈、规划子进程、RNG和事实历史，不能只调用PhysX pack/unpack后即开始正式采集。

## 提交与同步

每次提交前拉取并检查upstream，只暂存本轮明确修改的源文件和小型说明；英文subject与bullet正文。提交后立即推送当前upstream并核对远端包含该commit，再同步CCI主checkout。保护并行修改，禁止force-push、reset或clean。

本地与CCI的28项相关测试均通过，真实DEV50和两任务资格另做独立原数组/输入身份复核。执行状态与证据入口为`tmp/rebuttal/execution/20261108/EXECUTION_STATE.json`，原始数据、失败日志和聚合复核继续保留在Git之外。本批全部作业已退出；正式恢复/退让/成功率实验尚未开始。

## 后续批次更新

- 实验性`RMBenchQposBackend`已实现task/controller/cache/RNG及调用者策略指纹检查，分支内规划RPC立即拒绝。原恢复顺序与Battery抓取状态的native编码失败均保留；没有通过改seed或放宽容差掩盖失败。
- 显式`prepare_native_query`在观测/动作封存之前构造可幂等恢复的原生checkpoint。无物理step，最多5轮；要求task/controller/RNG不变并记录原live编码是否改变。两个任务各自reset与首个专家grasp状态共4项通过，均seed3407、H32 A-A-B-A，合计512条分支命令。Battery grasp原live CPU编码改变已明确记录，因此这些结果只资格化准备后的checkpoint协议。
- 四状态原始包、六类端点和RGB经独立核对，见`evidence/RESTORATION_V3_VERIFICATION.json`；双端归档SHA-256为`70165ee8d1a093bbfb25900ea97192721b2b773974674a0d1c8784ee030c9d8d`。实际WARM记忆、其他阶段和独立标签仍未取得科学采集资格，没有新增策略成功率。
- 当前本地54项定向测试通过；此前两端28项测试是首批快照验证，不将其写成当前源码的CCI54项验证。
- 固定模型成本评测已完成严格合同和100个DEV前缀输入预检，开始100个计时query加20次预热；完整`_replan`含预处理、检索、生成及telemetry，磁盘解码排除，初始化另计。独立guard记录其他GPU进程。作业状态、计数与统计须在退出后从原件验收，目前不报告尚未取得的延迟。

后续成本作业已退出并独立验收：WARM中位数/p95为0.530059/0.565220秒，FastWAM release计算参照为0.366320/0.381831秒；各100计时、20预热，120对query的原相机和预算匹配。GPU peak allocated为23.620066/23.307467 GiB。WARM bank为89,680 events，recent容量20但实际占用0–8；完整初始化含严格资产校验/AFS加载单列为854.93/144.29秒，不计入查询。两条实现的计时/数值预处理边界及计算参照身份明确保留，不据此声称匹配训练或SR收益。全部原件、guard和源码身份双端封存，详见账本21.2–21.3及两个`*_COST_VERIFICATION.json`。

首批`EXECUTION_STATE.json`保留其历史时点，最新后续状态写入独立快照。上述工程资格和模型指定不覆盖原论文结果，也不自动补齐raw/refined/null恢复率、自然缺技能退让或配对闭环收益。

最新材料为`responses/20261108_v2/`与`revision/WARM_ICLR2027_revision_v2.pdf`（均相对`tmp/rebuttal/`）。回复覆盖20项意见，并拆分15个内部评论块；修订正文10页/全文35页，新增Appendix C.4/Table21/p35成本证据，前34页与最终v1逐像素相同。可编辑源ZIP、原件哈希和全部渲染QA见账本22节；版本化源差异为`MANUSCRIPT_REVISION_V2.patch`。仍未外部发布。当前入口`execution/20261108/FOLLOWUP_STATE_20261108.json`明确区分已完成成本与尚未采集的效能结果。
