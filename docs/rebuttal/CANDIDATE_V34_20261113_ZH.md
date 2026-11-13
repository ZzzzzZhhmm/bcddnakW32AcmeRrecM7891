# v34 内部候选与恢复入口

本版是可审阅的内部候选，尚不能作为“关键证据全部闭合”的发布版。三位回复采用作者视角、以审稿人实际肯定的贡献开篇，逐项回应，并在结尾说明已经完成的修订；不向公开回复引入AC、评分或录用请求。20项原review字段、序号和hash绑定、17个可粘贴块和37个完整条件论证槽均保持。条件槽仍待证，不将理想方向当成事实。

## 已完成材料

- 回复：`tmp/rebuttal/responses/20261113_v34_namespace_q8`，包括Bbqv、rkWY、Hh4E全文、17块、20项映射与自审。
- 原LaTeX工程：`tmp/rebuttal/revision/WARM_ICLR2027_v34_namespace_q8`。
- PDF：`tmp/rebuttal/revision/WARM_ICLR2027_revision_v34_namespace_q8.pdf`，SHA256 `a2802e614a68b79718b01ed1819ee58f6c366c3932f52441bd16dc6f7c1daaf3`。
- 源ZIP：`tmp/rebuttal/revision/WARM_ICLR2027_revision_v34_namespace_q8_source.zip`，SHA256 `e1361fa91aa2541395db6adf6c1e8d54fa64f5b32c6b26c5fe62fa1ddd55134e`。
- 原工程、干净ZIP分别完整编译，49页/主文10页；0 undefined reference、0 overfull。前45页逐像素与v33相同，46–49页直接检查；标准分页使D.6标题与表43同页。45项既存字体警告保留。匿名性、AI声明、原表/公式/九图及实测RGB保持。
- 108成员封存包SHA256 `b4afc90390114c3853d088b9707740547170cb52db9f623b2a26d4b175100157`；AFS `writing_candidates/20261113_v34/verified`逐项验真。原v32/v33不覆盖。

## 新证据及严格边界

两组事前冻结PB namespace source均完成1500命令/375次推理、success=false，2298原件已独立验真。它们使用原稿训练run的固定step12000；原稿DEV所选step仍未恢复，不能称为原稿成绩复现。

两组q8各32候选均有终态，H32完整28/32和32/32；共同有效支持上的预测/历史/常量MSE分别为0.0502703/0.0499976/0.0119948、0.0612284/0.0609323/0.0178333。两组均未胜两个参照。case0四Full均12命令后TOPP-invalid，H32标签未知；case1四Full及32候选严格完成标签全0，pretruth gate=0.9296875。后者是该query的检索池、严格H32完成准则下的直接失配观察，不能推广为全bank缺技能或总体误激活率。746原件、72分支、76独立前缀已重算验证；D.1/D.4与Bbqv/Hh4E同步，未用别的指标替换预定主指标。

已有Battery Try固定q8的Full/null/Full物理稳定性证据保留，并与任务完成/总体收益区分。shared-final3000比较仍为82角色/41个0–0对、13个unknown尝试；原稿specialist、shared及新固定12000三种范围分开。C.6实测成本仍为75回合/15450次replan。

## 执行恢复

先读`tmp/rebuttal/planning/current_iteration.json`及Git外`ITER46_EXECUTION_ADDENDUM_ZH.md`。两原闲时ACP继续固定q48→q96，原UID和10:34:23/10:35:04UTC截止不改；0GPU审计继续等其余四池。不要复制、重跑已完成source/资格，或按结果换query。

显式q8 Full/null/Full新对照已104项原生CPU检查、CLI/真实plan预检通过，冻结在`PB_namespace2_null_20261113_v1`；只用一份闲时GPU、3.5小时含180秒退出余量、最晚11UTC。事前累计预留143.8054115725/144GPUh，不代表计费。控制台IAM已过期，已请作者登录；尚未点击创建、没有新UID。恢复后先核对任务名及AFS是否已有实际runner，再提交，不重复未知创建。

尚未闭合：预测优越性/完成型提案修复、自然撤回可靠性、完整同信息性能比较与相应区间、可忠实复现的命名baseline，以及作者独立推进的新真机数据。完成工程和负结果归档不等于这些问题得到正向支持。下一版只接入独立核验的真实数据；若结果仍不利，正面解释测量范围与限制，不填有利方向条件段。
