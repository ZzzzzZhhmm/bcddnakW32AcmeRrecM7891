# 原PB命名空间的两个独立source：事前扩展协议

本协议是新增取证范围，不覆盖physical seed 0的既有单episode诊断或其所有失败、未知、窗口和完整原件。旧PB q8/q48/q96继续按原协议核销，不能按新状态替换旧结果。

## 为什么追加

原PB静态bundle中的seed protocol为int64 [100,2]：第一列是episode ordinal，第二列是候选seed下界，root3407对应340800000起。`validate_official_seed_namespace`和bundle builder明确允许official expert setup跳过不稳定候选，actual accepted seeds另行保存。旧诊断取展平首scalar=0，确实记录了physical seed0，因此是一个额外固定状态研究，并非该命名空间的首个正式accepted rollout。该source完整1500步、375 replans、success false已如实保存；不将它或新增shared模型成功地板解释为原稿specialist成绩复现。

新增研究只按官方候选顺序冻结最先通过expert setup的两个状态，case IDs 0/1；从340800000逐一检查，最多2000候选。保存每次接受/拒绝/不稳定原因；runtime错误停止，不能跳过。expert名单冻结时learned-policy queries=0。此后所有策略失败与unreached query都保留，不更换accepted case。

## 不变项与直接对照

模型固定作者确认原训练run的PB12000，weight a765d987…、原source47277e5/4a1d05cb…、原static runtime和bank/normalization/数值设置不变；不是原DEV选点恢复。新policy episode ordinal和physical seed从名单显式绑定，不再展平二维数组。每source先做1 replan/4命令资格，再从fresh reset执行完整1500步或official success，首观测/模型输出/四物理命令须与资格精确。

每source仍取预定q8/48/96，未达到则记unreached，不改query。每query在truth前封存全部K32原候选/预测/历史效果/Full/RNG/context，并独立reset重放原factual前缀；不用physics unpack。Full A-A-selected proposal-A及pool-final共四个Full控制，保留全部32候选和所有invalid/unknown。原四token均值DINO delta、H32共同支持centered主指标及原计数rubric不变；早成功的H32 censoring明确保留。

不同source各有private runtime、cache、lease和output；两个闲时ACP并行，无任务自动重启。资格、整episode、候选效果、修复及总体成功各自有独立范围。两个新增source不能替代完整任务评测、多训练种子或大样本总体repair率。

## 新增资源阶段

原阶段119.8054115725/120 GPUh预留账、所有旧窗口和三个已生效credit保留，不从未知退出假造退款。此扩展另登记：expert roster最多0.5 GPUh，两路ACP各最多10 GPUh，新增最多20.5 GPUh；累计准入上界140.3054115725，新的总上界144 GPUh、总并发仍不超过6。此为预留上界而非实际账单，不能重置旧时钟或挪作旧RB45第三次尝试。

roster须通过CCI剩余bootstrap窗口准入；长source/pool任务交ACP，GPU只闲时，已确认镜像/AFS自动填充。每路从真正准入起最多10h，且统一最迟2026-11-13 11:00UTC结束，给北京时间23:00稳定候选留4h整理。新phase须先代码/CPU验证、英文commit/push与CCI同步、冻结快照，再实际登记运行；准备和排队不是已取得数据。

## 当前状态

协议已书面固定，namespace解析与expert-only选择检查正在实现。新GPU资源尚未申请，新roster尚未运行；以Git外注册/实际进程及收据更新本段。原q96在23:15:38UTC原窗口结束，原件/退出/CPU审计另行核对，不因本扩展改写。
