# Nov13 iter43：闭窗核销、RB45转交CCI及82回合验收

本轮独立验收累计82个完整策略回合、41个固定Full/Pooled配对，均为0–0；其中原allocation66回合、资源等价补充16回合。固定100对研究尚未完成，这组shared-final3000结果不建立优势或非劣效，也不替代原稿specialist成绩。历史unknown尝试仍为12个：cohort10、独立CCI加载1、独立资源guard1。

## 真实资源所有权

- **CCI唯一新任务**：`matched_outer_rb40_stage_iter43_v1`，原RB45 Pooled显式第二次→条件45 Full。21:27:45UTC launcher2587139，21:27:48真实runner/monitor启动，21:29 active child2587173、0生成，仍属加载阶段。不得把启动或加载计为策略推理完成。
- **原ACP已停止**：`warm-mf3k-rb40-tail-20261113-p43-idle`，UID6561c8f110b54afea9a0c79c9450fae2。GPU不足等待、无Pod IP/runner/AFS日志；21:21:11UTC浏览器回读“已停止”，之后双端核查无该阶段operator/job_logs，再转交CCI。不要恢复或复制它。
- 同一冻结登记21:02:23.699406→**23:32:23.699406UTC**完整保留，CCI转交不增加预留、不重置窗口。源acca89ba…/计划b6f57eea…/模型、样本、原161条已完成命令和1条无end命令的未知结局均保持；无自动第三次尝试。
- BT32及RB35的两个ACP阶段分别21:20:25、21:05:48UTC真实runner exit0；前一CCI BT4阶段21:03:51UTC正常结束。均不再视为活动任务或重复派发。
- 原唯一PB ACP85e374b421a24794ae0e9f76d0257378继续q96，截止23:15:38.613729UTC；原标准0GPU审计1ae90ef25fac4730a0ac38ba1405df73等待完整终态，截止23:25:38UTC。21:29 q96已进入rank18的父状态重建；未到整池终态，不作全池统计。

所有UTC时刻对应北京时间次日+8小时。恢复时先读最新Git外current_iteration/心跳/退出收据，不按本页时点重开任务。后续GPU只闲时，CPU可以标准0GPU/4CPU/16GiB。

## 预留账核销

三个已结束CCI阶段iter36/38/40保留原登记、真实终态和失败原件，仅一次性核销未使用尾窗，共2.5874595514 GPUh。原119.8928711239降至117.3054115725；新RB45独立登记2.5后为**119.8054115725/120 GPUh**、最多6槽。这是准入预留上界，不是实际GPU账单或新增配额。

`ITER43_RESERVATION_RECONCILIATION_V1`是应用前审计；`ITER43_CREDIT_APPLIED_V1`是唯一生效收据，三个credited IDs已经写入状态，**不可再次核销**。后续核销必须继承这条负调整，不能拿原纯正收费链重算后重复释放。纯CPU模块17项Windows/17项原生测试通过，全部44个既有冻结入口保持。

本次CCI转交的首个准入检查遇到CPU预检暂持lease而拒绝，未创建GPU runner；lease释放、再次核对无活动runner后才启动。两角色CPU预检各exit0。恢复过程另一次纯CPU重复预检因唯一收据已存在而拒绝覆盖，未产生策略forward或物理动作；原成功收据不变。

## 原件与科学边界

六个新增完整回合按V1/V2分别封存，4,212个原件在本机独立验证全部字节、有限H32、qpos及accepted物理链。三个新配对BT4/RB35/BT32的初态、共同contract、首query图像/proprio、Gaussian、bank和candidate payload完全对应。RB35 Full有1次TOPP观测错误，其官方执行轨迹、正常终态和success0原样保留，不能写成无TOPP错误。

RB35 Pooled还精确复现旧中断尝试的42个H32和167条物理begin/end记录，两个context文件60个Torch元数据/storage记录相同。旧中断结果仍为None。BT32旧attempt只有加载，没有可宣称复现的旧物理前缀。

AFS根为`/mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations/rebuttal_20261109_acp`；本机原件在`tmp/rebuttal/execution/20261113_iter43`。重点收据：`ITER43_ROLE_RAW_LOCAL_VERIFICATION_V2`、`ITER43_RB35_REPLAY_LOCAL_VERIFICATION_V1`、`ITER43_CREDIT_APPLIED_V1`、`ITER43_RB45_CCI_REASSIGNMENT_V1`、`ITER43_LIVE_V3`。完整原件hash见对应manifest，不以UI状态替代科学验收。

## 写作交接

封存v30 PDF/源ZIP仍47页/主文10页；20项意见、17块回复、37整段有利方向PENDING不放行。C.6成本样本固定**75回合/15450 replans**，不可改成82。新增两份已验真作者口吻补充供审核，非新完整回复、非公开评论；q48独立Markdown/LaTeX插入片段尚未合入PDF。q96终态核验后合并q48/q96，生成新的完整原工程及干净ZIP、编译并视觉验收。

四个作者确认原训练目录的58个有界文本文件未恢复DEV选点记录；固定12000身份保持，不能按新TEST结果选点。自然missing-skill、总体修复、忠实named baseline及作者新真机证据仍独立开放，不能通过82/41或已完成工程检查宣布全部补齐。
