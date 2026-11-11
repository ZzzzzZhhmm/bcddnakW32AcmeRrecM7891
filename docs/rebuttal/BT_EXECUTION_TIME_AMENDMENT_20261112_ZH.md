# BT配对评测：显式工程时限修订

## 原因与边界

原BT 1000步评测在单角色2400秒外层时限下，三个固定case分别于936、946、988步被终止，success均未知。它们不是模型返回无效动作的已知失败，也不能拿来计算完整SR。原件全部封存，不以另一个case替换。

新的工程入口`run_rebuttal_execution_amendment.py`在显式hash绑定的修订收据下，把BT两个角色共同的单角色外层上限从2400秒调为3600秒。原计划文件、case/seed、checkpoint、模型/collector源码、1000步任务上限、训练与推理预算、首次准入时间和总截止均保持原绑定。最多6个并行formal slot、原120GPUh总上限保持；已消耗时间不冲销。该修订允许在已有总资源窗口内完成原任务，不能重开过期窗口。

## 实现与可追溯性

入口导入原冻结orchestrator，只修改内存中的工程worker时限视图；子进程仍调用原冻结collector并读取原计划字节。新的attempt与campaign状态同时保存修订路径/hash和effective limit，避免静默改原计划。所有模式、训练损失、采样器、阈值和科学指标保持原身份。

原外层收到SIGTERM时，result文件可能停留在loading状态，而物理日志实际已到946步。只有原结果及实际exit文件与已封存终态收据完全相符，才能在独立只读视图中分类为unknown并显式续跑；原JSON不改写。原有next_request和两次attempt上限继续生效，不绕过unknown、不选优、不把重复控制算额外独立trial。

## 验收

CPU拒绝检查覆盖：只改变共同worker时限且不修改原plan；拒绝改变截止、角色公平性、科学字段或超出本次3600秒；没有实际终态/hash不允许转换stale status。正式启动前还须核查旧云UID/CCI组终态、campaign及native lease、AFS容量、原截止余量和source hash。

CCI采用最多一个新role的有界invocation；长批次放ACP。实际成功率、置信区间与rebuttal结论只用完整且独立核查的原固定样本。该工程修订本身不构成机制收益证据。

## RB外部中断的独立恢复类型

RB case11 Pooled的实际非bounded SIGTERM使用单独的terminal-status-amendment收据。它只在完整原件/exit/result绑定下把旧stale loading放入只读unknown视图；RB共同worker上限仍为原2400秒。该类型不能取得BT的3600秒额度，也不改变任何原样本、策略或截止。正在运行的BT修订入口继续使用其旧冻结源码，不受后来功能扩展影响。
