# v28：明确实验身份，避免同名query被误读

当前完整内部候选为 `tmp/rebuttal/responses/20261112_v28_task_identity`、`tmp/rebuttal/revision/WARM_ICLR2027_v28_task_identity` 及对应 PDF/源码ZIP。原v27、全部原件及执行增量保留；未发布。

## 本版修订的具体问题

正在测量的 Press Button q8/q48/q96 与已完成的 Battery Try specialist q8/q96、shared-model Rearrange Blocks q8 使用相同query序号，但属于不同任务、模型和状态。v27回复部分句子只写“original-specialist q96”，容易在接入PB结果后造成误读。

本版在三位回复、论文4.7与D.1的标题/图注/表注/起句中明确task/model/query身份。原shared q8的seal和online contract核对为rearrange_blocks、seed3407、Full权重b0669047…，不是凭文件名推断。Bbqv W1将“收益出现在哪里”与“条件rollout区间”分段，全部原数值保留。三位感谢、各自真实strength、we/our作者口吻及已完成修订总结保持。

## 保真与验收

- 全部三位回复的数值token、整段PENDING及其有利定性假设保持；20项原review字段/序号/hash绑定、17个评论块与完整文本同步。
- 原方法、全部旧表数据/公式、九幅原图与新增原始RGB保持；不删除原实验、不用新run替换未解决数字。
- 原完整LaTeX工程及独立干净源码ZIP均经含BibTeX的四遍编译：46页，主文含结论10页，全页文本一致，0未定义引用、0overfull。
- 新第10、42、43、44、45页直接视觉验看；其余41页与封存v27逐像素相同。45条既有字体warning保持。
- PDF SHA `6e4873c4d9f47c7aff032d211e7c58a97b80d85a67d5c2e6d55f5598d540f534`；源码ZIP SHA `bbd5531d386a006192118d64fdbf6807fe222250a148d40d80d2cbc30df358e8`。
- 85项写作包TAR SHA `4f15302b854c1334fcd0db2b5087307a2124d37a097ac1f217eb390294daa524`，manifest SHA `f612a27a4def006b45b689fbf53ac953d33c3c8e25dca827b3b08aaf14b4a384`；14:53UTC AFS逐项重读验真，`writing_candidates/20261112_v28/verified`。

自审遵循作者提供的SciCore笔记：分别检查全文是否易懂，以及剥离修辞后主张是否仍由相同证据支持；不把AI自评分当录用概率。37个待证槽及其方向没有因表达更明确而放行。

## 动态证据必须另读最新快照

封存v28的执行说明以14:28:52UTC的已验收51角色/25对为截止；随后新增BT47 Full于14:31:11退出并独立验真，当前累计52角色/26对，26对均0–0。原allocation50角色与两个资源等价CCI角色分别标注；8个matched基础设施unknown保留。这是shared/final3000补充研究，完整100对尚未结束。后续以current_iteration所指最新独立验收数为准，不能提前把尚未审计的新终态加计。

CCI当前已按同一outer自动进入48Full，原16:19:40UTC窗口保持，不重复派发。两ACP RB分片继续原名单；新BT00剩余角色的独立闲时任务UID `a61db50720a241ef9f3a852495231db3` 已在另一集群提交，按原next_request仅2Full、2Pooled、3Pooled、3Full、4Full，18UTC截止。旧bt00截止和一项旧加载unknown不变；不重跑旧未知，不占用第七个GPU槽。

唯一PB长任务UID `85e374b421a24794ae0e9f76d0257378` 与BT30抢占后继UID `427e8b34f97c4dbe96b0030a5717efe3` 继续保留。PB当前受到nodeSpec CPU/内存限制，新的BT00集群尚无空闲GPU；全局余卡不等于实际分配。已查看PB优先级修改按钮，实际disabled，未修改或绕过。新阶段仍为闲时1GPU/8CPU/128GiB，最大并发6、保守预留上界108.8975605514GPUh/120保持。任务只有实际runner、心跳、轨迹和退出验收后才计为结果。

PB已测q8零、q48超额NA、q96 invalid17/旧最终控制unknown与原BT完整q96不利误差，均分别保留；无新真机trial回传。完整原稿主张由原有证据及实际新增直接证据支撑，待证结论仍需正面补证，不能仅填写数值。
