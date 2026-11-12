# v23：逐项证据、积极作者论证与具名基线条件

内部审阅候选；尚未达到实证发布门槛，未发OpenReview或上传正式文稿。原v22封存不改；稳定版目标仍为北京时间11月13日23:00。

## 本轮改进

三位回复保持真实认可优点的感谢开场、we/our与具体已完成修订结尾。Bbqv W1/Q1先呈现已报告M(n)+5.8点及BT/PB任务收益，再准确讨论总体0.1/0.9差值的不确定性。所有数字、P1–P5整段条件论证及37空槽保留，不靠更强措辞填证据缺口。

20项映射各自列出review关切、原主张、已经回答的子问题、具体原报告/新证据、尚需直接实验及真实PDF位置，并绑定原review note、字段、序号与hash。独立核验20项源字段、20项回复标题覆盖、17评论块和全部修订label。修正此前沿用的v21版本字段与过旧的候选进度；不再用通用证据一句话概括全部问题。

针对Hh4E的具名成本问题，单列MemoryWAM成本与LIBERO复现。其[官方公开仓库](https://github.com/yangsizhe/MemoryWAM)在Nov12 00:41UTC固定main commit dc82a9d760e751ae88f2e0ae521af7c7ed597dfb仍将训练/推理代码列为待发布；完整tree只有README/LICENSE及四张asset图。原API/README/tree在Git外保全。回复与附录D.2说明这一时点公开分支条件，保留其原报RMBench结果，Pooled明确是自己的适配对照。该事实不构成新的性能/成本结果，也不涵盖私人渠道；正式发布前重新核验。

Wilson表的置信水平写为nominal，避免把名义95%表述为精确有限样本覆盖保证。原13任务统计、主文、九图、既有公式、表格与所有不利/未知结果保留。

## 材料与验收

- 回复：`tmp/rebuttal/responses/20261112_v23_issue_specific/{Bbqv,Hh4E,rkWY}.md`；17网页评论块位于`comment_blocks/`。
- 逐项JSON与自审：同目录`ISSUE_EVIDENCE_REVISION_MATRIX.json`、`ISSUE_EVIDENCE_AUDIT_ZH.md`。
- 完整原LaTeX工程：`tmp/rebuttal/revision/WARM_ICLR2027_v23_issue_specific/`。
- PDF：`tmp/rebuttal/revision/WARM_ICLR2027_revision_v23_issue_specific.pdf`，SHA256 `96069f57186468cac1617454966b659fdedbb88baa7e910b852dabb01d7be597`。
- 源码ZIP：同名`_source.zip`，SHA256 `d5ef75c5d81a2bbca802dd2cb800d7ed11710a66c74485f44682e2361defeb2f`。

完整工程及干净ZIP均含BibTeX四遍编译45页/主文含结论10页，所有页文字相同；0undefined/overfull，45继承字体warning。对v22只有43/44/45改变，三页直接验看，另42页像素相同。原九图逐字节保留，原数值与公式、表格内容核对；匿名及内部待证标识通过。首次包装检查因预期的表注文字变更触发旧严格相等检查；随后增加两处已审文字的精确重建守卫，并核对全部原数字和表数据后通过，失败时PDF另保留、字节与最终编译件相同，没有改数据绕过检查。

89成员123,951,134B包在本机逐项校验，AFS于00:52:58UTC另行逐成员/解包验真：`rebuttal_20261109_acp/writing_candidates/20261112_v23`。TAR SHA256 `99f68449da64d7b1e0ed5dc740893fab0dcbc55094a5d28334afd5109f96f41c`；manifest `98a6d0b710418f04a9165d0080ad02c2dc9f6d04784a59592a5a862ad7fdd577`。

## 实证状态

核验截止00:31UTC：30完整角色/15配对均0/0/6unknown，不能填最终100对SR/CI或收益。原BTq96已有20候选端点独立归档；全池未完成、父状态干预前掉落保持。真机由作者独立采集，收到0新trial。最新活动状态须读取current_iteration及原始收据，材料快照不是未来实时状态。
