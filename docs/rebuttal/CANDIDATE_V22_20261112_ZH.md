# v22：作者对话、原报告精度与干预时序

内部审阅候选；尚未取得实证发布资格。目标仍为北京时间11月13日23:00稳定候选版。原v21封存不改。

## 可审阅材料

- 三位回复：`tmp/rebuttal/responses/20261112_v22_count_precision/{Bbqv,Hh4E,rkWY}.md`；17个ICLR网页评论块在`comment_blocks/`。
- 20项逐条映射：同目录`ISSUE_EVIDENCE_REVISION_MATRIX.json`；逐项给出原证据、待证分组及真实PDF位置。
- 完整原LaTeX工程：`tmp/rebuttal/revision/WARM_ICLR2027_v22_count_precision/`。
- PDF：`tmp/rebuttal/revision/WARM_ICLR2027_revision_v22_count_precision.pdf`；45页，含结论主文10页。SHA256 `ade470a6a7ba9e1e7a346a150514790039c069f0ea902dc926f430ac151f6567`。
- 源码包：同名`_source.zip`，SHA256 `cf05bf28bd59ee027a1ddfe9e28335e7fa5a2bff5621441e00c7464c286cd92e`。

## 对话与论证

保持每位审稿人实际认可的优点开场，避免把Hh4E对问题与整合思路的认可写成认可机制有效性。主体用we/our说明我们的设计、证据及修订责任；结尾总结已落实的变化，公开文字不出现显式AC、meta-review、索分或录用目标。原稿核心贡献以既有机制、重复交互和完整事件干扰证据组织，弱实验继续补证，逻辑歧义精准替换。

新增原任务逐项精度：正式33页PDF明确九项RMBench各100次、真机四任务各20次。附录D.6从原报告成功数计算13项95% Wilson区间，SciPy独立复算最大差2.22e-16。它们以固定训练策略、任务内独立二项试验为条件，属于边际逐任务区间；新增试验0，不是原逐次日志恢复、跨训练种子区间、宏平均区间或paired baseline检验。主文4.2、Bbqv W1/Q1及Hh4E W1就近交代。

新增干预时序：附录D.1、Bbqv W3/Q2及Hh4E W5明确，原BT q96父状态在384步已掉落，随后零/零分支描述该不利状态后的行为。与q8中Full避免提案掉落分开解释；不据此判定不可恢复或一般失败原因。保留固定状态、全部标签和预定误差支持集，不作事后剔除。

## 验收与剩余门槛

完整工程及干净ZIP均含BibTeX四遍编译，45页文字逐页一致；0未定义引用、0overfull，45个继承字体warning。原九图逐字节保留，主文数值及既有公式、表格保留；对v21仅页7、42改变，新增45页，三页已直接验看，另42页像素一致。匿名及内部待证标识检查通过。

83成员、123,862,802B材料包本机逐成员校验，AFS于Nov11 23:41:58UTC另行逐成员和解包独立验真：`rebuttal_20261109_acp/writing_candidates/20261112_v22`。TAR SHA256 `fc8a2a8524c909c57d486659091f26709841f8273d62d1beffd99dd2fa1e5888`；manifest `451f3540e5ed37ac95c3ab091f4b94f3d0bbb83a64aec7026dab7463de6435f5`。

P1–P5整段方向与37槽仍待证，原有不利与未知不隐藏。当前完整对照29角色/14对全0/0/6unknown，case40 Full尚未配对；q96全池未完成；真机由作者独立采集、收到0新增trial。文字和排版验收通过不等于实证缺口关闭。最终发布前须按真实结果核对整个论证段，不是只填数字。
