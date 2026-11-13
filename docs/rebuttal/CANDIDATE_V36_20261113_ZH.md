# v36：新增撤回实测与内部审阅候选

本记录截至2026-11-13 12:35UTC（北京时间20:35）。v36已完成写作、完整工程编译、干净源码ZIP重编译、视觉核验和双端封存。它是可审阅候选；关键科学问题尚未全部闭合，不能把37个条件论证槽位直接发布。

## 先读哪些材料

- `tmp/rebuttal/responses/20261113_v36_namespace_withdrawal/REVIEW_GUIDE_ZH.md`：三位回复、实际新数据、剩余缺口的审阅入口。同目录有Bbqv/rkWY/Hh4E全文、17个评论块和20项原review绑定。
- `tmp/rebuttal/revision/WARM_ICLR2027_revision_v36_namespace_withdrawal.pdf`：50页、主文10页；D.1表43/44在第47页，D.2第48页，C.6第42页。原33页最终投稿PDF仍为原始主张的权威来源。
- `tmp/rebuttal/revision/WARM_ICLR2027_revision_v36_namespace_withdrawal_source.zip`：完整原LaTeX工程；直接用原工程生成PDF，未以重新排版的独立文档替代。
- `tmp/rebuttal/execution/20261113_iter50/V36_QA_V3.json`与`ITER50_V36_AFS_VERIFICATION_V1.json`：最终构建、视觉和封存收据。

三位回复保留各自实际认可的strength感谢、we/our作者视角、逐项回答及已经完成的修订总结。没有显式面向AC的评分诉求，没有把尚未执行的protocol写成已完成证据。全文与17评论块可精确拼接；20项原字段、序号和hash绑定保留。37个有利假设整段继续标记待证，不仅把数值留空。

## 这轮新增数据及其边界

q96两个固定32候选池分别完成14/19个H32候选，invalid均0，各1个中断、17/12个未开始；411原件、39封口分支及43条384命令前缀已双端验真。case0完成标签全0；case1因父状态预先超计数而标签不可识别。最后Full控制未开始，没有发布完整pool主指标或修复收益。

显式撤回的两组固定query/context/checkpoint/Gaussian已核验，干预gate为0、候选token为0，query-only gist和实际history保持。四个原先未开始的物理角色均已封口：case0 null为32步/label0、Full为12步后invalid/标签未知；case1 null与Full均32步/label0。60原件双端验真。此前两个Full-pre在12/32命令后封口失败的20原件另行保全，保持unknown且不重跑。这里支持的是精确移除候选影响和局部执行记录，不支持新任务收益、自然gate可靠性或整bank缺技能结论。

原稿specialist结果、shared3000的41个0–0配对/13次历史unknown、固定PB12000诊断三种范围始终分开。C.6效率分母仍为75回合/15450 replans。投稿specialist训练运行已由作者确认，但最终DEV选定step与正式成绩对应的评估manifest仍未恢复；旧V5工程审计区分V1–V4版本，不足以单独认定投稿使用版本或推翻原稿结果。

## 验收和封存

原工程与干净ZIP均BibTeX加四遍编译，50页文字与栅格完全一致；前45页与v35像素相同，46–50最终逐页查看。0 undefined reference、0 overfull box，45条既有字体警告保留；原正文、结果、公式、九图及实测RGB、匿名性、AI声明保持。

|封存物|SHA256|
|---|---|
|PDF|`96384b358046799ce5020766fecbba4f072b7d5140c18cc20269fb9b171fea4e`|
|源码ZIP|`fb78b7e802af1bc8bb936828cbce70f53f988e1d5197ae4d78553d89cc94b97a`|
|110成员写作包|`240e4491428eff5e6fb3479987baaf76d5f40f8d061d1497d3d99766d90d34e7`|
|逐文件manifest|`3926fca4f0369bd1f7d862af2f6f2462e7c5e621e8c6827125b814e34f650688`|

AFS封存在`WARM_evaluations/rebuttal_20261109_acp/writing_candidates/20261113_v36/verified`，110项逐项核验；v35的102项、旧49入口及所有冻结源码均保持。已封存文件不原地改写，后续变化另存新版本。

## 恢复后先做什么

当前科学GPU程序为0；两个最后runner分别于11:13:55/11:14:41UTC complete/0，已结束。旧0GPU审计也已complete/0，但只产出两个q8报告，不能当作六池全部完成。原11UTC窗口已关闭；唯一闭合余额credit只转用一次，独立12分钟阶段也已关闭，不复活旧任务、不重置clock、不重跑unknown。

SenseCore在12:09UTC再次回到IAM页，已经请求作者重新登录，尚未收到本次登录确认；原生终态与原件核验不依赖该会话。登录后只核查现有UID终态，再按新阶段的明确计划派发。GPU仅闲时、CPU为0GPU，自动使用已确认镜像和AFS；不重复索取标准配额。

后续最高价值是绑定最终成绩与DEV选定权重/配置、取得有辨别力的同信息完整配对、后果预测与修复的直接证据、自然非空bank缺技能及完整任务对照。真机新SOR/R2P材料由作者独立采集，当前收到0次新试验，不重复索取旧录像。P1–P5保持开放并与37条件段联动，不因文稿完成而宣称科学证据齐备。

未来默认采集器显存限额修复已在df5fef0提交并push：同进程曾加载policy时保留原.48限额，fresh physical-only才用.12。85项原生CPU通过，Windows69通过/16项Torch缺失明确skip；本轮实测仍来自冻结e075ccd，未以新代码重跑。最终Docs离线bundle与current_iteration的安装收据记录在iter50执行目录；同步前保护已知三文件stash，科学程序不访问Git远端。

未发布OpenReview评论、未上传正式稿、未向他人发送材料。
