# WARM v12内部审阅候选

新增原specialist BRT固定14k/seed0/query48的完整K32核销：24完整、8 TOPP unknown；同支持集centered MSE为预测0.0550445858、constant0.0307141708、historical0.0550926601，未观察到优于constant的预测优势。四Full anchor及全部fresh parent/RNG精确，343原件91d595a8…双端逐项验证，本机另用独立方差恒等式重算。原q8、原稿另一个机制cohort和shared-stage不利/未知记录全部保留，不换样、不缩放、不把工程精确性当作收益。

原RB六对确已到原生700上限，十二角色均0成功，余四对pending；62项审计de99e5ca…双端验证。修正先前“低于native horizon的有界700”概括，不更改结果、checkpoint、seed或不完整cohort边界；原BRTsource600/3500是另一种有界来源。此口径与原生正式比较冻结设计3484ad14…保持一致：RB/BT各50对、Full/Pooled固定3000、新100对/200回合，设计未派发，不补填旧Full/null或missing-skill结果。

三份英文/15块/20项位于tmp/rebuttal/responses/20261111_v12；Bbqv W3/Q2、Hh4E W5与矩阵指向Appendix C.2.5、Table23/page33。完整29源文件原LaTeX工程及ZIP独立编译、全部38页文本相同；主文10页与v11相同、参考文献从11页开始。所有旧tabular和九图bytes保持，引用、overfull、匿名性通过，46条继承font warning不增加。38页contact与全部变化页32–38单页视觉检查通过。成本表随新增表编号变为26/27，矩阵按实际aux全部刷新。

PDF SHAed11dae1012a54a7a04bdb8450ffd9be9ca1a768b0eac7084ecc59d7ab9860d1；source ZIP SHA8955a7e9a994aebe345e01d63621c58c8be60ecbeae9d863f1d206dc03d9e464。原33页与v11不改，真机新增延期。pool100原生10/10资格完成；Full/pool最终训练仍在运行，原q96已17:45准入独立384命令两遍审计、phase截止Nov11 01:45UTC，尚无新候选truth。直接修复/撤回、真实missing-skill、最终匹配性能及原DEV/cohort绑定仍未关闭。stable_release_ready=false；未发布OpenReview或正式上传。

18:06执行增量：q96两遍384前缀17:59:42正常退出，索引101/234拒绝精确保留，16原件4431cceb…双端核验；新标准UID492eb35f0d98462db196a624cf492d5c已实际启动，runtime hash2f3e5d29…与原合同全字段相同，正在首个anchor父状态重建，AABA和32池尚未完成。原phase01:45UTC不变，未重复创建或回退闲时。正式同信息100对的独立Full/Pooled两checkpoint原生collector尚待实现/验证，不能把旧Full/null入口直接交给Pooled；当前设计冻结不等于运行程序已就绪。

18:37终态增量1：q96实际18:20:13 exit1；两Full均在第27条命令遇相同TOPP拒绝，完整26条轨迹/parent/RNG/物理端点/DINO文件完全一致。属于完整H32资格失败，不是复现漂移；0候选端点/32未观测，不缩H、换rank或重跑。36原件45f4f00c…已双端核验，标准任务不再活动。最新三回复/15块/20矩阵转responses/20261111_v12_addendum1，matrix62fdfe7a…，44项包b384164b…双端验收。v12 PDF/源码ZIP及原95项包原样保留；论文没有q96结果断言，回复和矩阵已撤去pending表述，与论文数据范围一致。实际运行的是Full1680/pool966两路训练（18:32快照），final3000与新100对尚未完成，下一优先实现typed Full/Pooled原生collector。

20:01增量2：双checkpoint worker/campaign及formal准入已实现、46项检查通过，Full200/Pooled100实际DEV各十query完整；所有当前输入/query key/噪声、初始K32匹配。旧全query池相同资格仍失败，五处后继候选差异已双端独立重构，139原件6a98ce52…保全。新较窄共同外部信息/因果续接资格在TEST前单独冻结并需formal计划显式引用，不把旧失败改名通过，也不声称固定候选池机制隔离已完成。最新回复/15块/20矩阵在responses/20261111_v12_addendum2，matrix4fb14257…；PDF/源ZIP与原包均不变。训练Full2075/pool1370仍实际运行，最终3000、50-rosters、原生并发和100对性能待补，stable_release_ready=false，未发布/上传。
