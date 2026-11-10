# WARM v11内部审阅候选

新增既有100条PB DEV成本日志的嵌套检索阶段核算，原始13项及逐query数据完整保全。完整retrieval median/p95为38.44/43.65ms；mean524.27ms完整重规划=39.11ms检索+485.16ms其余工作。检索内部preprocess/DINO/exact CPU search/gather median分别9.21/5.95/21.34/0.13ms。mean区间可按定义相加，median/p95不可相加；剩余工作不当成WARM独立模块耗时，未补出MemoryWAM或新收益结果。

三份英文回复/15块/20项位于tmp/rebuttal/responses/20261111_v11；Bbqv W5/Q4、Hh4E W2与20项矩阵共同指向Appendix C.4、Table26/page38。完整29文件原LaTeX工程与ZIP均已编译，独立干净ZIP重建38页文本完全相同；主文10页与v10逐页相同，参考文献从11页开始。所有原tabular和9图bytes保持，引用、overfull、匿名性通过，46条继承font warnings不增。38页contact及两张变化页37/38单页视觉验收通过。

PDF SHA1e11e07ec4de5f71a91cf6b6e201c7eb84473449586d73a5d9f64e1df72941ef；source ZIP SHA8b8514877535651baea584a98d94b442e700f9ee082c00412857caf2bc402dd1。v10及原33页不改，原q8不利结果、shared-stage六query与全部unknown继续保留。实际Full/pool训练与原q48固定池还在进行，原native pool100仅作DEV工程资格；final3000匹配TEST、直接修复/撤回、完整missing-skill、原DEV/cohort绑定与模块/MemoryWAM成本仍开放。stable_release_ready=false；未发布、未正式上传，真机新增继续延期。

封存后增量：原生pool100资格16:41:20完整10/10并正常退出，真实因果history/all-K和8+4条件路径通过；0生成命令执行/0仿真/0效能结果。46项原件136b665c…及旧51项数据/失败原件双端核验。v11原47项审阅包c62f1a87…不覆写；工程增量26项b3428130…/20项matrix f68f902f…单独位于responses/20261111_v11_addendum1与AFS writing_candidates同名目录，三份回复与PDF保持。16:47实际Full1219/pool500更新、q48固定池23sealed/16完整独立端点；均有时间戳与未完成范围，不写成final3000/完整32池。QA继承的all37标签已更正为实际验证的all38。
