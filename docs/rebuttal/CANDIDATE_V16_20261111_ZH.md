# WARM v16 内部审阅候选

当前三份英文回复和15个纯Markdown/行内LaTeX话题块位于 `tmp/rebuttal/responses/20261111_v16`，3020词，逐项覆盖全部20项weakness/question。原LaTeX工程 `tmp/rebuttal/revision/WARM_ICLR2027_v16`、40页PDF和29文件源码ZIP均保留。原v15及全部旧候选不覆盖。

本版把上一版截止后完整到达的原BT q48结果真正录入回复和论文，补齐原训练目标与新后果测量的合同核对；没有把审计通过或轨迹完整写为机制有效。

| 固定原BT诊断 | q8 / frame32 / rank4 | q48 / frame192 / rank3 |
| --- | ---: | ---: |
| 完整H32候选 / unknown | 32 / 0 | 32 / 0 |
| 各状态Full精确控制 | 4 | 4 |
| centered MSE，预测 | 0.076844 | 0.207278 |
| centered MSE，历史 | 0.076292 | 0.206218 |
| centered MSE，常量 | 0.022408 | 0.128352 |
| 新交互标签1/0/unknown | 10/22/0 | 0/32/0 |
| 原selected / Full完成标签 | 0 / 0 | 0 / 0 |

两状态均没有centered预测优势；q48标签地板不能证明策略等效或NI。两query来自同一原来源episode，候选和重复控制不是独立任务试验，不估总体repair频率。q8的电池终点跌落/未跌落差异仍单列Table24；不把它称为完成型repair。原稿19%及expanded applicability cohort分母均独立保留。固定q96未准入，不因已见负结果替换或取消其原状态。

q48完整380原件49,017,472B档案SHA `6c92f2fe882749c3cf35c966dc6034722841cc7516552bc0ee30336d7bdec839`，manifest `2cb2ee082fe3560d47851ff11491b04ad90a1ab0904d8968080d02444e4f87b6`；前一轮已完成本机逐项物理/DINO/方差恒等式核验。现已同步Bbqv W3/Q2、Hh4E W5、Appendix C.2.6/p34/Table25和20项矩阵。

新只读CPU审计追溯全部64个候选的event ID/bank row到32个原feature-cache episode，原46,783-event bank的effect_pre/post与缓存起点/起点+32精确相同，按原uniform mean重算历史效果与事前sealed数组精确相同。原model action_horizon、bank window、demonstrated query teacher、新执行分支均32步，timing_dim4是维度；DINO cam_high/2×2 row-major pooling/BF16计算/float32存储、camera/encoder/M1/stats合同均一致。未发现这些合同错配，不改变已留存误差；历史事件factual重建不等于当前状态的反事实转移已识别。原实际retrieval tensor未保存，不声称恢复了该tensor。

64项2,440,194B审计档案SHA `49f3e061ae9a313202440c954daee6bbee673b3e80830ec0f07b5d2b5cc80edf`、manifest `168c5cc7ab6eae8f85a1235c23c3a3b23a550d4a340c5f1c5dcc45b0c96ce444`，CCI真实exit0；本机逐项全成员及64个历史均值独立核算通过。大bank/cache只在CCI核原hash，本机没有下载或冒称全tensor重hash。此次是已见负结果后的只读测量合同核查，没有新GPU/模拟/策略得分、投影拟合、指标替换或原件修改。

完整原工程与干净ZIP两次独立编译均40页/10主文，前33页文字及像素与v15一致；变化34–40逐页目视验收，无overfull/undefined references，46个既有font警告。九图原bytes和同一改动文件内12个原table块完全保留。r1有41页及一页大段空余，已完整封存后仅合并重复叙述，原数值和font/margin不变。新增Table25使后续表号顺移：C.4旧成本/检索Table28、29，C.5匹配总成本/模块区间Table30、31；回复和矩阵已按实际aux更新，成本数值不改。

PDF SHA `94c88d81d2e01777f34d8d9da7d0f78405b190e6b3c1eaecef65762a2fa45e87`；源码ZIP `427846e23c3e7f458cf9f23264ced224e971eeaf2acc66e057c27733d39dc894`；矩阵 `79a467417603892ac1a4dabf9255dae172c303445085905b882bd724623823de`。

65项127,561,493B审阅包SHA `e519361aedd3cffa103d9abe157c8d32a243d09eec0d8a6b2ff702f19c52800d`，manifest `833eba1a3703642783864b36e4cdafe4824851fda4ea8b7b78280b8ddfc5626c`；本机04:47:39、AFS04:50:07全成员落盘核验。AFS目录 `WARM_evaluations/rebuttal_20261109_acp/writing_candidates/20261111_v16`。包完整增加380项q48及64项审计原件；以SHA引用父v15/v14/v13的旧原件、完整训练state/CAS元数据及58+17成本档案，未静默替换。

科学截止04:29:36UTC。所有20项empirical_closure_claimed与stable_release_ready仍false。两final3000和静态合同齐备，固定100对/BTq96尚无正式准入；IAM会话仍待此前登录请求，正式时钟未开始。当前无已知活动rebuttal GPU程序；真实云allocation终态须认证后回读，不由CCI进程为空推断。真机新增/录像审计延期；未发布评论或正式上传。优先补直接策略/修复/撤回与自然缺技能证据，不重复已完成资格、计时或本次审计。
