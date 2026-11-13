# v39：实现归属一致的纯净回复与原工程修订稿

本版另存自封存的v38，不改原稿成绩、固定样本、失败或unknown。实证候选与独立理想结果演练继续分开；正文没有pending、do not submit、作者执行提醒或特殊颜色。演练值仍是构造例子，其段落/表格前保留源码注释，不能回填任何正式实测字段。

## 审阅入口

- 实证回复：`tmp/rebuttal/responses/20261114_v39_verified_clean`。Bbqv/Hh4E/rkWY分别1811/2338/818词，10/14/4个数据表；完整回复可由17评论块精确重构。20项原review字段、序号及hash保持。
- 实证论文原工程：`tmp/rebuttal/revision/WARM_ICLR2027_v39_verified_clean`；PDF为`WARM_ICLR2027_revision_v39_verified_clean.pdf`，50页/主文10页；PDF SHA `814e4cba2b7bf422e7d44d67014ef6b168709dbe9e14f99777bef457409c40b8`；源码ZIP SHA `2829b8cc927a6361e534f25ffc8d3705b159f11e5d643c2f7d3dbc5ed4ea3b4e`。
- 独立演练：`tmp/rebuttal/rehearsal/20261114_v39_hypothetical`，三位回复、17块、完整LaTeX/PDF。PDF51页/主文10页，SHA `eabf9fb68fca87bc233ea1ffcf447c5e7c52bcf03ea011b0a44282c4f7a05622`；源码ZIP SHA `6e55f66cb46c43344d99cb4c0f40afd438f0c2b093afde5441714dbb3f18d011`。

## 本轮实质修正

主文3.3/3.4保留requirement--proposal--influence贡献，明确history进入requirement/后果比较；将专门history-query与phase/ordinal投影、event cursor、repetition/接受规则归属到独立shared研究。Appendix A.2新增实现对照表和原specialist五特征门控公式；B.1、符号表及Hh4E W3自包含回复同向更新。没有把后续shared机制无条件归因给原specialist分数，也没有撤回共同的零gate机制。

原PB47277e5与BTf77c633的model/consequence两文件SHA分别相同：`d89796cc5da67d431d911e5f9191c5d60a14599bd3763cc64198e1384dc0b67d`、`6430424190242669d23685fdded0aff75615197cac3a9534e3c52c0497a6b819`。遍历原src/scripts未找到shared候选cursor字段；文中限定为model boundary，不凭字段缺失推断未恢复的整个原评估系统。

原PB12000 checkpoint CPU核验于18:22:40UTC complete/0，146.89秒：实际保存权重的门控首层为五输入，原context timing为四维，专门phase/history-query投影不存在；真实q8 history和proprio字段齐全，四个输入文件含12GB权重在前后全SHA/stat一致。它验证实现归属，不是完整新control策略、训练公平性、任务成功率或原DEV选点评测复现。V1因审计键名slash/dot匹配失败保留；V2只修正检查器，未改权重、输入或模型。

## 验收与保全

两套完整工程均BibTeX加四遍pdflatex，再从各自源码ZIP解包独立重编译；50/51页文本和像素全部精确。主文仍10页，10图、原47表环境及结果表保留，新增1张实现表；0未定义引用、0overfull、45条既有字体回退警告。全部页面接触表及主文5页、A.2门控和末页直接验看；匿名和AI声明保持。并发PDFium渲染发生已知Windows异常后，保留已完成独立编译，改顺序逐页验证通过，没有改数据或论文结果。

Git外iter56的`V39_QA_FINAL_V1`、`V39_REPLY_QA_V1`、`V39_FINAL_CONSISTENCY_V1`为验收入口。完整材料包67成员，含两个PDF、两个完整源码ZIP和两套回复，TAR SHA `d1e09bff71d77007de84f545342154c2710868fc46dcd512ad28a23d3b4a94bc`，manifest SHA `05d53ab7d96ccc6556dea8d815fa4e4e298407b5bf48a7936c8f30dabf5ac999`；本机与AFS逐成员验真，原生收据`ITER56_V39_FULL_AFS_VERIFICATION_V2`。v38两个PDF和两个ZIP哈希保持。

ABI原件13成员保留成功与失败：TAR SHA `c78a31bc06d447f3847470aa6b1d35d247e29dd2b9d61277fd1d008732f1c78b`，manifest SHA `8dd2efcb2a1a41ad56f502dbfbb10139ab75e13715d9428d9a06e3019dd58c5c`，本机/AFS精确。报告包排除了无关bootstrap进程参数，私有原始只读收据保持。

## 仍待补齐的证据

P1公平对照仍为41对0--0/13历史unknown；现有phase-free原数据小模块未训练，完整legacy policy/history/generation尚未资格通过，不能使用已有phase-aware control权重冒充。P2总体后果预测/修复收益、P3自然技能缺失可靠性仍开放；P4/P5等待作者采集的真机结果，新素材收到0。37个外部实证字段仍null/未放行，所有原科学资源窗口和预算保持，不把工程通过当效能收益。

下一步先完成原specialist control完整路径和真实history/proprio绑定，再为兼容训练与固定科学评估注册独立新阶段；不能重置旧窗口、重复unknown或直接重跑低效旧队列。18:19UTC ACP刷新返回401，已有表格不能算新认证核查；已记录需用户重新登录，未创建/停止/删除任务。CCI工程核验已完成，无新增科学GPU程序或策略动作。本版可审阅，未向OpenReview发布、未上传投稿。
