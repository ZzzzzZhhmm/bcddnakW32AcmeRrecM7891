# WARM v17内部审阅候选与本轮恢复入口

本轮先完成原件保全和存储迁移，再恢复计算，并把已独立核验的Stage A结果纳入三位审稿人的回复及原LaTeX。候选科学截止09:57:14UTC；之后到达的结果单列late intake，不静默覆盖已冻结材料。仍为内部候选，未发布评论或上传正式稿，20项科学closure和stable_release_ready仍false。

**11:35晚到补充：** q8全32/H32、三Full锚点、0unknown已实际11:21:38 exit0并本机独立验证；三份简短讨论补充及20项矩阵位于`responses/20261111_v17_native_q8_addendum1`，12成员44.260MB增量包3643bdef…已双端验证，原PDF/源码保持。centered预测/历史/常量.065133/.103004/.048779，absolute.183358/.187569/.201642；主指标未优于常量，32候选和Full局部标签均0，同状态null尚缺，不计repair/absence/NI。B holdout Full也已11:28:50正常700失败，与null成seed3408的holdout0/0配对并独立核验，driver现进入clean null。这些晚到数值不声称已进入v17已冻结PDF，后续稳定修订统一纳入；下文在跑进度保留其原时间戳。

## 本轮交付

- 回复：`tmp/rebuttal/responses/20261111_v17`，Bbqv/Hh4E/rkWY共3,281英文词，15块覆盖20项，最大块3,387字符；可直接粘贴的纯Markdown/行内LaTeX。
- 原工程：`tmp/rebuttal/revision/WARM_ICLR2027_v17`，29文件/9原图；PDF41页、主文仍10页。新Table22/p33报告固定final3000开发角色，原图/原数值及已有不利/unknown均保留。
- PDF SHA `a4e4f5d7859a5b23cf38ca979805e4c977c39a9f0d80f4ba7a8a7f0d739787a4`；源码ZIP SHA `5d57cdcabeff96271c8e33aa6921ca6ce1864dd13552e8e56cdf11380cbe9e12`；20项矩阵SHA `ab51206c471856a15a9f8820ac91c98c720d0ab07398e60b1908abc08e6d34bd`。
- 完整原工程及干净ZIP独立复编，41页文字一致；前32页与v16文字/像素完全相同，变化33–41逐页高分辨率视觉验收。13个旧table块字节不变，无overfull/未定义引用，46个既有font warning。
- 116项115,881,629B审阅包，TAR SHA `78783ccd6e6b2e0c9da76a1a8171a346b931469bc9fd25810e332f7b3fd10bc2`、manifest `95657d0f9d54db867eb5bcd84758dc66d83a3a3b6454bb921e24f9e97fde18ff`；10:55:53 AFS `writing_candidates/20261111_v17`全部成员落盘核验。原v16/父证据包保留。

## 证据解释

新固定final3000 shared模型的Stage A：clean Full/null各700命令/175H32，配对0/0；holdout Full同700/175失败；holdout null只加载，0已记录命令，退出未采集保持unknown。原06:29→09:29UTC窗口不重开。1,664项全部原件的TAR2eb7ac6d…已本机验真，旧实施失败/unknown保持。

null175次实际effective/source gate为0且source精确Gaussian；三个已存context的candidate slots为0，required gist/事实记忆保留，直接支撑“显式撤回候选影响”的实现边界。自然holdout Full仍为正gate，未取得独立allK适用性前，不计false activation，也不把单pair0/0计NI或收益。这些开发观察与原specialist成绩、提交时机制cohort、正式Full/Pooled比较分开。三个回复在BB-W4/Q3、HH-W4、RK-Q2使用同一范围、同一Table22位置。

## 已保全和释放的存储

旧Control200七state26.307GB已另复制到本机`tmp/rebuttal/storage_backup/20261111/control200/state`并逐项重读SHA；原38.438GB TAR重新hash通过，local/AFS压缩CAS保持。仅迁移两个已备份的大raw state副本26,307,288,751B，原root新增STORAGE_LOCATOR及恢复脚本。原评测weight/当前Full与Pooled final3000/数据/轨迹/其他checkpoint均不改。10:04:37实际AFS可用27,142,139,904B，后续以新live采样为准，不把逻辑字节当独占df增量，不继续删独一原件。

八个旧冻结科学源及selected operator的3,723成员9.122MB归档已本机验真；新实验源为557文件2fdbbfeed…，来自已push/sync的代码commit7309c9d。详见NATIVE_QUERY_AND_STORAGE_RECOVERY_20261111_ZH.md。所有大数据和runtime结果保持Git外。

## 计算已恢复：区分准入、排队和真实执行

CCI Stage B固定seed3408首次10:09:30.774636→13:09:30.774636UTC；plan a3f5a02e…，私有rb-40，monitor2444145/driver2444146。角色按预登记反序执行，不按Stage A得分换样本；只有B被动增加q48候选保存。10:51:16首个holdout candidate-null正常exit0，700命令/175H32/1,400物理记录、0 TOPP、success0；TAR1f095dfe…，542项及175次显式null边界10:57:51本机独立核验。它晚于v17截止，存于ITER17_STAGEB_FIRST_NULL_COMPLETE_INTAKE_V1，须在下一版连同配对结果整合，不能补成收益/NI。原driver已进入holdout Full，先查实际进度不要复制。

固定clean q8/allK首次10:15:26.693183→13:15:26.693183UTC，私有rb-30，monitor2447586/worker2447587；seal3d540098…、admission89679388…，Full/Full/K32/Full共35分支，完整公共支持才汇总。10:43已13候选+2锚点，仍在采集，无完整池新指标。为保护B首角色原2400s预算，10:45:52给B临时算力优先；名义300s、实际301.677s调度等待后10:50:54已自动SIGCONT恢复q8。两原窗口/样本/模型/指标均未改，记录于ITER17_BOUNDED_CCI_PRIORITY_V1；pause不是新科学结果或延时理由。

正式100对master首次10:11:36.607719UTC冻结，硬截止Nov12 18UTC；原设计3484ad14…、H861/C871、两final3000权重和首次12h分片窗口保持。准备器v3 2c29354d…已经冻结，勿改写。首六固定分片各10对已提交，rb30/rb40仍保留给CCI研究，后四正式分片尚未准入。

| 分片 | 当前唯一闲时UID | 首次分片截止UTC |
| --- | --- | --- |
| rb-00 | c1186e7f7fd24a5e9647e769a52ce0a8 | Nov11 22:12:01 |
| bt-00 | 96aa4fd9d473406fadfb3bd7915c9946 | Nov11 22:14:31 |
| rb-10 | ef42e49d7d9f419588312000cf075981 | Nov11 22:16:44 |
| bt-10 | bef818e6de314819bdb38ccbd4802459 | Nov11 22:16:44 |
| rb-20 | ac826871b0c642519fb7446c8d7e75ca | Nov11 22:16:44 |
| bt-20 | c74786dc829e44fa9d8898ad50d04573 | Nov11 22:16:44 |

已实际尝试两个标准集群：01e UID04e6e357…与01e_02 UID76b34e76…均“限额中”、无runner/campaign。预检查“限额不足”弹窗本身不是后端创建拒绝，已保留澄清；确认提交后获得UID、实际限额状态，10:26:59两标准申请均已停止，所有元数据归档后才转闲时，未重置分片窗口。原identity按已核销allocation保留，当前UID没有并发重复。

10:42实际云页六片都“启动中”，但AFS尚无runner；10:48 rb00 Worker明确“等待中”、无Host/Pod IP，事件`FailedScheduling: insufficient resource: nvidia.com/gpu 1`。这是云端GPU资源不足，不能写作正在GPU评测或性能已完成。保留六个唯一UID及已准备slot，不因排队再创建。实际调度成功后核对run_manifest来源/退出、物理链和完整归档，再计结果；只有云终态和实际runner/原件已核销才释放slot、准备余下四片。

## 下一轮顺序

1. 核对B和q8实际进程/心跳/退出及完整归档，保护13:09/13:15首次窗口。q8已恢复，勿再因旧pause记录保持停止。收齐固定allK再给覆盖/后果结论，负和unknown均保留。
2. 回读六UID实际worker/调度事件/AFSrunner，先核销而不复制创建；既定六slot/120GPUh/各12h保持。资源不足时继续可独立写作/归档，不把waiting当活动GPU。
3. v17已完成交付验收，接下来集中纳入B配对、完整q8和正式对照，而非重跑旧train/CAS/资格/成本/原BTq8/q48。新真机/原录像审计继续延期，不重复索取。
4. 每次提交前pull检查上游及共享文件，英文Type(module)与分点，push验证；CCI离线FF保护九个冻结源，运行程序不接触Git远端。每小时heartbeat仍ACTIVE，目标11月13日23:00稳定版不变。
