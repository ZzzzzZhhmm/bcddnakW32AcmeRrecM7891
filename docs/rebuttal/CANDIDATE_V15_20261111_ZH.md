# WARM v15 内部审阅候选

三份英文回复位于 `tmp/rebuttal/responses/20261111_v15`，共2930词、15个可粘贴话题块、全部20项weakness/question。论文由原LaTeX工程 `tmp/rebuttal/revision/WARM_ICLR2027_v15` 修订；完整PDF为 `tmp/rebuttal/revision/WARM_ICLR2027_revision_v15.pdf`，29文件源码ZIP同目录。原完整v14、首版简洁稿和所有证据均保留。

本版最主要的新证据是两个独立训练final3000模型的匹配计算成本，以及Full五个模块实际执行区间。它们直接补充效率问题；是否值得这些计算，仍须正式配对策略性能和机制直接证据，不由成本接近推出。

| 新记录观测成本 | Full | Pooled |
| --- | ---: | ---: |
| 固定PB DEV测量数/额外warmup | 100/20 | 100/20 |
| median/p95完整replan (ms) | 473.4/500.3 | 476.5/501.1 |
| peak allocated/reserved (GiB) | 23.62/23.74 | 23.61/23.75 |
| 一次构造 (s，排除于query计时) | 439.0 | 420.4 |

H100 80GB、BF16、batch1、H32/prefix4/NFE10/K32、recent/summary容量20/8、bank89,680；五个最早PB DEV回合435–439各20固定前缀。未来图像丢弃、只记录已执行action更新history，生成动作不执行。固定Full→Pooled顺序和单次描述性测量，不推统计提速、最大history成本或SR。Pooled在相同信息访问/数据/训练预算下去除四个机制及对应loss，是完整方法比较，不是孤立同目标结构消融或命名baseline复现。新模型和原530.1/565.2ms WARM/FastWAM参考独立，旧数值不改。

另一次Full在相同100观测/20warmup上观测五区域，均在全部100次调用一次。CUDA流区间median/p95 (ms)：source resolution inclusive14.895/16.107、event adapter inclusive3.133/3.290、完整effect方法0.565/0.610、candidate reranker0.530/0.561、confidence-gate forward0.692/0.739。gate行排除后续eligibility/blending。区域包含父子重叠、observer enqueue/host gap，不相加，也不视作删除模块的因果节省。instrumented outer477.2/506.5ms独立保留，干净总成本不替换。detached启动器未收集OS退出码，显式null；120完整观测及500次测量调用、统计和原件均独立验证，不冒充exit0。

58原件633,133B归档SHA `14e94a59634b66de53e8390914a3092bbd763af0d6d6a3f7b9832a989dcee74e`；17模块原件324,053B归档SHA `942c27396e13d1fd4454d4314accda485f1b2cead470d5d5c139b631bc1e0834`。本机逐项成员/模型/实际args/观测调度/quantile独立核算全部通过，原件Git外保留。

论文新增Appendix C.5/p40，Table29/30。完整工程和干净29源ZIP两次完整编译均40页/10主文，原39页文本及像素全部相同，九图原bytes保持；新增p40目视验收，无overfull/undefined references，46个既有font警告。r1的三行尾页完整留档后，以压缩新增冗余叙述收为一页，font/margin/数值不变。新增主文额度留待完整正式策略结果再使用，不为尚无数据的主表填数。

PDF SHA `1fe430921cd50d96b763604b4630287ed85b9c4e007e136051da2b0886a3b287`；源码ZIP `34953fea89ee7e02233f59664a1caba044744e79ea0c623ba5c296866278d041`；20项矩阵 `9a2c418ac0ead70751adcfddf8462a294807a524aebcce97e99602efe3c1d40c`。

95项82,764,031B完整审阅包SHA `e45f94bfbd0b6b0ffdce1ba5125635426fd1b23caf90adf50b0d93437b6914b2`，manifest `530b2b98ac14225a83ced9081d067b9656c8ba22539f549becf97adb381707c5`；本机04:04:24、AFS04:07:52全部逐文件核验。AFS位置 `WARM_evaluations/rebuttal_20261109_acp/writing_candidates/20261111_v15`。包引用不可变父v14/v13完整原件，新增两成本档案、Pooled完整状态/元数据/两task12静态文件和17项分析检查收据。大权重/state保留原AFS/CAS，未入Git或重复下载到Windows。

科学截止03:28:53UTC。之后04:00:18新完成的原BT q48全部380原件另行封存并04:02:39本机独立核验：32/32 H32、0unknown、四Full精确，预测/历史/常量centered MSE0.207278/0.206218/0.128352；32交互标签全部0，selected raw/Full0/0，无完成型repair/harm或预测优势。该结果在 `tmp/rebuttal/execution/20261111_iter14/BT_Q48_COMPLETE_EVIDENCE_INTAKE.md` 列明下一版录入位置，尚未写入本版不可变PDF/回复。下一版必须同时更新Bbqv W3/Q2、Hh4E W5、C.2.6和证据矩阵，不删除负结果或改分母。

两final3000完整state/CAS及typed合同现已齐，正式RB/BT各50对仍未准入。IAM登录墙阻断云创建，04:08实际formal根不存在、时钟/TEST未启动；q96首次11h/rank20也未准入，不重复q48已完成任务。当前没有已知活动rebuttal GPU程序，云UI资源终态仍需认证后回读。AFS约5.75GB，准入前核对bounded raw/context及归档余量，保护全部原件/checkpoint。所有20项empirical_closure_claimed和stable_release_ready仍false。真机新增/录像审计延期；未发布评论或正式上传。
