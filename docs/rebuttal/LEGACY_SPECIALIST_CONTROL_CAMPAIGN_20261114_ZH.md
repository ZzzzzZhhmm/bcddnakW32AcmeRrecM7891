# 原 specialist 同信息对照：正式训练与评测执行协议

此处是执行记录，不进入审稿回复正文。原稿逐任务 specialist、已有 shared3000 Full/Pooled，以及本轮新的原 v3 兼容 Pooled 是三个独立研究范围。新研究不能静默回填旧 shared3000 的数据槽。当前没有新增完整成功率结果，37 个实证字段仍未放行。

## 冻结的问题与配置

新研究针对 Press Button：在相同训练信息、共同 released base、有效 batch128、优化器和12,000次更新预算下，比较原 specialist Full 与显式消费全部 K 条记录及真实八个 history tokens 的独立 Pooled。它保留原 history、query、proprio、start-proprio warp 和双通路接口，去掉四类 assessment/adaptation/gate head 及其对应辅助损失。它是完整方法对照，不是同目标函数的单变量消融，也不是 MemoryWAM 复现。

原始 recipe SHA256 `32b8d67832c49894138fbe8808b51f8feca91a65741fb4f7056935da58cccae6`。共同 released base SHA256 `776475b22566a791854ecf31cf3b50f25e7d8d94c343132ec16eb94994aa9e63`；禁止用工程一步更新后的 pool 或已训练 Full/shared control 初始化。seed3407、AdamW peak LR0.0001、beta0.9/0.95、eps1e-8、weight decay0.01、BF16，warmup600步及全局12k cosine scheduler保持。原 Full 为4 H100×micro8×accum4，新单卡短段为1 H100×micro8×accum16；两者有效 batch 都是128，但硬件/分布式执行不同，必须披露，不称逐位相同的训练。

固定 Full reference 为 `WARM/runs/rmbench_official50_specialists/press_button-s3407-v2/checkpoints/weights/step_012000.pt`，SHA256 `a765d987b2df0d228782980592c39aaed0e837040c7f6f48c905e4641f79c285`。这是新研究事先固定的末步对照，不声称恢复了原稿 DEV 选步。

## 评测与续跑

`FIXED_EVALUATION_ROSTER_V2.json` 在任何新增训练更新/测试结果前冻结：100个配对 reset；simulator seed=340900000+i、paired policy seed=340700000+i，i=0…99。Press Button按原任务协议使用1,500 native commands，而不是 Rearrange Blocks 的700。草案 V1 的700在执行前已纠正，成员不变，旧草案完整保留。V2 SHA256 `de4dc7debcf0bf86994a9ad6bf8fd507fe321870289623caec8fdfcfba7f9c35`。先独立确认物理 reset 可重现，再进行固定12k/12k完整轨迹评测；不按结果替换样本。

全局预算12k与样本序列不因分段改变。短段只用于获取真实吞吐及可核验恢复状态；必须保存完整 weights、optimizer/scheduler/RNG/sampler/state 和 training attestation。只有 step、有效 batch、source/config/checkpoint 全SHA及恢复状态核验通过后，才能从该已知终态续跑。不得在活动 CCI 段存在时复制同一训练到 ACP。长段使用已确认镜像/AFS及闲时 GPU；ACP 当前认证超时，原登录请求保持，不重复索取。

## 阶段账与容量

新 campaign 为 `legacy_pb_pooled_s3407_formal_20261114_v1`，上限192 GPUh，绝对截止2026-11-17T23:00UTC；每个最初 CCI invocation 上限5,400秒、预留180秒退出，预约1.5 GPUh。旧143.8054115725/144 GPUh阶段与随后0.389627911/0.75 GPUh工程阶段均已关闭，不能重置或用其余额重启。新 campaign 的失败运行同样扣账，不重置上限/截止时间。192是全阶段上限，不是已申请/消耗192 GPUh。

AFS一般4GiB准入/512MiB运行保护保持；这次完整 state/weights 训练另设64GiB准入，估计56GiB存储峰值是估计而不是实测。空间以即时共享卷值为准，旧文件逻辑冷存量不等于净空闲。主机 PID1 不是 CCI 容器启动时钟；剩余四小时窗口未独立恢复，不宣称重开窗口。

## 已知第一次失败与 metadata 修复

R1位于上述campaign目录，2026-11-13T22:13:57UTC启动，22:25:03UTC failed/1，665.847315822秒，0优化器更新、0新任务命令。原训练收据代码要求冻结树存在 `.git`，在第一次训练更新之前失败。完整 source、日志、lease及所有配置保留；它不是新 SR，也不是训练完成。

R2为 `legacy_pb_pooled_s3407_formal_20261114_v2`。仅将 canonical eaf5e7d 中已存在的 `clean_git_commit` 容错函数回移到原冻结源码；不改模型、样本、训练损失或优化器。无 Git 元数据时收据的 git_commit 是明确未知的40个0，真实源码身份由完整1055文件 manifest逐项 SHA绑定，不能把0当真实commit或把组合快照声称为完整canonical checkout。无Git/脏树/绑定sidecar三个实际CPU检查均通过（3 passed）；仅本地 rev-parse，无Git远端请求或dirty gate。旧R1生成的四个Hydra运行文件在R2执行前精确移到独立preflight归档，原件保留，不计冻结源码。

准备的50步前缀在任何更新前缩短为2步，以先测真实吞吐并尽早保存完整恢复状态；12k全局scheduler与数据顺序不变。R2第一次准入因AFS约66.65GB低于64GiB而被拒绝，没有创建GPU进程或lease；门槛未降低。后续只在实测容量恢复、唯一已知前驱终态和source/config再次核验后准入；真实启动/终态以最新 `current_iteration.json` 和 admission/runner收据为准。

BT6000完成双完整副本/原生guard冷存后，22:55:47UTC实际free78,782,689,280B，唯一准入R2/operator2627762。最新source manifest为11d283fd34a4e46ca71ea9242dcb2aa3f569a33bfe33f3e4dd15e8093709907d；job646661cad1e29ccfa5841607a34b7a18824feffe2d2f25ee2abc3274436b90f5、configf34ae1688db30b82da16ec1d720e2e5ad42bce5b31f546a057b0614b8a085d2c，完整记录ITER57_FORMAL_GITLESS_SEGMENT2_ADMISSION_V3。启动不是已验证更新/12k完成，实际续跑只依赖已知终态和完整checkpoint收据。

## 首段已完成与当前续跑门槛

R2于23:13:54.079190UTC complete/0，实际1086.380175秒，完整训练收据确认global_step2/max12000、有效batch128、world1/accum16；source前后精确。原R1加本R2共1752.227490883秒，均计新campaign，旧两关闭阶段不变。训练日志证明五组正式trainable audit共929 tensors/1,069,047,916 trainable parameters有限，并按全局warmup600从initial LR1.667e-7开始；不是以peak LR直接走两步。

保存 `training/checkpoints/weights/step_000002.pt`，12,138,204,611B，SHA566c6170e44e5daec1144e81dd011163b132be56f9f968dd9c799decaa9a4880；旁有`.training.json`。完整state中model13,548,540,417B、optimizer12,828,654,705B、scheduler/RNG/trainer cursor均保存；cursor为epoch0/batch_in_epoch32，对应2×128样本。保存存在与实际成功恢复是两件事：独立0GPU full-byte审计在 `legacy_pb_pooled_step2_resume_audit_20261114_v1` 运行，随后真实resume仍须通过trainer自己的lineage/load_state核验。

AFS在保存当前weight/state后约40.25GB，低于新段64GiB准入。当前step2完整权重和state保护，不删除。新本机0GPU存储协调器V3/PID38268只对BT4000、BT2000、BRT8000做两完整副本及原生guard冷存；原V1/V2保持终态。待实际容量和完整恢复审计通过，再准备同data/global12k/有效batch128的有界续跑；不把两个更新当已训练12k control或新SR。ACP23:21重新导航明确落在signin IAM页，仍须认证，0新任务/0未知创建。

23:31:06UTC独立0GPU恢复审计complete/0，251.918秒：重新完整核验step2 weight/attestation和全部DeepSpeed state，tree SHA256 `a3f3d594a7876240b7354f44d55497427576a03e50e513406d51e07c141485c4`；前后stat不变，scheduler last_epoch2，Python/NumPy/Torch CPU/CUDA RNG齐全，cursor32正确。仍未执行未来trainer的真实load_state，不能把此预审说成已经续跑。

新2步的1089个小型source/config/log/receipt/resume-metadata原件本机与AFS全成员验真，TAR8c82292f3b9fb58c9a16b15b8511c86fe64f5885ffdb177aa106356a9b9fbafd、manifest3fc4c47768d27351450ea12be4cb5795a8cba0c920a150154afe8d1984f72def。大weight/optimizer/model payload不在该8MB包中，它们在AFS原路径保护并由weight/tree SHA绑定；不得误称全部39GB已复制到本机。

## 材料联动

v40实证与构造值演练分别封存，回复29/42表、17评论块/20意见。实证正文没有作者提醒；另有去除HTML源码注释的纯粘贴导出，不改可见论述或任何数字。50/51页PDF和完整原LaTeX ZIP已逐页文本/像素检查，主文均10页。新的训练工程进展不能回填任何成功率、自然missing-skill或完成型repair结论。


## iter58：实际续跑至66及注册下一段

正式step2→66实际续跑完成与全恢复状态核验

同一新原specialist campaign的R3（legacy_pb_pooled_s3407_formal_20261114_v3）于2026-11-13T23:59:13UTC启动，11月14日00:40:46.555653UTC complete/0，实际2493.209521595秒，新增64次更新、global66。日志00:19:27明确完成optimizer/scheduler/RNG/sampler恢复，继承epoch0/batch32；source1055文件manifest 11d283fd34a4e46ca71ea9242dcb2aa3f569a33bfe33f3e4dd15e8093709907d前后精确。共同base、全局12000/warmup600、数据、seed3407及名义effective batch128保持，原4卡和新1卡执行差异不隐藏。

当前step_000066.pt SHA5d4674ca350cb000eb99eb8742e568a9ed24adec53ee9241e20bab97e7e2620b，12,138,204,611B。00:56:27UTC独立0GPU审计complete/0，全部state前后stat不变，global66/epoch0/batch1056、scheduler66及Python/NumPy/Torch CPU/CUDA RNG齐全；完整state-tree SHA4a9c9835efbbe7f7e57f89db77ca503434fd2c14805f676e6b78f5812e9daf22。父step2树a3f3d594…及weight566c6170…严格绑定，step2与66所有大型payload继续保护。

1088个完整冻结source、config、日志、收据及小型恢复metadata在AFS和本机逐成员size/SHA全部通过：TARff94d46f247dfa68b66cc2fb9c1e845605dfd73785d19a8dcd97f8b7fd08cdb9，manifest e7d59a9b071d00ac715be322e6be17081ce63992b35b635edc80f2334cb97dd9。原生P/ITER58_FORMAL_STEP66_RAW_V1.tar；本机execution/20261114_iter58同名。三个大型weight/model/optimizer不在7.73MB包中，仍在AFS原路径，并由完整weight/tree SHA绑定。R1/R2/R3正式累计4245.437012479秒；不能称完成12k或产生新SR。

### 23.349 固定四场景初始重置：原失败保留与A–A–B–A通过

按已冻结100个PB种子的前四个case0…3，不替换seed，独立进行四组A0/A1/B/A2初始重置，B固定为下一个case modulo4。未加载learned model，0policy query、0native command、未做expert admission或指令生成。第一版未设置已知图形进程环境，16角色均未得到初始观测，complete/0仅表示错误核销完成，不能称通过。第二版恢复匹配580.95.05的进程局部Vulkan/EGL库后，因缺multiprocessing主模块guard，CuRobo spawn重入并竞争operator.json；该operator不可信，已知-15终止、339.157803秒，连同首版33.611074秒共372.784130秒，原0.25GPUh phase关闭，不重置。

第三版只增加main/freeze_support guard，保留同样case/seed、模拟器及图形配置。独立新0.25GPUh工程phase于00:58登记，01:12:43UTC exit0，实际保守计867.680049秒；16角色全部完整，4/4组A重复观测、任务变量、articulation qpos/qvel身份完全一致，B均改变身份。模拟器source前后精确；它只支持这四个初始场景reset可复现，不是完整100case policy/动态state恢复资格，不是SR、repair或memory收益。两次失败和全部raw/log/config一并保存，40成员TARf36d8c4fbc4815c2059d55a60ad0ab5a370965edbc8899374f22365b26bf741f，两端逐成员验真，原生P/ITER58_RESET_ALL_ATTEMPTS_RAW_V1.tar。

新campaign连同上述reset工程实际保守计5485.901191603秒/约1.523861GPUh，192GPUh上限及2026-11-17T23:00UTC截止不变；旧144/0.75阶段继续关闭，两个0.25工程phase独立记账，无复活或回拨。

### 23.350 step66→256注册、跨epoch游标与CCI窗口保护

下一R4仅改变output_dir、resume、run_steps=190，target256而全局scheduler仍12000；1055 source SHA不变，共享recipe d7ea9a8471ca54b226d280e0da72ef704330ce1dfe56d0bc933268b7ba4b1580。CPU配置/Hydra及真实ResumableTaskEventBalancedSampler+Accelerate1.12控制流资格通过：23913样本、micro8，从66/epoch0/batch1056到256将跨epoch0的2990 microbatches/global187，最终epoch1/batch1104。末尾不足完整microbatch accumulation的更新不称精确128样本；128为名义有效batch，不改变原训练程序。

R4于01:15:59UTC注册、未启动，job6e8d8b87b72e6112f467a899aed71534a4d667ec8c676f82118899f31ba4f816，configc5717be3edc181d58e06c7116e8452bbd7f415d0cfdc3c471c6ee3aac80bc2b0；控制器SHA44fd5fd5699b28bde336a78fe3c6fec9471d0ce20927f26cf8836c80c84c4c41、验证器befcfb6240ecdf1dfa3c92b1b65706e18b7c3ab814f047cf8757df7f2d69f7cc。预约2GPUh，训练5400秒+退出180+CPU验真900，实际仍计elapsed。

原生root bash2623968与主sshd2624502同namespace，start分别21:34:17/21:34:31UTC；它们作为保守四小时窗口代理，不把host PID1或新脚本启动当新资源起点。当前窗口不足完整下一段，准入明确resource_window_waiting_no_launch，0新GPU进程/0lease。独立本机只读协调器R4_FRESH_WINDOW_WATCH_V1最多45分钟等待真实新entrypoint/sshd与足够6480+120秒余量、64GiB及GPU空闲；只有ready后允许一次准入调用，回执丢失为unknown且禁止自动重试。读其current.json恢复，不重复启动R4。

ACP01:06:47UTC实际重新导航后落signin IAM，AUTHENTICATION_REQUIRED；仅保存状态、不读登录表单或任何input值，0新ACP/0未知创建。首次导航的HTTP200/空rows是重定向前中间态，不构成认证恢复。Tabbit正常finish、closed0、forcedCleanupfalse，用户登录页保留；原登录请求不重复。


## iter59资源clock纠正（覆盖此前SSH代理推断）

纠正SSH辅助进程作为CCI资源时钟的错误前提

01:56:15UTC只读核查明确：此前root bash2623968的子进程2624503实际是sleep12h，并非CCI分配起点或四小时倒计时。该SSH bootstrap的启动时间不能作为平台资源窗口。此前R4等待新entrypoint的门槛据此不成立，造成额外等待；不能继续将这项推断当硬件事实。原生ITER59_ENTRYPOINT_METADATA_KEYS_READONLY_V1保留实际sleep参数与进程身份（未打印任何环境值）；只读自Pod时间查询在标准域名/平台配置10.96.0.1端点均不可用，没有绕过TLS或扩大查询对象，平台剩余寿命仍未知。

01:57:40UTC核实唯一watcher62756命令及admission_calls0后明确停止。随后原生read-only再确认R4无OPERATOR_ADMISSION、ACTIVE_LEASE或runner；其注册、source/config全部保留，不能称执行失败或重复启动。下一R5从同一完整step66接续到256，不改12k、数据、seed、优化器、名义batch或source1055 manifest11d283fd…。

新增纯标准库cci_runtime_budget只约束实际可控制的单次训练5400秒+退出180+CPU验真900=6480秒，小于作者单次4小时限额；不假设当前时刻开启新资源窗口，不把SSH PID时间当分配clock。若确有带完整观察SHA的认证平台allocation deadline，按真实剩余时间拒绝；未知则明确记录未知并允许有界checkpointed调用，保留外部抢占/中断风险与全部失败收据。容量64GiB、512MiB运行保护、唯一运行、已知前驱、完整state验证、新192GPUh cap及11月17日23UTC截止均不变。

0GPU原生回归于02:00:51UTC完成10 passed，2.467秒；source SHA67f23e904f642a72569bd10c4e71b0ba63374b57421627e1a5b978be639d432e，test SHA39ce61e524cab7f258f46b1ca218c1c62bc6fcde92dc3395ec181db83e21e7a4。覆盖未知寿命不造clock、明确平台短deadline拒绝、SSH观察拒作allocation证据、完整退出/验真预算和单位/时区约束。本机缺pytest的前置失败保留，未临时安装或改模型。新helper只影响operator准入，不改原Full或新Pooled数学；模型source及正式已完成global66不变。回复/论文v40、37+6null和所有负结果保持，无新SR。


## iter59实际续跑与条件接续

### 23.354 R5实际启动与完整step66恢复后继续更新

R5 legacy_pb_pooled_s3407_formal_20261114_v5于2026-11-14T02:08:07.284731UTC在CCI唯一准入，AFS125,142,462,464B，operator2645124/child2645127/GPU worker2645174。jobc37042d7fdc2fc926f47dc8ae7f77aad26f60394824dbbe9d34a6931eee13a8c、config6ad0a9af10feb8e2ed9df117bd6d590033361b4250bbd39d8bb3c56a4ca41070，controller8465bf6ea87fa2e4badfd8ff780a85abbd924b904e1de3e639f7992e4dd106c7，source1055 manifest11d283fd…及helper67f23e90…均精确。只有output_dir/resume/run_steps=190变动，parent66→target256，全局12000/seed3407/名义batch128及shared recipe d7ea9a84…不改；5400+180+900=6480秒、预约2GPUh，新192GPUh/绝对截止及旧关闭账不变。

02:48:47UTC快照确认数据恢复epoch0/batch1056/sample_offset8448，并已通过load后state重新哈希、实际进入后续训练；日志最高step160/12000，GPU49,480MiB，梯度有限。原生P/ITER59_R5_LOG_SNAPSHOT_V2.log及本机同名SHA75303e56e44b574b78a4c80d1afd1ab56be7fdfce664088a3f8129b03ca075ba，是带时间的活动前缀，不是完整训练终态；最近完整持久checkpoint仍是66，不能把日志进度当256保存成功或新SR。

另0GPU只读读取既有scheduler.bin：step2 last_epoch2、LR4.994444444444444e-7；step66 last_epoch66、LR1.1148333333333338e-5，均保留warmup600、base LR1e-4、后续cosine11400，训练日志step120 LR2.01e-5继续延伸。此前4.994e-7是step2恢复时LR，不能误当step66学习率；不存在重开warmup或替换全局12k预算的证据。

### 23.355 事前冻结一次R5→R6自动续接，保护并发和未知状态

02:39:01UTC、R5仍训练且256终态未产生时，冻结R6 conditional plan SHAe2d8d5f81affa184ae2f5ea509def65f94c47bd4777afdebc1f510af6a7562fb。它只在R5 complete/0、step256及全部checkpoint审计通过后，尝试一次同source/recipe的256→640操作段（新增384更新）；这是按运行效率预定的保存端点，不是性能选步或改变最终12k评测。0GPU真实sampler/Accelerate1.12控制流预审256/epoch1/b1104→640/epoch3/b1264通过，跨epoch边界global374/561；实际state仍须复核。

本机隐藏follow-up PID6452于02:40:06UTC启动，最多7200秒只读等待，当前mutation_calls0、新GPU0。只有已知父终态后才发一次native审计/准备/准入；任何未知或回执丢失不重试，先查R6 root、parent audit和admission。R6单次训练9000+退出180+验真900=10080秒、预约3GPUh，仍小于4小时；不假设新平台窗口，64GiB/512MiB和原192GPUh/截止不变。读取execution/20261114_iter59/R5_TO_R6_FOLLOWUP_V1/current.json及P/ITER59_R6_CONDITIONAL_OUTCOME_V1，禁止agent另建同一R6或将等待程序当作活跃GPU任务。原R4完整未启动注册保留，旧watch已明确停止。

### 23.356 完整step2恢复包只读备份与材料/认证状态

02:26:08UTC本机隐藏0GPU backup PID64268开始完整step2 weight+DeepSpeed state备份。原生先全SHA复核weight566c6170…和state-tree a3f3d594…，再直接stream TAR到本机，额外AFS大型临时文件0；随后需逐成员验真并独立展开完整raw恢复树，只有两份全SHA都通过才记完成。当前状态native_hashing_or_streaming、记录时流入9471039488B，不能宣称两副本或释放容量。原weight/state、全部12000/14000/3k/CAS/raw/unknown不修改，不创建cold locator，不删除或迁出任何新checkpoint；未来保留策略仍无批准。

ACP02:52:14UTC实际重新导航后仍落signin IAM，未读input值/未存登录表单，0创建/0未知；正常finish closed0/forcedCleanupfalse，原请求不重复。v40实证/演练回复、两完整PDF/ZIP全部未改，37原字段+新study六字段继续null；工程进展不放行为效果、repair或SR。元数据打包的一次本机PowerShell转义/随后missing-file调用失败发生在读取执行前，模型训练未受影响，已改为UTF8脚本文件方式并成功保存快照；不计为新policy失败或GPU实验。


## 已完成256与实际接续640

### 23.357 R5已完成global256，完整state审计与一次R6实际准入

R5于2026-11-14T03:16:30.524961+00:00 complete/0，4102.885685593秒，实际新增190更新/global256，source前后精确。weight SHA43f685febae321e8235df119ce540bcf270539a7b11b6d84ed99eca11a7e4a52；随后原生0GPU完整state审计通过，tree SHAd299e25bde9584032a5226c8ebf35d23f548a45b4431991c44984282b250e600，global256/epoch1/batch1104/scheduler256、RNG齐全，符合事前CPU跨epoch预期。源文件、配置、日志、训练收据及小型resume metadata合计1090成员，本机与AFS逐成员size/SHA全部精确：TAR275ecae1c3d5073859177241b456b756d1e9487b863bdbbe2ad332ffe9f54364（7,772,160B），manifest7f0f2fc6e455c58b8b3b2248f8691e0812545cc7f4834a1b0f1195500f95db0b。原生P/ITER59_FORMAL_STEP256_RAW_V1.tar；此包不包含38.5GB大型weight/model/optimizer，三项在AFS原路径保护，不能误称已完整本机备份。

唯一follow-up于03:16:51UTC读到已知父终态后只调用一次，完成完整state审计、source/config/recipe/Hydra预检与64GiB准入后，实际启动R6；本机PID6452已退出0，mutation_calls1，禁止重启或再次提交R6。R6位于legacy_pb_pooled_s3407_formal_20261114_v6，2026-11-14T03:21:44.525106+00:00启动，parent256→target640/新增384更新，job SHAaac46793a022aa7b65c2ee7a23d4e2a919ea458b171362918acfa246bea8b24a，source1055/manifest11d283fd…和shared recipe不变，parent weight/tree严格绑定上项真实SHA。9000+180+900=10080秒、预约3GPUh、名义batch128、全局12000及原192GPUh/11月17日23UTC截止保持，不假设新平台窗口；启动仍不等于已完成640或已验证下一次实际load_state。

新campaign完成段累计9588.786877196秒（正式8348.322698072秒+reset工程1240.464179125秒），R6只记录当前reservation，待其实际终态再累加elapsed；旧144与0.75关闭阶段不改。完整已保存训练进度256仍不是最终12k对照或新SR，37原字段与新研究六结果继续null。另一次本机sealer生成命令转义失败在执行前被阻止，改用独立UTF8 builder后封存成功；没有新GPU调用、未改正在运行的模型或数据。


后续03:52:25UTC只读核查：R6日志明确显示parent256真实load_state、dataloader cursor恢复与run_target640/384新增更新，并已执行到日志step290（03:49UTC；日志进度不是持久checkpoint）。H100实际计算进程2648971、49,480MiB；source1055逐文件不变。最新完整已保存checkpoint仍256，目标640尚未完成；AFS可用79,476,994,048B。见Git外execution/20261114_iter59/ITER59_CAMPAIGN_PHASE_LIVE_V5.json。


## iter60在线评测前置CPU资格


### 23.358 固定原demo动作表示、真实保存边界及legacy online配置的0GPU资格

在R6继续训练时，使用同一1055/11d283fd冻结source开展只读CPU检查（新GPU/模型构建/任务命令/正式contract/SR均0）。已封存的原训练index0对应dataset0/episode270/frame0，原action tensor SHA6086604f…；直接读取同一parquet首32条真实动作，按原RobotwinQposZScore前向编码及反向解码，最大误差分别0/5.96046448e-08，均小于2e-6。此固定记录支持训练与旧在线action仿射表示一致，不支持物理成功/完整adapter正确；物理初始seed仍未核实，严禁把episode270当seed270或擅自teacher replay。

从真实冻结trainer AST提取原periodic/final保存控制流，4例通过：1999最终1份、2000最终复用periodic仍1份、跨2000再以2048终止需2份、12000最终复用仍1份。没有真实写checkpoint或修改trainer。下一段若经过2000，应预先固定在2000结束或实测足够双份峰值，不能降低64GiB准入。实际原生产derive_online_query_seed对固定100-case roster的0/4/1496共300 query检查通过：各case使用事前paired_policy_seed，独立CPU generator在无关全局RNG消费后Gaussian逐元素相同，300种子无冲突。这里仅证明seed函数/独立generator，不称Full/Pooled真实sampler配对通过；未来每case online contract应绑定该policy root，训练seed3407与物理seed独立，不改roster de4dc7de…。

新增独立configs/sim_rmbench_legacy_pooled_memory.yaml，仅指定已训练中的原v3 Pooled factory及四项移除loss=0。真实旧deploy composer（SHAbaabaeca…）的函数原文通过AST提取执行，Full/Pooled解析只相差factory+四loss这5项；data/camera/state/action/history/inference配置全一致，prefix4/history chunk4及policy root保持。标记路径仅用于0GPU配置工程检查，未制造或绕过正式contract/attestation，完整模型加载与online rollout资格仍须后续补齐。活动训练source不增加此文件、不变更11d283fd身份。

CPU intake V1的AST selector误选了两个同条件早退，V2 fake logger缺accelerator属性；两次已知工程失败原样保留，V3仅修harness后04:07:22UTC complete/0、13.372609秒。配置资格04:09:19UTC complete/0、1.688297秒。原件/失败/程序/解析配置53成员，原生和本机逐成员验真：TAR SHA2564a34e7c1cec58584c0e296581d9acf24b4a6691265524ece7ef4f35ce60af（276480B），manifest22d89f5a95a810d8b15724a9f1c02b8f1707eb97606a8afc9cc5e0d1b7f6f5f0。AFS P/ITER60_CPU_RAW_V1.tar，本机tmp/rebuttal/execution/20261114_iter60/CPU_RAW_VERIFIED_V1。04:10UTC R6日志已380/epoch2，实际GPU进程2648971/49,572MiB，最近完整保存仍256、target640未完成；AFS79,460,065,280B。37+6未测结果继续null，v40回复/PDF/ZIP不变，旧41对0–0/13unknown及负结果保持。step2完整备份仍只读进行，不计完成、不删除新checkpoint。


## iter61已完成640与永久可恢复存储


### 23.359 R6 complete640、完整step2双备份及可恢复冷存

R6于2026-11-14T05:13:00.574763+00:00 complete/0，6676.052649107秒、384新增更新/global640；weight SHAb32a3bdc242ce83c4c43d4da091d14479671c8f21d9c6c41b7426e04a8b25213，parent256 lineage/shared recipe/source1055/11d283fd前后一致，cursor epoch3/b1264。完整模型/optimizer/scheduler/RNG已写出，独立parent640 full-state audit正在进行；tree SHA尚不先填。该训练prefix不是最终12000对照或新SR。新campaign已完成累计16264.839526303113秒，旧144/0.75关闭账不变，192GPUh及11月17日23UTC截止不重置。

此前0GPU step2完整备份已04:21:05UTC complete，9成员、38515456772 payload B、两份完整本机raw/TAR；TAR 38515486720B、SHA7a14b0e5c616c89c99e89df5d174a55db7a778f449058ba53d0994bc2e4e240c，state-tree a3f3d594…、weight566c6170…保持。两副本位于同一本机物理盘，此处“两份”仅指两个完整文件表示，不声称不同失效域。iter61再次全量SHA重读两副本，proof SHAffb05ced1e3d968550ef82bd0668217cc619a0304cd0b846a86cdbe1798fd664；原生9件全SHA/stat再审209.652617秒通过。

按此前已授权的无损可恢复冷存流程，先建立永久full-manifest/双副本索引/恢复helper及原路径旁pointer，再检查无活动FD/不被当前parent使用，于05:14:44UTC仅释放R2已完成step2的三份冗余热存大文件，共38515399733逻辑B；其完整内容在上述两份永久副本中保留，6份小型原metadata仍在AFS，step66/256/640及所有原specialist12000/14000/shared3000/CAS/raw/unknown未动。这里不执行尚未批准的未来新快照不可逆舍弃策略。永久locator P/checkpoint_cold_storage/step2_full_20261114_v1/COLD_LOCATOR_V1.json SHA8de2c52ed11722a48840f74957ccae4306a321f124447a3b3d70b2b0f75bd3a8，helper SHA12648a18fef29a78222c594b4df1713971213c34a6562b2b202c26183ef2ed5e；完整路径、9件哈希、逐项释放收据和恢复staging路径均已保存。即时df缓存未刷新，不能按逻辑B推算可用容量；随后实际df显示76108910592B，重新满足64GiB。step2备份原current仍记录其04:21只读完成时现场，新冷存状态以独立收据和最新current_iteration为准。

### 23.360 真实checkpoint角色metadata资格及下一操作段固定

0GPU通过mmap读取真实原Full12000和新Pooled256的checkpoint metadata，调用实际原v3与新Pooled类的配置/schema验证器：正确角色2/2通过，错角色2/2被拒绝；05:13:27UTC complete/0、26.577122秒，未构造7b backbone、未初始化CUDA、未执行policy query，未再次全量读取12GB权重SHA（绑定此前完整权重证明与当前stat/attestation）。这只证明角色schema/config不混用，完整online权重加载、正式每case contract及sampler资格仍须补齐，不能写成online rollout或SR。

下一R7操作性保存端点640→1280、640更新在R6已知终态前发布（05:10:49观察parent resuming_or_training），plan SHA9b2475c4a6cb03ddb2b06d13ebf48690463949047e93d2fcd4c3c1ef49563441；真实sampler/Accelerate1.12的CPU游标测试已通过，预计1280/epoch6/b2528，跨748/935/1122三epoch边界，尾batch不声称完整128。11400训练+180退出+900验真=12480秒单次bound、3.5GPUh预约，仍小于4h；不创造新allocation clock，不改12k/scheduler/训练数据/roster。正式准入仍须parent640全审、64GiB实际free与GPU空闲，当前启动结果必须读唯一R7 outcome/runner，不能由本段计划推断。ACP05:02:20UTC实际新导航仍signin IAM，input值/表单快照未读未存，0创建/0未知，正常finish保留用户tab，原请求不重复。


## iter61实际640→1280接续


### 23.361 parent640全审、唯一R7实际准入与step66只读备份

parent640 full-state audit于05:20:31UTC complete/0、257.211185秒；tree SHA0b15bec45bb839b8f62c8e181fbe4ab5f4abdf9c5e78ad9ed735ec5f428aa82c，global640/epoch3/b1264/scheduler640、五类RNG键齐全。R7完成source1055逐文件身份、同shared recipe及Hydra检查后，仅一次实际准入于05:21:16UTC，operator PID2652395，job SHAe371f49ea5613ca368494e55c52a65b06cdc1c260284413e16c02d1c3f71cee4，config550c5dc5…；parent640→target1280/新增640更新，11400+180+900=12480秒bound、3.5GPUh预约，phase192GPUh/绝对截止不变，不假定新平台窗口。05:25实际GPU进程2652445/13,618MiB正在初始化，尚未在日志看到实际parent640 load_state，不把进程启动当恢复完成或1280终态。准入时AFS74,773,770,240B，05:25实测72,757,465,088B；原step2内容永久双副本、原66/256/640热存保护。

R6完整小原件+原始1055 source、parent640全审、R7冻结注册/初始化观测、checkpoint角色metadata资格及可恢复冷存索引/收据共1104成员，原生和本机逐成员size/SHA验真：TAR bc55c73586e42cae630f6f9574f1880f2172c1c5de19cf68e42750b5ea95f476（8017920B），manifest 7cf71619d362f705d926fe95d15f35b003f14dc420421ec561396304a93c38a9；原生P/ITER61_STEP640_RAW_V1.tar，本机execution/20261114_iter61/STEP640_RAW_VERIFIED_V1。该小包不含任何38.5GB完整模型/optimizer payload，不能与永久完整备份混淆。

本机0GPU step66完整只读备份于05:22:23UTC启动，PID1756/SSH55692，native full hash后直接stream到本机，不产生额外AFS大型临时文件；新采集bound9000秒，当前未完成、不计容量，原生checkpoint66不改动。读取storage/20261114_STEP66_FULL_STATE_v1/current.json；不要重复启动。完整step2恢复路径另记docs/rebuttal/FULL_CHECKPOINT_COLD_RECOVERY_20261114_ZH.md。37原字段+新study6字段仍null，v40回复/论文/ZIP未改，角色metadata2+2与游标CPU通过均不作为效果、修复或SR数据。


## iter62独立CPU完整checkpoint资格


### 23.362 R7真实恢复并更新、CPU完整模型加载资格与一次已知工程修复

本轮05:56只读日志已确认R7真实恢复parent640的模型/optimizer/scheduler/RNG和epoch3/b1264游标，随后step650/660/…/710产生实际更新；后续06:22观测持续运行，最近完整checkpoint仍640，target1280未完成，不能把日志步当新持久checkpoint。R7 job e371f49e…/model source1055/11d283fd…/192GPUh及绝对截止保持。未重复创建任何GPU任务。

新增0GPU独立CPU资格，固定已保存的新Pooled640 checkpoint b32a3bdc…、同一原Press Button q8封存输入/历史/Gaussian，以及已核实的原index0记录（episode270/frame0）。目标是完整7b模型严格加载和一次生产sampler，不执行任何native命令，不用于正式12k/100-pair的效果或效率结论。活动训练source保持不变；CPU source在原1055冻结文件上只替换资格CLI并增加测试，共1056文件/manifest4f34a6e0…。准入要求cgroup可用80GiB，CPU自身RSS上限72GiB、虚拟地址96GiB、运行余量16GiB，4 CPU线程/nice10/5400秒bound，AFS4GiB准入/512MiB运行；CUDA_VISIBLE_DEVICES为空。

CPU V1于06:13:15唯一准入，模型实际构造后于06:19:23 failed/1、0 policy generations，原因为资格caller把FastWAM底层的strict switches传给了WarmSourceFastWAM的窄公共接口。原公共入口本就强制完整strict_model_state/proprio/legacy rejection/metadata预检和加载前后SHA，并仅在全部成功后授予checkpoint identity；因此修正资格caller为原生model.load_checkpoint(path)，再验证其授予的SHA和九类模块逐项值，而不修改或绕过模型/公共loader。原错误、0生成、配置和全部1056 source完整保留：1064成员TAR c77d2032513bc56b3d3a36e0c100a59b7445df9fd3970e3760fdfa1c741af8e3（7546880B），manifest ca4280d8addf1424cde90755397d912ffd262aa760250bdad5e15065cccc869c，原生P/ITER62_PARALLEL_RAW_V1.tar及本机CPU_V1_RAW_VERIFIED_V1逐成员验真。此为caller工程错误，不计为新policy任务失败或模型效果证据。

另一次先后顺序错误：prepare返回“仍运行”时，曾提前执行native测试（实际发现0条）和准入（读取尚不存在的job.json即失败，尚未写intent/启动模型）。0测试不计通过，错误单独保存；随后等待prepare明确exit0才逐项测试、准入。修正后新增严格公共接口回归：未授予声明SHA必须拒绝、原loader失败原样保留、路径型公共API可正常调用；合计6项本机/原生Linux均通过（0.629s/1.161s）。这些边界测试不声称完整模型已经加载。

已知V1终态核销后，CPU V2在同checkpoint/输入/协议上，仅改上述caller和3个新回归，manifest75dbfbad…/1056，job ba39677aa1b0e45e7f8b8d8ed971ee67c4868bb1c1df7009a37cd537df14541f，于06:30:53UTC实际准入，operator PID2654495、0GPU、cgroup可用123976818688B。完整加载和sampler仍须读该V2真实终态，不复活V1、不混入其0生成原件。未使用工程一步权重作为初始化，未改变任何训练/eval样本、阈值或最终12k选步。step66原只读备份继续以原9000秒bound运行，不扩展或重置；若发生已知部分流失败，保留原件并从同9-member manifest校验prefix后准备独立尾段续传，不覆盖失败收据或重复未知调用。ACP06:19:22实际新导航仍IAM auth，原请求不重复。v40和37+6未测字段不变，无新SR。


## iter62 CPU完整加载及事实采样完成


### 23.363 CPU V2完整模型加载与一次事实输入生产sampler完成

CPU V2于2026-11-14T06:40:48.371502+00:00 complete/0，591.220980132秒，原始输入/冻结source前后不变。实际构造6773524296参数模型，严格公共WARM入口加载Pooled640/b32a3bdc…并授予相同SHA；9类retrospective模块keys和值（按声明评测dtype）逐项一致。随后在封存原Press Button q8真实图像、文本、proprio及因果历史上执行且仅执行一次生产FastWAM sampler，输出有限[32,14]动作；原Gaussian逐元素一致，source与conditioning的Full-null-Full重复一致，null候选tokens为零、共用8个history tokens不变，实际pool输入梯度有限，历史/proprio/query梯度非零。这里Full/null重复检查是source/conditioning层，未额外生成两条完整policy动作序列；没有native task命令、optimizer更新或CUDA初始化。

这关闭了“已保存legacy Pooled完整权重可严格加载并在真实输入运行生产sampler”的实现资格；它是prefix640的CPU工程验证，仍不是最终12k权重、RMBench完整online contract/物理rollout或性能/效率结论，不能用于替换37原字段或新study6个结果。591.221秒包含多次完整checkpoint/输入SHA及CPU模型构建，不能写成部署单次时延。实际full_model_load和sampler1/1已完成，V1失败0生成保留，两者不混并。

完整输出、源代码、冻结配置及逐项证据1078成员，两端逐成员size/SHA验真：TAR 4f8b581e49119e82128f97aa4e02303e21a697784fae19b20092a76300b8e06d（11929600B），manifest 2d28f1624a786b0a423dfc9b3090d3cf286363b943a14a9406d0399597d1ad76；原生P/ITER62_PARALLEL_RAW_V2.tar，本机execution/20261114_iter62/CPU_V2_RAW_VERIFIED_V1。小包不含12GB模型checkpoint payload；其原生文件仍受保护。相关CLI/test已commit并push/native offline FF至3da94cdd07fc95620af066f3d2536b682c527d0d，字节与CPU1056-file manifest75dbfbad…中的两个overlay完全一致；整个CPU组合source由完整manifest定义，不声称等于该Git提交的纯checkout。

CPU已终止，不再启动相同资格。R7仍为唯一GPU训练/target1280，最近完整保存640，实际日志进度见06:47快照；step66只读全备份继续原9000秒bound/不提前计容量。浏览器备用会话核查的Cua和Sky均在初始化时因kernel-assets路径不存在(os error3)失败，尚未观察到任何其他browser的认证状态、未作Windows UI输入或读取登录值；不能据此称其他会话不存在。已有Tabbit06:19实际IAM状态保持，原请求不重复。合作专家目录本次检查最新mtime仍11月9日，无新材料，不重复采纳或追加冗余实验。v40回复/两PDF/ZIP未改，37+6仍null，原41对0–0/13unknown与全部负结果保持。


## iter63：下一操作段与完整存储接续

R7仍唯一CCI训练，07:45实际1180但完整保存仍640；新R8事前固定1280→2000、720更新，复用periodic2000，不换global12000或测试样本。真实生产sampler/Accelerate CPU游标目标2000/epoch10/b2080已验，计划485d9c1b…已发布且未创建R8 root。Windows唯一一次接续PID64140/07:17起7200秒，须先known1280 complete/full-state audit、64GiB和GPU空闲。原预算、绝对截止、平台余寿未知与所有旧失败保持。

AFS实测65.47GB低于下一段门槛，0GPU完整恢复副本与可逆冷存按独立bounded流程推进：66原9000秒copy；256新9件manifest36f807ef…/state d299…，复用20.12GB、传18.40GB；priority2657492最多900秒只协调只读reader、不signal GPU或延长任何clock；冷存PID56224每项最多一次，双完整副本再核和原生保护全部过后才释放冗余热存，所有内容永久保留。未知调用只读核销，未成功不计容量。详见SPECIALIST_CONTINUATION_AND_FULL_RECOVERY_20261114_ZH.md；没有新SR，37+6字段继续null，v40不变。


iter63 07:57覆盖：R7日志1220但完整仍640；66原9000秒copy known failed、38.02GB前缀及原件保留，priority已恢复4/4 reader。独立rescue66 V2/PID61956（3600秒stream/1800秒native）先复核再生prefix SHA再补尾；66原冷存56224已跳过/mutation0，V2独立冷存12732（5400秒）只等新两完整副本，不拥有256，不重试未知操作。每项仍full-SHA/stat/noFD/非parent/永久locator/intent，未完成不计容量、不减64GiB门槛，所有checkpoint内容永久保留。v40/37+6未测字段及固定12000/100对不变。


## iter64终态与存储连续性

R7已08:17:54 complete/0、global1280（640新增更新），weight3c7cef40…、epoch6/b2528/source11d保持，1086小原件两端验真TARaa12e3d7…；大state/weight原生保护，独立1280 full-state再审仍待R8准备器，不称12000或100对完成。累计26861.683004秒/192GPUh与截止不改。唯一接续64140原09:17 bound、一次准入，等实际容量与GPU空闲，不复制。66独立恢复2完整副本与保护后冷存均complete，旧失败保持；256完整备份08:39 complete，其冷存看实际receipt。0GPU无损独立容器pilot仅处理3个已完成历史冷存包、保留一切原路径/locator，7项软件完整性测试过，未生成科学结果，读SELF_CONTAINED_RECOVERY_CONTAINER_PILOT_20261114_ZH.md。v40与37+6字段继续不变。


iter64 09:01接续覆盖：256完整备份及可恢复冷存均complete，实际AFS约103.08GB；独立1280 full-state audit tree9c35d933…/cursor6,2528/scheduler1280通过。唯一R8已08:50:43在CCI准入1280→2000/720更新，job a2556a20/config4ad1809f/source11d，旧接续64140退出0/mutation1。GPU2661880初始化中，尚未观测parent1280实际恢复，最后完整仍1280；不重复R8、不重置12480秒/3.5GPUh/192GPUh/11月17日23UTC。pilot63360为独立0GPU只读归档，7Win/7Linux软件测试过，所有原件/索引保持，没有新科学SR或final12000/100对完成。v40与37+6字段不改。


## iter65：R8未收据中断与短段R9、已批准保留策略

本轮运行恢复以实验台账23.372–23.375及current_iteration为准。R8在1350日志后失去进程，无退出收据；两次独立非活动检查通过，但退出码/原因保留unknown，未执行先前考虑的controlled stop。最后完整1280不变，以3493.744733秒上界保守扣账后prior30355.427737秒，不能把未知耗时改写成精确实际值。R9已09:57:21UTC在CCI仅一次准入1280→1780/500更新；独立完整1280 state审计和真实CPU游标到epoch9/b1552通过，source1055/11d、数据、seed、全局12k和固定100对不改。初始化不等于实际restore，后续读真实日志；原12480秒总单次bound/192GPUh/绝对截止保持。

作者已明确批准后续新快照保留策略，授权原文及scope见CHECKPOINT_RETENTION_PROPOSAL_20261114_ZH.md与Git外ITER65_RETENTION_AUTHOR_APPROVAL_V1.json。最近两完整640/1280、活动父1280及原稿/CAS/raw/失败未知继续保留；本轮eligible0/删除0。新的R9只登记未来可能产生的1780，不能预先判定可删除。所有旧冷存/两完整副本保留，先做符合条件的后继验真/实际恢复，再处理更旧新操作性快照。

无损自包含pilot已complete、两份均全SHA并恢复导出，原路径/locators未改；ACP实际列表访问恢复，不再按旧IAM截面重复索取登录。37+6实证仍null，无新SR，v40基线保持。优先在R9实际稳定恢复后备份非活动640，再准备用ACP并行的原production online合同及独立Full固定参考评测。
