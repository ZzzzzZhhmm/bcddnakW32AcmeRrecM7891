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
