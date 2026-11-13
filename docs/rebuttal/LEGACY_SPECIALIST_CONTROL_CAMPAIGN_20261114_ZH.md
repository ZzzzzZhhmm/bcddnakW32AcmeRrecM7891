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

## 材料联动

v40实证与构造值演练分别封存，回复29/42表、17评论块/20意见。实证正文没有作者提醒；另有去除HTML源码注释的纯粘贴导出，不改可见论述或任何数字。50/51页PDF和完整原LaTeX ZIP已逐页文本/像素检查，主文均10页。新的训练工程进展不能回填任何成功率、自然missing-skill或完成型repair结论。
