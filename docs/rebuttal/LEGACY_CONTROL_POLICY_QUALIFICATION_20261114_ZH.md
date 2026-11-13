# 原 specialist 兼容的独立 Pooled control

2026-11-14（北京时间），iter57。原稿 specialist、已有 shared Full/Pooled 3000 和这个新 control 是三个明确范围，不互换权重、训练收据或评测分母。

## 已实现与直接资格

显式工厂 `fastwam.research.legacy_pooled_memory_prior.create_legacy_pooled_memory_prior` 只在原 v3 冻结源码中使用。model/consequence 源码分别必须匹配 `d89796cc5da67d431d911e5f9191c5d60a14599bd3763cc64198e1384dc0b67d`、`6430424190242669d23685fdded0aff75615197cac3a9534e3c52c0497a6b819`。不修改当前默认 Full/Pooled 或原 specialist。

新 control 接收原 bank 所有 K 条动作、4 空间 pre/change 单元、777 维 context、timing4，以及真实 query、proprio 和八个 history gist tokens；四个 learned queries 汇聚全部记录。保留原 start-proprio warp、history requirement 和原双通路接口。粗排第一有效候选提供动作先验，固定权重 0.25，四个 pool tokens 直接进入条件路径；没有学习 reranker、effect assessor、action adapter 或 confidence gate，对应四个辅助损失必须显式为零。该对照的意义是完整消费相同可用信息，并不把它称为原 Full 的等价实现。

checkpoint schema 为 `warm.research.legacy-phase-free-pooled-control.v1`；原 specialist/shared checkpoint 均不能作为此 control 的训练结果载入。单元检查保留共同模块初始权重与 RNG、一致 null、teacher firewall、非首选记录可达性及反传等；原冻结源码中 **14 项通过**。最初两空间单元的测试 fixture 与真实四单元契约不一致，已只修正 synthetic unit fixture，保留首版 4 passed/9 failed；真实 bank 未改。

AFS 根目录为 `/mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations/rebuttal_20261109_acp`。`legacy_pooled_factual_CPU_20261114_v3` 已实际构造 6,773,524,296 参数的生产模型，从共同 released backbone 初始化，新 pool 5,763,072 参数；读取原训练 index0 和封存 PB q8，完成一次实际 FastWAM sampler，输出有限 `32×14`，原 Gaussian 精确。后置检查误按 `1×32×14` 校验，故该 attempt 保持 failed/1，不追改为成功。

独立 v4 仅续检该次已完成 sampler 的结果，未重复生成、未重复训练样本。20:21:28 UTC complete/0，340.588 秒；重建八 history gist 逐位精确、原 Gaussian、Full/null/Full source/conditioning 重复、null 候选条件归零、共同八 history 保持、实际 history/proprio/query 梯度非零均通过。没有补 phase、history 或 proprio 占位。原输入完整 SHA/stat 前后不变；新 control 的紧凑初始模块独立保存为明确未训练资产。

上述是完整生产接口与数值检查，**不是已训练 control 或成功率收益**。0 环境命令、0 优化器更新、0 新 GPUh；原负结果和 37 个正式未测字段不变。

v1–v3 和两套 unit attempts 共 5,309 原件，原生/本机逐成员核验：`ITER57_LEGACY_CPU_COMPLETED_RAW_V1.tar` SHA `0d37b8f2b8c3132f1d6ac617ab2e8e327416e29ac17eaeef48ee4d6a3e024840`，manifest SHA `9492ebc9635d1d98e51fb24ae0283ea24b6a6321410fc1963126eabac8edfcf4`。v4 原件保持独立，按新的 seal 收据封存，不覆盖旧失败档案。

## 下一阶段的不可混淆边界

先用 `scripts/preflight_rebuttal_legacy_training.py` 在真实 index0 上检查一次 action+video forward/backward 和 optimizer update；微批量 1、累积 1 明确只是有界工程资格，不是 batch128 的匹配训练。只保留小型 pool 工程权重，不晋升、不作为完整训练初始化。独立登记新阶段及窗口，保护已关闭 143.8054115725/144 GPUh 原账；运行程序不访问 Git 远端。

原四个训练收据均为 AdamW peak LR 0.0001、beta0.9/0.95、eps1e-8、weight decay0.01、BF16、seed3407，4 H100 × batch8 × accumulation4 = **128**；BT/PB/Cover 12,000，BRT14,000。正式新 control 必须使用原训练数据、共同初始化、同更新预算与 scheduler；若换 GPU 数，须显式匹配 effective batch 并披露运行环境，不冒充完全相同分布式执行。原稿 DEV 选点/最终 evaluator manifest 未恢复，不能据训练收据声称已经复现原准确率。

20:22 UTC ACP 已重新加载实际列表，旧 unstarted0/1 云端均成功；应用的 progress-listen 请求出现 HTTP2 错误仅为页面订阅，未进行任务创建。后续 GPU 创建必须在冻结源码、实际磁盘准入和新计划登记后进行。已确认只用闲时资源；排队与实际分配/GPU 程序分别记录。


## CUDA AMP 的实际续检（20:58 UTC）

新阶段在 CCI 单卡实际运行，原始预算/窗口保持，独立阶段上限0.75 GPUh。首个真实 action+video forward 于20:46:56 failed/1，0优化器更新、0环境命令；错误是 gist 输入dtype不一致。工程检查不是任务失败标签，原件/lease保持，不静默重写。

精确的小型 CPU AMP 检查15项通过，但 CUDA 的同一bf16测试重现失败：bridge world_tokens 是bf16，semantic_tokens 是fp32（CUDA autocast LayerNorm输出），其余真实接口使用bf16。这解释了CPU资格不能代替CUDA训练资格。保留 CPU pass 与 CUDA 1pass/1fail；仅在新 control 的严格 gist/pool 边界对齐到 native action dtype，不修改原 Full、bank、teacher、选样或阈值。非AMP同dtype路径上的cast是恒等。

修复后的独立冻结树在H100上 **16项全部通过**（含CPU/CUDA bf16 forward/backward），新训练入口同时封存实际训练样本、source输入和bridge/gist dtype，便于任何后续失败直接定位。仍须真实生产 action+video 完成一次优化器更新，才可称训练资格通过；16项小型检查不替代此项。续跑为新attempt，沿用同一个新工程阶段0.75GPUh和02:00UTC绝对截止，原失败不覆盖，所有工程更新权重禁止作为匹配训练初始化。


## 真实生产训练闭环已通过（21:14:24 UTC）

R2使用canonical eaf5e7d、source manifest2821f011…、job c6483da6…；与R1原训练index0的所有样本tensor SHA逐项完全一致。实际联合action/video forward、backward和一次optimizer update已complete/0，worker675.515秒，runner691.009秒。loss1.5941695，action1.2917665、video0.2014629、aux0.1009401；action expert、bridge、gist、pool、video adapters五组均有有限非零梯度，pool5,246,304个元素实际改变，峰值allocated22,226,856,448B。实际production hook再次确认bridge semantic为FP32、严格gist入口已统一BF16，支持修复定位。

该阶段只是微批量1的生产数值/优化资格：0新rollout、0新SR、不晋升工程权重、不作为正式匹配训练初始化；12k/14k预算、effective batch128和完整评估尚未完成。新0.75GPUh工程phase累计实际保守计费1402.660秒/0.389628GPUh，包含R1失败、两次CUDA检查和R2；该阶段终止、GPU程序为空，原144GPUh关闭账及全部旧窗口保持。

1073原件含真实样本、source输入、dtype、grad/optimizer指标和独立工程pool，AFS+本机逐成员全验：TAR5643f807312caf858b93a70ebb41adba50bfce63dacb3c4355f0252af7c50eb2、manifest23da9cbe29fd342533efbfc2d58b4467adcac8644663f7a693242b341a523f03。R1/CPU/CUDA复现及修复另4252原件全部保留，TARb02406fa8df766880dcec30b59b7b0329fa4b9cf7656326bbe6db2865c042071，不能用R2成功覆盖R1失败。

下一步为原specialist同信息控制的正式训练与固定评估，最新执行读 `LEGACY_SPECIALIST_CONTROL_CAMPAIGN_20261114_ZH.md`。本段原24.9GB及复制中状态已成为历史：PB2000、BT10000/8000、BRT10000均完成双完整本机备份/可恢复冷存，全部12000/14000及shared3000物理保持，V1/V2协调器不再活动。后续恢复覆盖另存，不追改旧失败。

新正式训练第一段在第一次更新前因旧Git attestation要求.git失败，665.847秒计入新campaign；只回移当前canonical已有的容错函数后，无Git/脏树/绑定sidecar三项CPU契约通过。真实source由完整SHA清单绑定，git字段为明确未知哨兵。R2首次准入因共享AFS66.65GB低于此次64GiB而被拒绝，0新进程/lease；先完整备份旧BT6000。是否已再次准入必须读最新admission/runner，不按准备完成推断训练活动或12k已完成。

ACP20:22页面列表展示不是独立认证API200或最新Pod终态证据；此前该句表述过强。20:32登录超时和新创建0为当前状态，原登录请求不重复发送。原生终态由各runner单独证明，不借缓存页面替代。
