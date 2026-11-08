# WARM 实验记录与论文回填索引

更新日期：2026-11-08。本文是持续维护的回填入口；原始记录、数组、运行输出按各条目的实际身份与可用性登记。每次取得数值，先保存，再核验，再更新本文。失败与不支持假设的结果同样保留。以下各历史条目的状态属于各自记录时点；投稿版本清点见第17节，作者来源澄清见第18节，rebuttal首批新诊断见第19节。下方09-24等历史快照不代表当前投稿结果或运行状态。

**最新稿件回填映射：** 作者09-24新提供的28页PDF使用表1–11、图1–4；详见[最后24小时审计与计划](PLAN_24H_20260924_ZH.md)。本文旧条目中的表17–25等是早期论文表号，数值范围不变，不应据此误判新版缺口。当前C.2.2已包含forced-null 50/50记录；表4仍未实测。09-24 14:42 UTC再次只读检查AFS，训练指标仍停在02:03:32 UTC修改时间，未见stage1最终summary/15k/natural-gate产物；没有新增论文SR。

**2026-09-24 12:56 UTC现场快照：R2四卡续训已实际恢复并记录到step3680；新增338条训练指标及1000/2000/3000三个checkpoint。训练日志最后更新于02:03 UTC，尚无最终退出/完成收据；新模型DEV50及source诊断没有产物。** 这是部分训练结果，结束状态待ACP平台确认，不能把旧`running`字段当作实时存活证明。详细数值、原始证据及适用范围见第14节；此前各节的“未启动”是相应历史时点记录。

**目前没有新增可直接回填为当前论文正式主实验SR的结果。** 已完成实现与梯度诊断、新增记录前缀在线成本测量，并核验了一批与稿件数值不一致的历史 LIBERO 结果。下表的“可用范围”是回填限制，不得省略。

## 1. 已核验数值

| 实验 ID / 指标 | 实测值 | 分母与范围 | 可用范围 / 论文位置 |
|---|---:|---|---|
| N02-DEV50 / forced-null 通过数 | 50 / 50；失败 0 | Press Button、Put Back Block 各 5 DEV episodes，每 episode 5 固定前缀 | 表23/24相关实现自检；仅 v8 smoke300、离线生产推理核心 |
| N02-DEV50 / 动作最大绝对差 | 0 | 同输入重复与更换候选后，完整 H32×14 模型空间输出 | 不能推出 learned gate 拒绝能力或任务 SR |
| N02-DEV50 / 峰值已分配显存 | 12.8348 GiB | H100 80GB、BF16、batch=1、NFE=20 | 该离线 probe 的显存，非正式在线性能表 |
| N02-DEV50 / kernel median / p95 | 0.5663 / 0.6059 秒 | 150 次调用；未设正式预热排除，含首次调用 | 描述性记录；不回填表25端到端 replan 延迟 |
| N02-DEV50 / 作业时间 | 610.33 秒 | 子进程 584.99 秒；wrapper 计入启动后开销，未含启动前输入哈希 | 约 10.2 分钟；先前 15–25 分钟是预算 |
| N06-R2-DEV50 / 自然门控激活 | 0 / 50；mean g=0 | 同一10 episodes / 50前缀，三个独立source进程 | 有效激活覆盖为零，不能证明source内容收益 |
| N06-R2-DEV50 / 三种source差异 | source RMS=0；动作RMS=0；mean[c²]=1 | Full、scale-only、Gaussian均相同 | 自然g=0下的退化结果，不是三种方法等效性证据 |
| N06-R2-DEV50 / 作业时间 | 1662.94秒（27.7分钟） | complete / exit0，前后源码哈希相同 | 服务器与本地数组复算均通过 |
| W08-DEV200 / 在线replan median / p95 | 0.4999819407 / 0.5537580627 秒 | 单H100、BF16、batch1、NFE10、H32、K32；5 DEV episodes×40次，另20次预热排除 | 表25/C.5的受限成本记录；smoke300、记录观测/动作历史回放，非闭环SR |
| W08-DEV200 / 峰值allocated / reserved显存 | 23.620100 / 23.740234 GiB | 同上；CUDA allocator统计，含常驻模型与在线检索编码器 | 非整进程NVML显存；不能与NFE20离线kernel测量直接比较 |
| W08-DEV200 / bank与历史规模 | 89680 events；每query32个候选；历史token 0–64 | 实际历史event 0–14、action summaries 0–8 | 这是历史读入token数，不是完整DiT序列总token数 |
| W08-DEV200 / 自然gate激活 | 0 / 200；alpha恒为0.119140625 | stagnation范围0–0.6328125；原阈值0.15 | 描述本次工作负载，不补独立标签FA/TA；不与N06重叠前缀合并样本量 |
| W08-DEV200 / 作业时间 | 943.15秒（15.72分钟） | complete / exit0，源码前后相同；含初始化，未含单独contract准备 | 模型脚本内部elapsed为911.99秒；预检与失败尝试另记 |
| R2-stage1 / 已记录续训进度 | parent300 → last logged3680；338条，步距10 | 原四卡shared recipe，目标15000，总schedule30000 | 已有真实优化记录；非目标训练完成，终止状态待确认；见第14节 |
| R2-stage1 / action loss | 首20条0.049257 → 末20条0.009741 | step310–500与3490–3680的记录点算术均值 | 训练轨迹描述，不是DEV泛化或闭环成功率 |
| R2-stage1 / raw learned gate | 首20条0.096301 → 末20条0.177411 | 同上；训练期`warm_learned_gate_mean` | 不等于新模型DEV自然g、TA/FA或接受能力 |
| R2-stage1 / 数值与强制拒绝泄漏 | 338/338条全部数值有限；记录的forced-source-leak均为0 | 每10 optimizer steps记录一次；不声称检查了每一步张量 | 支持这段已记录训练的数值/诊断检查，不替代forced-null推理测试 |
| L19-Spatial / SR | 488 / 500 = 97.6%；失败 12 | 历史 step019100，10 tasks，root seed 3407 | 历史结果，稿件归属待核实 |
| L19-Object / SR | 496 / 500 = 99.2%；失败 4 | 同上 | 同上 |
| L19-Goal / SR | 483 / 500 = 96.6%；失败 17 | 同上 | 同上 |
| L19-LIBERO10 / SR | 469 / 500 = 93.8%；失败 31 | 同上 | 同上 |
| L19-四套件合计 / SR | 1936 / 2000 = 96.8%；失败 64 | 每任务50次，40 tasks；一个评测 root seed | 描述这批历史记录，不能写成多训练 seed 均值 |
| L19-pilot / SR | 47 / 50 = 94.0%；失败 3 | LIBERO10 task0，seed17，pilot-s17-v4 | 单独保留，不并入2000次，也不作为第二完整 seed |

N02-DEV50：候选选择在推理前冻结，不按结果挑样。每个前缀先重复相同输入，冻结 `max(1e-6, identical max error)` 容差，再替换候选；source 和 candidate conditioning 双路强制归零。前缀50个来自10 episodes，不能当50个独立episode计算区间。无SR、无训练seed均值/标准差。

N02-pilot01 也保留：1 episode / 1 prefix，通过；最大误差0，子进程500.28秒，单query三次推理5.55秒。它与50前缀检查有重叠，不合并增加样本数。

## 2. 论文汇总与历史原始记录存在差异

| 套件 | 当前文稿/仓库汇总 | 本次恢复的历史原始计数 |
|---|---:|---:|
| Spatial | 492/500 | 488/500 |
| Object | 500/500 | 496/500 |
| Goal | 493/500 | 483/500 |
| LIBERO-10 | 489/500 | 469/500 |
| 总计 | **1974/2000（98.7%）** | **1936/2000（96.8%）** |

相差38个成功、1.9个百分点；这是不同归属记录之间的算术差，**不是配对方法改进量**。没有证据证明这批历史评测就是论文所用运行，不推断稿件结果错误，也不能把98.7%标为已复现。下一步需要找到1974个成功所对应的逐episode产物及checkpoint；找不到时，应明确撤回“原始证据已验收”的表述，并由最终选定的可追溯实验决定回填值。

## 3. 身份、核验与存档

服务器公共前缀 `S=/mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations`。

本地备份前缀 `L=C:/Users/admin/.codex/visualizations/2026/09/21/01a0c353-7ccc-7c51-9310-339904508ea2/warm_sprint`。

### N02-DEV50

- 原始输出：`S/nonreal_full_null_20260921/dev50`；日志：`job_logs/20260921T115134Z-3c3f9b80`。
- 完整归档：`S/evidence_archive_20260921/full_null_complete.tar.gz`；本地同名文件位于 `L`，并已解包为 `L/nonreal_full_null_20260921`。
- 归档 SHA-256：`82817555e20472543dbb918459a5679d325b6d704497e670accd3fe09bbbc570`，服务器与本地一致。
- 模型：shared v8 step000300；checkpoint SHA-256 `ab4042b1a761f53ec5f5379ce461f5a2f6f3decf89e3f3a16322b66894918a1d`。
- 运行源码集合 SHA-256：`a40222fade015a5c238add92cf6135b77b24ea3b4a9dd229f8a22d81ec8073fe`，运行前后相同；源码已随归档保存。对应实现提交 `4a7c864`；源码集合哈希才是此次运行的精确身份。
- 配置、normalizer、bank、query corpus、prefix manifest 的哈希保存在 `dev50/manifest.json`；实际配置、50条逐query记录、100份probe和动作数组均已备份。
- 实际验收：wrapper complete / exit0；本地独立读取动作数组复算50个误差，核对前缀身份、probe哈希、source==Gaussian、g=0和conditioning一致，全部通过。
- 复算命令：`python scripts/verify_nonreal_null_archive.py <L>/nonreal_full_null_20260921`。不使用 `python -O`。

### L19 历史 LIBERO

- 原始批次：`S/batches/step_019100/{spatial,object,goal,libero10}-formal-s3407-v2`；pilot批次为 `pilot-libero10-task0-s17-v4`。
- 归档：`S/evidence_archive_20260921/libero01`，本地镜像 `L/libero01`。`files.json` 记录每个原文件路径、字节数、SHA-256和压缩备份路径；`audit.json` 保存41个任务运行的复算结果。
- 每个任务的原始 result JSON（含逐episode/replan telemetry）、配置、online contract、兼容补丁身份均已压缩保存；共约127MiB。174条文件索引、136个去重文件已在本地解压并逐一复核字节数和SHA-256。不上传原始运行产物至Git。
- 逐文件原始哈希与旧completion记录一致；success/failure集合完整覆盖所有episode，无重复，逐episode布尔结果、终止原因、replan数与汇总一致。正式40任务与pilot1任务全部通过这些计数检查。
- checkpoint 身份由原始runtime/contract共同记录：`de324de810df74bd7ff16114c4cf22e90c470bcdadcc7995fdbac9b494e1bb64`；本轮未重算12GB checkpoint实体哈希，不能把记录内一致性写成checkpoint独立验真。
- 历史代码：`c4763a975298de6f00939360551616af7902d57a`；正式批次中32任务兼容文件SHA-256为 `f392a0ac5d13cce7156a32d8a091d6682564a2f5df5966a054750c44b7673a59`，LIBERO10 task2–9这8任务为 `44fdee2d72d7b34aa11f1570bb7b4ad1f6aa7a2ef1b0e396ecc723b328b92d69`。两份文件仅launcher commit不同，声明的patch、patched source和namespace相同；没有把它们隐藏成单一launcher版本。pilot使用另一版patch，单独记录在索引中。
- `comparison_kind=full_retrospection_single_checkpoint`，`source_policy=fixed_context_top1`。当前方法/稿件等价性尚未核验；不能仅凭运行名称写成最新Full WARM主表。
- 审计脚本：`scripts/archive_nonreal_results.py`。这是原记录计数/哈希核验，没有重放模拟器，没有baseline配对，也未核验历史数据泄漏边界。

## 4. 已完成：N06-R2-DEV50 source 局部诊断，激活覆盖为零

**首轮失败记录：**`S/nonreal_source_20260921` 于2026-09-21 12:15:50 UTC以exit1结束，运行513.51秒，没有有效跨模式配对结果。原因是单进程切换source违反模型“首次推理后控制项锁定”的生命周期约束；不是方法负结果，也不是任务失败率。原目录完整保留；服务器备份 `S/evidence_archive_20260921/source_failed_v1.tar.gz`，本地 `L/source_failed_v1.tar.gz`，SHA-256均为 `fd818d343434deaf127031522d685dcd0601930230af2eae3d7bdbccd59d36b8`。

**修复后完成：**CCI单张H100，2026-09-21 12:22:17–12:50:00 UTC，1662.94秒，exit0。运行根目录 `S/nonreal_source_r2_20260921`，输出 `dev50`，控制台 `dev50.launcher.log`。每种source使用独立进程，启动前固定模式；没有解除模型锁或改动线上策略。原预算30–40分钟，两小时硬上限。

使用与N02相同的预声明DEV选择规则、50前缀、NFE20、BF16和smoke300 checkpoint。首先独立Full进程对每个前缀运行Full和重复Full，保存容差；随后分别运行scale-only和Gaussian进程。三份过程输出独立保留。配对脚本核对科学身份/前缀哈希，并逐项核对Gaussian draw、realized g、candidate selection、adapted actions、valid mask和conditioning完全不变，再报告source/动作输出差异、mean g、**mean[c(g)^2]**。Gaussian source噪声方差为1，其conditioning gate仍保持原值。

相关模型生命周期/source/归档测试在服务器23项通过；跨进程配对与拒绝混杂测试本地3项通过。完整模型三进程运行已完成，服务器和本地均已从数组独立复算，50个前缀的固定输入、gate、conditioning一致性与汇总全部通过。复算入口：`python scripts/verify_nonreal_source_archive.py <run_root>`。

归档：`S/evidence_archive_20260921/source_r2_complete.tar.gz` 与 `L/source_r2_complete.tar.gz`；SHA-256均为 `9f3628909a82e2448660bf7ca7b96b65d86f108cc74a83f35251f6583fbf8605`。运行源码集合SHA-256为 `35a97cdf8209f0d32afa123dc0eac0abd7c919129c08bb8680e39cea325100b8`，运行前后相同。checkpoint与N02相同。

**自然gate诊断：**50/50都有有效候选、v_det均为true、stagnation均为0；原始alpha min/mean/max全部为0.119140625，低于配置阈值0.15，因此有效g全部为0。Full、scale-only、Gaussian三种source与动作相同，mean[c(g)²]均为1。这不支持“source内容先验有效”，也不证明其无效，因为此次未覆盖激活区域。不会降低阈值以制造正结果。

checkpoint的gate输出层权重范数为0.00516087，16个权重均非零；输出层bias仍为-2。不能把constant alpha直接解释为“模型从未训练”。下一步检查训练梯度/optimizer更新及BF16数值行为，并优先恢复成熟checkpoint。CPU检查产物位于 `S/evidence_archive_20260921/diagnostics/{source_r2_gate_activation,smoke300_gate_tensors}.json`，本地亦已保存。

本次已完成的启动命令如下，勿重复提交；无需多卡ACP：

```bash
RUN_ROOT=/mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations/nonreal_source_r2_20260921
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
WARM_NONREAL_OUTPUT="$RUN_ROOT/job_logs" \
bash "$RUN_ROOT/code/scripts/acp_nonreal72h.sh" "$RUN_ROOT/dev50.job.json"
```

只验证固定checkpoint的局部source机制，对应表22的局部诊断说明，不回填独立训练控制器的SR。若全部g=0，如实记为零覆盖/未激活，不能从输出相同推断内容先验无效；不事后强行抬高gate制造正结果。

## 5. N00：训练更新与 FP32 master 诊断

**已完成的只读核验：**原 shared smoke300 的完整四卡 ZeRO-1 状态仍在，位于 `WARM/runs/rmbench_official50_shared/shared-official50-dev45-s3407-v5-smoke300/checkpoints/state/step_000300`。按该 checkpoint 附带的 DeepSpeed `zero_to_fp32.py` 分片顺序重建四个 gate 参数的小切片；四项 FP32 master 转回 BF16 后均与保存模型精确一致。

| 保存状态中的量 | 实测值 | 解释边界 |
|---|---:|---|
| global step / data-parallel world size | 300 / 4 | 原始分布式训练状态；不是本次单卡恢复 |
| gate 输出层 bias，FP32 master | -1.9985454082489014 | 相对初始 -2 已有累积变化 |
| 同一 bias，BF16 模型 | -2.0 | 差值 0.0014545917510986328；舍入后仍是 -2 |
| gate 输出层 weight，master min / max | -0.001498857862316072 / 0.0014154266100376844 | 16 个元素均非零 |
| 历史 step300 gate 梯度日志 | 0.06348787411878672 | 旧训练日志记录，不是本次重算 |
| 历史 step300 learning rate | 1.002666666666668e-5 | 训练配置总步数30000、warmup1500；smoke尚在warmup内 |

因此，**BF16 bias 仍为 -2 不能证明 gate 没有更新，也不能据此诊断原 DeepSpeed 丢失小梯度更新。** 当前证据更支持先完成训练链路核验、再寻找或恢复成熟模型；尚不能证明后续训练会激活 gate 或提升 SR。

产物：`S/nonreal_train_update_20260921/zero_gate_master.json`，本地 `L/zero_gate_master.json`；两端 SHA-256 均为 `16b04a17036602d89d97659dd1a48cb49bd13338b2ceb024ac65c416142eee31`。检查脚本 `scripts/inspect_nonreal_zero_gate.py` 只读可信本地训练状态，核验四个 gate 切片，不代表全 optimizer 完整性或分布式 resume 验收。原训练日志已备份为 `S/nonreal_train_update_20260921/historical_training_metrics.jsonl` 与 `L/smoke300_training_metrics.jsonl`，SHA-256 为 `79dbdba3913bc30b941d1024e8f34ca27a3df0a7d53ed81d383de12ec8d5108d`。

**首轮失败已保存：**2026-09-21 13:30:28–13:43:22 UTC，773.37秒，exit1。运行根目录 `S/nonreal_train_update_20260921`，输出 `probe01`，日志 `job_logs/20260921T133028Z-71cd4c61`。源码快照 SHA-256 `2dcd8db10fa2cf8e197eeb6b2651a7ef43c376e3e60a2faf0736b5f51ae40f84`，前后相同。首个forward在gist输入检查报 `all token groups must share device and floating dtype`，没有取得有效backward结果。诊断脚本额外开启native autocast，与服务器安装的Accelerate+DeepSpeed BF16路径（`native_amp=False`）不一致；已改为显式关闭native autocast，生产模型未改动。原始目录、源码和日志完整归档为 `S/evidence_archive_20260921/train_update_failed_v1.tar.gz` 与 `L/train_update_failed_v1.tar.gz`，两端SHA-256均为 `dfabe73c500c9ebd05c0ab0cfe6b3c91d51a375540df9de1ca1b3e9aa98537fe`。这次是诊断实现失败，不计作任务失败率。

**R2已完成并验收：**2026-09-21 13:47:25–13:59:59 UTC，754.05秒（12.6分钟），wrapper complete / exit0，源码前后哈希相同。根目录 `S/nonreal_train_update_r2_20260921`，输出 `probe01`，日志 `job_logs/20260921T134725Z-185052cc`。源码SHA-256 `1f526ffe0584b34eefabdf23fd1b8501e357409c1399a326c964a4531e7ffe45`。单张H100、原生BF16、每任务一个固定TRAIN前缀、batch1、真实完整training loss/backward；使用trainer原有参数注册逻辑，比较隔离的gate-only fresh AdamW BF16与FP32更新，不修改生产模型/checkpoint。

| TRAIN诊断指标 | Press Button | Put Back Block |
|---|---:|---:|
| episode / frame | 270 / 268 | 90 / 156 |
| 总loss | 0.2869138718 | 0.5340017080 |
| gate输出bias梯度范数（clip前） | 0.005950927734 | 0.015991210938 |
| 全局梯度范数（clip前） | 1.4765625 | 3.84375 |
| fresh FP32 bias单步最大变化 | 9.775161743e-6 | 1.025199890e-5 |
| fresh BF16 bias单步变化 | 0 | 0 |
| 记录的单batch耗时（秒） | 2.8995 | 0.5377 |
| 峰值allocated显存（GiB） | 14.79725 | 14.79725 |

两个batch的四个gate参数均有有限非零梯度，全部在trainer参数集合内；生产gate张量未改动，诊断小张量保存/加载精确一致。Press Button此次训练corruption强制拒绝，Put Back Block正常source exposure=0.119140625，后者是**训练时**软gate，不等于先前DEV推理阈值通过。两条记录不能用于估计总体梯度分布或训练吞吐。

完整归档 `S/evidence_archive_20260921/train_update_r2_complete.tar.gz` 与 `L/train_update_r2_complete.tar.gz`；两端SHA-256均为 `fa71a81358044b11ad97ca4311007ffbd15b6a40474f846238090e79302452f5`。该诊断已完成，后续不重复运行。

此probe只检查梯度路径、数值更新和小gate张量保存/加载。它不恢复原四卡optimizer，不执行生产模型optimizer step，也不构成完整训练恢复或论文SR。单元测试在相同服务器环境 **3 passed**，覆盖不改原参数的小更新对照、跨rank边界切片与越界拒绝、关闭并恢复外层autocast上下文。现有证据不要求修改模型或推理阈值；下一步转向成熟checkpoint恢复及尚缺的在线成本测量。

## 6. W08：记录 DEV 前缀的完整在线链路成本

运行根目录 `S/nonreal_online_profile_20260922`。预声明 Press Button 五个 DEV episodes，每个前40次重规划（frame 0,4,…,156），共200个测量query；另20次预热排除。NFE10、H32、replan4、K32、BF16、batch1，使用与N02相同的smoke300权重。测量范围为CUDA同步包围的完整部署策略 `_replan`，包含预处理、DINO、真实检索、历史读取、模型推理、动作转换与标准telemetry写入；排除磁盘解码、模型加载及模拟器。

这是记录观测及记录动作历史的回放，生成的动作不执行，不能作为SR。数据集要求读取5帧观测对应4步动作；仅保留当前观测给策略，其余未来观测在缓存前丢弃。每个episode的40次测量不是40个独立episode；median/p95仅为该预声明工作负载的描述性耗时，不附伪独立置信区间。

**准备阶段失败已保留：**第一份online contract错误地把评测代码commit写作训练commit，末尾身份复核拒绝；已改为使用training attestation中训练commit，未关闭权重或数据校验。失败准备目录归档 `S/evidence_archive_20260921/profile_bundle_failed_v1.tar.gz` 与 `L/profile_bundle_failed_v1.tar.gz`，SHA-256 `e095527ed12bed55ddd97530e2a3f3a9e866bc7008e4c02a09cae3c5fec69174`。修正后的bundle已通过校验，manifest SHA-256 `2e3c429777336ea4080e1338945130da57c4142eec07ee39e9a814fbd702dc1c`。

**第一次测量入口失败：**2026-09-21 17:00:47–17:01:16 UTC，29.46秒，exit1，输出 `profile200`，日志 `job_logs/20260921T170047Z-5c9df7ea`。数据集拒绝obs_size1/action_size4，GPU模型尚未加载，无有效耗时数据。源码前后SHA-256均为 `54a097875f9d4fc7da9bc94cc841323f5cb21ccfb670c66add6177b00ecc0853`。原源码与输出完整保留，修正以独立 `code_r2` / `profile200_r2` 启动，未修改运行中的快照。

**R2数据预检通过：**200/200固定前缀的episode/frame身份、未padding动作、相机shape/dtype、动作normalization均通过；DEV episode catalog SHA-256 `d4a13afc421d27846925450e41ff316c26a1db781168556a8a5b3bab9d22392f`。预检输出 `data_qualification_r2`，exit0。

**R2已完成并双端复算：**2026-09-21 17:03:25–17:19:08 UTC，943.1466秒，complete / exit0；日志 `job_logs_r2/20260921T170325Z-f6968086`。精确源码SHA-256 `686fe25cee9e8176f9d63db3fac9aea00bb7db379442e30e4e21591290246c03`，运行前后相同。输出 `profile200_r2` 保存220条计时、6组episode边界（含预热）、全部replan telemetry、manifest与summary。200条正式结果均按预声明episode435–439及frame顺序覆盖；数值见第1节W08行。银行加载为CPU NumPy数组并做CPU精确cosine search，选中payload转到GPU；原manifest中的“numpy/mmap”是泛化描述，本次实际输入为NPZ，不据此宣称mmap性能。

所有正式query均有32个有效候选、自然g=0。虽然历史有增长、stagnation可达0.6328125，本模型alpha仍低于阈值。因此这是当前配置完整代码路径的成本，尚未覆盖自然非零gate工作负载，也不是成熟模型、recollection-only对照或真实仿真逐帧吞吐。episode_end的success=false仅是“没有闭环outcome”的封存字段，绝不能统计成0/5 SR。推理日志有已有的slow image processor提示及readonly NumPy转换警告；本次没有更换processor或修改生产模型处理这些警告。

环境：H100 80GB HBM3，driver580.95.05，Python3.10.20，Torch2.7.1+cu128，NumPy1.26.4，Transformers4.49.0；OMP/MKL/OpenBLAS线程各8。bank内容SHA-256 `758bfee0225ef33fe1e158502949f296318144438cff531a84fe0c0ebc0c336c`，online contract SHA-256 `8dbf3cc5c0264e1e7a076e9c63dfd4b4fd5650d980d4e1c20a69a4ff5335dcb5`，原训练配置文件SHA-256 `c09e2431597760a2e42a56af0e8488586214619a5554a649a7686a54d08ea077`。完整环境、配置与attestation已备份。

完整归档（含此前准备/入口失败及两个源码快照）：`S/evidence_archive_20260921/online_profile_complete.tar.gz` 与 `L/online_profile_complete.tar.gz`，两端SHA-256 `2117471260fd4681112f34ba6fb35498f703f59a5446d9010774fc1649451d32`。本地解包为 `L/nonreal_online_profile_20260922`。复算命令：`python scripts/verify_nonreal_online_profile.py <run_root> --profile profile200_r2`；核对wrapper、前后源码、checkpoint/contract/seed、计时与telemetry一一对应、预热排除、episode计数及全部summary指标。复算与拒绝篡改/缺失记录测试通过。该smoke工作负载成本已完成，不重复运行。

## 7. 四卡训练前置任务：已准备，待用户提交

`S/nonreal_resume_20260922/stage1/plan.json` 已绑定原四卡step300完整状态、weights、配置与源码哈希；仅将输出目录、resume和本段run_steps改为300→15000，其余30000步总schedule及global batch128不变。CPU身份核验、Hydra解析和相关测试已通过，真实四rank恢复仍待ACP现场检查；不能写成训练已完成。

申请4×H100 80GB、24小时，首段预计16–20小时。提交命令及完整身份见 [ACP_RESUME_ZH.md](ACP_RESUME_ZH.md)。结束自动生成 `stage1/summary.json` 与 `stage1/EXPERIMENT_RECORD.generated.md`，保留完整loss/gate/gradient/LR/速度记录及checkpoint路径；验收后再并入本记录。该任务用于获得当前schema兼容模型，不产生主表SR，也不保证gate或SR改善。

## 8. 后续记录规则

每项实验记录：ID、claim、训练/推理干预身份、code/config/checkpoint/split/bank哈希、种子、硬件、开始/结束/exit、原始产物、k/n与失败/unknown、统计单位、复算命令、可填论文位置、限制和下一步。运行失败时先保留失败目录，再以新ID重跑。

当前仍未验收：RMBench 755/900原始episode，六行消融训练身份，五轮干扰donor记录，真实snapshot/restore backend与renderer。未拿到这些原始证据前，相应论文行继续标记待核验，不填零或虚构区间。

## 11. 2026-09-23 新版论文缺口与36小时预算审计（非性能实验）

审计绑定作者提供的 `tmp/paper/WARM_ICLR2027_new.pdf`，36页，SHA-256 `e6d5bd7293a16072c8512572be82015f91bdfaba5f911a583a77c06620b9c2e8`；不是第9/10节另存的修订输出。活动LaTeX逐表清单见 [PAPER_GAPS_20260923.json](PAPER_GAPS_20260923.json)，逐表用途、资源与截止条件见 [PLAN_36H_20260923_ZH.md](PLAN_36H_20260923_ZH.md)。本节仅记录版本、覆盖与运行条件，不新增SR、TA/FA、MSE等性能数值。

| 本轮核验项 | 实测/核对结果 | 证据范围与论文处理 |
|---|---|---|
| 字面TBD | 8张表、74处标记：表11=19、12=8、14=6、17=5、18=11、20=8、22=11、23=6 | 标记数不是标量数。PDF的第75次TBD出现在表9标题说明中，不计数据缺口 |
| 隐藏数值占位 | 正文表4有10个红色数值；真机图3有12个红色成功计数 | 不是实验结果；表4纳入P0，真机由对应团队提供真实trial |
| 已完成且新版已填 | 表19 forced-null 50/50；表21 source诊断50前缀自然g=0；表24 replan median/p95 500.0/553.8ms、peak allocated 23.6201GiB | 引用第1/4/6节已有原始证据；不重跑、不扩大为正式模型效果或SR |
| CCI实际renderer复查 | 2026-09-23 14:37:17–14:37:22 UTC；1次尝试，exit1，`failed to find a rendering device` | 仅证明当前环境未通过图形验收，不能执行正式候选端点评测；CUDA可用不等于图形可用 |
| 四卡stage1现状 | 14:39:44 UTC限定扫描仅有plan/config/Hydra解析，未见started、训练输出；源码哈希与已准备plan一致 | 未在四卡恢复。若另有ACP输出优先验收、避免重复训练；限定路径扫描不是全服务器不存在模型的证明 |

最高优先级共用S1候选分支与S4三次requirement读取，争取表4/17/18/23合计32处占位；这是有前置条件的目标，不是已完成数量。预声明两任务各5 episodes、每episode4 queries、每query全部最多32合法候选。必须先验收真实renderer、完整state恢复、H32执行与独立适用性标签；全零gate不支持“选择性接受有效”，缺标签/零分母不填0。匹配训练表20/22暂不承诺36小时内完成；原表11/12/14只有找到原联合记录才可补旧身份、覆盖与配对CI。

原始归档在本地 `L/paper_20260923` 与服务器 `S/nonreal_deadline36h_20260923`（L/S完整根见第3节）。保存输入PDF、完整源包ZIP、输入身份、renderer失败日志与现场状态。源包ZIP SHA-256 `eba8099f2a948d0a121424415fbb814251c7cb353ac12e10e76c2dc560e51042`；`renderer_recheck.json` SHA-256 `90ffce06edd566b77feba3a2645a44bbca63f52b7e8d13f43c218711fc472db4`；`server_state.json` SHA-256 `9a946497af43dcc14555a0fef085b4950c4b7d99f8b25513f711a1a37cf36f36`，两端一致。此前四卡不可变源码集合仍为 `665d0620a2a8a15e1d3daaa2ddad38e6ca7c383f8af5aa407ef0784aff797bda`；当日未重读25GiB父state，不声称重新完成该全量核验。

后续每一实测单元格同时追加新版表号/标签、值与单位、k/n、聚类/CI、失败与unknown、T/I身份、checkpoint/config/source/split/bank哈希及原始路径。ACP继续自动生成仓库外 `EXPERIMENT_RECORD.generated.md`，验收后合入本文，并注明哪些结论受到支持或需要改写；不得把新cohort数据悄悄替换为原主表或五轮干扰结果。

## 12. 2026-09-23 单命令ACP流水线交付（工程准备，未启动训练）

按作者要求将当前已具备入口的步骤合并为 [ACP_BUNDLE_ZH.md](ACP_BUNDLE_ZH.md) 的一条命令：原四卡300→15000续训、目标checkpoint验收、CPU gate权重检查、新15k模型DEV50自然门控诊断，以及有非零自然g时的两个独立source干预进程。Full只采集一次，三个模式从原始数组配对；若DEV50全零则不重复退化对照。旧step300诊断及已填表19/21/24不重跑。

预声明新诊断范围：Press Button/Put Back Block各5 DEV episodes，每episode5个固定前缀，共50；NFE20、seed3407、固定bank/query corpus/normalizer/prefix manifest。该诊断无适用性标签、闭环SR或独立训练身份，不填表18 TA/FA或表22训练SR。15k仍是固定新预算，不按结果选择checkpoint。表4/17/18的仿真恢复/标签、表23的requirement-only干预、表20/22的匹配训练、表11/12/14的原记录仍明确标为未完成；流水线正常完成也保持`paper_evidence_complete=false`。

服务器目录：`S/nonreal_bundle_20260923`。申请4×H100 80GB、24小时，整体预计17–21小时，入口预算23小时；同一提交先用4卡训练，再用其中1卡做诊断。原训练目录与快照不变。独占锁阻止重复进入，完整阶段须校验收据/哈希后才复用；已启动但不完整的训练及半截输出不覆盖。每阶段自动更新`run/bundle_summary.json`和`run/EXPERIMENT_RECORD.generated.md`，保留原数组与逐阶段日志，随后验收合入本文。

验证：本地19项相关测试通过；CCI相同19项全部通过（10.72秒），CPU身份预检及bash语法检查通过。没有执行四卡恢复或15k真实推理，没有新增论文性能数值。原始CPU预检、测试日志与plan双端保存在服务器上述根及`L/bundle_20260923`。准备时AFS拒绝tar恢复属主导致首次解包非零，改用`--no-same-owner`后完成相同部署包的解压与核验；未启动任何GPU实验，也未修改原训练快照。

新汇总器对已有N06-R2的50条Full原始数组完成读取兼容性核验，复算10 episodes、g非零0/50、alpha均值0.119140625及mean c²=1，与第1/4节一致。`reader_compatibility.json`明确标记`new_experiment=false`；没有重新采样/推理，不新增样本量或替换已有表格结果。

新源码集合SHA-256：`70f316e0e2e46718ba33b2dea2d0bb5836ce0fcb446ecf895fb98aa127757265`；部署包SHA-256：`348643a15b022e57c6760230b4bd093ff90e687ab4bc726e041b0832a7f68451`；`bundle_plan.json` SHA-256：`10f6c1bd390dc15a782a4f1644337c9f79b54595a8ec827af646557d3695e37a`；服务器`tests.log` SHA-256：`f7afd6f4c680496df60bdb5421565c866fecc2342ab5b83a09bee1bca504c83a`。仍调用已验证的旧训练源码`665d0620a2a8a15e1d3daaa2ddad38e6ca7c383f8af5aa407ef0784aff797bda`，不让Git同步改变运行中的源码。

## 13. 2026-09-24 ACP启动失败修复与R2交付（未开始训练，无新性能结果）

作者上传`logs-acp-20260924T010802.txt.gz`，SHA-256 `a3a996c4c88b7df86781f255d818494238a3ed27e24973c39f3fbcf6b5c9e5fe`。服务器原任务的renderer阶段exit1（仍缺图形环境）；训练子阶段`training-e4a44aaf`于2026-09-23 16:18:59.824–16:19:00.835 UTC运行1.01075秒后exit127，原`warm_server_common.sh`第2行报`$'\r': command not found`。该旧helper含194个CRLF换行，SHA-256 `f8546127e29f3044a665a6b9cecae14f94d98f082c4e8787ff9d81ee2e31c3c5`。这是Shell启动错误，Vulkan警告不是训练退出原因。

现场确认原stage1只有plan、config、Hydra解析及lock，没有`started.json`、job或训练输出；原Python恢复程序尚未执行，新增optimizer steps为0。本次不是训练性能失败，不报告loss、SR或gate新数值，也不重跑已完成诊断。原失败根`S/nonreal_bundle_20260923`保持原样，完整失败归档`previous_failure.tar.gz` SHA-256 `1051a8772ee60593a2284bc7177adfb4c5ad9e61d71093efc9f7f2b88d3dd390`，已下载至本地。

修复：新入口在已配置好离线模型路径、线程和本地缓存的bundle环境中直接调用原不可变`nonreal_resume.py run`，不再source旧Shell；原源码SHA-256仍为`665d0620a2a8a15e1d3daaa2ddad38e6ca7c383f8af5aa407ef0784aff797bda`，原resume plan SHA-256仍为`dd4a8332becec36d2b64cacc4bf3ea6ef723fb0b9f85e9959e1874c603a0fdb9`。训练配置、父weights/state、总schedule、随机种子及target15000未改变。新增真实Shell `--preflight`、原Python CLI启动检查、活动Shell字节检查和Git的Shell LF属性；禁止就地转换旧快照破坏其身份。

R2已部署在`S/nonreal_bundle_20260924_r2`，输出独立。实际在CCI执行最终Shell命令加`--preflight`通过；Linux全部22项测试通过（10.90秒），Windows本地21项通过、1项Linux专用集成测试跳过。新增用例覆盖旧CRLF依赖被绕过、真实Shell初始化和环境变量。准备时新增Linux测试夹具的临时plan层级有误，修正后全通过，第一次测试与plan保存在`preparation_r1`；生产R2 Shell预检在修正夹具前后都通过。真实四rank恢复及15k推理仍待ACP现场，不声称已完成。

R2精确源码SHA-256 `976b74014e46451a84ca311fb374d3dcbc5468c9eb1f969940ce5a641450319f`；R2 plan SHA-256 `d6e3ef2b07e4b673948c5f3f8b7e8ba8957e948bdf7533e7edf072b02e0d74e9`；最终部署包`code_final.tar.gz` SHA-256 `a4408a97f4c663d8c0d49daee3fa0a28e054713aeb9eb093992538b9cd4123ba`。原始失败日志、失败目录归档、`failure_audit.json`、plan、实际Shell预检与测试日志双端保存于R2根和`L/bundle_fix_20260924_r2`。最新重提命令见 [ACP_BUNDLE_ZH.md](ACP_BUNDLE_ZH.md)；申请4×H100 80GB、24小时，仅提交R2一次，不再运行旧Shell入口。

## 14. 2026-09-24 R2实际续训的部分结果（完成状态未确认）

本节来自通过CCI SSH读取共享AFS产物的独立快照，审计完成时间为**2026-09-24 12:56:06 UTC / 北京时间20:56:06**。没有修改活动训练目录、源码、plan或checkpoint，没有启动续训或推理。本文中的运行阶段是`shared`；不是blocks specialist或新15k模型的评测。

### 14.1 已启动、已恢复，但没有完成收据

- R2实际命令为`bash S/nonreal_bundle_20260924_r2/code/scripts/acp_nonreal_bundle.sh`。外层训练job为`run/jobs/training-dc2b6885`，开始于09-23 21:47:48 UTC；内层`S/nonreal_resume_20260922/stage1/job`开始于21:52:03 UTC。
- 已通过实际四卡runtime前检。训练日志显示加载optimizer/random states、恢复dataloader progress，并于09-23 22:17:31 UTC记录`Verified formal WARM resume after ...`。后续真实optimizer-step指标和checkpoint证明此次越过了旧CRLF启动故障；不是此前只有CPU预检的状态。
- `training_metrics.jsonl`共338条，global step严格递增且无重复，范围310–3680、步距10。相对parent300，至少已有3380个新增optimizer steps进入已记录进度。全部已记录数值有限，没有据此声称每个未记录step都已检查。
- 最后一条训练日志/metrics修改时间为09-24 **02:03:32 UTC / 北京时间10:03:32**；内层heartbeat最后写入02:03:23 UTC。快照时已约10小时53分钟无更新。bundle与两层manifest仍为`running`、exit_code为null，但均缺终态收据，不能据此认定ACP仍存活。
- 没有stage1 `summary.json`或`summary_error.json`，没有step015000权重、`natural_gate.json`、新checkpoint inspection或source输出。训练console中未发现Python traceback、CUDA OOM、ChildFailedError、FloatingPointError、SIGTERM/SIGKILL或`[done]`。因此记录为**部分训练、结束状态未确认**；现有文件不能确定平台回收、外部终止、挂起或其他原因。CCI进程列表不能证明另一个ACP worker的状态。
- renderer子任务有明确终态：6.498秒、exit1，`RuntimeError: failed to find a rendering device`。它发生在训练前且未阻止后续训练，不能用来解释数小时后的日志停止。

### 14.2 可以回填的训练数值

以下首/末窗口各取20个记录点，分别为step310–500、3490–3680，均为**记录点的简单算术均值**。这是同一次续训的描述性轨迹，不是固定验证集评测、所有optimizer steps的加权平均或独立样本置信区间；未筛选有利记录。

| 原始指标 | 首20条均值 | 末20条均值 | step3680单条值 |
|---|---:|---:|---:|
| `loss` | 0.434301 | 0.275376 | 0.311095 |
| `loss_action` | 0.049257 | 0.009741 | 0.011242 |
| `loss_video` | 0.249260 | 0.186167 | 0.219631 |
| `loss_warm_effect` | 0.828948 | 0.443242 | 0.434192 |
| `loss_warm_gate` | 0.489502 | 0.677344 | 0.743164 |
| `warm_gate_brier` | 0.057871 | 0.116354 | 0.124023 |
| `warm_learned_gate_mean`（raw alpha） | 0.096301 | 0.177411 | 0.160889 |
| `warm_gate_mean`（训练source gate） | 0.069142 | 0.132458 | 0.153687 |
| `warm_normal_source_exposure_mean` | 0.119666 | 0.227197 | 0.234131 |
| `warm_grad_source_gate` | 0.067948 | 0.120777 | 0.125555 |
| `warm_forced_source_leak_mean` | 0 | 0 | 0 |
| `steps_per_second` | 0.269730 | 0.248895 | 0.249250 |

step3680的learning rate为`4.928882364186164e-05`、grad norm为`1.0966973304748535`。338条记录的source-gate梯度范数范围为`0.0614487877–0.1474911482`，记录的forced-source-leak全部为0。

这些结果支持“已恢复训练，记录的action/video/effect loss下降，gate分支有非零梯度、训练输出不再固定在smoke300附近”。同时gate loss和Brier的窗口均值上升，不能只报告有利趋势；训练batch、corruption及动态soft target不固定，不能由此单独断言校准变好或变坏。训练期alpha/source exposure也不能与此前DEV推理g=0直接比较，更不能当成新模型自然接受率。

**论文回填边界：**本节可用于训练进度和优化诊断说明；没有新增可填主表SR、表18 TA/FA、表22匹配训练SR或表23历史收益的数值。既有表19/21/24的smoke300证据保持原范围，没有被本节替换。

### 14.3 Checkpoint与运行身份

已找到step1000、2000、3000三个发布权重及各自`.training.json`、`trainer_state.json`、scheduler、四rank optimizer shards和四份random states。元数据中的step、resume300、parent weights/state、shared recipe、world4、GBS128及seed3407一致；每个列出的state文件均存在且非空。

最新发布权重为`S/nonreal_resume_20260922/stage1/training/checkpoints/weights/step_003000.pt`，大小**12,148,647,926 bytes**；本轮实际完整读取权重计算SHA-256为`68c7f3840921241df45a5b6997b772db3a69a31584644aa0e66929b841b845bf`，与attestation相同。step1000/2000仅核对元数据及文件存在性，没有重算其大文件哈希；没有重新哈希或反序列化约25GiB的step3000完整optimizer state，也未执行新一次四rank恢复。故step3000是最新已找到并验过权重字节的发布点，不把step3680当作已保存checkpoint。

运行身份：4×H100 80GB、BF16、ZeRO-1、per-device8、gradient accumulation4、GBS128、seed3407；目标15000、总scheduler30000、AdamW LR上限`5e-5`、warmup1500。父weights SHA-256为`ab4042b1a761f53ec5f5379ce461f5a2f6f3decf89e3f3a16322b66894918a1d`；parent-state为`48be391eb0c223f862e5197e1fc77455b67225ce1df98b7773b3478ecd5af0cf`；shared-recipe为`0461b9fac327974d5dd4fa11c810a07aafe4785f8949d0d8c0d13916cea69f6b`。本轮复算续训YAML字节哈希`34f5dc0a6f4cc354da61c0ba7bdc394b6c9652c7ef7706e72e8cbb683ec9a00c`与plan一致；attestation中的resolved config哈希采用另一种序列化，不能与YAML文件哈希混同。

本轮重新计算冻结源码集合：R2的414个文件为`976b74014e46451a84ca311fb374d3dcbc5468c9eb1f969940ce5a641450319f`，训练的407个文件为`665d0620a2a8a15e1d3daaa2ddad38e6ca7c383f8af5aa407ef0784aff797bda`，均匹配预声明plan。step3000 attestation的Git commit为40个零；精确源码身份由这些冻结文件哈希和已保存配置/manifest提供，不把零值或当前main commit当成训练commit。

### 14.4 双端原始证据与后续条件

本次使用新的本地目录，**不是第3节另一台机器的L路径**：

- 服务器归档：`S/evidence_archive_20260924/r2_partial_20260924T1255Z.tar.gz`。
- 本地归档：`F:/WARM/result/nonreal_bundle_20260924_r2_audit_20260924T1255Z/r2_partial_20260924T1255Z.tar.gz`；同目录保留`audit_partial.py`、receipt及解包文件夹。
- 压缩包大小2,512,489 bytes，双端SHA-256均为`9e3cae437e97a704b7217c08a70aa78c3f2d561b2ebec9dde0b0420c200db7a9`；含858个按文件清单登记的原始文件及独立审计报告。两份冻结源码、原始metrics/console/heartbeat/manifest、renderer失败、checkpoint sidecar/trainer-state/scheduler均保留；大型权重与optimizer shards留在服务器原路径，没有声称已本地备份。
- 原始`training_metrics.jsonl` SHA-256：`66b6609e93a5ba7bb8430f67c2168e70c4634d49697ae445d5ee332a3ef2b06a`；`audit.json` SHA-256：`efa4f97e0db83fcfa1deaedf373aafe70ccdddff477bf9f6ce4154f7a09c3f01`。
- 本地解包后逐一重算858个文件的字节数/哈希全部匹配；通过独立PowerShell读取338条JSONL，复算首尾窗口均值，与服务器Python报告一致。归档为现场切片，不冒充最终完整run归档。

下一步先取得ACP平台终止状态/事件，确认worker是否仍在运行，再决定恢复。原stage1已存在`started.json`，**不得直接重复提交旧命令、删除锁/started标记或覆盖原目录**。如果确认中断，需先验收最后完整state，再为剩余3000→15000准备新的续训输出与plan；step3000之后未保存的更新不能当作已恢复。新15k模型验收后才执行预声明DEV50自然gate及条件source比较，不因当前训练alpha上升就跳过该测量。

## 15. 2026-09-25 回填前的服务器寻证与真机数据审计（非新增SR）

按作者提供的范围，经CCI检查`/mnt/afs/task3_2/L202500276_lwz/projects`及实际存在的`datasets`（单数`dataset`不存在），并重点核对WARM real/piper、历次评测与训练根。定向清单有3,361个匹配元数据/日志文件、9个symlink，扫描错误0；跳过大权重、媒体、依赖缓存和重复源码快照。未将“未找到”写成全服务器绝对不存在，也未将计划文档中的数字当原始结果。

扩展关键词搜索于9月24日16:09:29 UTC完成，遍历76,118目录、读取112,926文本文件、跳过2,409个超过2MB文件、访问错误0，记录168个归档路径。最大深度15且有媒体/依赖/重复源码等排除；2,230份命中含其他项目与计划配置，不构成2,230次WARM实验。仓库外`F:/WARM/result/paper_fill_20260924/claim_search.json`保存路径/行号/片段；没有由此找到支持目标主表和消融/干扰的完整结果链。

- 找回的历史LIBERO汇总仍为spatial488/500、object496/500、goal483/500、long469/500，共1936/2000。主结果根为`S/batches/step_019100/{spatial,object,goal,libero10}-formal-s3407-v2/summary.json`，逐任务在`S/results/step_019100/`。仍不能支持稿中1974/2000，也不能未经模型身份核验换成当前Full WARM成绩。
- 本轮已检查路径未定位到RMBench755/900、六行组件消融及五轮干扰的完整原始run/episode/donor链。作者提供4×H10080GB可用、RMBench仍不能渲染；本轮没有重跑任何GPU实验或修渲染环境。
- 2026-09-24 **16:00:27 UTC / 北京时间9月25日00:00:27**再次读取R2 bundle：summary仍`running`，最后更新时间9月23日21:47:47 UTC；训练metrics仍最后修改于9月24日02:03:32 UTC、step3680。该标记不是ACP平台状态，不能据此安全重提。未发现新的15k完成证据。

### 15.1 真机数据与memory：可填数据设置，不可填trial成功率

绑定版本：`/mnt/afs/task3_2/L202500276_lwz/projects/WARM/real/piper/processed/pilot_20hz_joint`。25条遥操作示范，20train/5dev；B2B4/1、D2B4/1、R2P8/2、SOR4/1。处理后12,333个command-bearing frames，train9,976、dev2,357。原始25条在`WARM/tmp/WARM_real/pilot_20hz`，`WARM/real/piper/raw/pilot_20hz`为同一目录symlink，不重复计数。

事件manifest记录1,356事件；本轮实际读取bank的dataset/episode索引，20个不同source episodes全部在catalog的train split。bank manifest文件SHA-256为`1b4fc6087b98863056a055711139c0a2ee1a123921c9468cb40a4a93f6e4a4e0`；manifest记录content hash `a5cf005dd30627efdb2edd7bdc3110708944e4f48e64a1e688e721e8bb3c4de3`。本轮没有重算整个事件payload或原始视频哈希，不把读取metadata扩大为全量内容重新验收。

数据配置：external/wrist两视角，声明raw640×480、各自224×224、水平224×448，nominal20Hz、H32、K32、context772。关节版本的动作是6个absolute joint targets(rad)+absolute gripper width(m)；**proprio仍为TCP xyz+rotation vector+gripper width(7维)**。依据服务器`build_piper_joint_action_cache.py`、conversion与prepared配置，不能把同为7维误写为joint proprio；该版本delta mask只处理padding。camera型号、实际标定误差、部署command rate、执行prefix及decoder未由这些文件验证。

`success` outcome的notes标记JointCtrl teleop/sidecar，不是自主policy trial；本轮未找到三方法×四任务的真实k/n日志，因此图3仍不能填。episode-level dev划分不证明独立采集session/未见布局；holdout命名不等于物理采集独立。

### 15.2 真机训练：配置与时点记录，而非已完成实机评测

该版本三个`warm_from_stage_a/run_20260924_*`配置均为BF16、batch1/accum1、LR1e-4/cosine、weight decay0.01、clip1、8epochs、training seed42（corruption seed3407）。归档时100346目录记录至13340、state13200；122118记录至6260、state6000；150003仅配置未归档到metrics。不能相加为最终训练量、声称全部8epochs完成或擅选最终部署checkpoint。训练配置中eval_num_samples=0；内部eval NFE字段不构成实机推理设置证据。

### 15.3 原件与回填交付

双端原始622文件归档：服务器`S/evidence_archive_20260924/paper_fill_20260924T1545Z.tar.gz`；本地`F:/WARM/result/paper_fill_20260924/paper_fill_20260924T1545Z.tar.gz`。2,468,470bytes，SHA-256 `ddba6c7a277714635ebe9cf44539cad621afc27e5d96f763428f372c26229f59`，逐文件字节/哈希核验622/622通过。`files.json`保存原服务器路径；同级`real_analysis.json`和`real_bank_check.json`保存复算。额外小文件在本地`supplement/`保留joint转换源码、conversion及stats；没有大权重下载或原始数据入Git。

[PAPER_FILL_AND_RUNS_20260925_ZH.md](PAPER_FILL_AND_RUNS_20260925_ZH.md)给出新版精确文件/label位置、四个可粘贴LaTeX片段、哪些值不可回填，以及按4H100/无renderer的24小时优先队列。真机数据说明可新增到C.4.1，表11仅部分配置可填；W08成本可新增到C.2但保持smoke300/g全零/记录回放限制；R2训练动态只属附加诊断，不替代主结果或表4。没有直接改作者LaTeX/PDF，没有把旧数字自动改成新主表成绩。

## 16. 2026-09-25 最新表4现场复核与统计收尾修复（未产生新测量）

核对 `WARM_ICLR2027_new (1).pdf` 第9页和ZIP中的表4/附录C.2.1，要求固定query的adapted
H32候选分支实际执行、独立效果和适用性标签。当前CCI主机
`app-5e8f0e533ef54feeb96451ac86bb4108-6fbfcff76d-m5gxp` 的H100 80GB空闲；重新执行
runtime check（press_button，未跳过renderer）仍报 `failed to find a rendering device`。
标准Vulkan ICD/图形驱动库目录未找到所需挂载。没有更改系统驱动、安装包、提交ACP或启动新训练。

R2 summary仍是旧running标记；weight目录仍仅1000/2000/3000，最后metrics仍step3680。
本轮只核对文件清单与末尾记录，没有重新哈希大权重，也没有将step3000认定为论文最终模型。
通用BranchBackend尚无真实RMBench实现及验收，不能以toy tests或renderer通过代替恢复合格。

本轮新增专用表4汇总：三行相同的query标签支持集、逐任务coverage和配对episode bootstrap，
缺失/未知不补零，缺少输入时不导出LaTeX数值；修复初始读图异常的父状态恢复与事件可变引用。
本地CPU测试33 passed，0 failed。这是工程检查，不是新效能结果。
详细设计、3840分支预算敏感性、仅环境验收的ACP命令及待确认模型见
[TABLE4_EXECUTION_20260925_ZH.md](TABLE4_EXECUTION_20260925_ZH.md)。

### 16.1 作者改用LIBERO权重；环境恢复验收通过（非表4效能结果）

作者随后明确取消R2，采用LIBERO权重step019100。固定旧模型配套代码，不混入新RMBench head。
CCI H100 + MuJoCo3.3.2 + Mesa EGL软件渲染已生成两路有效256×256RGB；
环境层32步A–A–B–A重复/顺序检查通过，物理atol1e-10，两路图像逐像素相同。
四分支实测3.5063/3.4851/3.4757/3.4688秒；这不是模型性能或表4指标。

原件在`S/table4_libero_20260925/branches-egl-v2/`，本地
`F:/WARM/result/table4_libero_20260925/environment.json` SHA256
`a4993bd43702ff296a641260bbf6cf5851d223865eae8e7e8dbc0ee90a8d1684`；
主相机PNG SHA256 `2935c1e3265e333203312ee25267e3cc6452d3e4768c15ef3d1fd845fa5cf1ee`。

预声明正式采样：LIBERO-10所有10任务、initial states0–9、factual frames80/160，seed3407，
最多200query/6400候选H32分支。独立标签定义为端点新增BDDL子目标且保留此前已完成目标，
严格称32步goal-progress，不将其夸大为通用适用性或整任务SR。
任务0/init49/frame80保留作采集器资格验收，不进入正式统计。

首次模型资格验收保留于`S/table4_libero_20260925/qualification-v1*`：
真实模型加载成功（约559.6秒），进入factual rollout后发现缺少历史评测兼容层，
动作签名21/15维不匹配。新增显式legacy兼容适配复用原eval-v2语义，不改权重或学习模块。
修复版资格验收另存v2；未得到完整结果前不宣称表4已完成。
当前相关CPU测试44 passed；既有job/bundle/resume/archive回归30 passed、1项Linux-only skipped。

### 16.2 LIBERO真实模型采集器验收完成，4卡正式实验待作者提交

第二版保留`qualification-v2*`：模型加载553秒，进入frame80并封存32个真实proposal，
随后state fingerprint对slots dataclass使用`vars()`报错。`plain_fields`已支持dataclass
fields，并补充pending动作deep-copy测试。第三版固定在`S/table4_libero_20260925/release-v3/`，
未修改任何运行中源码快照、旧checkpoint或旧模型worktree。

第三版`qualification-v3`退出码0，`execution.completed=true`。任务0/init49/frame80，
1query、32候选均执行32步、32个complete_horizon、32/32父状态恢复、768维observed effect。
实际模型下A–A–B–A通过（physics atol1e-10、RGB exact），DINO重复编码最大绝对差0。
封存proposal→attempt/result→独立标签→assemble的真实数据读取全部通过。
该验收query的goal-progress标签为0/32正例；属于窄子目标定义下的工程验收记录，
不纳入正式结果，不据此重选query/initial state或修改标签。表4正式效能仍未测得。

实测：model_ready561.5秒；32候选执行107.71655秒；总798.41003秒；观察显存约24,969MiB。
正式预声明200query时仅候选分支约5.98 worker小时，单卡不满足作者<=4h条件；
交付4卡ACP（各worker10任务、reset互斥），预计3–5小时但CPU/AFS可能延长，
每worker设6小时安全上限。未提交正式ACP，未启动r2或新训练。

双端证据包：服务器`S/table4_libero_20260925/qualification-v3-evidence.tar.gz`；
本地`F:/WARM/result/table4_libero_20260925/qualification-v3-evidence.tar.gz`，SHA256
`eaba59476cd895396ea8b52ef68683b44cf4a636fb2b3f9b55a453eaed13d68e`。
包内有完整console、plan、compatibility、proposal、index、qualification、branch/label及assembled记录。
权重SHA256仍为`de324de810df74bd7ff16114c4cf22e90c470bcdadcc7995fdbac9b494e1bb64`。
最终本地相关测试45 passed；之前既有回归30 passed、1 skip，合计75 passed/1 skip。
远端ACP脚本bash语法检查通过。源码及说明已在本地修改，尚未提交Git；冻结运行包已上传CCI。

论文适用边界：此旧LIBERO head使用warped action/context预测effect，执行的是含adaptation的proposal，
与最新稿`stats(mu_i)`输入式并不完全一致。附录需说明实现版本、suite和goal-progress标签，
不能把表格直接称作新RMBench head或主表最终模型的机制验收。

### 16.3 2026-09-25 ACP返回：仅1/4分片完成，实测部分结果与草稿结论不一致

作者提供`logs-acp-20260925T164510.txt`和最新LaTeX包`WARM_ICLR2027_new (2).zip`。
ACP输出根`S/table4_libero_20260925/formal-20260925-041226`；平台日志最后一行为
`ERROR: at least one shard failed`，不是完整成功。
启动脚本全局固定`MUJOCO_EGL_DEVICE_ID=0`，而worker1/2/3分别仅可见CUDA1/2/3；
robosuite在导入时因EGL编号不在CUDA_VISIBLE_DEVICES中断言失败。
这是上一版启动器的设备映射缺陷，单GPU0资格验收未覆盖多GPU可见性情况。
worker1/2/3都没有进入采集，仅留下compatibility.json。原件均保留，不覆盖失败目录。

worker0的execution.completed=true，耗时8545.255秒（约2.37小时）。
覆盖全部10任务、每任务initial states0/4/8和frame80/160：30个episode、60个query、
1920个候选分支。1920/1920父状态恢复通过；1852个完整H32端点、68个提前成功端点。
终止标记success共70个，其中2个恰好执行满H32，因此不能把70与68混淆。
10任务的A-A-B-A和DINO重复性检查全部通过。与预声明100episodes/200queries相比，
还缺70episodes/140queries；未将merge完整性门槛下调，未把部分报告伪装成完整表格。

对全部已完成样本使用原来冻结的统计口径复算：三参考共享支持集、candidate→query→episode→task等权，
task内episode-cluster bootstrap10000次、seed3407。MSE/ordering/ties使用59个有完整端点且
有可比较候选对的query（28,441对）；选择率使用60query。标签已知1920个，正例275、负例1645，
定义仍是H32端点新增BDDL子目标且保留已成立目标，不能称为整任务成功率。

| Effect reference | MSE | Ordering | Tie rate (%) | Goal-progress choice (%) |
|---|---:|---:|---:|---:|
| Historical | 0.0551252164 | 0.6344244688 | 0.0033602151 | 18.33333333 |
| Query-only | 0.0426805201 | 0.5 | 100 | 14.32291667 |
| Predicted | 0.0551812770 | 0.6351804493 | 0 | 18.33333333 |

predicted-minus-historical：MSE +0.0000560606，95%CI[0.0000437719,0.0000690308]；
ordering +0.0007559805，95%CI[0.0003694879,0.0011591384]；choice差0，区间[0,0]。
预测效果的MSE略差，ordering提升很小，choice相同；不能保留草稿中的
0.084→0.052、0.63→0.76、64%→78%的有利联合提升结论。
旧checkpoint/head版本和窄标签的适用边界依然成立，且每task只有3episode的CI需谨慎解释。

双端原始证据包22MB：`S/table4_libero_20260925/formal-20260925-041226-evidence.tar.gz`及
`F:/WARM/result/table4_backfill_20260925/formal-20260925-041226-evidence.tar.gz`，SHA256
`081d355b18283dada70cea8242dc2e474c5773a94ed58b561df386b5dd02cff8`，本地复核一致。
本地复算脚本`analyze_partial.py`验证实际query矩阵、probe元数据/数组SHA、branch身份与恢复，
报告`analysis_partial/table4_partial.json`明确`collection_status=partial_1_of_4_shards`。
没有运行新GPU实验，没有变更服务器或启动器；等待作者选择明确以30episode报告还是补齐原100episode。
最新独立`4_experiment.tex`与ZIP中的同名文件SHA256一致，尚未改动作者原稿。

### 16.4 作者要求继续ACP：设备选择修复及只补缺失分片

作者要求提供继续运行命令。已建立独立`S/table4_libero_20260925/release-v4/`，
保留release-v3、旧失败分片和成功shard-0，未覆盖任何旧数据。
进一步读取当前robosuite1.4实现后确认：其EGL索引选择直接耦合CUDA_VISIBLE_DEVICES，
仅改MUJOCO_EGL_DEVICE_ID=1/2/3仍会在只有一个Mesa EGL设备时失败。
修复为显式进程内software EGL selector，使用MuJoCo的独立EGL选择函数；
CUDA_VISIBLE_DEVICES从不被改写，只有导入时暂移除robosuite的EGL/CUDA成员断言输入。
不改任何site-packages、权重、动作、seed或统计参数。

CCI实际验收两种环境：CUDA_VISIBLE_DEVICES=3（该单卡CCI无CUDA3，仅检查非零可见性下的CPU渲染）
与CUDA_VISIBLE_DEVICES=0（require-cuda检查实际H100）。两者均渲染成功、H32 A-A-B-A通过，
两路RGB与旧`branches-egl-v2`逐像素一致。不能将该检查称为CCI四卡硬件验收；
ACP启动器会逐worker在真实可见GPU下检验CUDA分配、渲染和恢复，全部通过后才加载模型。
原件在`S/table4_libero_20260925/egl-v4-cuda{0,3}/environment.json`及相邻console。

新增resume planner，在旧结果上实际通过proposal SHA/branch身份与状态恢复/标签/逐任务资格记录验收，
确认旧states0/4/8，缺失1/2/3/5/6/7/9；四worker分成[1,6]/[2,7]/[3,9]/[5]。
旧shard仅只读复用；新worker加载权重SHA必须等于旧shard；最终merge仍要求完整采样矩阵，
不接受重叠或漏采。汇总protocol.episodes修正为0..9，另保留各shard原始plans以免误标样本范围。
新数据输出`resume-<timestamp>`，预计2–3小时但取决于CPU/AFS，6小时为每worker安全上限。
本轮未提交ACP、未运行新效能实验；由作者使用已上传release-v4命令提交。
本地相关单测53 passed，bash语法检查通过。源码仍为本地未提交修改，冻结运行包已同步服务器。

## 17. 2026-11-05 审稿前通读与最终投稿版本确认（非新增实验）

作者在当前对话明确确认：**33页的 `11572_World_Action_Retrospecti.pdf` 是最终投稿版本，以该版本为主。** 本轮工作为论文、方法核心实现、相关配置和测试的静态阅读，没有启动训练、ACP、仿真或真机试验，没有复算服务器原始测量，也没有把历史实验状态当作当前运行状态。

### 17.1 原件、工作目录与源码身份

- 最终投稿原件：`tmp/first_submission/11572_World_Action_Retrospecti.pdf`，37,342,016 bytes、33页；SHA-256 `aa04f9519881983b9f6f408dcfb5b912c9290b883dbe31567f671136fa864594`。
- 原样备份：`tmp/rebuttal/baseline/11572_World_Action_Retrospecti.pdf`，本轮实际重算字节哈希一致。这些路径均在 Git 忽略的 `tmp/` 下，原始 PDF 和生成阅读材料不加入 Git。
- 当前读取的代码 HEAD：`5eb321d95fe7f5e0db720e1f60df4d3926a9b1e7`；432份已跟踪源码、配置、脚本、实验入口和测试文件的逐文件哈希及集合摘要保存在 `tmp/rebuttal/reading/source_manifest.json`。该集合摘要为 `2e45333888eb23e8b20af92a6f67a2063250f52d7fcb58058b36135724bd6e49`。当前 checkout 身份不自动等于每项投稿结果的运行源码身份。
- 工作路径：`tmp/rebuttal`；后续论文修订放在 `tmp/rebuttal/revision`，必须从核对后的原论文 LaTeX 工程编译 PDF。当前阅读笔记不是论文 revision。
- 本记录修改前的副本保存在 `tmp/rebuttal/baseline/EXPERIMENT_RECORD_ZH.before_20261105.md`；原有工作区其他文档删除状态保持不动。
- 已定位的历史 LaTeX 工程编译件为29–30页，尚未确认最终33页稿的完整源码。不能将旧工程的附录直接当作最终投稿底稿。

### 17.2 最终稿主张与现有记录的对应范围

`tmp/rebuttal/reading/claim_evidence_map.json` 记录最终稿 Table 1–4、Fig. 3–5、Tables 12–19及命题的主张、数值、cohort和证据状态。所有论文数值标为 **paper-reported，本轮未独立复现/验收原始测量**，不升级为新实测结果。

- 最终稿报告 LIBERO 98.7%、RMBench 83.9%（M(1) 81.2%、M(n) 87.3%）。本轮只核对了表内算术：九任务成功整数合计755/900，组计数406/500与349/400；这不是一次新的评测。
- Table 4 的三参考数值、Tables 12–15 的独立PBB/PB/BT分支cohort、Tables 16–17的gate分布，以及Table 18的三训练种子constant-utility对照，必须保留各自源码、checkpoint、config、query/branch/label/trial记录，不能用第16.3节旧LIBERO部分采集替代。
- 已有第2/3/15节的旧LIBERO 1936/2000、N02 forced-null、N06/W08全零自然gate等记录继续保持原历史身份与边界。它们不否定也不独立支持最终稿的不同运行；本轮未联系服务器查新产物。
- 最终稿真机17/16/15/15 successes out of20属于稿件报告；本轮没有获得或核验对应自主trial原件。25条示范、20/5划分、1356事件与已有历史数据审计的数值对应，但不据此证明自主成功率或当前硬件标定。

### 17.3 独立内部对应问题

这些是审稿前内部核验事项，不是实际Reviewer Concern，也没有据此修改论文或算法。

1. 最终稿Table 8统一写write-threshold range为[0.08,0.85]，当前RMBench dataset factory和policy代码使用[0.015,0.25]。需由投稿运行的resolved config确认版本及分平台设置。
2. 当前research/online probe开关在训练期被明确忽略；它们支持固定checkpoint干预，不能单凭开关配置证明Table 3或Table 18的独立训练对照已实现或完成。
3. 在本轮定向读取的`src/scripts/configs/tests`和历史活动LaTeX中，尚未建立最终稿扩展centered-effect、full-selector和constant-utility研究的完整运行对应。搜索范围有限，不推断其他机器或团队不存在材料。
4. legacy payload corruption与最终稿完整事件random/similar replacement的干预定义不同；后者需核对独立donor计划及实际覆盖。

目前真实首轮审稿意见、评分与confidence仍不可见；约四名reviewer为作者预期。本轮不产生录用概率、模拟审稿结论或正式rebuttal。

## 18. 2026-11-07 外部专家材料对照、候选源工程与作者来源澄清（非新增实验）

本节记录本轮取得的源码与结果来源说明，没有启动训练、ACP、仿真或真机评测，也没有复算原始测量。第17节末尾的审稿状态是11-05历史时点；当前已读取三份官方审稿。

### 18.1 新材料与原件保存

- 已读取并对照 `tmp/rebuttal_wb/` 的逐条分析、统一补强、写作建议及新增七天计划；它们是参考意见，不代替原始实验数据或本对话授权。
- 新定位的本地候选工程为 `tmp/first_submission/WARM_ICLR2027_new/`。已核对活动附录链：主文件 → `sec/999_appendix.tex` → `sec/appendices/c_additional_results.tex:90` → `c_rebuttal_p0_p1_draft.tex`。完整编译与最终33页稿的精确对版仍未进行。
- `c_rebuttal_p0_p1_draft.tex` 第1–5行保留了旧 synthetic/TBD 说明，第323行也有 synthetic seed inputs 注释；可见单元包含与稿件C.2.2/C.2.3一致的候选中心化误差、完整选择率、配对区间及learned/constant utility对照数值。这是来源核对线索，不是新测量。
- 该文件本轮读取时的SHA-256：`1af74b4f8f09ca2b1e360338cd732e6b688cbf341180a870d2de01335fe6eba0`。
- 原工程的主文件、README和三份活动附录文件按字节保存在 `tmp/rebuttal/expert_reference/20261107/primary_source_snapshot/`；文件身份与作者澄清见同目录的 `primary_provenance_context.json`。本节追加前的实验记录副本为 `EXPERIMENT_RECORD_ZH.before_20261107_provenance.md`。原始工程与最终PDF未改动，原件及快照均不加入Git。

### 18.2 作者明确澄清与证据等级

作者在本对话于2026-11-07明确说明：**该文件夹中的数据均为真实可信结果；synthetic 是未及时删去的旧标记，并非虚假数值；原始结果存放位置已不可考。**

- 当前登记为：**作者确认的真实既有结果；原始产物路径不可追溯，本轮未独立重算原始记录。**
- 不再把“模板数值是否被当作结果”列为待作者确认的问题，也不因旧注释推断数值虚假。
- 可按既有投稿结果引用；不得写成“本轮已重新复现/已复核逐样本日志”，也不能承诺提供当前无法定位的原始run产物。
- 清理旧注释应在原工程归档、修订版本核对后进行；清理注释不改变数值及实验身份。本轮未执行该清理。

### 18.3 对论文主张与后续补证的具体影响

该澄清主要涉及引用C.2.2候选区分/完整选择器证据以及C.2.3学习utility对照时的来源状态，也更新全稿结果的作者确认口径。既有数字继续保留，不用历史smoke或其他run替换。

新增验证应优先服务HH-W5的候选敏感性、BB-Q2的生成修复/闭环贡献、RK-Q2的自然缺技能退让和公平成本收益。新实验必须保留checkpoint/config/source/data/split/seed、原始标签/输出和聚合过程；与旧结果不一致时并列记录、说明条件，不能用新运行倒推出旧运行身份或静默替换原表。

额外纳入WP2的设计注意点：完整初始化同时改变动作均值和噪声尺度，可使用现有`scale_only`推理probe做固定条件的敏感性诊断；该probe只在infer生效，不等同于匹配训练基线。本轮只有源码核对与方案补充，没有新增实验值。

## 19. 2026-11-08 Rebuttal首批CCI诊断与论文工程对版

### 19.1 运行端与不可变身份

- 用户明确授权在单张H100 80GB的CCI执行补实验。本次现场观察为H100 80GB HBM3、81559MiB；容器PID1启动时间为2026-11-08 06:50:59 UTC。初始GPU空闲，后续发现同卡其他WAMO作业，保留该作业；本次耗时不作为独占GPU的正式成本证据。
- 起始本地、GitHub、CCI代码均为`5eb321d95fe7f5e0db720e1f60df4d3926a9b1e7`。CCI工作树干净；本地既有文档删除未恢复、未暂存。
- 新输出根`S/rebuttal_20261108/`。运行源码为`baseline_5eb321d/code`，原Git archive SHA-256 `6964c7f257550acf5a5e921b11f07b374b60375dbc08ee313f5f60b0b4f66ca5`；runner源码集合SHA-256 `75cbcc171be0669f0808cacdc6317225c30836eaea49add85a1b32b29397c829`。首次tar解包因AFS不允许修改归属而报错；随后使用不改变归属的解包完成，运行前冻结源码，未影响既有实验。
- 只读盘点254个现有checkpoint/config/result元数据项，错误0；盘点不证明最终投稿模型身份。已找到的当前v8发布点为step3000，本轮不把它标为最终投稿模型。

### 19.2 V8-step3000 / DEV4 自然source pilot（工程证据）

- checkpoint：`S/nonreal_resume_20260922/stage1/training/checkpoints/weights/step_003000.pt`；SHA-256 `68c7f3840921241df45a5b6997b772db3a69a31584644aa0e66929b841b845bf`。resolved训练配置`resume_train.yaml` SHA-256 `34f5dc0a6f4cc354da61c0ba7bdc394b6c9652c7ef7706e72e8cbb683ec9a00c`。权重在运行前按字节验证，未修改checkpoint。
- Press Button、Put Back Block各1个DEV来源episode，各2个预冻结前缀；共2episode/4query，H32×14、NFE10、seed3407、BF16。调用自然Full并作同输入重复；没有强设门控，也没有动作适用性标签。
- **4/4 query自然g非零**；mean g=`0.18896484375`，mean noise-scale-squared=`0.7218399047851562`。失败query为0，模型脚本summary complete、runner exit0且源码前后身份一致；模型脚本elapsed=`497.01972905546427`秒，含初始化，不作为端到端策略成本。
- CPU只读检查确认checkpoint version=8，gate末层权重非零且bias不再等于初始-2；这只是张量身份/训练变化检查，不证明校准、收敛或任务收益。
- 原始输出`S/rebuttal_20261108/pilot_source_full/`；job `pilot_job_logs/20261108T070354Z-2367897a/`。archive `S/rebuttal_20261108/archives/pilot_source_full_evidence.tar.gz`，SHA-256 `8f4eb4a6719e830b7b29a6d709ac817e9263f53c7a8936d479ae6760c733ddad`，本地副本`tmp/rebuttal/execution/20261108/evidence/pilot_source_full_evidence.tar.gz`双端一致。
- 本地`verify_source_pilot.py`实际重算probe JSON/NPZ哈希、4个prefix身份、自然门值、source尺度和重复输出容差，结果见相邻`PILOT_VERIFICATION.json`。
- **支持的准确主张：** 当前step3000固定模型能在该DEV4小cohort产生非零自然source影响，具备继续做共同噪声source控制的条件。**不支持：** 适用性TA/FA、生成修复率、闭环SR、无技能退让、匹配训练比较、训练seed稳定性或最终投稿模型复现。不得把4个query当4个独立episode。

### 19.3 原稿对版与修订v1（非新增效能结果）

候选LaTeX工程已独立编译为33页。72dpi逐页渲染比较显示第2–33页逐像素相同，第1页只差OpenReview附加审稿提示；文本抽取差异主要是数学字体ToUnicode编码。源工程及原件未改动，完整副本和比较记录在`tmp/rebuttal/revision/`。

本地修订v1编译为34页，正文结束于第10页；只做监督语义、指标/cohort、已报告机制证据、M(1)负差、已知任务/桌面范围和实际AI用途澄清，保留原实验数值。新DEV4工程pilot没有回填为论文结果。仅在修订副本清理作者已解释的旧synthetic/TBD注释；原工程完整保留。源差异版本化为`docs/rebuttal/MANUSCRIPT_REVISION_V1.patch`。

最终v1交付件为`tmp/rebuttal/revision/WARM_ICLR2027_revision_v1.pdf`，SHA-256 `96ea251702bcc7020a6dee49fcb22a5142ecbf315afe9b2d3304bea40c788315`；完整可编辑源包为同目录`WARM_ICLR2027_revision_v1_source.zip`，SHA-256 `ea973e283dbaa2af294ff8af8d2f4820b31cd9ad3d8bb76b5a537e3b763d75b1`。34页均已渲染检查；原四张主表、C.2数值表和九个图文件保持原样。修订v1并非新增实验完成后的最终稿，未上传OpenReview或写入Inscrive。

### 19.4 DEV50 Full / scale-only / Gaussian共同噪声诊断（工程证据）

- 固定代码`S/rebuttal_20261108/dev50_23d86dd/code/`来自提交`23d86ddd4a21fbb07d46f7ccfcd020a939d702b7`；Git archive SHA-256 `cba0684dc30e5e32d82ace889be5bb062594b996d14cfcee5c2725f98e98a30a`，runner源码集合SHA-256 `ff6a66121cbede70ae6bcb52a8070eab969cf274871e0d48ffc1f472c9bd5951`，运行前后一致。checkpoint与训练配置同19.2，均在运行前核验哈希。
- Press Button、Put Back Block各5个DEV来源episode、每episode 5个冻结前缀：共10 episode / 50 query，每query三种source模式。H32×14、NFE10、seed3407、BF16。DEV4 pilot属于该DEV50集合，不将二者相加称为54个query。
- 逐query核对共享Gaussian噪声、conditioning、自然g、alpha、候选索引、adapted actions和valid mask一致；Full与scale-only的噪声尺度也一致。scale-only仅移除source动作均值，Gaussian另外恢复单位噪声尺度。这是固定checkpoint推理干预，不是匹配训练消融。
- **50/50自然门控非零**，三模式mean g均为`0.1885546875`。Full与scale-only的mean noise-scale-squared均为`0.7214434814453125`；Gaussian为`1.0`。
- 相对Full，scale-only的source RMS差均值为`0.16606547594070434`，生成action RMS差均值为`0.08441442349925637`；Gaussian分别为`0.2231895759701729`和`0.08490878326818346`。另外直接比较Gaussian与scale-only的action RMS差均值为`0.01029347077012062`。这些单位均为归一化模型动作坐标、先逐query求RMS再取均值，不能解释成物理动作误差或收益。
- 数据身份：bank `758bfee0225ef33fe1e158502949f296318144438cff531a84fe0c0ebc0c336c`；query corpus `9b568fad0eee677cf6df4d6335fe61c4b248d0bafa3941667596001a10a887ab`；normalizer `2cac1e1b434b2a339f2e4eddeab7815682e243d59428bc7dcaddca14f173746d`；冻结prefix manifest `9a03542fc9dc532558848a50ed39ee732a6854cd02b3511a488a441abeabf1c6`。
- 三模式子进程均exit0；job `dev50_job_logs/20261108T073515Z-4606f9d1/`，elapsed `1614.6069943904877`秒，包含三次初始化及同卡资源竞争，不作正式部署成本。原输出`S/rebuttal_20261108/dev50_source_modes/`；证据包`S/rebuttal_20261108/archives/dev50_source_modes_evidence.tar.gz`，SHA-256 `ebf1f1f6e4332d338ba78d35cb1e905f5d4ec505eb914031bd0d45043540ecb9`；本地同名副本在`tmp/rebuttal/execution/20261108/evidence/`。
- 本地从归档原数组重新运行`scripts/verify_nonreal_source_archive.py`并计算额外RMS，结果为`evidence/DEV50_VERIFICATION.json`。JSON/NPZ哈希、共同输入、源码身份、样本矩阵和失败状态均通过核验。
- **支持的准确主张：** 当前v8-step3000在这10个DEV来源episode的事实历史输入上，对source动作均值存在生成输出敏感性；固定方差时差异仍非零。**不支持：** 候选适用性、生成修复/损害率、闭环成功率、自然无技能退让、匹配训练公平比较或最终投稿模型复现。本次没有独立分支标签或仿真策略执行，不把50个相关query当50个独立episode，也不将诊断写入论文效能结果表。

### 19.5 进程内图形运行环境修复与相机读回（工程资格检查）

- 原有RMBench runtime checker先报`failed to find a rendering device`；现场driver为`580.95.05`，MIG Disabled。容器内缺少对应Vulkan/EGL组件；未修改内核、系统驱动、设备或挂载，未重启容器，未终止其他项目作业。
- 从NVIDIA官方取得精确版本`NVIDIA-Linux-x86_64-580.95.05-no-compat32.run`，325,359,760 bytes，官方及双端文件SHA-256均为`482130b27454864f3873729debbffa9e317badd1c56271dd811219c2385da575`。服务器直接下载出现EOF/慢速，保留失败partial；改由本地核验下载后上传。**只运行extract-only，不运行安装器。** 运行时提取位置`S/rebuttal_20261108/runtime_verified_client/extracted/`；大体积二进制不入Git。
- 新增`scripts/setup_rebuttal_graphics_env.py`，验证提取库版本与当前驱动一致，只在`S/rebuttal_20261108/graphics_5809505/`写进程专用ICD/EGL JSON和`graphics_env.sh`。现有不同配置拒绝覆盖，不改全局环境或site-packages。GLX库SHA-256 `ffcafdde84c33cd14ca52d06873fce20eed7d26e10516ec61fe025a3d9fe704e`；EGL库SHA-256 `7210a8a23fad4bc81cca1337bd5f5e2fabba8e27e42770201e2444460a9510f9`。
- 在单次命令中source该环境后，原有未修改的`check_warm_rmbench_eval_runtime.py`通过：Python3.10.20、Torch2.7.1+cu128、SAPIEN3.0.0b1、MPLib0.2.1、H100、press_button、renderer=ok。该检查不是完整任务重置或策略评测。
- 追加实际相机读回：128×128场景、原运行端rt shader / 1 sample / path depth1 / oidn设置；读取`Color`为有限且非均匀的float32 RGBA `(128,128,4)`，RGB标准差`0.10467271506786346`，原数组SHA-256 `656f17804a1a90ccaeb2aaaf1a615687ce5c8f08287cb94deaaf90b6f9b6d1a3`。场景含ground与红色box，无机器人、任务标签或策略执行，不能用作论文视觉或实验结果。
- 证据包`S/rebuttal_20261108/archives/graphics_qualification_evidence.tar.gz`，SHA-256 `6394494630c74189690d8c514ec523ac3af72db5133d151f641aad79447d1734`，本地副本在`tmp/rebuttal/execution/20261108/evidence/`；包含修复前后日志、实际读回数组/图像、配置、库哈希和资格脚本。原官方驱动包单独保存在运行目录及本地私有cache。
- **资格边界：** 只证明当前环境可初始化渲染并从相机读回。完整RMBench任务重置、controller/task/RNG/history的全状态恢复、A-A-B-A、独立分支标签以及正式模型身份仍未验收；不能据此声称恢复cohort已开始或已经获得任务成功率。

### 19.6 完整任务资格首次失败：cuRobo CUDA kernel（工程失败保留）

- 使用新脚本`scripts/qualify_rebuttal_rmbench_task.py`，预声明`rearrange_blocks`、seed3407、1次重置、最多32条保持当前qpos命令。它不是学习策略，不计算修复率或任务成功率。官方`envs/task_config`复制到`S/rebuttal_20261108/rmbench_qualification_source/`，assets仅链接既有固定仓；源码集合SHA-256 `a97e308ca053c3aeb1bfec8e3c4c4f550aac7f6ae753e36bbce82e9522292379`，运行前后一致。
- job `task_qualification_job_logs/20261108T083310Z-1cfb152b/`，elapsed `61.417216777801514`秒、exit1。在`setup_demo`内部，cuRobo `MotionGen.warmup`的CUDA kernel报`illegal instruction`，随后规划子进程退出、父进程收取开夹爪规划结果时报`ConnectionResetError`。重置未完成，**相机任务观测0、保持命令0、有效策略episode0**；不能登记为已通过的完整任务资格。
- 原日志和失败summary原样保留在`task_qualification_rearrange_blocks/`及相邻job目录；双端证据包`archives/task_qualification_failed_evidence.tar.gz`，SHA-256 `e1ca1cb30f28e449362f68d82a0034bcc3bba1527db5d296329d5c64bcb4751b`；本地副本位于`tmp/rebuttal/execution/20261108/evidence/`。未以另一个seed重试掩盖错误，未绕过官方规划器，`battery_try`尚未启动。
- 环境排查定位现有系统nvcc12.4，以及包含sm_90的已安装cuRobo扩展；尚不能仅凭此证明根因。参考[cuRobo维护者的H100兼容性讨论](https://github.com/NVlabs/curobo/issues/481)，正在保留原site-packages的条件下，以官方CUDA12.8编译器、相同固定cuRobo源码和相同编译选项建立私有扩展。此诊断不关闭正式实验的模型、全状态恢复或标签缺口。
- CUDA编译器包来自NVIDIA官方Ubuntu2204仓，七个deb逐文件按官方Packages SHA-256核验，只解包到本次输出根，不通过apt/dpkg安装、不修改系统CUDA。包清单及哈希见`runtime_verified_client/package_manifest.json`。首次`dpkg-deb --extract`因AFS禁止chown而失败，保留`cuda128_private/`和原日志；r2改为fsys-tarfile加no-same-owner解包到新目录，未覆盖第一次失败。

### 19.7 私有CUDA12.8扩展编译完成，原seed任务复验（工程资格）

- 以私有CUDA12.8.93编译器、Torch2.7.1+cu128、sm_90、原cuRobo源码及编译选项构建五个扩展；原site-packages、系统CUDA和官方模拟器源文件不修改。213份原源码/构建文件逐文件核对全部未改变，包括算法实现；结果见`curobo_source_build_verification.json`。无Git元数据的复制工程先因setuptools-scm版本解析失败，保留r2日志；r3仅设固定版本元数据0.7.8后构建成功，没有更改优化算法或关闭CUDA graph。
- 构建证据包`S/rebuttal_20261108/archives/curobo_cuda128_build_evidence.tar.gz`，SHA-256 `32f4c83781c564d4183e133e93a299b6cef72c15918ce454fad1968ff1306778`；本地同名副本在`tmp/rebuttal/execution/20261108/evidence/`，哈希一致。包内含所有构建尝试日志、七包官方来源/哈希、构建脚本、未改动的源码及五个新扩展和逐文件哈希。编译器二进制及deb另存私有cache和运行目录，不入Git。
- 验证进程只增加私有cuRobo源码路径和可写Torch kernel cache，沿用19.5的进程专用图形环境。首次单独import扩展的preflight遗漏先import Torch、报libc10缺失；该次未启动任务，日志保留。修正import顺序后实际import路径指向私有源码和新LBFGS扩展。
- `rearrange_blocks`复用同一seed3407、同一冻结官方任务源码和配置。job `task_qualification_job_logs/20261108T084851Z-c17d620a/` **complete / exit0**，elapsed `82.05043125152588`秒；任务重置、4路240×320 uint8 RGB、32条当前qpos保持命令及末端观测通过。官方任务源码与runner源码运行前后均一致。此复验将环境缺陷与seed差异区分，但不单独证明所有cuRobo CUDA路径均已合格。
- `battery_try`随后使用同一seed3407、同一冻结源码和私有扩展。job `task_qualification_job_logs/20261108T085123Z-f71f135b/` **complete / exit0**，elapsed `78.30343174934387`秒，同样完成重置、4路相机及H32关节保持。合计为两个任务各1次工程资格、64条保持命令；不计作WARM评测episode。
- 修复后任务证据包`S/rebuttal_20261108/archives/task_qualification_cuda128_evidence.tar.gz`，SHA-256 `39e1964069d0e39e9ccb72c8df3d34c15c4e15bba932510eabba72a29ec14313`；本地同名副本在`tmp/rebuttal/execution/20261108/evidence/`。本地独立验证两份runner exit0/源码前后身份、官方源码逐文件哈希、entry及五个扩展输入哈希、64步计数、16份相机原数组、PNG逐像素一致和summary标准差；结果见`TASK_QUALIFICATION_VERIFICATION.json`，两个head-camera初始图已人工查看。原环境1次失败与私有扩展2次通过分开保留。
- SAPIEN当前公开API包含scene `pack_poses/unpack_poses`及PhysX `pack/unpack`；只记录API可用，不把这些API等同于完整快照。任务具有stage_id、press_cnt/flag、eval_success等状态，控制器、规划子进程、任务反馈、RNG及事实历史的恢复仍须另行实现并通过A-A-B-A。
- **论文边界：** 这些是初始化/图形/关节命令工程资格；没有调用WARM策略或独立动作标签，保持命令的`eval_success=false`不构成策略失败率。恢复率、自然无技能退让和配对闭环SR仍未采集，DEV50诊断仍不替代它们。

### 19.8 本批提交、验证与后续入口

提交`23d86dd`已推送并同步CCI；随后`20fd522`同样推送并验证GitHub/CCI HEAD一致。两端定向28项测试通过；DEV50与两任务资格均另做真实归档/数组复核，不以单测代替实测资格。论文源差异通过对归档底稿的git apply检查，最终PDF/源ZIP哈希同19.3。原有本地文档删除未暂存，未触碰真机主路径、数据或checkpoint。

首批工作稿是三位审稿人20项意见的回应及正文10页/全34页修订v1；仍未在OpenReview发评论或上传PDF，也未改写Inscrive。下一批先实现并验收完整状态恢复、冻结独立标签与正式模型/cohort，再采raw/refined/null、非空缺技能配对闭环和可控成本。所有工程资格作业已退出，不将排队、构建成功或GPU占用宣称为正式实验在跑。

## 20. 2026-11-08 后续完整状态恢复工程探测（未取得科学采集资格）

### 20.1 新后端、隔离范围与首次恢复失败

在第19节首批完成后，继续实现实验性`RMBenchQposBackend`。只允许已静态检查的Rearrange Blocks / Battery Try、CPU PhysX和绝对关节命令；显式保存原生物理包、scene pose cache、关节drive targets/velocity targets、qf、机器人/TOPP值状态、任务变量、相机/观测缓存、Python/NumPy/Torch/CUDA RNG和调用者提供的策略指纹。任何分支内cuRobo RPC均立即报错，不将未捕获的规划子进程状态默认为已恢复；官方TOPP异常或空轨迹也会使探测失败。该后端没有改写官方任务、训练或真机路径。

本地相关测试38 passed，其中3个新状态/guard夹具仅验证字段恢复、opaque拓扑变化拒绝及RPC拒绝，不能代替真实SAPIEN资格。真实探测从`93a827f`基线加三份明确overlay文件构建不可变源码：r1 archive SHA-256 `500a52da50b8910bc364efbbfcca197fe216c03ddb427500db296c17ccbe7d4a`，实际双端一致；文件清单见`restore_probe_round1_manifest.json`。

r1 job `restore_probe_job_logs/20261108T090814Z-8c6c0b8b/`在seed3407重置及32条保持命令后，**首次parent restore即失败**，没有开始A-A-B-A分支；控制器、任务、RNG、观测缓存及无策略占位指纹匹配，physics指纹不匹配。密封原件为`S/rebuttal_20261108/archives/restore_probe_r1_failed_sealed_evidence.tar.gz`，SHA-256 `b1208e903164a3b249ed2b62e533023b6eb4fe63bf7781631cda8f2a8afd8668`，本地同名副本在`tmp/rebuttal/execution/20261108/evidence/`。

r2只增加恢复前原始state落盘，保持同一seed、动作和严格标准；再次在parent restore失败。原始包逐字节比较：**PhysX CPU包832 bytes完全一致；scene pose包2044 bytes有26 bytes不同**，两次restore的差异一致。该发现只定位到scene pose cache，不把它解释为策略误差或放宽容差后的合格结果。r3正在检查两个原生恢复API的执行顺序；所有失败目录和源码快照保留，正式恢复率和闭环采集仍未启动。

### 20.2 恢复顺序修复、重置状态通过及抓取状态的任务差异

r3调整为先原生PhysX unpack、再scene pose unpack后，parent及最终恢复均精确匹配；测试随后在零位移保持命令的官方TOPP空轨迹处停止。r4将资格动作改为双臂±0.005rad交替小幅轨迹，B在左臂另加0.01rad，使验证实际经过TOPP且B物理端点不同；没有放宽恢复或端点容差。完全相同的hold path允许官方零时长轨迹，非零路径空轨迹及异常仍失败退出。

- Rearrange Blocks r4：job `20261108T091827Z-3a26fb6d`，exit0、130.65657329559326秒；Battery Try r4：job `20261108T092254Z-830bc879`，exit0、131.114102602005秒。runner源码集合均为`dc421016f5c20e536c856d7225b41d9d59a6ac17f9263971b04c734c3474bdc4`，运行前后一致。两任务在重置后保持状态完成H32 A-A-B-A，报告物理/controller/task端点及RGB精确一致。每任务一个seed/一个状态，不代表接触、插入或任务反馈阶段已合格。
- 新脚本`probe_rmbench_grasp_restoration.py`在官方专家的首个grasp macro后检测机器人—目标物的正接触冲量，再运行同一严格探测。Rearrange Blocks job `20261108T092713Z-13eeb1fd` exit0、135.77174854278564秒；检测到fr_link7/8与box接触，并通过该单一抓取状态的H32 A-A-B-A。没有运行学习策略。
- Battery Try同一抓取探测job `20261108T093136Z-8165cafc`失败：抓取接触存在，但首次parent restore时CPU包832 bytes的4 bytes变化（offset728/732/736/740），scene pose包2100 bytes完全一致。其他五类指纹匹配；没有开始A-A-B-A分支。因此不能将前述reset或Rearrange通过扩展成Battery接触阶段合格。
- 两次grasp源码集合均为`1f720ee9c93d76e1a8e403a2e99e5eb5adf32a08b436c3fb815ffea66fb73585`。更严格的、要求原native bytes不变的v2 query canonicalization也在Battery抓取状态失败（job `20261108T093926Z-1844328d`，exit1、77.20895624160767秒），原件保留。
- r1–r3封存包`archives/restore_rounds1_3_evidence.tar.gz`，SHA-256 `15d10227524546105f4189c21cb7ac018dcbddcb453a99fc48825b2764960fc5`，服务器与本地`tmp/rebuttal/execution/20261108/evidence/`均保留；后续各次独立输出继续保留，不覆盖失败。

### 20.3 显式原生checkpoint查询协议（工程开发，不替换既有结果）

为区分“原始live pose编码保持不变”和“所有分支从同一可幂等恢复的native checkpoint出发”，新增`prepare_native_query`，**在任何查询观测/动作封存之前**最多进行5次无物理step的native roundtrip，保留每次CPU/scene pose前后原包。要求task/controller/RNG全程不变，最后两次CPU与pose包精确相同；不收敛即停止。若原live CPU编码改变，proof必须明确记录`original_cpu_encoding_changed=true`，**不能声称恢复了原live编码**。这是另一个明确标识的采样协议，不是放宽bitwise检查，也不据模型分数、标签或收益选择状态。

v3完成两个任务各自的reset和首个官方专家grasp状态，共4个工程状态，seed均为3407。每状态执行H32 A-A-B-A、5次parent恢复；合计512条分支命令。六类端点状态与RGB精确一致，B物理端点不同。两个grasp状态均检测到机器人—目标物正接触冲量；没有运行学习策略、没有独立动作标签，也没有任务成功率。

- v3固定runner源码集合SHA-256 `04d0eaf9d3c62872ab4dc147ee64f3aba68e010cd539a4e2863ddb7d160391f7`，四次作业运行前后一致。Battery grasp job `20261108T094526Z-5d8f3d83`、Rearrange grasp `20261108T095016Z-31dd5fc7`、Rearrange reset `20261108T095235Z-7214672e`、Battery reset `20261108T095445Z-0b625621`均exit0，runner分别耗时142.5017、137.1166、128.5012、127.9041秒。这些耗时不是策略成本。
- 四状态均在2次native roundtrip后达到固定点。Battery grasp明确记录原live CPU编码改变，另三状态未改变；所有分支均从准备完成后封存的checkpoint出发。task/controller/RNG不变由实际运行代码断言；准备阶段未完整输出这些状态，因此不声称已对该断言作独立原件重算。
- 双端证据包`S/rebuttal_20261108/archives/restoration_native_v3_evidence.tar.gz`，SHA-256 `70165ee8d1a093bbfb25900ea97192721b2b773974674a0d1c8784ee030c9d8d`。本地`tmp/rebuttal/execution/20261108/evidence/RESTORATION_V3_VERIFICATION.json`独立核对原生固定点包、五次parent恢复、六类A端点、RGB和B差异。未使用重新pickle后的跨NumPy版本哈希代替原数组比较。
- 原v1三通过/一失败的封存包`restoration_v1_evidence.tar.gz` SHA-256 `b9efda92e446c0979acc14f05d0979d5e09d1f32663319afd1192d6886016edc`与v2失败继续保留，不用v3通过覆盖失败身份。

当前本地54项定向测试通过；状态夹具、固定点与预算拒绝测试只验证软件逻辑。四状态的真实工程探测不能代替其他seed/任务阶段、实际WARM记忆与独立标签集成。`full_scientific_qualification=false`保持不变，正式恢复/退让/闭环成功率尚未采集。

### 20.4 作者固定本轮补实验模型

作者在本对话明确说明最新训练模型就是论文模型，并指定在当前只有v8-step3000的情况下使用该模型。本轮接受此指定；第19节关于尚未确认模型的表述为此前时点，已由本节更新，不再向作者重复索要同一信息。

- 本轮RMBench补实验固定checkpoint SHA-256 `68c7f3840921241df45a5b6997b772db3a69a31584644aa0e66929b841b845bf`，训练配置SHA-256 `34f5dc0a6f4cc354da61c0ba7bdc394b6c9652c7ef7706e72e8cbb683ec9a00c`；路径同19.2。选择记录为`tmp/rebuttal/execution/20261108/MODEL_SELECTION.json`。
- 对应训练attestation SHA-256 `4254bf00d70872d959a8cb62f0e14771a0ca183f6184ae7c4e653dfe243e5b4b`；其resolved-config摘要`35691ae9c35f5959d2d6a94c9fb72505c4dd21117de414911ed212338551ee83`与保存的YAML文件哈希承担不同角色，不混用。attestation中git_commit为全零快照标记，原件不修改；实际推理源码身份单独记录。
- 实际训练元数据为version8、shared stage、step3000/configured max30000、训练4卡/effective batch128、BF16。单卡推理沿用recent-event容量20、action-summary容量8、summary chunk4、H32、prefix4和NFE10。这些事实不单独证明训练收敛，也不重建历史900回合的逐trial链，不将RMBench14维模型当LIBERO7维模型。

## 21. 2026-11-08 固定模型记录观测成本评测

### 21.1 输入、合同与计时边界

采用作者指定的v8-step3000，预声明Press Button DEV来源5个episode，每episode20个固定前缀，共100个计时query，另20次预热。当前图像与状态来自记录前缀，未来观测在输入策略前移除；事实动作仅用于重放已执行历史。模型生成动作不在模拟器执行，不能计算或推断成功率。H32/prefix4/NFE10、BF16、K32，CPU精确cosine检索与已配置的近期记忆均保留。计时以CUDA synchronize包围完整部署`_replan`，包含预处理、检索、模型生成、队列与telemetry，排除磁盘解码；初始化单独记录。

- 严格online bundle构建job `cost_job_logs/20261108T100307Z-94e9e390`，exit0；bundle manifest SHA-256 `11a2d9b6a80caebae65acbcf82c04db169036c46d1cb76759550e100afc08c96`。Press Button合同canonical SHA-256 `d85addd002492d4b1d654639ad458ca6a4956898c116845d0d4e8f1f9c8505f5`，runtime projection SHA-256 `1d0a900e42ced039c726bd7810fc253a74376f2226256fda0e0bb904f5e33da2`。构建遵守原训练attestation，不修改或绕过合同检查。
- data-only preflight job `20261108T102110Z-fb140d7c` exit0，100个记录前缀全部通过身份、无padding、相机尺寸及动作归一化检查；catalog SHA-256 `d4a13afc421d27846925450e41ff316c26a1db781168556a8a5b3bab9d22392f`。
- GPU评测job `20261108T102525Z-1bb6fa74`，10:25:25 UTC启动，max_seconds1200；输出`S/rebuttal_20261108/step3000_cost_warm_dev100/`。新增独立GPU进程guard，在起始无compute PID的条件下每5秒记录，若观察到其他进程则不作为可控成本。该抽样不等于硬件独占保证。
- 核心policy/builder固定来自`dev50_23d86dd/code`；计时脚本是另一个不可变overlay。import harness先核对核心模块实际路径，再运行overlay；`step3000_cost_profile_source_binding.json`保存双方身份。这样保持原sim-config相对根目录和严格合同projection，无全局变量替换、合同伪造或活动源码改写。

当前尚未验收100条计时、20次预热、策略telemetry覆盖及原始数组；不把数据预检、GPU占用或模型加载写成已取得成本结果。先前smoke/共享GPU时间不替代本次测量。FastWAM成本参照仍需相同输入、预算与清晰checkpoint身份；成本比较不等同于匹配训练性能比较。

### 21.2 WARM测量完成与独立验收

上段为启动时状态；本次job现已complete/exit0，2026-11-08 10:25:25–10:41:22 UTC，runner elapsed957.1772秒。源码集合`71e5393385d1f326092e3b94a26ab487c2c666ca7651b0f634116821e9e7e170`运行前后一致，核心policy集合仍为`ff6a66121cbede70ae6bcb52a8070eab969cf274871e0d48ffc1f472c9bd5951`。本地从原始timings和policy telemetry重算100计时/20预热、完整顺序、6次episode边界、每次事实执行历史游标和checkpoint/合同/seed/task身份，通过全部检查。

| 项目 | 本次实测 |
| --- | --- |
| 计时cohort | Press Button DEV，5个来源episode × 20前缀；另20次预热 |
| 部署预算 | H100 80GB / BF16 / batch1 / H32 / prefix4 / NFE10 / K32 |
| 全重规划中位数 | 0.5300586130470037秒 |
| 全重规划p95 | 0.565220080781728秒 |
| peak allocated / reserved | 23.620065689086914 / 23.740234375 GiB |
| bank规模与检索 | 89,680 events，CPU NumPy精确cosine检索 |
| 配置recent / summary容量 | 20 / 8 |
| 实际recent / summary占用 | 0–8 / 0–8；recent容量未饱和，不是最坏历史长度测试 |
| 策略构造 | 854.9277074038982秒；含本次AFS部署的严格资产校验与加载，单列，不计入query延迟 |

- 100/100 query自然gate非零；learned gate范围[0.1845703125,0.193359375]、stagnation[0,0.296875]。这些是计时覆盖信息，不证明适用性或效能。预热只排除预先声明的20次，没有按延迟或门值选择计时query。
- guard保留189次采样，间隔约5秒，无foreign compute PID、无采样错误，exit0；仅支持“采样未发现其他计算进程”，不声称硬件独占保证。
- 双端原件包`S/rebuttal_20261108/archives/step3000_recorded_dev_cost_evidence.tar.gz`，398,866 bytes，SHA-256 `024a2186e56e0e0f768a3419d085b40d1a601c096758d1182aee25b2b201aabd`。本地副本及`STEP3000_COST_VERIFICATION.json`在`tmp/rebuttal/execution/20261108/evidence/`，实际重算数值和哈希一致。
- 冻结计时脚本的summary scope遗留了旧`smoke checkpoint`短语；原summary保留，不改原件。checkpoint哈希、合同和作者指定均明确为v8-step3000，正确范围由manifest和本节勘误确定。当前源脚本已移除硬编码身份短语，不改变本次数字。
- **准确论文主张：** 指定v8模型在上述DEV记录观测和实际占用范围的完整部署重规划成本。**不支持：** 机器人控制频率、实时闭环稳定性、模块耗时分解、满20近期事件的最坏成本、MemoryWAM成本、匹配训练优越性或成功率。

### 21.3 FastWAM参照准备与在途状态

新增`scripts/profile_nonreal_fastwam_replay.py`，严格复用已完成WARM成本cohort的全部前缀、预热和原相机哈希，冻结相同H32/prefix4/NFE10/BF16。使用官方baseline部署及其数值预处理；WARM计时含内置telemetry，baseline外部证据落盘不在计时区间，二者边界明确保留，不对两种路径作逐模块归因。

发布checkpoint为`W/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt`，预期SHA-256 `776475b22566a791854ecf31cf3b50f25e7d8d94c343132ec16eb94994aa9e63`。这是推理计算参照，不标为同数据微调后的性能基线。job `cost_job_logs/20261108T104340Z-b70b4ff2`，max_seconds300；冻结core与脚本overlay独立绑定，离线加载，未修改任何活动模型或论文数字。当前尚待该作业退出和原件验收。

该job现已complete/exit0，10:43:40–10:48:35 UTC，runner elapsed295.3029秒，在预算内完整结束。源码集合`2aae21a35e7f99586bbfb8a9bac184c5455f745fad4c17792274c83a08aef81d`运行前后一致；正式baseline/core policy来自23d86dd的同一不可变快照，外围计时脚本单独记录。与WARM逐行核对120个包含预热的query：episode/frame顺序、三路原相机哈希、GPU/BF16/batch/H/prefix/NFE/task、catalog和冻结episode均匹配。生成的4×14执行前缀均有限且保留哈希，没有在仿真执行。

| 实现 | median / p95（秒） | peak allocated / reserved（GiB） | 策略构造（秒） |
| --- | --- | --- | --- |
| WARM v8-step3000 | 0.530059 / 0.565220 | 23.620066 / 23.740234 | 854.927707 |
| FastWAM release compute reference | 0.366320 / 0.381831 | 23.307467 / 23.580078 | 144.288312 |

中位数之比为1.4469813479305644，allocated显存峰值差0.31259822845458984 GiB。这是两条现有部署实现的成本对比，WARM计时包含内置telemetry、baseline证据写出在计时外，保留各自官方数值预处理；不能将差值全归因于某一个模块。GPU数值不包含CPU bank内存或系统RAM峰值。构造耗时受AFS/校验与加载路径影响，不当作跨机器通用初始化benchmark。

FastWAM guard有58次约5秒采样，无foreign compute PID、无采样错误。双端证据包`archives/fastwam_recorded_dev_cost_evidence.tar.gz`，68,048 bytes，SHA-256 `ce80f99930c89c268306ded82fe1537bb4d166280a44c449f5ecc6671d04712f`；本地独立验收见`FASTWAM_COST_VERIFICATION.json`。原strict v2恢复失败也补做双端密封包`restoration_strict_v2_failure_evidence.tar.gz`，SHA-256 `305dc1407a105e485495ce9d773417c2ef3dbb258755d7c2692ba3a6c821dd10`，未覆盖此前失败。

**本批支持BB-W5/Q4、HH-W2的明确成本数值，尚不关闭MemoryWAM成本、匹配训练简单策略及成功率—成本权衡缺口。** 不把release计算参照的模型身份改写为匹配训练性能基线。当前本地55项定向测试通过；服务器对f5f70de版本的54项通过，新baseline脚本由上述完整真实运行验证，最新55项服务器复验另记。

## 22. 2026-11-08 成本证据整合到回复与修订v2（本地工作稿）

### 22.1 回复、修订与原件保护

- 三位审稿人的20项意见继续完整路由，新增BB-W5/Q4和HH-W2的成本数值及边界；草稿在`tmp/rebuttal/responses/20261108_v2/`，合并源副本为`docs/rebuttal/RESPONSE_DRAFTS_V2.txt`。同时拆为15个内部concern评论块，每块少于4000字符；这是内部编辑预算，不声称已核实本届单条评论表单上限。尚未发布任何评论。
- 修订v2复制自保留的原LaTeX工程派生v1，只新增Appendix C.4 / Table21（p.35）的记录观测成本。正文仍10页、全文35页；前34页与最终v1 PDF的重新渲染逐像素一致。原效能数字及九个图文件不变；没有把DEV source敏感性、工程恢复命令或cost query当作任务成功率。
- PDF为`tmp/rebuttal/revision/WARM_ICLR2027_revision_v2.pdf`，43,124,215 bytes，SHA-256 `76d51dbbee3ef0c5db2fd2bee19ebefe25a540e5cc0044c2070e8f4c3911a123`。完整可编辑LaTeX源包`WARM_ICLR2027_revision_v2_source.zip`，35,900,721 bytes，SHA-256 `b7622e239d0db2f30166d77e6994c5aa5fae6678c6d5ee3a586abf89dfaf60d6`。原投稿工程、v1及旧QA图像全部保留；v2未上传OpenReview，未写入Inscrive。
- 全部35页均渲染；新增成本页及notation table分页已按最终PDF检查。先前`qa_revision_v1`的21–23页PNG早于最终notation分页，本次另从交付PDF重新渲染全部34页，使用`qa_revision_v1_final_recheck/`比较；没有把旧PNG冒充最终PDF。最终日志无overfull、missing glyph、未定义引用或待重跑标签。详情`revision/DELIVERY_QA_v2.json`。
- 小型版本化源差异`docs/rebuttal/MANUSCRIPT_REVISION_V2.patch`已通过对归档原稿副本的`git apply --check`。生成PDF、源ZIP、图像、逐query日志、checkpoint和bank继续留在Git之外。

### 22.2 当前入口与后续效能工作

当前工作索引和执行队列已更新；后续快照为`tmp/rebuttal/execution/20261108/FOLLOWUP_STATE_20261108.json`，交付哈希清单为`FOLLOWUP_DELIVERY_MANIFEST.json`。首批`EXECUTION_STATE.json`仍保留其历史时点，不静默重写。最新模型指定已解除模型选择缺口，全部本批GPU作业已退出并完成原件归档。

后续科学采集仍需实际WARM memory集成、逐任务阶段资格、预冻结独立进展标签和query cohort，再执行sealed raw/refined/candidate-null分支、自然非空缺技能及配对闭环。成本数据与四工程状态不关闭这些缺口，也不建立主表总体领先的显著性。外部专家模板的证据分级与可追溯包建议继续采用，但不采用为了避免数值差异而回避必要复验、或以定性方法表替代匹配训练对照的建议。
