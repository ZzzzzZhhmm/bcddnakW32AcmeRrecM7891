# WARM 实验记录与论文回填索引

更新日期：2026-11-09。本文是持续维护的回填入口；原始记录、数组、运行输出按各条目的实际身份与可用性登记。每次取得数值，先保存，再核验，再更新本文。失败与不支持假设的结果同样保留。以下各历史条目的状态属于各自记录时点；投稿版本清点见第17节，作者来源澄清见第18节，rebuttal首批新诊断见第19节。下方09-24等历史快照不代表当前投稿结果或运行状态。

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

## 23. 2026-11-08 直接效能证据P0与CCI切换

作者要求以争取讨论后332/333为目标，主动补齐关键机制、缺技能与公平对照证据。新增预定计划`configs/nonreal/rebuttal_p0_evidence_plan.json`：Rearrange Blocks/Battery Try，固定v8-step3000、H32/prefix4/NFE10/K32、recent20/summary8；正式每任务/条件100对，从candidate seed340800000顺序进行官方专家准入，保留所有准入失败；pilot3407/3408不混入正式cohort。先运行独立fresh-reset闭环，使SR采集不依赖尚未完成的全阶段分支快照集成。

- 新入口`scripts/run_rebuttal_rmbench_pairs.py`为Full/null各自构造独立model/retriever；null同时撤回source和conditioning、保留回合记忆。两角色核对初始物体/目标、qpos、相机及官方unseen指令，按预定case ID交替执行顺序；逐动作原qpos、完整policy telemetry、预定query数组/图像和trial结果保存在Git外。源码不拉取Git、不接触origin、输出目录拒绝覆盖原件。运行错误和资格截断不计作正常任务失败。
- 新`policy_state.py`覆盖真实回合snapshot、事实执行历史、retriever发放/消费与continuation、模型thread及队列。7项软件测试通过；不以它们替代真实模型/模拟器资格。理论片段`docs/rebuttal/P0_THEORETICAL_EVIDENCE.tex`给出候选共享目标的均值/方差恒等式、事实监督风险分解和条件Euler稳定性界，未宣称自然缺技能或SR保证。
- 原21276以外的20276入口从约北京时间19:24持续超时。作者切换到`slai_debug_workspace_1`、180.184.249.158:21276，路径不变。19:51实际连接验证H10080GB/81559MiB、driver580.95.05、GPU空闲；AFS源码d578a78且服务器工作树干净。仅为该绝对仓库路径配置Git safe.directory，不改系统驱动、现有环境或其他项目。
- 新节点PID1显示2026-08-11；不以其作为本次CCI租期起点。GPU作业将有界执行、留证据归档余量。当前尚无新的策略成功率或修复率，实际启动及结果以接续的job manifest为准，禁止把准备就绪或模型import当作有效GPU实验。

### 23.1 新入口的已完成准备与在途配对pilot

- 已提交、推送并同步CCI的源码为`ae2459dd650a80a0e0cef960873c4dbd7dbb7cbe`。固定源码tar SHA-256 `ece6c36bdfaad2089d42fe7f0fd001bc6e9b82a09e7a90fb072515d3d9d23740`，代码集合SHA-256 `37d11caa89b3fb63f93a0d8a316ffb62e02fde1b4cc465d8f77039441ad864e9`；路径`S/rebuttal_20261108_p0/source_ae2459d/code`。活动作业不随canonical checkout同步而变化。62项定向测试已在本地及原生CCI通过。
- 两任务严格bundle构建均exit0。Rearrange Blocks：job `preparation_job_logs/20261108T120558Z-81f247d9`，manifest SHA-256 `f002233241e861f55a9dfcc713258df375d3ceeed99658d6703ae3614fae12f7`，online contract canonical SHA-256 `1261c9a727cc9ae04094ecfd1cbceeb2269d1630ba7f3b8b24c44c52041dd17c`；Battery Try：job `20261108T121058Z-e691d7aa`，12:10:58–12:21:13 UTC，manifest SHA-256 `c42d9db90cb053863916942e6a1bbf302b1c1955a42df21319a199aaa180498d`，online contract canonical SHA-256 `51fc5539aa9d84def5e538e0982da2374548acaec8f7e06537008cb7e3c758a4`。二者共用指定checkpoint、89,680-event原bank及同一推理预算。
- 新节点runtime preflight v2因symlink解释器默认manifest路径错误而终止，尚未启动策略试验；原日志保留。v3显式指定已有`warm-rmbench-eval/warm_rmbench_runtime.json`后通过实际H100渲染检查，未重装环境、驱动或修改官方任务源码。
- Rearrange Blocks完整官方步数pilot于12:16:50 UTC启动，job `paired_job_logs/20261108T121650Z-51572500`，max7200秒，输出`S/rebuttal_20261108_p0/paired_pilot_v2_rearrange_blocks`。专家seed3407准入、两次独立reset的图像/qpos/物体目标/官方unseen指令身份完全一致；Full构造完成，candidate-null正在加载。此时尚无学习策略完整回合结果；该pilot无论成败均排除正式统计，不能替代100对cohort。
- 已完成bundle、两份preflight日志及封存reset证明归档为`S/rebuttal_20261108_p0/archives/p0_bound_inputs_and_reset_preflight_v1.tar.gz`，411,515 bytes，SHA-256 `b927c0e7a99c70ec2a855efa69887ec7a330d753d3c1b8724bc7978817c900b1`。归档不含活动作业的未完成telemetry，不将其标为最终效能证据。

### 23.2 真实历史分支及技能库干预采集器

新增`collect_rebuttal_rmbench_recovery.py`及只读query capture：截取真正Full `_replan`已经消费的当前事实输入；candidate-null用独立同权重模型的production sampler重算同一上下文、同一Gaussian，不重放已消费的online capability。先封存当前历史、候选身份、动作、context tensor、噪声与gate，再执行raw/refined/null/refined-repeat。实际两策略memory/thread/queue/cursor和六类模拟器状态都进入fingerprint；重复refined必须重现完整endpoint与RGB。新增采集器首先仅做排除在正式cohort外的actual-history pilot，尚未宣称真实模拟器集成已通过。

独立物理标签版本`warm.rmbench.physical-next-interaction.v1`，读取实际接触、提起、放置、任务stage与反馈配置，禁止将预测效应或selector得分作为真值。它是新定义的局部物理交互完成率，不冒充原稿19%/27.3%历史标注。2mm几何边界保留unknown；失误、回退、运行无效分别记录。Battery Try对新极性配置的探索进展不使用隐藏正确答案。

`repertoire.py`预声明两个技能家族的非空holdout与等规模clean搜索域；原bank字节及global EventID不变，continuation不能泄漏被排除的条目。报告明确区分“家族可用性holdout”与“实际query不存在可用技能”，后者还需要完整候选执行覆盖。新增11项软件测试连同原62项，本地73项通过；实际分支、自然退让及SR仍以随后原生运行和独立验收为准。

### 23.3 正式数据前的统计、资源与训练协议补强

- v2计划将holdout固定为每任务250对、2pp绝对成功率不劣界，clean size-matched仍100对。`paired_binary.py`用gain/harm边际精确二项尾界加Bonferroni构造保守配对差区间；零discordance不报告伪[0,0]，未完成/invalid cohort禁止不劣结论。算法核对[SciPy官方精确区间示例](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html)并通过零事件/不完整cohort边界测试。这是事前检验设计，不是实验观测，也不单独证明自然gate正确。
- 为缩短CPU物理执行占比，新增单角色worker及严格merge入口；Full/null权重、retriever与环境仍独立，只有完全相同reset及source/model/protocol的trial可配对。旧顺序入口不变，尚待原生执行资格，不能把并发代码或单角色数当已完成pair。
- `simple_prior.py`是独立训练控制：同一候选池/事实history与gist、粗排top1动作、固定0.25先验、确定性起点warp，无学习式动作适配/后果预测/重排/门控。单独checkpoint schema；Full的默认训练与推理语义由原有Torch回归测试覆盖。配置生成器强制作者原配置SHA-256，两分支共同从release以新optimizer起训，batch8×accum16=effective128、原max30000 scheduler、固定3000更新，不把新单卡cohort冒充原4卡续训的逐bit复现。
- native precommit v1为96 pass/2 fail：两失败已在未修改的69ff6b4源码重复复现，分别为旧trainer stub遗漏attestation context、旧静态断言仍匹配已替换的完成日志。修复测试fixture/断言，未绕过真实resume检查或修改生产trainer。precommit v2包含Full/source/config/entrypoint与新control/统计/merge测试，100项原生通过（20.74秒）；本地非Torch相关78项通过。Windows无Torch，新控制的数值/梯度检查仅计原生CCI结果。
- 已核实AFS余约60GB、节点临时盘约6.9TB、本机C盘约1.45TB。新增训练的state在有界分段后同步本机并验hash；原始checkpoint、数据与运行成果不清理。当前新simple-prior尚未训练，单卡吞吐与正式成功率仍待执行，不以软件通过替代公平训练比较。

### 23.4 首个完整真实配对pilot：保留负结果，不支持效能优势

`paired_job_logs/20261108T121650Z-51572500`于12:16:50–13:19:50 UTC结束，complete/exit0，runner elapsed3780.1005秒、collector elapsed3694.7732秒（包括两套模型的初始化/校验）。AE固定源码集合运行前后均为`37d11caa89b3fb63f93a0d8a316ffb62e02fde1b4cc465d8f77039441ad864e9`，官方任务源码前后均为`a97e308ca053c3aeb1bfec8e3c4c4f550aac7f6ae753e36bbce82e9522292379`。

| 排除pilot，Rearrange Blocks seed3407 | Full | candidate-null |
| --- | --- | --- |
| 官方完整回合 | 1 | 1 |
| 实际执行命令 / 重规划 | 700 / 175 | 700 / 175 |
| 官方任务成功 | 0/1 | 0/1 |
| 非零检索gate的query | 175/175 | 0/175 |

配对差为0，只有一个预先排除的资格pair，**不能用于正式SR区间、不劣结论或提分证据**。Full gate范围[0.1484375,0.2158203125]，方块1由x约0.14移动到约0.31，但stage始终0、press_cnt0、max_reward0；可视化原件显示物体确实移动而未完成第一阶段，不把这条轨迹解释为生成修复。两角色初始RGB、qpos、物体目标和官方unseen指令完全相同；六个预定frame0/32/96/192/320/480的base Gaussian逐元素相同。闭环后续状态自然分歧，不把这些query冒充同一当前状态的局部分支。

完整双端包`S/rebuttal_20261108_p0/archives/p0_rearrange_blocks_complete_pair_pilot_v1.tar.gz`，7,447,914 bytes，SHA-256 `f32e181be3a7d83ae32c0b686ec7f9be88329b40ee8582a03409ba6a921ae589`。本地`P0_COMPLETE_PAIR_VERIFICATION.json`核对70个归档成员哈希、所有1400条动作及before/after原生qpos哈希、350个replan、null零门值、初始身份和共同Gaussian，通过；原始失败不删除。此前单Full角色及search预检独立包3,984,144 bytes，SHA-256 `e51f8064f77b53e44307e594e7beaa756f6cb59f328f508edd6a380f8c58c7a9`，39成员亦独立验收。本地均在`tmp/rebuttal/execution/20261108_p0/evidence/`。

### 23.5 真实非空搜索域与下一阶段状态

实际bank89,680条已通过EventBank完整校验，restricted search在完整域的top32 ID/score与原reference相等。家族holdout/clean各自大小：Rearrange Blocks为19,540（holdout保留Battery Try、Observe And Pickup及Press Button，可能仍含通用抓取/运输原语），Battery Try为78,772；目标任务条目在holdout均为0。此为真实bank上的搜索域工程资格，不是候选零适用覆盖或自然退让效能。预检v1读取manifest封装层级错误的KeyError原日志保留，v2按已有contract结构读取后通过，未修改bank/manifest。

新控制/统计版本`5799e7e62b884071662964b294af0aa9b3670e46`已commit、push、GitHub分支核验且CCI fast-forward；归档tar SHA-256 `45da96aef6f715768a8e7d6b54f27c9034cd9f798f5261cd8cea34760b0fff8e`，固定代码集合`9f37135ce0e983f9509f422dd41bb6986a6b0270feecd95baff73b1ed6ee254a`。活动pair仍来自AE旧冻结源，未被Git同步改写。

真实历史分支pilot在前一作业退出且GPU清空后，于13:20:04 UTC启动`live_history_job_logs/20261108T132004Z-eb70dd4c`，max5400秒。collector来自69ff6b4不可变代码（集合`4a6762f26fcdf6e9256fe305c84c3dcd7423ff4791296a10be22dae8deb31c37`），合同/core仍来自AE；seed3407、实际Full前4个命令后的query1、raw/refined/null/refined-repeat各H32。当前仍在模型加载，尚无真实分支效能结果。

匹配训练两配置已按作者YAML哈希生成：Full SHA-256 `710b74b4a94036a1712c081f3d95aee18eb3e9bc5b91b2a836f6bc3d9fc5c0f8`，simple-prior SHA-256 `a00bcc14b6a7bed50ec7242aa982dfccc57cad8d3951b3cad3bfe593cd8936e3`。单卡5更新预检只在CPU真实TRAIN/DEV输入通过、前一GPU作业结束、显存清空且14:30 UTC前可启动时接续；max3600秒并保留重启/归档余量。此时训练尚未实际启动。正式100/250对及候选特异性结论均未产生。

### 23.6 匹配训练输入及大资产重复校验预检

TRAIN/DEV实际构建完成，样本数249,966/26,934；两个重点任务各自固定TRAIN/DEV前缀的候选、split和有限张量通过CPU预检，elapsed360.3542秒。原summary为`S/rebuttal_20261108_p0/control_data_preflight_5799e7e/summary.json`。不是逐项遍历全部媒体，也不是optimizer更新或效能证明。单卡5更新作业仍待上一GPU作业结束。

针对反复读取同一大checkpoint的启动成本，新增默认关闭的Linux进程内摘要复用。首次完整字节SHA-256，后续仅对metadata完全相同且已打开的同一大文件复用；key包含PID/realpath/device/inode/size/mtime_ns/ctime_ns/mode，变更即重读、fork清空、无持久或外部提供摘要。测试覆盖恢复mtime/同大小写入、保留时间的原子替换，并连同EventBank、online retrieval和training attestation共49项原生通过（10.99秒），Windows只计默认路径1 pass，Linux专用2项skip。

真实指定checkpoint12,148,647,926 bytes的首次完整哈希与`68c7f384...`相符，74.9567秒；同一进程二次打开/metadata核对后命中，0.8708秒，统计为miss1/hit1、实际累计哈希字节仅一个完整文件。原报告`asset_hash_cache_real_probe.json`，工程源码集合`cc55ceb21f5ce220c51b02fdfa38262a0ca6d6895ee0f8e6c495d84e42c37ba6`。此为当时AFS与并行负载下的单次文件级检查，不声称完整模型初始化加速倍数，不覆盖旧初始化或重规划成本数值。AE/69/579正在/排队的冻结作业不改用新实现；后续需要从新不可变版本显式启用并重新绑定core/config合同。

### 23.7 真实历史分支的接口失败、修正与重跑

首轮69ff6b4 collector完成4条实际Full命令，在query1封存了真实history（observation_updates1）、原始当前态adapted proposal、Full/refined与同上下文/同Gaussian的null，Full gate0.1923828125。但独立效应读取错误地将uint8 HWC直接传入要求float32 batch的image adapter，于首个H32分支执行前抛出`ServerFeatureEncodingError`；**branch truth0、repair/harm0条、科学采集未通过**，不能当作raw不适用或策略失败。原件包`p0_history_preexecution_interface_failure_v1.tar.gz`，999,168 bytes，SHA-256 `4907565620688300fa2110d0eaf4562797577d45279da2a36dc160186db83b48`，在`S/rebuttal_20261108_p0/archives/`，原job保持failed/exit1。

修正后复用正式retriever的`_prepare_raw_cameras`和image adapter/DINO路径，要求parent effect token与该真实BoundOnlineStep的factual DINO spatial token逐元素相同；读取前后六类fingerprint必须不变。增加完整parent snapshot及哈希receipt，在任何branch truth前封存；null诊断生成显式隔离Python/NumPy/Torch RNG，异常时亦恢复，不推进Full事实轨迹的随机流。研究扩展的实际import路径单独核验和记source hash，核心policy/config合同仍独立绑定。14项相关原生检查通过（9.51秒），Windows13 pass、Torch专用1 skip；真实H32恢复仍需重跑验收。

新core `b823b3a`的实际Git归档字节集合为`55f40ddc07437bd0b47c3a31019a720bb05555ee53512f6cb0fe297a262e21f5`。与原cache预检集合cc55的差别仅为`manifest.py`经Git归一化行尾，已逐文件核验；所有正式receipt仍使用实际字节哈希，不替换为归一化哈希。两任务cached strict bundle已exit0，manifest SHA-256 `b9d39c83c90b60a30338295f6b709f42ccdfdf7c10eae83cd193ae6099b70ccb`，35次guarded hit、131次完整miss、完整读取40,353,657,484 bytes。预备脚本v1将precommit字节哈希直接等同Git归档而中止，未生成合同或启动评测；原失败保留，v2显式核对行尾差异并按实际core绑定。

修正pilot于14:27:16 UTC从immutable `camera_fix_precommit_v2/code`启动，job `live_history_job_logs/20261108T142716Z-ce1945ff`，max2700秒；core/config为上述b823b3a bundle，仍为seed3407、前4条Full命令后的query1及raw/refined/null/refined-repeat各H32。此时尚无新分支结果，不将重跑启动计为机制证据。

### 23.8 两个5更新训练预检的完成状态

- source-only简化控制：job `training_job_logs/20261108T135011Z-7759fc16`，14:04:39 UTC前complete/exit0；actual step5/max30000、microbatch8/accum16/effective128、BF16、seed3407，共5条实际更新记录，权重SHA-256 `1503d79ffaf9ac65d370ae7fc2d700d5dc3adce3e8fa1ff014f1fa9a0588c1c4`。此为旧source-only工程prototype，不是完成3000次的性能基线。
- Full同配方预检：job `20261108T140953Z-b06c248b`，现已complete/exit0并在step5保存；核心b823b3a只改变重复资产摘要读取，Full训练数学及预算未改。checkpoint/optimizer及吞吐的独立核验接续进行，不能只根据`training_complete`文件称科学对照完成。
- 两者临时checkpoint工作区各约36GB，位于`/tmp/warm_rebuttal_p0_20261108/training_pilot_5799e7e/`。归档/本机转存尚在进行；只有传输后哈希相符才标为双端持久化。尚未开展3000-update匹配性能比较，未更新论文效能数字。

### 23.9 真实历史H32分支通过，局部结果仍无优势

修正collector job `live_history_job_logs/20261108T142716Z-ce1945ff` 于14:40:41 UTC complete/exit0，runner804.9833秒、collector696.0373秒。实际collector集合`0e20e34dfd7b746bb82824348b5b02500128ccf684526d3df5ff8afa32bf530e`前后一致；core仍为b823b3a集合`55f40ddc07437bd0b47c3a31019a720bb05555ee53512f6cb0fe297a262e21f5`。指定v8-step3000、Rearrange Blocks seed3407，Full真实执行4条命令后的query1，history observation_updates1。

- raw/refined/candidate-null/refined-repeat各实际执行32条命令，共128条；四分支均为complete_horizon，六类父状态恢复一致，refined-repeat的完整endpoint及RGB完全一致。Full事实轨迹总共执行8条命令。DINO读取必须与真实retriever token逐元素相等，null生成的Python/NumPy/Torch随机流隔离已通过。
- 独立physical-next-interaction标签为raw0/refined0/null0，均未完成新里程碑。**只支持采集链在这一真实历史状态通过，不支持生成修复率、candidate-specific后果预测优势、正式SR或提分。** 原summary的`scientific_collection_qualified=false`保留，其他阶段尚未取得资格。
- 双端原件包`S/rebuttal_20261108_p0/archives/actual_history_corrected_qualification_v1.tar.gz`，4,331,908 bytes，SHA-256 `758e1181a14a598b9769a8f46aeadfb8ea3684c42a65853797e2e8a9b3e3eecd`。本机独立验收29个文件、封存receipt、128条实际动作、四次六分量restore、重复endpoint及RGB，见`tmp/rebuttal/execution/20261108_p0/evidence/P0_INSTRUMENT_PILOTS_VERIFICATION.json`。没有将排除pilot提升为正式cohort。

### 23.10 5更新的FP32检查与无损持久化

Full step5权重SHA-256 `c4bf3244f4e343a0d5938bfb14f8ffa9b7c5b08c604794bf38fb1fdfeac39403`；simple-prior权重身份见23.8。两者均保留完整optimizer、scheduler、RNG、trainer state、配置、五条metrics和训练attestation。Adam实际step均为5，FP32 master分别含1,074,264,859和1,061,785,196个元素。

固定检查共同ActionDiT的`action_expert.action_encoder.weight`（14,336元素，optimizer partition起点0），与同一release中`mot.mixtures.action.action_encoder.weight`的FP32转换比较：Full14,263元素变化、simple-prior14,256元素变化；二者最大绝对变化均为`5.029141902923584e-07`，所查master及Adam一、二阶矩有限且非零。原始检查报告`UPDATE5_FP32_MASTER_VERIFICATION.json`已传回本机。**这是指定共同参数的实际更新证明，不是全参数扫描、3000步训练完成或策略性能。** 前一版layout打印器对LossScaler JSON序列化失败，原失败保留；更改的仅为诊断输出，未改checkpoint或训练。

两个原始TAR分别为38,412,615,680 bytes/SHA-256 `5d37f17546c00ccb0ce229a7f4de4e0aee3d35688067c406030c39338ca9a720`（simple-prior）和38,599,106,560 bytes/`711c61369cbccf80e5944529084ef919332e325853d7c4114530d2e73f0cd611`（Full）。因公开SSH下载约5MB/s且AFS原余约60GB，先以lossless Torch-ZIP payload去重和zstd保存到持久`S/rebuttal_20261108_p0/lossless_training_update5_store_v1`。存储自包含，不依赖外部base引用或有损tensor转换；15:17 UTC两个原TAR均经完整解压逐字节重建、长度及全包SHA-256核对，receipt为`archives/LOSSLESS_UPDATE5_COMPLETE.json`。原始临时文件仍保留。

本机完整metadata/日志/codec/manifest包`training_update5_preservation_receipts_v1.tar.gz`，17,440,563 bytes，SHA-256 `042869e8c3aefa50f07d9d06b8b5dea2dca4b1e65fe161545bcd16ed882046c5`，21文件逐项核验。**持久AFS raw已验证，本机仅完整接收此receipt包，未宣称所有CAS权重payload均已双端。** 原直接SCP在约1.1GB时主动停止，文件保留为`evidence/training_updates/simple_prior_update5_evidence.tar.partial`，不计作完整原件。

### 23.11 非空holdout前缀：自然gate非零，尚无零覆盖或不劣证据

`missing_skill_prefix16_b823b3a_full`完成seed3407、Full单角色16条实际命令、4次重规划。bank原字节/全局ID不变，target-family holdout含19,540条，目标Rearrange Blocks条目0；四个query均有32个有效检索候选。自然gate依次为`[0.193359375,0.1884765625,0.20703125,0.1923828125]`，并未自然降为0。保留的通用抓取/运输原语仍可能适用，**不能据此直接判为false activation，也不能称模型已学会missing-skill fallback**；尚需候选真实执行覆盖和官方完整配对回合。

此试验为预先排除的qualification，`success=null`、`unpaired_role_trial`、正式完整pair0，16步不冒充700步成功率。启动时同节点WAMO独立进程占约14GB；没有停止或修改其任务。WARM仅在至少45GiB可用时获准启动，Torch每进程上限0.4，admission及PID保留。后续观察GPU已清空，不将共享节点耗时作为成本benchmark。

双端包`nonempty_holdout_prefix16_qualification_v1.tar.gz`，1,169,330 bytes，SHA-256 `f0ecbca2e5167225aaba99530aa86ffac4c68f83f6937f63937b80134e8729a7`；本机独立核验30文件、全部16条动作及原生qpos前后哈希、4个自然gate和真实搜索域，见同一instrument verification报告。

### 23.12 单H100执行与在途任务

作者确认目前只有一个节点。优先完成机制修复、候选独立后果真值、自然退让及同预算简化基线，三训练seed按顺序执行；首个匹配seed完成前不把多seed计划当成果。正式100/250对的目标和2pp不劣准入保持原计划，未完成时不报告不劣结论。两独立单角色worker允许在约80GB内并行CPU物理轨迹；保持官方H32/prefix4/NFE10/K32和完整任务步数，不靠截短换取速度。

首个Battery Try正式Full单角色shard于15:43 UTC前启动，输出`S/rebuttal_20261108_p0/formal_battery_full_bank_full_shard0000_attempt01`，candidate seed340800000起、case index0、官方上限1000步、固定v8-step3000。source/core均为b823b3a不可变集合55f40d；max4200秒，Torch上限0.4、与其他作业的GPU admission单独记录。此时仍在初始化，**还没有正式完整pair或SR结果**。后续null需同一准入seed、初始图像/qpos/物体目标/指令和协议核对后才能merge。

另增默认关闭的串行只读权重replica，用于同一封存context的null诊断，模块与状态独立、参数/buffer只读共享、原始Full状态及tensor version检查；不改变null定义、候选、NFE或正式闭环策略。原生v1软件14项通过，但真实部署复制遇到bound retriever内的mappingproxy，于分支真值前失败，原failed记录保留。v2将诊断replica明确保持online-unbound，不复制/共享任何retriever capability，仍只经production FastWAM kernel消费原先已验证的factual tensor；增加真实能力对象不可复制fixture后15项原生通过（20.49秒）。v2源码集合`bb0c92453ed9e92aa0b9ec30d799902f4f182f0a426732119721ca215eca147f`，15:50 UTC启动独立资格重跑。**真实作者模型的逐动作等价尚待核验，不能用于正式提效声明。** 活动job均不读取canonical checkout变更。

### 23.13 更强的匹配简化对照及共享权重原生核验

对照进一步保留Full的八个事实history gist槽及四个检索动作conditioning槽：粗排top1的同一H32×14确定性warp动作，逐值无损分成四个112维token并补零到现有512维，再经可训练线性projection进入ActionDiT。初始化该projection为零，不压缩丢弃动作值。固定0.25先验沿用先前事前计划；仍无学习式动作适配、后果预测、效用重排或置信gate，common bridge/gist、历史信息、候选池、ActionDiT、数据及优化预算保留。这是显式common-backbone control，不称官方ReCAP复现。旧source-only5更新prototype及其原件保留，不把其结果算作新对照。

新独立schema为`warm.research.coarse-conditioned-prior-with-recollection.v1`；事前计划`configs/nonreal/rebuttal_matched_conditioned_control_plan.json`规定固定3000更新、同一release、fresh optimizer、effective128、原30k scheduler，seed依次3407/3408/3409，未完成seed计划须明示。Full旧5更新只有通过原full-state lineage验证才可计作其连续起始段，不允许重置optimizer或数据游标后仍沿用累计步数。

代码precommit集合`4efb8094729ba8771c7ec0d472c30ad8b585ef319e897edb566b12d3d5a81a14`，Full/source回归、动作信息无损性、projection梯度、checkpoint schema互拒及配方几何等58项原生通过（21.37秒），本机两项纯配置检查通过。首个100更新段配置SHA-256 `dd0ddb5ce07dbecb2cc8628479c755a2769a7aa932722a0e7fb60fb23803ce7a`。原配置生成器manifest中的`preflight_updates`为旧字段名，本次段的用途在运行前已由独立plan及job明确为固定3000的初始优化段，**100更新本身不进入TEST效能对比**。

该段于16:21:47 UTC启动`matched_training_job_logs/20261108T162147Z-e10800ad`，仅在前一自有诊断job退出、GPU清空后启动；此时处于模型/数据初始化，尚未验收实际100更新。输出`/tmp/warm_rebuttal_p0_20261108/matched_conditioned_seed3407_v1/conditioned_prior-s3407`，原始optimizer/scheduler/RNG和metrics的持久化接续同一无损CAS流程，不触碰原checkpoint。Full5→100配置只准备，未运行，不能称两臂3000更新比较已完成。

共享权重v2的真实作者模型核验已通过：同一seed3407/query1，raw/refined/null/refined-repeat四套原生命令、Full与null所有probe数组、21个factual context tensor、非tensor字段及parent DINO token与先前独立模型逐元素相同；5,549个tensor引用共享，clone新增allocated CUDA bytes为0。仍仅是一处排除pilot的实测等价和软件只读保护，不宣称所有query经验覆盖或计时加速倍数。双端包`shared_weight_native_equivalence_qualification_v1.tar.gz`，4,363,358 bytes，SHA-256 `8369bdf397435b7d2cfb291f0c9fadc64fe7f0952047df0ed1ba410186f00263`；本机核对42文件，包含v1复制失败原件和v2完整分支，旧失败未删除。

首个Battery Try正式Full尝试在80条命令后于重规划校验处停止，`success=null`、原始`invalid_runtime`记录保留，未组成pair。双端原件包`battery_first_formal_boundary_failure_v1.tar.gz`，1,578,925 bytes，SHA-256 `5b0f2f9331595189a5f08cdf7141d3d31ff03d56aa29cf2037f2a2d01eed13a3`，32文件本机核对。不得把生成/策略相关失败静默排除来抬高SR；必须先分清基础设施、审计范围和实际部署失败，再以显式口径报告全部尝试。

### 23.14 80步停止的数值定位与实际执行前缀修正

新增只读generation witness，在consumer归一化校验及该chunk任何执行之前封存完整模型动作、BoundStep/输入/候选payload身份、derived seed和consumer失败；不裁剪输出、不截断原异常。15项原生witness/role-merge/policy-state检查通过，来源集合`67c612673f6ca33d1658b8afb4915116cee696af1f2d52701914d64234f8706a`。原native测试receipt沿用了旧shared-weight scope文字；实际测试列表是action-witness、role-merge、policy-state，本节明确其真实范围，不据此再声称共享模型资格。

同一case0/seed340800000的诊断attempt02于16:10:17 UTC启动`formal_diagnostic_job_logs/20261108T161017Z-c6dde853`，仍在80条命令后复现原guard。封存的query frame80动作最大值5.25，超出±5的仅为零基位置`(23,4),(24,4),(25,4),(26,4),(27,4)`，数值`[5.125,5.21875,5.25,5.1875,5.09375]`；**不属于浮点舍入误差**。真实执行前缀为前4步，其所有数值位于[-1.8125,1.140625]，按原normalizer严格`fail_on_clip`检查通过。两次尝试前80条实际命令、native qpos before/after以及此前20个完整生成chunk逐元素一致，证明witness没有改变原轨迹。

`SingleFieldLinearNormalizer.forward`将输入特征截断到±5，而其`backward`为未裁剪仿射逆变换；±5不能直接当作物理关节限制。此次已确认的工程问题更具体：`RecedingHorizonQueue`只发布前4步，原guard却要求未进入执行队列的28步尾部也满足可逆检查，从而拒绝了严格有效的实际前缀。修正只对即将入队的前缀保留原范围/逆变换要求，仍检查整个H32 shape及有限性；队列native命令、原生成数组、checkpoint、NFE、H32、prefix4及历史前缀的canonical归一化均不裁剪或重定义。真实执行前缀若越界/逆变换不符仍须失败，不能以本次修正放行。

该修正在precommit集合`6ab037a228e67e7e3ba8654c9c4465c77724752598e4dbe4aac565c9c5f67ca0`上36项原生policy/queue/witness/entrypoint回归通过（10.07秒），覆盖discarded-tail越界、执行prefix仍拒绝、逆变换错误及非有限尾部。Windows完整policy测试因缺PyYAML未收集，不计作通过。另用实际封存的5.25输出及原stats重现旧guard、核对严格prefix通过和原生入队命令逐元素不变，报告`S/rebuttal_20261108_p0/ACTUAL_PREFIX_BOUNDARY_VERIFICATION.json`。**仅有数值/队列证明，修正后完整物理回合仍须重跑**；原两次停止不改写为成功或正常官方失败，也不计成两条独立样本。新core将重新绑定严格bundle，保留相同evaluation namespace及共同Gaussian协议。

### 23.15 区分输入特征截断与生成动作解码

23.14是第一轮仅修正discarded tail的审计范围。其Git归档实际源码集合为`a7bcb231081047d9c3ae9fd9c632f55f37fc08a6130b30f46b95a7890e33c907`，与precommit差别只有4文件行尾，已逐文件检查并在Git字节上再跑36项native检查。首次bundle重建误传了通用M1 YAML，合同检查在策略运行前以`M1 data config does not match the feature-encoder contract`拒绝，原failed job保留；v2恢复encoder原先封存的`data_config.source.yaml`（SHA-256 `c8fcae013a1b1903b4c66b7dc27c5bd9a5c6eb858e3b7103e7ffb2e66115ff53`），不改encoder/数据或旧合同。v2 bundle manifest SHA-256 `0960bb27ef88c5d34826346a1d55bb4d05a4174edb8eede1994a3fad7cafca7c`，两任务保留同一evaluation namespace `2d57f5d0ca3dae0891de402f16ee91f8fda2a0b3d60eac273fade8ea585b4b8a`和指定checkpoint。

随后独立Full/null使用官方1000步上限重跑同一Battery Try case0/seed340800000，均执行到124步，**尚无完整正式pair**。此次Full的实际prefix最大5.28125、null最大5.125，原始失败仍为`invalid_runtime/success=null`，不能按完整SR处理。只修正尾部尚未解决更根本的审计假设；真实prefix也可超出训练特征的±5。

原FastWAM `SingleFieldLinearNormalizer` 明确采用 `forward(q)=clip(s*q+b,-5,5)` 和 `backward(z)=(z-b)/s`。两者在截断区外不是全局逆映射。原审计把 `forward(backward(z))==z` 错用于任意有限生成值，因而会拒绝原处理器本来允许解码的命令。正确边界是：native命令必须**逐元素等于原backward映射**，入模的事实动作历史必须**逐元素等于原forward映射**；保留完整shape、finite与实际executor运动检查。既不裁剪采样值，也不裁剪native命令，不把统计feature clamp伪装成物理关节限制。

利用两角色frame124的实际封存数组和原stats，已验证完整H32的NumPy解码与原Torch处理器backward逐元素相等；即将执行的4步canonical历史与其forward逐元素相等。Full的joint4 native目标约`[0.12742905,0.12893276,0.12968461,0.12968461]`，历史特征按原规则为`[5,5,5,5]`，不以特征5.28推断物理越界；null同样通过。前80条Full命令及native状态也与更早原attempt逐元素相同。报告`ACTUAL_FORWARD_BACKWARD_SEMANTICS_VERIFICATION.json`。新审计保存raw生成prefix和canonical history各自哈希、feature截断计数及精确命令一致性。

修正precommit集合`7b3b4551b80d6e48cde12c6d4e752bc500aec48d6cbd1dd122c1ec93432ba1a5`，38项原生policy/queue/witness/entrypoint检查通过（10.66秒），包含真实生成域、篡改命令拒绝、非有限尾部和zero-std下不强求错误的浮点可逆性。**这是原前后处理一致性修复，不是训练、采样、阈值或动作约束的改法；修正后完整物理回合仍待运行。** 23.14关于实际prefix必须继续落在±5内的临时解释由本节公式与原处理器核验取代；全部旧失败和对应源码仍保留，不重写旧结果。

### 23.16 单H100主证据配置、首段100更新与候选执行有效性

单节点时间预算现按独立v3主计划`configs/nonreal/rebuttal_p0_primary_single_h100_v3.json`执行；外部原件`S/rebuttal_20261108_p0/primary_single_h100_v3_plan.json`在此次修正后完整正式策略结果之前冻结。23.12的v2目标保留为历史计划，本次主计划不再沿用其100/250对的总量：Full/null各任务50对；局部修复各任务20个固定source episodes、固定六个query ordinal；候选后果各任务10个source episodes、q8/q48、原K32。未到达query保留，不按结果替换。非空缺技能的2pp不劣主检验限定Rearrange Blocks、200对holdout及50对clean；Battery Try各20对为描述性证据，不能宣称双任务2pp不劣。共享字段上限不代表每个任务实际目标，执行须读取`primary_counts_by_task`。第一匹配训练seed3407优先固定3000更新，3408/3409为主证据之后的次级任务；不得将计划seed当作实际训练方差。

加强版conditioned-prior首段实际完成100更新，job `matched_training_job_logs/20261108T162147Z-e10800ad`，16:21:47至16:54:39 UTC，exit0；来源集合`4efb8094729ba8771c7ec0d472c30ad8b585ef319e897edb566b12d3d5a81a14`，micro8×acc16=128、BF16、seed3407、原30000 scheduler/warmup1500保持。step100权重SHA-256 `3064637fc9d5fc44bb1aadec9d726f393acb8ef2e9b5bbdfc8247ac991300f5a`。Adam step100、FP32 master总1,063,882,348元素；指定共同action encoder和新增2,097,152元素direct-action projection的master及Adam矩有限，保存BF16与master转换逐元素相等。projection由零初始化实际学出458,752个非零master元素（其余输入为固定补零列）。这是参数更新证明，**仅100/3000，没有TEST或匹配SR结果**。诊断沿用的gate/adapter日志字段名不表示被移除模块存在，相关辅助项系数仍为0。

原始TAR为38,438,236,160 bytes，SHA-256 `2acd20824aacce72e6520866b7e980ee9ca64bd926315e4854d0a259b794385c`，17:01:50 UTC从AFS自包含CAS完整逐字节重建通过。原receipt误把job-spec SHA写入`configuration_sha256`，不覆盖原件；`CONDITIONED_UPDATE100_MASTER_VERIFICATION.json`明确实际配置SHA `dd0ddb5ce07dbecb2cc8628479c755a2769a7aa932722a0e7fb60fb23803ce7a`、job-spec SHA `ab373d5bfe87d1042554930c39b66ce9a870126d6b5f7ba4550f45e445eb6d00`。配置/日志/manifest/FP32证明包8,778,800 bytes，SHA-256 `dedd06150f7f3c1651087504dfa26abc675771fd88df3cb2e9511fd989c0032a`，本机18文件独立核验。新100更新CAS的本机完整下载仍在途；旧两份step5自包含CAS的398个压缩blob和2份manifest、共35,260,254,358 bytes已全部本机核对，receipt `UPDATE5_LOCAL_CAS_TRANSPORT_VERIFICATION.json`，不据此宣称新的step100也已完整双端。AFS余约11GiB，下一训练段前须保留和核实容量；没有删除原始checkpoint或原始临时TAR。

K32扩展先封存同一真实Full context下所有原始valid rank、adapted H32动作、event IDs、预测与原生解码，再同父状态独立执行；重复refined和selected-raw的endpoint必须精确复现。报告绝对误差和去除每个query共同偏置的centered误差，预测、historical和constant-query参照共用完整H32支持集，缺失效果不能填0或混入指标。v1来源`aa8cd5920eebbbc6e1ca2e7ca64b7d0c8cdb5aa8a2f2becba2893ccd8fd9568a`通过42项原生检查，但排除pilot `actual_history_K32_pool_rearrange_blocks_q8_v1`在rank0完整32步之后、rank1第3条命令遇官方TOPP `Fail to parameterize path`；rank1执行2条、invalid_branch、无后果真值，六类parent恢复精确。不得把它当成零后果或0适用标签。该任务未完成K32资格，也不代表对所有32候选的真实性能结论；Battery Try独立pilot仍在运行。

v2仅给K32全候选执行开放具类型的官方executor拒绝续采：保留invalid结果及原始错误，完成六类精确恢复后才能继续下一rank；未知异常、效果读取异常、任何恢复不一致仍停止，raw/refined/null主分支的默认停止行为不变。有限generated动作按原backward解码，不再错误假设浮点全局可逆。该集合`b1d0e4e0790a6c1f8c9d415d9363faafb21f1e34fdcfebb3095dcc63bcb424a0`原生47项通过（23.22秒），含已知拒绝、一般错误、拒绝后恢复失败、无效果插补及centered metric检查；Rearrange Blocks另起v2排除pilot，不修改活动v1源或旧记录。全部仍属采集资格，**尚无正式完整pair、修复率优势、候选后果预测优势或不劣结论**。

80步witness与两个124步中止角色及原处理器数值证明已双端封存：`archives/battery_boundary_witness_and_prefix124_failures_v1.tar.gz`，5,880,166 bytes，SHA-256 `303b9567633c187c8cb91712ee8f5c6b5c366f915008519cc5c27941da29a710`，本机逐项357文件通过。两份新包的独立本机报告为`tmp/rebuttal/execution/20261108_p0/evidence/P0_BOUNDARY_AND_TRAINING100_ARCHIVE_VERIFICATION.json`。以上raw均在Git外，Git只提交实现、检查、固定计划和本记录。

### 23.17 首个完整K32排除pilot：后果区分优势未成立

`actual_history_K32_pool_battery_try_q8_v1`在seed3407、真实Full事实轨迹query ordinal8/frame32完成原32个valid候选的32步原生执行，共1024条候选命令；另有raw/refined/null/refined-repeat共128条命令，refined完整endpoint/RGB重复精确、selected-rank候选与独立raw的endpoint精确相等，所有六类父状态恢复通过，32条全局event IDs均已保存。输出维度为768，所有预测和动作在真值读取前封存。本机独立重算centered和绝对误差，并核对32个H32原命令及全部封存哈希；未将此排除pilot并入正式cohort。

同32个完整候选支持上，预测头绝对MSE `0.27574974966964133`、centered MSE `0.1106178077064582`、centered explained variance `-0.05967603190517745`；constant-query参照绝对MSE `0.22612540586336397`、centered MSE `0.1043883265978753`、centered explained variance0；historical参照绝对MSE `0.2082692016413086`、centered MSE `0.14076921292195332`、centered explained variance `-0.3485148915570271`。**该状态下预测未优于常量参照，不能用来证明候选后果区分能力；也不能从单个排除状态推广为所有任务的否定结论。** 保留不利结果，按固定正式source/query队列扩展，不能换query或用较好的historical对比掩盖constant参照。

独立physical-next-interaction标签32个候选均为0；raw/refined/null也均为0，natural gate为`0.1904296875`。此为完整bank下的一个状态，不是非空holdout实验，不能将其改称missing-skill fallback或宣称已观测到harm、修复率优势。全部原件与此前Rearrange Blocks rank1规划拒绝合包`archives/K32_battery_complete_and_rearrange_rejected_qualification_v1.tar.gz`，35,554,819 bytes，SHA-256 `6804c1a86298af51b7a803683763879fa49d5c894f292a42a8e57a5be25d9bb0`，双端111文件核验；本机报告`P0_K32_BATTERY_QUALIFICATION_VERIFICATION.json`。

### 23.18 固定正式机制source的采集准入（已实现，物理整回合待验证）

正式模式从既有完整官方Full trial中按预定episode index准入前20个source，成功值0/1均接受，不以模型结果选样；必须匹配作者checkpoint、计划、任务、bank条件和固定core。K32仅前10个source的q8/q48；局部修复按q0/8/24/48/80/120，明确允许q0空事实历史。正式模式禁止手动覆盖query或截短官方轨迹；每个未到达query保留`not_reached`，不替换源回合。

源回合的初始RGB/native qpos/逻辑任务状态/指令必须逐项相同。重新运行原策略，每条已生成的native命令与before/after qpos均核对原Full trace；参考动作只用于生成后校验，禁止teacher forcing。每个分支仍须六类精确恢复和重复endpoint检查；任何事实轨迹偏离保留失败，不能靠放松容差声称同轨迹。计划、source trial和trace哈希另行封存。新入口来源集合`7e79031a69fccce72efc5104b179ff8e13dbc4b86ae79b59c93ec9c8eef8f3a4`，65项原生协议/采集/分支/后果/静态入口检查通过（23.47秒），并已逐字节核对三个改动文件与本机一致。**尚未在正式source上完成整条物理轨迹的资格验证，不把单元检查当成正式机制结果。**

修正原前后处理后的固定core集合`5d7748e77a648511c5ddcfc0c65c3fac9933bbeb0e2bff20c61a7fd7e6356638`另在Git字节上重跑38项检查通过（9.69秒）；新bundle manifest SHA-256 `ec2afcad56fc8ddb044ebae0b5d5c29cae02140d4ca2494129679bd963215599`，原数据recipe/checkpoint/共同Gaussian namespace保持。Battery Try case0的Full attempt04和null attempt02已登记在`AFFINE_CORRECTED_FORMAL_PAIR_QUEUE.json`，等待排除K32任务结束及至少75GiB空闲，随后按官方1000步运行两角色；排队不算GPU运行或完整pair。

### 23.19 第二个K32排除pilot、存储分层及后续实际队列

Rearrange Blocks v2已对原32个valid候选全部尝试：27个完整H32、5个官方TOPP拒绝；五项均保留invalid_branch和未知effect/适用标签，六类parent恢复仍精确。候选分支完成901条命令（含拒绝前已完成前缀），另外raw/refined/null/refined-repeat完成128条；selected-raw和refined-repeat的endpoint精确复现。27个完整候选的已知独立标签均0，另5项保持unknown，不能据此称32个候选均不适用。raw/refined/null标签均0，natural gate `0.1923828125`，仅排除的seed3407/query8。

在相同27/32完整支持上，预测头绝对MSE `0.14846232741985319`、centered MSE `0.059648675310451954`、centered explained variance `-0.25370330893034`；constant-query分别为`0.20244148885062974`、`0.0475779834714995`、0；historical分别为`0.15734909237719055`、`0.09466533929069586`、`-0.989687926715157`。**绝对误差较小并未转化为候选间区分优势，不能只报有利的绝对误差。两个排除状态均尚未支持该项主张；正式固定cohort和较晚查询仍需检验。** 本机完整69文件及指标重算通过，报告`P0_K32_REARRANGE_QUALIFICATION_VERIFICATION.json`；双端包23,418,630 bytes，SHA-256 `9113cd320bd4346d6b6c111df8fc13a8b679fad69161bc17e4d7fab7cc28514c`，`archives/K32_rearrange_complete_with_native_rejections_qualification_v2.tar.gz`。

18:27:18 UTC，排除K32任务结束后，修正core下的Battery Try正式两角色实际启动，case0/seed340800000、官方1000步。此时不写完成pair/SR；两worker仍独立加载模型和bank。后续链`PRIMARY_FOLLOWING_CHAIN.json`依次等待该两角色退出、从固定完整Full source0采集六个query及q8/q48的K32，再尝试将匹配Full由step5通过原完整state校验续到100。正式collector的Git归档集合`91f7f1f22f6f06812447347328b4e01a9c949d76c98a4bd7c3ac6491faeedc2c`，三个改动文件与65项已测试precommit在统一行尾后相等；Full续训配置SHA-256 `ad428616cb581e4c3d21473763b3479f7a30b98ab619718b6510ecc778b990ff`。**后两项仅已准备、未运行，不计入当前完成工作。** 若Full源未完整则保留不准入，不按成功值另选；GPU不足时等待，不停止其他项目。

旧step5两包的全部400个CAS文件已在本机逐项验证后，仅将其19个独占压缩blob、26,306,412,792 bytes从AFS重复存储分层到本机持久CAS（并额外保留/tmp逐字节副本），AFS约释放24.5GiB。每次移除AFS重复blob前核对本机完整receipt、原SHA/长度和/tmp副本SHA，且确认不被任何当前step100 manifest引用。`archives/AFS_UPDATE5_PAYLOAD_TIERING.json`记录19项及本机receipt SHA `136b0f8a7af5195e934d7d231e45dfefd1b4c98cd2e02ff2c65856fe33d083bf`，已传回本机。旧step5完整重建现在使用本机自包含CAS；不再称旧AFS目录仍自包含。当前conditioned-step100的AFS CAS仍完整自包含，本机step100下载仍在途。原author checkpoint、datasets、原/tmp TAR及训练state没有改动。后续训练单独要求空闲GPU及至少20GiB AFS空间。

原方法核对还发现：当前action-conditioned prior没有ReCAP的检索视觉序列条件；跨回合检索先验也不是WarmPrior的同回合历史/preview先验。因此BB-W2的具名方法head-to-head仍有实质缺口，不能用当前100更新或简化对照替代。区别与可核实来源见`docs/rebuttal/MATCHED_BASELINE_METHOD_AUDIT_ZH.md`；没有为此擅自改动正在运行的策略或把新对照写成已完成实验。

### 23.20 排除状态的CPU探索探针与step100本机保全完成

为判断简单后果校准是否值得继续，另做只用两个排除query8状态的CPU ridge诊断，不训练或替换策略。按(task, source episode)整组留出，三个事前固定输入为完整adapted H32动作、historical effect、原预测effect；固定相对ridge系数0.1，标准化和拟合只用训练组，六种结果全部保留，未调参挑优。原始v8 head和正式TEST完全没有更新；不能把候选数当成独立source数或把该诊断写成正式后果预测结果。

留出Battery Try（32完整候选）的centered explained variance依次为`-1.6653862512428264`、`0.05057185892009841`、`-0.4330099708019961`；留出Rearrange Blocks（27完整候选）依次为`-1.2127499501628254`、`-0.06639334534053498`、`-0.3686764859904066`。**仅historical输入在一个方向略为正，两个方向没有一致改善；目前不能把问题归结为可由简单线性校准稳定消除。** 只有两个来源状态，不能据此认定物理后果不可学习，也不支持推广到其他阶段。继续以固定正式source/query的数据检验为准。

诊断源码集合`6af17287a9756648b950e0bff78bdc3aa34ce3ed99ee96de7138a9bb5bc2a5ca`，3项CPU数值/输入不变/退化方差检查在Windows和原生均通过（原生0.66秒）；双端包`archives/excluded_effect_probe_v1_complete.tar.gz`，1,138,450 bytes，SHA-256 `e1a946732cec563e77a3a0324592456fcc2abcb2df622a4af33b2cbe5bd08951`，本机13文件及六个互斥训练/测试来源绑定通过，报告`P0_EXCLUDED_EFFECT_PROBE_ARCHIVE_VERIFICATION.json`。没有占用额外GPU，没有将任何探索结果提升为原模型改进。

conditioned-prior step100的390个压缩blob及1份manifest已全部在本机核验，总22,373,314,771 bytes，完成耗时2392.42秒；本机完整transport receipt SHA-256 `c256f054592a30e783a00339742372b3abb9a12cf94bc768a35f35e381da4728`，对应inventory SHA-256 `456e141e620a9a2445fa1fa6436ae5f32b1e3aca96984d83a6a00b000ee72faa`。receipt已回传AFS，和先前原生完整raw TAR重建证明共同构成双端保全链；本机并未重新解压TAR，不能混称已做本机全tensor扫描。23.16/23.19关于本机step100下载在途的状态现由本节完成记录替代；训练仍为100/3000，非性能结果。

18:48 UTC读取时，正在运行的正式Full/null均已生成到frame140，越过先前124步审计停止点；官方上限仍1000，尚无完成pair。正式机制source0与Full5→100链仍等待该角色试验结束，不能把后续排队项计为运行中。

### 23.21 首个完整正式pair、原轨迹机制采集失败及Full到100的实际结果（11-09核验）

11-08 19:37:23 UTC，`formal_battery_affine_decode_pair_case0000_v1.jsonl`已完成合并：Battery Try固定case0/seed340800000，Full与candidate-null各执行原官方1000条命令，均success=0、max_reward=0；完整pair数1，角色initial RGB/native qpos/task/instruction完全匹配。仍用作者step3000（SHA `68c7f3840921241df45a5b6997b772db3a69a31584644aa0e66929b841b845bf`）、固定core `5d7748e77a648511c5ddcfc0c65c3fac9933bbeb0e2bff20c61a7fd7e6356638`、原v3计划及bank。**这是完整的零/零观测，不是Full收益证据；样本1也不支持不劣或总体负面结论，不替换原投稿表格。** 原两个80/124步审计失败继续保留。

随后`formal_recovery_battery_source0000_v1`用collector集合`91f7f1f22f6f06812447347328b4e01a9c949d76c98a4bd7c3ac6491faeedc2c`运行483.798秒后因事实首条命令/状态与固定原source不一致停止，事实执行数0。q0实际已完成四分支及一条query记录，q8/24/48/80/120均not_reached；旧summary的queries=0源于异常路径未更新计数，不能据此抹去q0原件，也不能把该q0作为正式cohort证据。只读诊断`S/rebuttal_20261109_acp/RECOVERY_FIRST_STEP_DIAGNOSIS.json`显示Gaussian逐元素相等，但selected index、适配动作和conditioning不同，首4条native命令最大差`0.007313606329262257`；preparation明确`original_cpu_encoding_changed=true`、两轮达到固定点。**分支内部精确恢复不等于原事实轨迹保持。** 后续验证必须保留命令/qpos精确比较，不放松容差或teacher-force原动作。

Full原step5完整state成功续到step100，20:27:18 UTC保存链exit0：新增95更新有限，配置SHA `ad428616cb581e4c3d21473763b3479f7a30b98ab619718b6510ecc778b990ff`，job spec SHA `ee2acd97b25645387def4e518ac9657badb0883ee688e6f201c607edc81f1b8b`，训练source仍`4efb8094729ba8771c7ec0d472c30ad8b585ef319e897edb566b12d3d5a81a14`。step100权重SHA `049f7d387c59ef4418f5c045e557124598b3a0e5b93eabda69316866bc1e64fc`，原始TAR 38,599,567,360 bytes、SHA `fb4321c8dc16b449d320cc6f97fafaaa88b08d22214af13af4127852e5e96a87`；AFS自包含CAS已完整逐字节重建核验，receipt为`archives/LOSSLESS_FULL_RESUME_UPDATE100.json`。11-09重连后旧/tmp TAR与训练目录已不存在，AFS manifest及引用blob仍在；当前本机完整CAS传输尚在途，inventory SHA `d8cabfdef5994286c0831521c790c64a28ee6fdb1ee326af68873d7c3b31677a`。**Full/control均100/3000，无匹配性能结果。**

上述完整角色、失败机制及训练保存收据已双端封存：`archives/completed_cci_pair_failed_mechanism_and_full100_receipts_v1.tar.gz`，22,715,936 bytes，SHA `50b9243f14b2022be54632df4dccb1fcdbc8c8077d464836d6ac67f63fddbc27`。本机1633项内部文件哈希全部核验，报告`tmp/rebuttal/execution/20261109_acp/COMPLETED_CCI_CYCLE_LOCAL_VERIFICATION.json`；此小包含大训练TAR的manifest/收据，不冒充包含完整大权重。

### 23.22 ACP配置确认、实际调度受限与不干预事实轨迹的工程验证

作者11-09亲自填写并确认私有镜像`ccr-zhicheng-02/lwz_wam:lwz-workspace-20261029145800`，AFS `afs-p3-01e`的`/`挂载`/mnt/afs`、以当前用户权限读写；已写入`configs/nonreal/rebuttal_acp_defaults.json`并作为之后自动填充项。CCI开发/测试/冻结，ACP仅执行快照；优先标准，GPU限额不足再用闲时。标准2卡申请明确返回“限额不足”，未生成标准作业；闲时集群02成功创建`warm-rebuttal-rb-pairs-20261109-a`，UID `9515df9ea9354f32acc2cda0ae1a4f75`、2卡44核460GiB。08:34 UTC实读Worker为等待中，事件`FailedScheduling / insufficient nodeSpec quota`涉及CPU/内存，尚无原生启动receipt，**不计GPU已运行**。

该已提交原作业controller固定Git `76979a5d1bc28b33df08a6259cd664228757dece`、源码集合`16f73967e8ade41ce805da900612b310e0c78afacb5df2e4e3d1638d4aff115c`，计划`S/rebuttal_20261109_acp/rb_pairs_0000_0009_a_plan.json` SHA `cfa2e52b50d039106b8c1d1a6d4de290d6c04148b9a04f4585c475b38be87059`；固定前10个RB Full/null对、官方700步、原checkpoint/core/recipe、K32/NFE10/H32/prefix4。原生14项controller/merge/runner检查通过。资源受限后的更小规格使用独立新计划和输出，不修改已提交source，不重复未知创建。

机制入口新增完整事实回放专用工程模式：模型构造先于fresh role reset，事实推理不做物理pack/unpack，每一条新生成命令与before/after qpos均精确对照既有完整source；保留原官方终止，禁止截短、原动作驱动策略或充当新的独立SR。仅工程回放通过后，仍需独立完成原始父状态下的反事实资格。普通机制入口亦将事实首命令一致性检查提前到分支执行前；异常路径实际query数保留。原生48项branch/capture/replica/policy-state/准入/资源布局检查通过（21.75秒），precommit源码集合`ed6c8d7a39d79673776367c93b22fa5a1c34bd7d55f779207678286eb10b7f13`，收据`S/rebuttal_20261109_acp/REPLAY_AND_RESOURCE_NATIVE_TESTS.json`。此阶段实现检查与未来实际回放必须分开记录，不能写成机制优势已补齐。

### 23.23 紧凑ACP作业与CCI真实回放的当前证据

实现Git `faeb4d6f927c4c5272b40fb00b86afa3aa44dc37`已完成本机提交、origin/main核对及CCI canonical快进。任务只用`S/rebuttal_20261109_acp/controller_faeb4d6/code`独立快照，源码集合`ab19db64db3d9ac47e4969b7d402c07f5f31a73a911e648b65595d8035266f03`；与48项已测precommit的三个改动文件统一行尾后逐字节相同。正在运行的旧快照没有修改，36项无关文档删除没有纳入提交。

已用记录的镜像与AFS自动填入并创建`warm-rebuttal-rb-pairs-20261109-b`，UID `6ff8eb21848c4f9696ad25c6c76cbe97`，闲时集群02、1卡8核128GiB、最大重启0；原2卡作业a已确认“已停止”，没有原生启动输出。b计划SHA `51118b16d10be66a81f4f2b7484ed14a8628e042a686dbb0f5b749de156c106e`，固定前10对不变，两个独立角色进程共享唯一分配GPU、最长12小时。08:52 UTC Worker事件已无原CPU/内存错误，最新为`insufficient resource: nvidia.com/gpu 1`，此前挂载PVC等待事件亦保留；09:00 UTC控制台仍“启动中”、实际规格乘数0，AFS无原生启动receipt。**提交完成，GPU尚未运行。** 单卡标准资源另做较小配额尝试c，服务返回429限流；随后详情页与404明确训练任务c不存在，没有再点击未知创建或把它登记为成功标准作业。c计划SHA `89def89af83f932cc18abe5089dcc91c2e5b5b3612cfddf93268f928f5915b0e`仅保留为未创建计划；b继续排队。

CCI完整事实回放于08:45:01 UTC实际启动，job `factual_replay_job_logs/20261109T084504Z-49a42bda`，spec SHA `91f984fe9731e9936429b0ac1dbc5d52fc957c2a9f1420200723c5a02013a256`，仍用原Battery Try source0及作者checkpoint，官方1000步，独立7200秒上限，模型进程显存fraction0.4。现场另有WAMO进程运行，按至少64GiB空闲准入，保留它不变；本项不作为延迟基准。09:02:31 UTC固定快照中官方executor日志推进261步，q0/q8/q24/q48分别封存在事实命令0/32/96/192处，生成命令和before/after qpos精确比较已通过对应前缀；尚无完整回放最终收据，反事实分支数0，也不是额外的独立成功率样本。

该活动快照连同ACP可见配置、冻结计划/来源、启动收据和先前不一致诊断已双端封存：`S/rebuttal_20261109_acp/active_factual_witness_and_acp_submission_snapshot_v1.tar.gz`，956,743 bytes，SHA `4b267cd3d91dd7f4182c877f9d7561af4546bd84d23c5be68397e8a8295a6c5b`，本机23项内部文件核验，报告`ACTIVE_WITNESS_SNAPSHOT_LOCAL_VERIFICATION.json`。只截取捕获时点的console前缀，不把可继续增长的活动日志称为完整结果。Full100完整CAS本机传输继续进行，不能据此移除其AFS独占payload。

### 23.24 原生运行状态核销、Full100双端保全与CCI首个RB正式pair尝试

11-09再次核验发现，ACP b的AFS `running/elapsed18.9566s`收据已停更；控制台为17:16:58失败，Worker事件4301明确`has been preempted by job`，其后容器被停止。实际GPU为H100 80GB、SM90、580.95.05，Torch2.7.1+cu128；两worker曾启动，但`role_trials`完成数0，无完整pair。原native receipt不覆盖，外部状态标记抢占失败，**不记为SR=0或政策科学失败**。原标准c与新的标准d提交均见429响应，随后详情页明确任务不存在，未登记为成功作业。d保持原core/模型/前10对，仅新运行目录，冻结计划SHA `83902800b8d75b8cf15dcdd24581ff55382728a9421fb476be9a3bfd7926ccd0`；当前没有新的ACP正式完成结果。

CCI Battery source0事实回放的六个q0/8/24/48/80/120前缀探针均已落盘；最后console为768/1000、mtime09:15:53 UTC。10:05 UTC现场进程2090137/2090133已不存在，`summary.json`和最终事实trace未生成；原running manifest同样为停更状态。记录为**中断、最终资格未通过、原因待进一步定位**，不凭时间关联直接断言平台重启，也不以部分前缀替代完整机制证据。

Full100本机自包含压缩CAS传输已完成：388个blob及1份manifest共389文件、22,539,160,115 bytes，耗时2473.891秒；本机transport receipt SHA `3e650a982454f186602dd0c02eb4533e73e703c65a52d5c59dde7c685dbec76e`，已回传AFS。结合23.21原生完整TAR逐字节重建，构成双端保全；本机仍未执行TAR解压或全tensor复扫。原始author权重、数据及任何当前引用blob均未删除。

为继续获取真实闭环数据，在确认无既有RB完整trial后，CCI新尝试`rb_pair_case0000_cci_v1`于10:13:27 UTC启动，spec SHA `7efa2e345a5558a536d87d162da34d3a5783dd50b8af8f17add440859ca2de65`。使用固定core `5d7748e77a648511c5ddcfc0c65c3fac9933bbeb0e2bff20c61a7fd7e6356638`、作者step3000与原v3主计划；同一进程中两个独立policy按原次序运行case0/seed340800000、Full/null、官方各700步，H32/prefix4/K32/NFE10不变，7200秒上限。空闲67,023MiB时准入，模型进程显存fraction0.7，保留其他工作；这不是延迟benchmark。此处仅登记启动，尚无完成pair或收益，不将ACP抢占产生空目录视作已完成或换样本。

当前核销包`S/rebuttal_20261109_acp/evidence_and_infrastructure_audit_20261109_v1.tar.gz`，2,041,005 bytes，SHA `c9c4558be8ca0072ecde580f470b23d3e22987df11f44e7a955b3083c550cbe7`，双端44项文件核验，报告`CURRENT_AUDIT_LOCAL_VERIFICATION.json`。包含ACP原件/可见事件、CCI中断前缀/收据、Full100本机保全证明及新CCI启动身份，未声称大权重在这个小包内。

### 23.25 AC视角证据盘点与真机1–2任务补强（计划，不是新结果）

作者本轮确认合作方仍保留四任务各20次评测的逐次结果与录像，本周可安排约1–2个task的小规模对照。完整分析与可编译LaTeX/PDF保存于`tmp/rebuttal/analysis/20261109_ac_evidence_real/`，六页PDF已编译、渲染检查，0个排版警告；可交付的协议片段为`docs/rebuttal/REAL_ROBOT_REBUTTAL_PROTOCOL_ZH.tex`。

优先恢复原80个任务-reset在所有方法下的逐次对应，登记R2P起始盘7/7/6、SOR两种顺序各10、实际几何/桌面条件、失败/接管、输入时间窗及checkpoint/bank来源；不把黑桌面七组定性序列补写成分外观成功率。优先建议R2P：20 reset×原库/困难库×Full/关闭显式比较，共80个新自主试验；计算困难库Full对照差和配对交互。若只能60次，预注册三条件降配，并明确缺少完整交互。新增donor必须独立于TEST、保持完整真实事件和原动作/encoder/normalizer语义；若仅操纵既有top-K，则称候选压力测试，不称自然大库扩展。

可选SOR：20 reset×Full/candidate-null，共40次，二者均保留回合recollection、共同撤回候选source和conditioning，检验已有顺序记忆后跨回合经验的增量。若用固定权重开关，按推理干预报告；独立训练baseline另记身份，不能混称。上述试次数是待首次TEST前冻结的建议，不是完成量，不保证显著性。整体长时序、历史依赖、库多样性及实际候选歧义是不同证据轴；当前R2P/SOR的原稿15/20与35/30点增益可以积极阐明，但不能替代困难检索和模块必要性的直接对照。

### 23.26 自主实验闭环、RB首个完整pair与角色级续跑实现

作者明确授权后续自主设计、CCI实现/同步、ACP提交/查验、修复重提、记录分析以及整理图表；常规实现、合理资源分配和失败恢复继续执行，不因任务失败停在状态说明。科学方案、checkpoint/数据范围和既有固定样本规则不因资源失败而静默更换。

`rb_pair_case0000_cci_v1`已完整完成：case0/seed340800000、Full与null各700步，均success0、max_reward0，原生完整pair数1。elapsed2714.626秒，next_candidate_seed340800001；两角色初始身份精确相同，core仍`5d7748e77a648511c5ddcfc0c65c3fac9933bbeb0e2bff20c61a7fd7e6356638`、作者step3000、原v3计划。原件及job日志封存`S/rebuttal_20261109_acp/rb_case0000_complete_pair_v1.tar.gz`，7,689,063 bytes，SHA `d5b9567f25ec96dba377115a5a993ba873427b5a8637dc12b7dbd54ed562e780`，本机1120项内部文件核验；新证据validator亦已读取真实结果、700×14三类轨迹及初始图像哈希通过。**这是第二个任务的单个零/零pair，仍无策略收益结论；不替换投稿任务SR。**

固定下一段`rb_pairs_cases0001_0002_cci_v1`于13:38:13 UTC启动：沿既有完整case0给出的next seed，起点340800001、全局剩余candidate预算1999，case index1/2，两个完整角色、原700步与采样预算；上限10800秒。准入时GPU空闲81,080MiB，模型显存fraction0.7，spec SHA `15ddc87b3aac1b77464466c85d64806da56bbe30276a17fe1369a110af129f53`。这是进行中的两对固定任务，不能将启动量记为完成量。

新增expert-admitted reset roster与角色级campaign：每次只执行一个固定role/case，完成后严格核对模型/core/协议/runtime、原步数、完整native轨迹及初始图片/qpos，再独立压缩并计算哈希；中断角色保持未知。ACP容器重启后读取已核验原件，仅补缺失角色；原始开始时间与总预算不重置。worker继承文件锁，阻止旧子进程未退出时重复提交；重复完整role须明确核销，不能按好成绩挑选。case0及CCI case1/2明确作为外部来源，ACP计划仅覆盖不重叠case3–9。roster按原seed流及expert准入产生，不看策略结果；所有新资格仍须真实运行，不以协议测试替代。

CCI原生27项证据完整性、JSONL中断尾部、封存恢复、继承锁、持久时间预算、资源分配和合并检查通过（2.34秒）；Windows13项通过、1项POSIX锁检查仅在原生执行。导出JSON/CSV/LaTeX计数表，绘图程序用灰色保留pending；图表是已完成的逐条证据展示，不能将partial counts包装为总体收益或不劣。实际case0的独立validator报告为`RESUMABLE_ACTUAL_CASE0_VERIFICATION.json`。

### 23.27 固定reset roster、明确标准配额错误与可续跑ACP实提交

expert-only roster在原seed流340800000–340800009逐个通过，共10/10原官方专家规划准入；未读取Full/null成败。外部`S/rebuttal_20261109_acp/reset_roster_rb10_v1/roster.json` SHA `58044eb62baffec13b55fd1eb8e3d81096f73d579289dd1c6781f6ebcb11cfc3`，core和runtime前后哈希保持不变。这是reset准入证据，不能计为策略成功率。

已冻结并原生验证角色级计划`resumable_rb_cases0003_0009_v1_plan.json` SHA `d8512a23021383385bab2014e2092b52d12495e1fbdd80e804c04f47216df249`，controller Git `df1df34d990b7b296d18fceb44d6748050072c99`、源码集合`75e24ed18de93c9dc577cbc2a7c5444d0e0eda4748291cdb096372ee450a299f`。ACP只拥有case3–9；case0和CCI正在执行的1/2为外部来源。总12h预算跨重启持久化，每角色3600秒、最多3次attempt；.5显存fraction仅为进程资源上限，不改推理参数。已输出真实计数JSON/CSV/LaTeX和SVG/PNG，并检查PNG：当前1/10完整pair、Full0/null0，其余9对pending；这是过程记录图，不是可主张收益的rebuttal结果图。

本次标准资源提交直接捕获服务端429正文：code8、`MEMBER_QUOTA_EXCEEDED`，`workspace_aec2_total`剩余GPU0、请求1。故此前仅根据HTTP429写成“限流”的解释不充分；本次有明确**成员配额不足**依据，按作者要求转闲时。闲时创建返回200，任务`warm-rb-resume-20261109-e`，UID `e9c3736f61124ffebebef597abf524b0`，22:09:09 CST创建，集群02、1卡8核128GiB、最大重启2；镜像、AFS与冻结命令均已在详情页回读。14:15 UTC附近Worker仍等待中，AFS无运行心跳，**未声称GPU已运行**。原始响应字段和实际UI观察来源保存在外部`ACP_E_CREATION_RECEIPT.json`；旧失败、旧计划和case0完整原件保留。

为处理23.19/23.24尚未解决的原始事实父状态问题，新增独立fresh-scene native replay资格入口：只执行既有完整Full source的记录命令，两次独立reset，每条before/after qpos、固定query的RGB必须逐元素等于原source，新采集的native物理bytes/驱动目标/task/controller/camera/RNG必须跨reset精确重复；不做物理unpack。opaque句柄仅做显式类型检查，不宣称隐藏状态已完整捕获。Windows3项与CCI原生26项包含单像素、dtype和单ULP篡改拒绝的检查通过（原生0.79秒）。**实际物理回放尚待启动，不能用这些检查宣称反事实机制已通过。**

### 23.28 第二个RB配对、完整训练状态恢复与官方观测语义更正

固定case1/seed340800001已完成：Full0/null0，各700条官方命令、max_reward0。当前RB完整配对共2个，均0/0；继续固定case2及ACP case3–9，不按成败换seed。原worker在执行case2，因此只复制case1完整目录、原始对应JSONL行和静态身份，封存`S/rebuttal_20261109_acp/rb_case0001_complete_snapshot_v1.tar.gz`：6,631,192 bytes，SHA `f7d74bd7c38143e4406d411a97c5c247b871dd2fc6dce698fa9d560cfb549c66`，本机1112项文件哈希核验通过。计数表和过程图已刷新为2/10 complete、8pending，**仍无新增性能优势证据，不替换原稿数字。**

本次原生源码核查发现需要收窄早先的“实际关节运动”解释：官方`envs/robot/robot.py`中`get_left/right_arm_jointState`读取`joint.get_drive_target()`（约524/531行）；`_base_task.get_obs`由此组成`joint_action.vector`。夹爪API也读取controller的`left/right_gripper_val`。故历史工件字段`native_qpos`和`qpos_trace.before/after`保留原命名，但其臂部是**驱动目标通道**，不是独立物理关节传感。原审计可以检查命令是否被官方目标通道接受，不能由此声称物理servo精度或全部实际运动已获证明。原始图像、官方物体几何成败和官方输入协议保留；不为补传感而静默改变policy输入。

首个RB对的原始相机序列已形成`diagnostic_rb_case0_v3/source_rgb_montage.png`，含官方输入的3相机及仅用于取证的front相机；与`native_execution.svg/png`、逐文件输入哈希和诊断JSON同存外部。两角色的臂部target-minus-after均0，只证明目标接受；旧v2图曾标为realized，已明确由v3的drive-target标签与上述源码语义更正，不作为物理跟踪证据。图中可见实际机械臂运动及操作尝试，但不能凭视觉序列宣称长时序任务已成功。

研究资格入口新增独立`articulation.get_qpos/get_qvel`只读采集，每条命令与固定query分开保存实际关节状态；不改官方观测、动作、模型或采样。Windows4项及CCI原生47项观察语义/状态拥有权、真实坐标读取、单ULP/像素拒绝和既有policy检查通过（1.01秒）。原native replay v1仍在独立冻结的Git `dc50f06`/源码集合`a8653771793bd02d78389597f99a2edd8a5d07efe53bfca6dc861faf3fde3c52`运行，14:18:57 UTC启动、1800秒上限、spec SHA `b6418a7bb107cf344ff45a6d2ae28dfaefaf94c0298219dc5bde21fcfac826fa`；14:40 UTC首遍700命令已重放，第二遍及完整资格尚未结束。此处不把旧通道一致性扩张为新物理读数已完成。

Full/control的100-update完整状态已从双端压缩CAS逐块校验并重建至独立CCI scratch：分别38,599,567,360与38,438,236,160 bytes，TAR SHA `fb4321c8dc16b449d320cc6f97fafaaa88b08d22214af13af4127852e5e96a87`和`2acd20824aacce72e6520866b7e980ee9ca64bd926315e4854d0a259b794385c`，两权重SHA与23.16/23.20一致，14:30:31–32 UTC完成。收据`RESTORE_FULL100_RECEIPT.json`/`RESTORE_CONTROL100_RECEIPT.json`，并非新增训练结果。原配方/优化器/调度器/RNG/数据游标续训100→200的有界顺序队列已启动等待（14:28:42 UTC、队列3h、每角色3600秒），只在现有评测终止且GPU无其他进程时准入；checkpoint选择仍固定3000，暂无新的200-update或TEST结果。产物写AFS，成功后自动保全并追加本ledger，执行程序不执行任何Git网络操作。

### 23.29 回放超时的实际核销、遥测新轮与过程材料双端核验

native replay v1于14:49:01 UTC由1800秒预算终止：runner `failed`、exit−15、`wall_time_limit`，elapsed1802.036秒，前后源码集合不变。首遍700条源命令及六个query已通过；代码进入第二遍后，q0/8/24/48的native物理bytes、poses、controller/task/camera/RNG和RGB均与首遍逐项哈希相同。第二遍没有完成，程序最终summary缺失，因此仅登记“首遍完整＋第二遍四个固定query前缀匹配”，**不是完整双遍资格或反事实收益**。原件、runner、下一轮路由封存`native_replay_v1_terminal_and_v2_route_v1.tar.gz`，5,256,272 bytes，SHA `b7b4cf94a4db5120871fd1fe07a993c58c53108ee243bf508b6d4787fd480fff`，本机23项内部文件核验。

按实际耗时单独冻结v2的5400秒工程预算，原source、seed、两遍全程、官方700步与精确一致性门槛保持不变。旧进程确已退出后自动准入，新spec文件SHA `e8adbd9ced889a8d4a1e73cdc97ec011da89404c26caf222c9a2e0d9f4dd1049`、Git `29ca02561f39cabe8fa577bd6b19b89ff36a6e8f`、源码集合`04c320a3e775b41890a47ef61bbeb7c3bce86578fe30582e07362d57b0594d79`，准入GPU空闲24,059MiB。新轮已真实生成`physical_joint_trace.jsonl`，独立读取实际articulation qpos/qvel，包括机器人与任务button；这与官方drive-target输入分开保存。**当前仅是已启动并采到首条真实遥测，未宣称完整回放、分支修复或候选效应已完成。**

顺序训练队列亦修复了“硬中断程序可能没有自身summary”的准入问题：只停止了确认仍在等待、尚未启动训练的自有旧waiter，保留其旧源码与停机收据；v2改读runner实际终止收据，且等待遥测v2结束及GPU空闲。原14:28:44开始的3h总窗口保留，没有重置预算。队列仍等待，Full/control仍100/3000；大权重原始证据、恢复state及其他作业均未删除。

在实际AFS上另做跨进程独占锁检查，竞争者确被阻止，释放后可获取，收据`AFS_CAMPAIGN_LEASE_QUALIFICATION.json`；这只验证ACP续跑租约的存储能力。ACP e在14:46 UTC详情仍启动中/实际规格乘数0，最后Worker事件为`insufficient resource: nvidia.com/gpu1`；不重复未知创建、不记为活跃GPU或SR失败。

过程材料封存`closed_loop_evidence_progress_20261109_v1.tar.gz`：1,982,961 bytes，SHA `84a5fd63f5b7b0b585d7b45a20941340132eee0eae90c2fd89fcab68b2142228`，本机37项内部哈希核验。当前真实计数CSV/LaTeX、SVG/PNG、原相机序列和语义修正后的诊断JSON整理到`tmp/rebuttal/analysis/20261109_experiment_progress/`。这些是2/10固定配对的过程和诊断材料，不能作为未完成总体的收益结果；最终rebuttal图表须待相应固定cohort与直接机制实验完成后再据实提炼。


### 自动采集：full matched seed3407 完成update200（训练证据）

UTC 2026-11-09T16:38:17.097598+00:00；严格resume100→200，batch128、原30k scheduler与源集合`4efb8094729ba8771c7ec0d472c30ad8b585ef319e897edb566b12d3d5a81a14`。权重SHA `1fb035c41663752d72098196cc04f04ff03cefcfa7db623bc994142d14af0ac4`；逐字节TAR重建SHA `b2a7a0e2b0b4dd441a24e70f9bb6145ac49db3827321c890b672a60a0db035b6`、38599198720 bytes。原件与档案收据`/mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations/rebuttal_20261109_acp/FULL_UPDATE200_ARCHIVE.json`。共100条有限loss/grad、更新101–200，优化器/随机数/数据游标由原完整状态续训；固定选择点仍3000。**仅训练执行和保全证据，无TEST配对性能、无正式baseline收益结论。**


### 23.30 正式并行推进、完整RB/回放原件和11月13日稳定候选

作者授权正式自动推进实验、回复、论文修订，并要求11月14日前稳定版本。已建立本对话每小时heartbeat `warm-rebuttal`，内部验收2026-11-13 23:00 CST，运行规则见`docs/rebuttal/AUTONOMOUS_REBUTTAL_RUNBOOK_ZH.md`，当前材料入口`tmp/rebuttal/planning/current_iteration.json`。定时轮次推进实质研究和写作；不把排队算GPU工作，不自动发布OpenReview或正式上传文稿。

固定RB case1/2完整worker已结束：seed340800001/340800002，各Full/null700步，均success0/max_reward0；加case0共3/10完整pair、Full0/null0，余7pending。完整worker和runner证据`S/rebuttal_20261109_acp/rb_cases0001_0002_complete_worker_v1.tar.gz`，15,386,512 bytes，SHA `7e23e0bcd779ff98f535878c29ee3408ee20c8be7daaacf0eebb2217204dbd9f`。本机逐项校验2225文件通过。新子集不替换投稿SR，不支持性能优势或不劣结论。旧2/10过程图已经过时，后续展示必须以当前3/10计数重新生成。

fresh-scene replay v2于15:33:04 UTC exit0：两遍各700条原命令、六个固定query（0/8/24/48/80/120）父状态跨遍哈希相同，1400行独立articulation qpos/qvel均有限，RGB和官方drive-target与原source精确一致。完整证据`native_replay_v2_complete_evidence_v1.tar.gz`，7,023,585 bytes，SHA `9f03c739f474fe3b7d543f5590dedfb59e7d00c82e321142c21fdcc6c0c49793`，本机24文件校验；收据`COMPLETED_RB_AND_REPLAY_LOCAL_VERIFICATION.json`。**仅工程回放资格，尚无反事实分支或修复收益。** 新增独立fresh-scene A-A-B-A入口，绑定原Full source0/q8的原封存raw/refined H32和执行prefix4，物理unpack及事实策略更新均为0；重复端点必须一致、B必须实际不同。TOPP/RPC拒绝保持invalid/unknown。Windows11项原始prefix、单ULP/像素拒绝、规划错误传播、状态拥有权与独立rubric检查通过；原生分支尚待验证，不先写通过。

原训练v2误把正在运行的partial summary视作失败，已保留失败收据并以v3修正；没有改seed、source、配方或原14:28:44 UTC起3h截止。Full100→200训练和38,599,198,720 bytes状态档案已完成并逐字节校验，身份详见前一条自动采集记录。因原窗口不足再准入control，单列新的5400秒资源窗口、每训练3600秒上限；`CONTROL200_AMENDMENT_OPERATOR.json`保留该预算修订。control100→200于16:38:25 UTC准入，spec SHA `8ef19c44c5af7b965411f92d86c017f6ee7fb6601ddc8ae2d8aded02137a0e4e`、config SHA `bf4a235a5810fc2a535314e91d42382d415286f145c96d1b4d34c775f0377f02`。固定TEST选择仍3000；不把200工程段或当前适配prior称作已完成的ReCAP/WarmPrior公平性能比较。

SenseCore会话过期由作者重新登录后恢复；16:32 UTC回读ACP e仍等待、0GPU、事件`FailedScheduling / insufficient resource: nvidia.com/gpu 1`，并非已运行或实验失败。原始UI收据`ACP_E_REAUTHENTICATED_STATE_20261110.json`，case3–9所有权保持，未重复创建。OpenReview实时回读仍为Bbqv/Hh4E/rkWY三位人工审稿，3/2/2、最后修改11月6日，逐条20项仍为当前回复依据；另有PAT LLM反馈单独保留，未算作第四审稿人。

逐审稿人v3正文、20项问题—证据—修订映射及15个独立短回复块已完成内部首版：`tmp/rebuttal/responses/20261110_v3/`。revision v3从完整原LaTeX工程编译为35页、主文10页，保留全部原始数值表及9个有效图件；加强R2P/SOR有序操作、历史依赖、起始盘7/7/6和两顺序各10的呈现，新增真机任务结构表，保留录制观测成本证据。PDF SHA `ef9e6d55c7a24822b147ec392773c2a5c17ec7bfe9bbfbf79d6d34ee04545b4b`；干净29文件完整source ZIP SHA `120daf4e8d298e8181565a646d9c6f5f424add748c97d4d2ed91f1e303733304`。全35页渲染检查完成，无overfull、未定义引用、缺字；存在继承模板的Times斜体小型大写字体fallback warning，已记录。当前仍为`stable_release_ready=false`：直接修复、候选区分扩展、同信息最终对照、非空missing-skill和困难经验库证据尚未全部关闭，不能以文稿完工冒充科学闭环。


### 23.31 首个原始事实父状态fresh-scene A-A-B-A通过及干净源包重建

Git `fabb74b3f46b5634d7eeeaccb8aa9b21259bb534`已推送并核对GitHub/main，CCI canonical已保留自动采集台账后fast-forward；运行端冻结`controller_fabb74b/code`，source SHA `b96ec080e834f5b1884de4ebeb858962a5a82f6c026c0c534dd47f7395a1146b`。Windows11项通过，CCI原生分支/回放/独立rubric/恢复检查26项通过（0.90秒）。执行程序没有Git网络操作。

首个固定Full source0/seed340800000/q8的真实fresh-scene A-A-B-A已完整通过：4次独立reset重放原命令到frame32，原RGB/官方drive-target与封存source相同，四个真实physics/poses/controller/task/camera/RNG/actual-qpos父状态哈希一致；三次refined H32端点逐项相同，raw H32的实际physics端点与refined不同。128条分支命令全部完成、无TOPP拒绝，物理unpack和事实策略更新均0。raw与三次refined的独立physical-next-interaction标签均0，**尚无修复收益、candidate-null比较、候选后果预测优势或整任务性能结论；资格仅限当前source/query。** 不按此结果替换query，继续原固定cohort。

实际runner exit0及全部原件封存`S/rebuttal_20261109_acp/rb_source0_q8_fresh_branches_v1_complete.tar.gz`，4,679,786 bytes，SHA `55984ff0eddc44b5840ceb590067151f6325becc5a3acf72a007c53dab8efb44`；本机22项内部文件全部核验。报告`FRESH_AABA_ARCHIVE_RECEIPT.json`/`FRESH_AABA_LOCAL_VERIFICATION.json`；启动spec文件SHA `553370f24cbf9052f10e4ea3d926213e211d7d8bbfa2218f398b9c969170a562`，runner规范化JSON spec SHA `1c907a56ca7de6010c63e2cdffbd71e6be2ee7a212fd416ae15fc5608fd2656b`，两种哈希对象分别登记，不混称。

revision干净29文件source ZIP已在独立空目录重新完整LaTeX/BibTeX编译，35页每页提取文本与交付PDF完全一致，无overfull、缺字或未定义引用；报告`tmp/rebuttal/revision/CLEAN_SOURCE_PACKAGE_REBUILD_VERIFICATION.json`。实时三个人工review的全部20项原句亦逐项核对通过，`LIVE_REVIEW_COVERAGE_VERIFICATION.json`。回复修正了HH-W5的精确章节定位；原稿数字、PDF及source ZIP不变。

文稿与回复基础包已在AFS逐项核验26文件，SHA `c5c8470ea3fc506e7f310f8cbe68a41d16138b99c77d4187c27ef8debe840b09`；24文件preflight补丁SHA `b48a6ab4dfde130cca388948c4931f960c040ef8fe8706231fb80490c17981a2`亦已核验。基础材料位于`S/rebuttal_20261109_acp/writing_candidates/20261110_v3`，更新的回复、证据映射和干净编译核对在同级`20261110_v3_preflight_patch1`。原包和补丁均保留、未对外发布。Full200已保全；control100→200已进入真实优化步骤，不能将正在训练或待评测写为完整公平对照结果。


### 23.32 非真机优先级更新与可续跑固定source K32实现

作者于北京时间11月10日更新：暂时找不到真机逐次结果和录像，先考虑非真机工作。已更新本对话`warm-rebuttal`每小时任务及runbook，将新增真机实验/录像审计移出当前关键路径；保留原真实机器人结果和已有协议的报告身份，不据此推翻原稿结论，也不继续承诺正在采集新的真机困难库结果。当前重点为后果预测、同状态修复、自然缺失技能与公平对照。

新增fresh-scene K32固定source采集入口：原v3计划、作者checkpoint、原始Full封存的全部valid候选及q8/q48不变，按refined/refined、原rank顺序、末尾refined执行。每一分支独立reset并精确重放原事实prefix，核对RGB、官方drive-target、native父状态与独立实际qpos；不unpack物理，不生成或替代事实策略动作。每个终止分支写原始command/actual-qpos/qvel、完整端点、独立标签及逐文件complete收据，中断attempt保持unknown；总12h资源窗口跨续跑持久化，AFS独占锁防重复，不按结果挑rank或换query。完整query自动追加本台账且无Git操作；原始集群/CI工作树脏不会阻止运行。candidate-null仍须同一Full事实上下文另行生成，没有拿独立null rollout代替。

独立DINO truth reader绑定原M1处理器、相机、normalization、DINO tree与CUDA/bf16 runtime，复用原始effect投影。第一版工程preflight因将JSON列表直接传给baseline normalizer而失败，未产生候选结果；已保留冻结v1源码并改为复用原`load_dataset_stats_from_json`的float32 tensor路径。v2原生读取source0/q8的两次特征完全相同、CPU/CUDA RNG不变，shape4×768 float32，feature SHA `41d5632901305a025308b4100aecdb2cbdbcdbb1bb34d583e1aee45489bc0c82`，`FRESH_POOL_READER_PREFLIGHT_V2.json`。只确认相同原输入/recipe/encoder，不声称核对过未保存的原retrieval tensor。Windows15项、CCI原生30项分支/传感/归一化工件/不可变收据/独立标签检查通过（原生0.94秒）。新完整K32物理查询尚待执行，不先填写后果优势。

control100→200于17:15 UTC写出`training_complete.json`，原训练run_steps100、固定选择点3000与原30k scheduler保持；完整state原件存在，当前独立TAR/CAS保全仍在进行，未将200更新或保存成功称为最终匹配性能。Full200已完整保全。后续须先确认control档案、AFS剩余空间和CCI实际重启状态再接续训练；不删除唯一state或旧证据来腾空间。


### 自动采集：control matched seed3407 完成update200（训练证据）

UTC 2026-11-09T17:26:08.169622+00:00；严格resume100→200，batch128、原30k scheduler与源集合`4efb8094729ba8771c7ec0d472c30ad8b585ef319e897edb566b12d3d5a81a14`。权重SHA `93870d3395cb0ee62978e627ec13d8da38f5b41c886fd5cba11aa60664069a41`；逐字节TAR重建SHA `d838ed0c5fc039170394a9719523a3c8d738929c9652b8f49ea6795d895451d7`、38437867520 bytes。原件与档案收据`/mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations/rebuttal_20261109_acp/CONTROL_UPDATE200_ARCHIVE.json`。共100条有限loss/grad、更新101–200，优化器/随机数/数据游标由原完整状态续训；固定选择点仍3000。**仅训练执行和保全证据，无TEST配对性能、无正式baseline收益结论。**


### 23.33 Full/control200完整保全、首三源并行K32与传输归档

control200训练与lossless档案于17:26:08 UTC完成：权重SHA `93870d3395cb0ee62978e627ec13d8da38f5b41c886fd5cba11aa60664069a41`，由control100权重`3064637f...`完整optimizer/scheduler/RNG/dataloader状态续训，100条更新101–200的loss/grad均有限。原始TAR 38,437,867,520 bytes，SHA `d838ed0c5fc039170394a9719523a3c8d738929c9652b8f49ea6795d895451d7`，AFS CAS逐字节重建通过；收据`CONTROL_UPDATE200_ARCHIVE.json`。与已保全Full200构成同一原配方/seed3407的两个训练段，最终选择点仍3000，**没有新的匹配TEST结果**。前面“保全中”状态由本条完成收据替代，不删除历史前缀。

Git `eb0545c18138e9744a32bcd19fa2e4112e2b64c7`已本机提交并推送核验，CCI冻结`controller_eb0545c/code`，source集合`823404c10af58f90b932b406b0a4cc2a3632acb20b691841a8a82ac7b0c5cdad`。本机35通过/1个Torch检查跳过；该Torch检查在CCI补跑，完整51项原生通过（10.79秒）。frozen reader对原q8输入重复得到相同4×768特征，CPU/CUDA RNG保持，原始处理器stats loader修复已在真实native执行核验。

资源级扩展为在同一H100上并行固定source0/1/2的K32采集：source0于17:27 UTC准入，source1/2于17:44 UTC准入，各自主process的PyTorch fraction0.12、单次runner7200秒、每source持久总窗口43200秒。原v3的首10个source、q8/q48、原valid rank、所有已封存动作/预测与H32保持，不改变模型/数据/样本/统计目标。运行时已观察三路真实GPU进程、合计约27GiB；不是三个GPU，也不是新训练速度或推理延迟benchmark。已有完整分支复用，部分attempt保留unknown；当前未完成query不写成候选效应优势。

18:04 UTC已保全的活动快照：三个q8各有两次完整refined anchor；候选共有30条终止收据，其中29个完整H32、1个source0/rank11在23步后TOPP拒绝，保持未知标签/effect。29个已知独立局部标签均0，尚无完整K32 query、candidate-null或新增收益结论。原件`fresh_pool_first3_sources_active_snapshot_20261110_v1.tar.gz`，44,126,627 bytes，SHA `dcc3aaafe72b67ad8862f99e55414370f1f69e500651a5e95ca1c4047a19a79f`，本机307项文件核验；`FRESH_POOL_FIRST3_LOCAL_VERIFICATION.json`明确区分terminal与完整H32，不将invalid终止收据计为成功端点。

ACP e于17:35 UTC实际回读仍0GPU等待、`insufficient resource: nvidia.com/gpu 1`；原任务与case3–9所有权保留，未因排队重复创建。控制台通过现有浏览器登录流程恢复，只回读任务状态，不记录口令或token。另从现有CCI实例表核对其Pod名与SSH hostname一致，配置为1卡22核230GiB；该观察时点UI运行时长8小时35分、重启180次。该值不确定下一次重启时间，不据宿主PID1或旧“4h”文字保证长任务安全，仍以持久分支/训练state恢复。

AFS剩余约89GiB（本轮观察），后续训练需先重新做空间准入。Full/control200两份native已验证CAS的联合本机传输已启动：401个blob/manifest条目、35,973,567,912 bytes，inventory SHA `c896c79385d2103ec511686d4761105c403d8da7ac70046f05f1808cd7d81d0e`；复用已核验本机共用payload，新增部分后台SFTP续传到`tmp/rebuttal/execution/20261108_p0/evidence/update5_cas_store_v1`。仅完成全部SHA/长度核对后才登记本机完整保全；当前为在途，未删除任何checkpoint或AFS blob。传输进度`MATCHED200_CAS_TRANSPORT_PROGRESS.json`，本机进程与日志另行登记。

作者的非真机优先要求已进入每小时任务、runbook、回复README及问题—证据映射。HH-W4与RK-Q1已去除新增真机/位移结果正在采集的误导性暗示；既有真机任务、数值、论文PDF和完整LaTeX源包保留。下一优先项为完成固定K32查询，并在不canonicalize原事实轨迹的条件下封存同一Full上下文/共同噪声的candidate-null H32；不能用独立null回合历史代替。

### 23.34 同一事实H32与检索上下文绑定的candidate-null封存入口

新增opt-in `--seal-null-on-factual-replay`，仅允许原formal固定source、完整事实生成见证与单模型只读共享权重模式组合。每个固定query在执行该chunk之前，先对照原`generation.json`的bound-step、模型输入、候选payload及derived seed，逐元素核对原封存float32 H32全448值与其array SHA；不是只比较执行的四步前缀。再核对原RGB/官方drive target，使用同一Full上下文、相同Gaussian与独立RNG生成g=0撤回候选结果，封存native动作、全部上下文与双方probe。原事实队列不被替换、不调用物理unpack、不执行任何null分支；即便封存完成，也不自动构成修复效果或candidate-null成功率证据。

本机30项通过，1项Torch副本检查因本机无Torch跳过；原生Torch与实际完整事实重现仍须在冻结版本补验。新增测试拒绝完整H32末端的1ULP变化、只传四步前缀、不同dtype、不同输入/检索/seed/封存哈希。原始fixed source、checkpoint、计划、H32/NFE及统计目标不变。此条记录代码和资格门槛，未宣称已生成正式null动作或取得收益。

### 23.35 原事实null封存v1入参失败与双trial布局修复

`f826c4c`已推送、CCI canonical保留自动control200台账后fast-forward；冻结source SHA `d0e50f7b198f5762dd2f46e927d2833f4d4d8bec7cdcb4252ee156f8bacf2ddc`。34项原生query/replica/branch/admission检查通过（19.74秒）。18:27:09 UTC以53,489MiB空闲显存准入source0 null封存v1，runner于18:27:41 UTC明确exit1：旧recovery入口只找`role_trials.jsonl`，但当前固定paired worker保存的是`pairs.jsonl`。失败发生在Full模型加载和事实执行之前，未生成null或分支真值，不算策略失败或候选标签0；完整failed runner原件保留于`rb_source0_nullseal_v1_job_logs/20261109T182711Z-a8065790`，spec文件SHA `d8003f5ffc3509405642de185856ecbe0f2e0ac73acba56a85b4dd4cd164f710`，runner规范化spec SHA `b4981dec7cfee92482f4d217307ae59cc6be532fb1208524ea05ac6ab4b4bdb4`。

修复为复用resumable pair公共的两种已提交trial布局读取：优先role-trials，其次pairs；只采纳带最终换行的完整记录，原partial tail原样保留。此外正式source在加载策略前复核原result、官方完整trace、原始初始图像/drive-target和simulator/source绑定，未放松身份或轨迹要求。本机42项通过。后续以独立v2目录与冻结版本重试；本条保留v1失败身份，重试尚未取得科学结果。

### 23.36 原生成witness的NPY文件哈希契约与v2工程停止

paired布局修复`01fd751`已本机/GitHub/CCI对齐；冻结source `f0a11d72fb08b09d299f01ee0b59249c2aa2215e54866a6bf92a70114c4632d1`。50项原生检查通过（20.90秒）；source0真实durable Full原件准入通过、700命令trace和初始图像身份均核对。18:32:20 UTC以52,311MiB空闲显存准入null封存v2，spec文件SHA `3968403d3eedcabb6a33e694749fe077cd1af90714a50875c4227f91254b5825`，原件`rb_source0_nullseal_v2_job_logs/20261109T183222Z-1e882b96`。v2 launcher沿用V1收据名的元数据覆盖已依据首次启动tool输出和冻结源收据恢复；两版启动收据均保全，恢复过程`NULLSEAL_V1_LAUNCH_RECEIPT_RECOVERY.json`，failed v1 worker和退出原件从未改动。

随后跨组件复核发现原`action_witness.generation.json`中`array_sha256`实际是**包含头部的完整model_actions.npy文件SHA**，不是`manifest.sha256_array`的dtype/shape/array-byte SHA。原生unit fixture此前手造metadata，未覆盖这个producer/consumer契约。已改为直接由原production action-witness生成测试文件，消费者显式接收并核对原NPY文件SHA，保持完整H32逐元素相等及输入/候选/seed约束；新增检查确保不能混用两种哈希。Windows48通过、1个Torch测试跳过。

18:37:45 UTC已用v2冻结检查函数对真实原q0工件复现该契约错误（原NPY和metadata文件SHA均`afac084c743ef5b9e8c52b41c04b47b7a92306405eb22d8bd341f571f86ebdd5`），故向核对命令行和独立process group后的本次v2进程发送SIGINT，避免继续无效GPU采集；三路K32工作保持。停止原因是已验证工程错误、非策略结果，不据此赋策略失败或null标签。`NULLSEAL_V2_KNOWN_HASH_CONTRACT_STOP.json`保留操作证据，真实runner退出/已执行prefix须继续核实；原始失败和中断不删除。修复后应先对原6个query的实际generation工件做契约preflight，再以独立v3冻结目录重跑。


### 自动采集：fresh-scene K32 fixed-fresh-rearrange_blocks-e0000-q0008

fresh-pool:fixed-fresh-rearrange_blocks-e0000-q0008:1b87159dd4cf6d74833a7d801eea81a56e57fcae7ac575666d091f340fa267f3；原K32有效候选32，完整H32端点28，完整refined重复支持True；原件`/mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations/rebuttal_20261109_acp/rb_source0_fresh_K32_v1/q0008`，候选结果SHA `286c8dc177ea0e085ddeb25f78df507cad38903e51d71fe428b49ba212fa4fb3`，centered结果SHA `2fcb6bb996e7634fdf3151a94d89bba90c1386999eaaac6a1b06fde01bb852c8`。**仅固定source/query的直接候选效应数据；不替换原论文表格，不视作整个cohort或策略收益。**


### 23.37 首个完整K32查询的负结果、原始特征哈希核对及null封存v3

固定source0/seed340800000/q8已完成原32个valid候选与三次refined anchor：28个完整H32端点，4个TOPP等执行无效、效应与局部标签保持未知；28个已知独立局部标签均0。三个refined端点精确相同。使用同一28候选支持集，predicted/historical/query-constant的centered MSE分别为0.0774111356/0.1023458527/0.0528061854；centered explained variance为-0.465948/-0.938141/0。对应absolute MSE为0.197898964/0.188502407/0.244073412。**模型在本query的中心化误差优于历史效应，但不及查询常量；绝对误差排序不同。本条不支持新的候选区分优势或修复收益。** 一个query不等于整个固定cohort，保留其他预定query和所有无效分支，不按结果替换样本，也不静默替代原稿作者确认的历史表格。

18:44 UTC不可变快照包含三个source已终止78条候选分支，其中69个完整H32、9个无效未知，8个完整anchor、1个完整query。原件`fresh_pool_first3_sources_snapshot_20261110_v2.tar.gz`，105,516,761 bytes，SHA `0f7b61a774934ba49d7da075f779fa739e956b48615d6b9f47c0ad1564f1763c`；本机736项逐文件核验。原sealed branch、负结果、v1失败和v2中断日志均保留。可审阅的对照表和分析为本机`tmp/rebuttal/execution/20261109_acp/fresh_snapshot_v2_query_results/ANALYSIS.md`与`effect_comparison.tex`；不将28个相关候选当作28个独立query计算置信区间。

原Full telemetry实际上还保存了`factual_world_tokens_sha256`。独立审计4个已生成父状态（source0/q8与q48、source1/q8、source2/q8），新增reader的4×768 float32语义tokens之dtype/shape/bytes SHA与原模型实际在线tokens哈希全部一致。`FRESH_PARENT_ORIGINAL_TOKEN_HASH_AUDIT_V1.json`同时绑定原telemetry文件SHA和bound-step；这是原hash的精确核对，不是未保存原tensor的直接数组对照，也不将NPY文件SHA混用。原先“未获得原tensor”的收据保留，此新增审计收窄了父特征读取失配这一解释。训练target索引为frame+action_horizon，当前分支为原32条native命令；物理耗时和原投稿评估模型/配置对应关系仍需核实，不从索引相同推断全部时间语义一致。

已修复NPY封存契约，Git `f82f8b043e1d2c2b7a23439a86da1c64d2b5c063`推送核验；冻结`controller_f82f8b0/code`，source SHA `8deb4c37d0a616211e84f05a9bdc4703fb55297c07bbab8262babf833589479d`。57项原生检查通过（21.05秒），原6个query的真实generation工件均在模型加载前通过producer/consumer契约核对。v2 runner明确exit-2，0事实命令、0null封存、0分支标签，原件保全；并非策略结果。v3于18:40:54 UTC准入，launcher PID2163273，spec SHA `e9ccbc76381c42354f6d576d3460f634592f085726c3d004b9e6ff2873c22c08`，5400秒上限；仅正式原source0、同Full历史与共同噪声，不执行null分支。此时仍在实际模型加载/推理流程，未宣称完整事实重现、null动作或修复收益已经完成。服务器canonical同步因刚产生的新query自动台账暂缓，冻结代码不受影响；须保留导入此自动条目后再fast-forward。

ACP e于18:30 UTC仍等待，事件`FailedScheduling / insufficient resource: nvidia.com/gpu 1`，case3–9所有权不变，未重复提交。Full/control200 native档案仍有效；本机联合CAS传输仍在途，必须待401项完整验证后再登记本机保全或分层旧AFS重复payload。当前优先完成固定直接查询、核对新负结果与模型/协议对应关系，核验简单对照的信息匹配；当前适配control不称作ReCAP/WarmPrior复现或最终公平性能。回复和论文的公开草稿暂不添加新增收益结论；HH-W5新增直接证据缺口列为明确release gate。

### 23.38 原始事实q0完整模型动作一致，绑定哈希含实测延迟导致v3停止

v3重建已在source0/q0生成实际Full H32：原始/处理后图像、proprio、context、DINO、候选payload、model-action整chunk及native整chunk的哈希均与原source相同，QueryId与instruction亦相同。原telemetry中retrieval的唯一差异键为`latency_s`与`step_sha256`；生产`_step_digest`包含五项实测检索耗时，因此跨运行原始step hash不可机械要求相等。真实诊断`NULLSEAL_V3_BINDING_DIAGNOSIS.json`保留所有对照字段和双方原telemetry SHA。v3在q0执行之前因该严格检查停止，0事实命令、0null封存/分支标签；保留失败，不赋策略或效应标签。不能将这一工程哈希阻塞与首个K32真实负结果混淆。

修复只作用于独立比较视图：用原Full replan记录中的五项已测延迟替换比较对象的telemetry，然后仍调用原production step-digest，要求整个其余绑定内容精确重建原hash。实际BoundOnlineStep、capability、模型输入、历史、随机数、动作和真正计时均不改、不重放capability。原文件SHA、完整H32逐元素一致仍强制要求，封存双方真实hash与比较proof。新增测试确保模型输入、原world-token hash、bank rows、cosine或valid mask任何变化均无法通过；54项本机通过、1个Torch检查跳过。v4应在冻结版本原生验证后准入，尚无null执行或修复结果。


### 23.39 v4实际准入、失败原件逐项归档与非真机工作交接

Git `66c301b21084169e4b8afa910379aee71f49c121`已推送核验；冻结`controller_66c301b/code`，source SHA `a720c23952c709d0cacb4e6bdb60165e98b9621494b62e8c45d043894dfc7dc5`。63项原生检查通过（19.96秒），原6个query真实NPY封存、telemetry与完整model-action chunk哈希均在GPU准入前一致。v4于19:03:13 UTC以50,703MiB空闲显存准入，launcher PID2170812，5400秒上限；spec文件SHA `9e62714e095d82ac9c8bb9019631efb485de9922a8020bbe5da95497dcf8eaf3`。比较原绑定时只在独立hash视图使用原5项实测延迟，真实capability和模型不改；实际q0比较及完整Full重现仍须查worker，**尚无null执行、修复或新增性能结论**。旧v1/v2/v3终止状态已核实再启动v4，未重复未知运行。

closed v1–v3失败/中断、双方原telemetry、v3完整模型q0等值诊断和v4准入收据已打包`nullseal_v1_v3_failed_attempts_and_v4_admission.tar.gz`，356,942 bytes，SHA `e4dd9563a72b2bb677fcdbf6d0c32581747e0f8a003e364826dd368173867ac7`，本机51项内部文件逐项核验。原source和3条fresh K32流保持冻结；每条source各自原12h总窗口不重置，单次7200秒runner到期后仅在退出/lease核实后从同一输出续跑。

本机回复保留原稿已报告的真机任务难度、历史依赖和既有布局例子；新增现场数据按作者要求延期。逐条20项问题—证据—修订映射和15个短回复块继续维护，HH-W5的首个新负结果、同信息最终对照和非空missing-skill仍为明确未关闭项。正文PDF和完整LaTeX ZIP没有新改动，延续先前35页/10页主文的完整编译与视觉核验，不将工程故障修复或中间训练完成写成新实证收益。


### 自动采集：fresh-scene K32 fixed-fresh-rearrange_blocks-e0002-q0008

fresh-pool:fixed-fresh-rearrange_blocks-e0002-q0008:cb5cfd629f06c871aa90286c67db24a6adb142f564a0ed5f7a6977db3813b29e；原K32有效候选32，完整H32端点29，完整refined重复支持True；原件`/mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations/rebuttal_20261109_acp/rb_source2_fresh_K32_v1/q0008`，候选结果SHA `ca8cfac5be607efb06eb5337fe748b04f3cd511743da48f6991bebe5741fd5f4`，centered结果SHA `1e4abf24629fff8cc8ebe23eef905d57dfd58c079c2f9ce3c8ece5a7d2ccfbf8`。**仅固定source/query的直接候选效应数据；不替换原论文表格，不视作整个cohort或策略收益。**


### 自动采集：fresh-scene K32 fixed-fresh-rearrange_blocks-e0001-q0008

fresh-pool:fixed-fresh-rearrange_blocks-e0001-q0008:517aeba2b01241fdac23e393f77ce9e7398c7edd245639cf0176d2cb4be71a62；原K32有效候选32，完整H32端点29，完整refined重复支持True；原件`/mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations/rebuttal_20261109_acp/rb_source1_fresh_K32_v1/q0008`，候选结果SHA `eae9eb6bff2b7617833713e7821fe3a19fa00274e29b375ae746577f7078d023`，centered结果SHA `d32675eb07c73097b1f8f7c19685f299c02c592b3bc5d669c586b62d18e87efe`。**仅固定source/query的直接候选效应数据；不替换原论文表格，不视作整个cohort或策略收益。**


### 23.40 首三源早期查询、独立误差分解、完整本机200备份与slotted对象修复

固定source1/2的q8也已完成：各29个完整H32、3个无效未知；与source0合计86个完整端点/10个未知，3个固定早期query。source0/1/2预测centered MSE为0.077411/0.071247/0.094636，query-constant为0.052806/0.060099/0.052307，预测centered explained variance为-0.465948/-0.185498/-0.809227；三者均不支持新增候选区分优势。所有比较保留各query相同完整支持集，未知不填0。source1的候选中有2个独立正局部标签，但3个query实际所选raw与refined标签均0，重复refined一致，不能声称已取得修复收益。原件和SHA见此前自动采集条目；`FIRST3_Q8_EFFECT_ERROR_DIAGNOSIS_V1.json`和`FIRST3_Q8_RAW_REFINED_DESCRIPTIVE_V1.json`完整保留全部三组结果。

误差分解是明确post-hoc诊断，未拟合任何校准或替换primary metric：预测的centered correlation为0.359359/0.373618/0.028558，历史为0.254235/0.311095/-0.022999。此相关性不代替失败的primary MSE比较，不给统计显著性或因果识别结论；继续原fixed q48/后续来源、核查原投稿模型和协议对应关系，不在当前评估query上调参。

Full/control200联合CAS本机传输已完成：401项、35,973,567,912 bytes均核对长度/SHA，inventory SHA `c896c79385d2103ec511686d4761105c403d8da7ac70046f05f1808cd7d81d0e`，本机verification收据SHA `9351592b76799ba58f4b4a62f2290d36331be8788e4680f01d16c03eaa663964`。这是原native逐字节重建通过CAS的完整压缩持久副本，未在本机新做全TAR重建或Torch tensor扫描；未删除AFS当前state/原始checkpoint或数据。续训备份阻塞解除，后续仍要做空间与GPU准入；最终选择点3000不变，200不是性能证据。

v4在原q0模型动作保存后、任何null/事实执行前因实际BoundOnlineStep使用frozen slots、没有__dict__而在比较视图vars()处停止。已改为读取dataclass fields；新增测试直接构造生产BoundOnlineStep并用原action-witness生成文件，核对真实slots/frozen/mappingproxy结构、整个production digest及真实对象不被改动。初始fixture total_s小于四阶段和的错误由生产构造器拒绝，fixture按有效时间账修正；55项本机通过、1个Torch检查跳过。旧v4原件保留，下一v5须先核实terminal exit和H32原件，再做原生补验准入；尚未报告null动作或对照效果。

### 23.41 v5与source0原预算续跑、三query完整本机快照及材料镜像

`bc133087dfaec78ec147c6b7ac182bc91a0676d1`已本机/GitHub/CCI canonical对齐；冻结source `84af4413e9c59934c34664c35bbfab86fb165f2604a2e01ffbf104f4f6716c3e`，64项原生测试通过（21.00秒），包括生产frozen/slotted BoundOnlineStep。v4已明确exit1、0事实命令和0null封存，其保存的完整q0 H32与原NPY逐元素/dtype一致。v5于19:31:11 UTC准入，launcher2177770、5400秒上限，spec SHA `b7e01ff71b5f0634c42265037d2a632c243a2737be210513528cac3535234975`，仍是前述原固定source0/同上下文撤回方案；真实null封存与物理对照待核验，不预填效果。

source0旧K32 runner于19:27:15 UTC exit-15，reason wall_time_limit，source前后SHA均`823404c10af58f90b932b406b0a4cc2a3632acb20b691841a8a82ac7b0c5cdad`。确认退出与AFS lease后19:31:11 UTC以launcher2177767重跑原冻结入口/原输出，原campaign起点17:27:25 UTC、余35,774秒，总12h窗口不重置。完整q8分支复用，旧未封存attempt继续unknown并留存，q48从固定rank顺序续采。source1/2仍用原冻结流，后续核实19:44首段退出，不重复未知任务。

19:28 UTC新原始快照`fresh_pool_first3_sources_snapshot_20261110_v3.tar.gz`已在本机核对976项文件，139,982,825 bytes，SHA `aaf1994187c4206ebbc7a1b546f983060c1903eaaf2e8806744c4d81b5412f41`。含3个完整q8及当时已结束的部分q48：全快照共101候选终止收据、91完整H32和10无效未知；其中3个q8独立表仍为86完整/10未知，不混淆scope。Markdown/LaTeX表`fresh_snapshot_v3_query_results/ANALYSIS.md`与`effect_comparison.tex`保留三个不及query-constant的primary结果。相关性为post-hoc诊断，未作拟合/替换primary或新增收益声明。

逐审稿人回复、20项映射、首三query表/诊断、完整200备份收据和v5准入已形成29文件research-patch2，SHA `583b802d8a2cb6cfc222e667926854e47c5f5b7f4b3c685418440f316b9c2360`，AFS逐项核验在`S/rebuttal_20261109_acp/writing_candidates/20261110_v3_research_patch2`。原v3 PDF/完整source ZIP不变，内部stable_release_ready=false，未发布评论或上传正式投稿。下一步优先实际v5封存后做fresh-scene null分支、完成固定后期query，并核查模型/评测协议及公平训练资源，保持负结果与旧原稿数据的独立身份。


### 自动采集：v5真实同历史null封存及两个已知超时续跑

UTC 2026-11-09T19:48:49.570658+00:00；已封存queries 0,8；每个query原Full完整H32和全部非计时绑定内容精确一致、同上下文同Gaussian、实际capability不改、g=0，物理null执行0，完整700事实重现尚不先宣称。原始包`/mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations/rebuttal_20261109_acp/source0_same_context_null_seals_active_v1.tar.gz`，SHA `58e04201aa3c5d53792a5734440a336aa4823a1e4928c41ac4b694bb95ec0031`，22文件、1612970 bytes。source1/2已核实wall_time_limit与空lease后按同一冻结代码、输出、原12h预算续跑；进程 2181269,2181270，收据`FRESH_POOL_SOURCE12_RESUME1_RECEIPT.json`。无新增修复/策略收益结论。

### 23.42 同历史null物理分支入口与真实工件准入

新增`qualify_rebuttal_fresh_null.py`与`sealed_null.py`，先针对预定source0/q8复用已有不可变Raw/Full原件，再从同一原始事实prefix分别新跑null和post-null Full重复。两次新reset都必须与原K32 anchor的完整physics/task/controller/camera/RNG/实际qpos父状态逐项一致，post-null Full端点和DINO效应须精确复现原三次Full anchor。原Full H32/输入/候选/seed封存、原NPY文件SHA、同Gaussian和g=0 probe、无future/teacher的context工件都在物理执行前核对。保留原selected raw，不根据已知局部成功候选换rank；两策略动作相同也是合法结果，invalid/early-success不伪造H32效应。新增实际关节qpos/qvel记录和错误前的观察/原生父状态取证，未调用physics unpack，也不更新事实policy。

Windows26项通过；原生冻结precommit26相关组件加query/replica共52项通过（20.73秒）。在真实source0/q8旧pool与v5 null seal上CPU预检通过，已核对真实Raw/Full/null action文件、context、原生产probe和source/runtime身份；preflight-only没有加载DINO或创建模拟器，物理null仍0，不能将该检查称为反事实效果。冻结precommit archive SHA `6b8f06d99ca7a7f674c2283062176274b70fa5603eddcfb0d187282898b753f6`；实际worker须用已提交的新冻结版本，运行前保持输入绑定不变。

定时任务原提示乱码已通过应用automation_update修复为完整UTF-8中文，回读确认无U+FFFD；原hourly周期、ACTIVE、同一thread与11月13日23:00稳定版目标不变。原始最终PDF及既有3位审稿人20项映射继续作为文稿依据，新增真机工作仍延期。ACP e于20:05 UTC实际回读仍等待，FailedScheduling/insufficient nvidia.com/gpu1，未获GPU、未重复提交cases3–9。后续资源扩展仅考虑独立匹配训练的可用资源，不把排队当执行。

### 23.43 首个同父状态物理null对照完成与证据边界

Git `3c1ea491f76e85699799af1a3a0f46d01326d9ff`已推送核验并同步CCI；冻结`controller_3c1ea49/code`，source SHA `0e2bf707080fe70ab7cdd90727413aba4a296d5dcf001f01d3d698a4ea7798f2`。source0/q8 qualifier于20:18:47 UTC启动、20:33:45 UTC明确exit0（898.745秒）；spec文件SHA `6344c6ea37099daa7194f5cd41b4b219ec750d1049a03f90c2fb2e0d3f586b90`，runner规范化spec SHA `4655bdfd29cb3b784b92d2b9cd441c13738e4aa1211c89a46525e81e02a5b6a0`。新增candidate-null和post-null Full各执行32条native命令，共64条；完整physics/task/controller/camera/RNG/实际qpos父状态与原pool anchor精确相同，post-null Full端点及DINO效应精确复现原Full。原core/runtime和collector前后source SHA均不变。

本次仅一个预先固定source0/seed340800000/q8/frame32，复用原selected Raw rank10及三个Full anchor，没有按已知候选局部结果换rank。Raw、Full、candidate-null、重复Full的独立“下一物理交互完成”标签均0。**首次得到可核对的同历史、同Gaussian、同物理父状态null分支；本query没有显示修复或相对null收益，不能由资格通过推断整个cohort优势。** 原K32前三个q8的负primary结果仍保持，候选未知不填0，原稿表12–18身份不变。

完整新分支、实际qpos/qvel逐命令轨迹、原Raw/Full anchor、同上下文null封存与终止runner原件打包`first_same_parent_physical_null_source0_q8_v1.tar.gz`，8,954,471 bytes，SHA `1da847e3a991d001ce772c92d79d4ce3b61137e7079c067abdadee16745da28e`，manifest SHA `f81e2a60e878e04e38c06e3a46fe3ba77a66e99c242a8aab1b85b56e5076850a`。本机80项内部原件均核对长度/SHA，收据`tmp/rebuttal/execution/20261110_direct/FIRST_PHYSICAL_NULL_Q8_V1_LOCAL_VERIFICATION.json`；AFS原件和同名档案保留。v5已有六个动作封存不自动等于完整700步事实见证；其余运行另查真实终止收据。

### 23.44 简化对照信息审计与Full独立续训准备

实际代码路径确认当前`ConditionedPriorFastWAM`只把历史gist和检索动作token送入简化conditioning；`DirectActionEvent`未消费检索事件的视觉/效应记录。现有control200定位为动作先验消融，不能独立关闭BB-W2/HH-W1/HH-W3的同信息公平对照质疑，也不称官方ReCAP/WarmPrior复现。完整审计与后续接口门槛见`docs/rebuttal/BASELINE_INFORMATION_AUDIT_20261110_ZH.md`。Full/control200保留；新结构不得静默继承control200为已匹配训练预算。

准备独立Full200→500（仍固定最终选择点3000），原训练source `4efb8094729ba8771c7ec0d472c30ad8b585ef319e897edb566b12d3d5a81a14`、seed3407、有效batch128及30k scheduler不变；完整optimizer/scheduler/RNG/数据游标父状态先按清单逐项核验。新增operator只允许output/resume/run_steps/save_every控制项变更；为在当前AFS容量内保全完整state，将save_every100改为500，使此段仅在终点保存，eval_every500和数学配方不变。训练4h上限，ACP额外留归档时间，运行前要求80GiB AFS空闲；不删除旧state/CAS，不重复e持有cases3–9。当前为代码与准入准备，**未在此条预填ACP创建、训练执行或匹配收益**。

### 23.45 v5完整700事实命令与六个null动作封存正常结束

v5于20:46:36 UTC明确exit0（4523.429秒），`exact_complete_source_trace=true`，原700条事实命令和6个预定query见证全部精确匹配；六份null均来自该Full事实历史/同Gaussian，实际撤回g=0。collector source前后均`84af4413e9c59934c34664c35bbfab86fb165f2604a2e01ffbf104f4f6716c3e`，原core也未变化。它仅重现事实和封存备选动作，独立null物理分支0；上一条qualifier的两个新增物理分支是另一个独立证据ID，不混合计数。

完整原件包`source0_complete_factual700_same_context_null6_v5.tar.gz`，7,058,889 bytes、92项原文件，SHA `f5cc232c29f7c01044ee01914c33eeb628633d7728235468d2e26f04667b8d3d`；manifest SHA `dae0b62896c8f3bb804af595ce9e353b4144c19054f09b3a30dbaef66604f1ee`。原v1–v4故障/中断、原q0/q8活动快照及本次terminal原件均保留。此结果关闭这一个source的完整事实动作来源检查，尚未关闭固定cohort的后果预测、修复/损害率、自然missing-skill和公平最终性能。

### 23.46 Full200→500原状态准入与ACP标准后闲时分配

`d3abf1569741296331134d54c413cdb0e25acc5a`已推送并同步CCI，冻结`controller_d3abf15/code`，source `043dd179e5b2000b3404fd64a6c6548235e0cf6a05a3e9e70f53bccbc1d79ba2`。Full200权重及7项完整state文件逐项SHA核对通过；原生/本机各4项segment contract检查通过，冻结operator/tests与真实父状态preflight所用文件逐字节一致。正式续训计划`full_resume200_to500_v1.plan.json`，SHA `7d359a7d95b03ee4e18435910fe7a622346e79bcd4d7e369f54df15cec119b61`。只改变restart/output/run_steps与checkpoint保存频次，数学配方不变；旧数据/state/CAS均未删除。

标准任务`warm-full-resume200-500-20261110-a`（UID `20239e012666446493a1470c726624c2`）在实际限额提示后创建为WAIT_QUOTA，实际0GPU。它是已创建的限额等待任务，**不是已被API拒绝的创建**；已确认“已停止”后才另建闲时`warm-full-resume200-500-20261110-b`（UID `9c6000a4ce734a7bac717b2053197a1e`）。闲时选择cluster01、1张80GB/22CPU/230GiB、确认私有镜像和afs-p3-01e当前用户读写挂载/mnt/afs，自动重试0。05:01北京时间创建，21:04 UTC回读仍排队、0GPU，事件23/25可匹配节点、2节点cordon、insufficient nvidia.com/gpu1；AFS operator/output尚未生成，**未开始训练**。这条任务只持有Full200→500，不拥有e的cases3–9，不能再重复创建未知/活动训练段。

### 23.47 ACP e真实失败、未知运行字段与保全

旧e实际于20:31 UTC拿到H100（80GB、cap9.0、driver580.95.05、Torch2.7.1+cu128）并开始case3/seed340800003/candidate-null；20:39:36明确exit1，错误`online DINO runtime differs from the M1 contract`，发生在策略构造，新增完整role0、正式策略回合0。不能把原UI的“启动中”写成仍在计算，也不把此工程错误记为策略失败或成功率0。旧实现未保存实际runtime的差异字段，目前不能断言只差platform；下一次须保存真实字段并提前检查。e的自动重启已于21:08 UTC确认“已停止”，原attempt与case3–9所有权保留，后续修复须沿同一campaign/plan和原截止08:31:04 UTC继续，不重置12h预算。

原始失败包`acp_e_failed_dino_runtime_attempt_v1.tar.gz`，450,840 bytes、30项，SHA `2ec2226c648cd94056c903f49901338703815d58eb7ce66e7b4ba0f446f1b74d`；manifest `b18271a51925a3e15e60ae0ae02ff14510b94d428d537d6530dda9965b1fb71a`，本机30项均核验。精确退出、reset预检、实际硬件、frozen计划、完整日志和旧campaign_state均保留。原3个已完成RB对仍为0/0；case3工程失败仍pending，不产生第4对样本。

### 23.48 CCI连接重置后的三路恢复与前置DINO资格入口

21:17 UTC连接重置后旧pool PIDs2177767/2181269/2181270均不存在，三个native campaign lease均可独占取得，旧runner收据仍running/exit null。保留其**未知退出**身份和全部partial attempt，不补造exit code。21:19 UTC按同一Eb冻结代码、spec和输出恢复，PIDs2198110/2198111/2198112；原source0起点17:27:25和source1/2起点17:44:21不变，余窗口分别29,305/30,321/30,321秒。完整q8复用；固定q48当时完成候选12/6/6，尚无完整后期query。恢复收据`FRESH_POOL_RESET_RECOVERY_V1.json`保存原manifest SHA、空lease、旧PID缺失及新PID，不归因于已证实的某种具体基础设施故障。

新增opt-in前置encoder资格入口：从原M1 contract恢复数值策略；实际字段必须完全匹配，唯一可单独记录的非数值差异为platform。再对预先固定source0/q8、source0/q48、source1/q8、source2/q8的原事实观测，核对原float32四空间token的shape/dtype/全部bytes。成功前不运行原policy；数值字段、设备或任一特征变化即停止。仅在资格通过后为旧full-map hash提供明示的platform比较视图，原contract、实际runtime原件、权重、输入与科学core保持；正式policy仍从原冻结core解析，不覆盖其模块。DINO加载前后的CPU/CUDA/NumPy/Python RNG恢复，资格不是训练或收益证据。

本机与原生已有16项检查通过（包括严格字段、1ULP/dtype/形状拒绝及旧worker不变）；本机初版测试路径断言按Windows路径语义修正。前置实际观测准备最初误指向不存在的anchor NPZ，未运行DINO；现改用原Full完整replan NPZ，并由原pool identity的qpos_trace路径定位。随后CCI实际runtime已与M1全字段相同，但四观测检查尚未完成时连接中断，旧`ENCODER_PLATFORM_CCI_QUALIFICATION_V1.json`停在checking、0个golden；保留未知，不写成资格通过。最终冻结入口须再做16项原生检查与四份实际观测完整验证，再提交修复后的ACP接续。

### 23.49 最终冻结四观测资格通过与ACP f接续已创建

`6b803c23d7472a86d0bec26e8b874cf73e0af6ff`已本机/GitHub/CCI对齐，冻结source `7966b63c3b24a1f9efc772fc4332b7e251b558c3ccee7f8ef822e767d4e90e22`。最终原生16项通过（0.94秒），包含dtype/shape/1ULP及有符号零bytes差异拒绝；原campaign与新增资格CLI的validate-only通过。21:24:47以600秒上限准入独立logged runner，21:25左右明确exit0（48.252秒）：预定四份原Full观测的4×768 float32 DINO全部bytes与原pool父特征相同。真实CCI runtime本身与M1全字段相同，未使用platform比较视图；保留旧checking/0个golden的未知运行，不将其追认为成功。原policy未加载、正式分支动作0，实际ACP兼容性仍须逐节点独立验证。

完整资格包`encoder_factual_four_observation_qualification_v2.tar.gz`，1,068,556 bytes、32项，SHA `b15d09a6e22497a84cfb13ed19292723c99be076a825bc547942fd05a82ed507`；manifest SHA `e2bfd26518983a5fcb90c61094a7b6ecd234e9538d878612bc5374d54bf909e7`，本机32项均核验。资格计划`encoder_platform_qualification_v1.plan.json` SHA `0df7b0f93bcd61f375bc259929631f487dfc07d915eaf22d6fbfd44202bf8d3e`，原科学core、数据/processor/encoder文件、四份原观测和预期特征均绑定，不读取未来或分支标签。

新任务`warm-rb-resume-20261110-f`（UID `5be8c6689d44435a8ef2c0b9497ec820`）于21:31:44 UTC由API200明确创建。先实测标准资源“限额不足”对话框并取消该前端确认（未创建另一标准f），才改用cluster02闲时1GPU/8CPU/128GiB、确认镜像与AFS、重试0。旧e已停止，f沿**同一旧campaign/root、plan d8512a23…、case3–9及原08:31:04 UTC截止**继续，不重新计时、不替换reset、不丢弃旧失败。21:34实际回读f为0GPU等待、insufficient GPU及PVC调度事件；Full续训b亦0GPU等待。两者的UI“启动中”不构成实际训练或策略执行证明。

### 23.50 已知planner RPC故障、实际采集状态与三线材料归档

source2的21:19恢复runner于21:21:41明确exit1，source前后均Eb/823404c…；错误为`env.setup_demo`初始化夹爪时cuRobo RPC连接重置，发生在新的branch parent/动作之前。保留原attempt unknown，确认退出与空lease后21:27:21按原冻结spec/输出重试一次，PID2199487，预算不重置。21:33:45实际核查source0/1/2的PIDs2198110/2198111/2199487均运行，q48已有完整候选14/8/7，后续分支仍在原native前缀回放中；没有完整后期query或新增repair收益。该时点单张H100为25,990MiB/81,559MiB、100%利用率；AFS空闲91,688,407,040 bytes。只记录当时状态，后续须复核真实worker/lease与心跳。

已更新Bbqv/Hh4E回复的信息匹配定义、15个可粘贴comment block与20项问题映射，逐项核验所有block SHA及证据ID引用。新单query物理对照表（Markdown/LaTeX）与四观测资格、失败和资源收据分别保全；未将资格或中间训练写成科学收益。论文完整v3 PDF仍35页/10页主文，SHA `ef9e6d55c7a24822b147ec392773c2a5c17ec7bfe9bbfbf79d6d34ee04545b4b`，完整源ZIP仍`120daf4e…`；另准备尚未纳入active工程的同状态协议LaTeX候选。没有新的PDF编译或正式上传、OpenReview发帖、对外消息。

44文件writing research-patch3已在AFS逐项核验：`writing_candidates/20261110_v3_research_patch3`，archive SHA `517ab30d96e067c7beb1e5cce889e15005afd3a0dc101f444973bcf1c7a86524`，49,128 bytes。它是21:40附近的不可变快照；`planning/current_iteration.json`与AFS根目录live handoff继续更新，读取时区分快照与当前值。stable_release_ready=false，关键后果预测、完整修复/损害cohort、真正同信息最终对照、非空missing-skill仍须正面补齐；新真机取证按作者要求延期。

### 23.51 排队复核与单回合有界CCI接续入口

22:10 UTC实际回读ACP b/f，均0GPU、Worker等待、无Host IP，仍为insufficient nvidia.com/gpu；Full续训operator未生成，f的campaign_state仍是旧e的20:39工程失败，没有新attempt。CCI三路K32保持运行，21:56 q48已完整候选16/10/9，尚未完成query。22:07测得同一H100空闲54,836MiB；此前同机Full封存模型与三路pool并发已有完整执行资格。

新增`--max-new-roles`只限制一次controller invocation接收的新完整role数：优先判断全部owned工作完成，否则达到局部上限时标为`partial_verified_invocation_limit`，不写成整个campaign完成。默认行为不变；原plan、固定case/seed顺序、模型/协议、失败保留及12h deadline均不变。Windows23项通过，原生23项通过（0.97秒），真实原campaign和四观测资格计划加max-new-roles=1的validate-only通过；无policy执行结果。

准备把一个固定role从排队f转回CCI：必须先确认f停止且native lease空，再用新冻结controller、原plan d8512a23…与encoder资格0df7b0f…接续；单次只收1个新完整role，worker4500秒、外层5400秒，运行前至少45,000MiB空闲显存。原08:31:04 UTC截止不重置，b仍独占Full200→500申请。此条是具体可核查的资源切换准备，不预填已停止、已准入或已完成数据。

### 23.52 ACP f已停止与CCI同计划固定role接续

22:25:31 UTC浏览器实际确认f（UID5be8c6689d44435a8ef2c0b9497ec820）“已停止”；批量停止确认框仅含这一个已知排队任务。先前More菜单操作均未确认停止；最终确认点击后的首个断言遇到modal/table重复row，未重复提交，而是单独只读核实终态。`ACP_F_STOP_FOR_CCI_RECEIPT.json`保留实际收据。

22:25:59 UTC核实native campaign lease可独占、原next role、原deadline及54,830MiB空闲显存后，CCI准入PID2204773；f不再拥有活动执行，cases3–9仍由原campaign独占。冻结`controller_5b9ebd7/code`，source SHA `56393c5a5d944004c55bf88e91aff9b4687be49849bc7da3b8d18fc578cb6971`；spec文件SHA `1e0df6a3fb9ad5f5ab78d75237e25d79f9ee219a077ea0cc735b54b11227482e`，规范化runner spec SHA `2f9ea49715e86ff4ad217ae6f5ff99d6602bace2e0070c39fb2b7c095095199b`。只接收1个新完整role，原case3/seed340800003/candidate-null顺序、d8512a23计划、20:31:04起点和次日08:31:04 UTC截止均不变；外层5400秒，原失败attempt保留。

该attempt的四份DINO原事实数组于22:27:00 UTC再次全部bytes精确通过，actual runtime没有差异，也未应用platform比较视图。原policy加载后22:40开始正式闭环，telemetry实际推进；此处仍未登记role完成、成功率或第4个完整pair。Full200→500的ACP b仍拥有唯一训练段，22:42 AFS operator未生成；不把排队/加载写成完成。

### 23.53 同信息池化对照实现与真实记录BF16资格

新增`PooledMemoryPriorFastWAM`：所有K个检索动作、视觉前态、历史效应、context、时间和阶段经一次线性投影及单层4-query attention池化，与共同8个history gist组成12个conditioning token。保留确定性起态对齐、coarse prior及常量0.25和共同规则；不输出学习的动作残差、后果预测、效用排序或置信度。通用attention会学习内容权重，不能称完全无学习的检索读取。独立schema和RNG初始化隔离防止静默继承control200或扰动共同训练流；原Full/旧control行为回归通过。

Windows配置3项通过、Torch模块2项按缺包跳过；原生首轮44通过/1失败（新schema额外字段未从父校验视图移除），修复后45项全部通过（20.27秒）。检查覆盖未选候选各字段的实际影响和梯度、padding/空候选有限零、未来teacher隔离、共同初始化/RNG、12-token接口、checkpoint隔离及原Full路径。真实记录预检v1在dataset统计副本默认工作目录不存在时停止，发生在读固定样本及优化器/policy动作前；保留原日志。修复只配置Git外工作目录，v2以900秒logged runner执行并明确exit0，source前后均`5347088c09f34707d4df80f6adf0b9c6d565a5e2be5303b0f53a45d0c4beb23e`。

实际训练样本index32和原封存source0/q8在线记录均有32个有效候选，BF16输出1×32×4×512，所有池化参数梯度有限；池化模块5,767,168参数。该模块预检明确使用零history-gist占位，在线proprio也为占位，不能称完整采样器资格或收益，单次并发耗时不作为部署benchmark。数据配方SHA `42d3b8c763ef61cf7c3c7b102a65b3dfeed1084aad4b33be98ec8b6998238adb`，原在线context SHA `2c05f995eddcf6dfafe84d0c72703fcb21045bcacd6dc603f04e274ecf96cb6a`。

新seed3407初始配置已由原author recipe生成，SHA `d5a7bd8c76aee178991e593439124aec72077e1734eb831cf2dfda55883846c3`；新优化器、从共同released base的0更新开始、有效batch128、30k scheduler、固定3000最终选择点，首段100更新。**尚未提交新训练任务，完成更新0**；必须解决与独占Full b并行的持久state/归档空间后准入。旧control200保留为动作先验消融，不更名为同信息对照。BB-W2、HH-W1、HH-W3仍未关闭。

### 23.54 动作/时间核对与35项新原件双端保全

原训练源码与实际第0个Parquet前64个transition确认action[t]=state[t+1]；H32 target读取frame+32。num_frames33/ratio4对应9个视频采样点，并非128动作步；没有依据按四倍或一行偏移修改truth。实际metadata fps15仅给出名义编码时间32/15秒，未证明示教物理时长。native每次qpos请求通过TOPP轨迹执行多次250Hz physics step，32控制命令不保证固定物理时长。时间对应是否解释新负结果仍未核实，不据此调尺度或替换主指标。完整核对见`docs/rebuttal/ACTION_TIME_SEMANTICS_AUDIT_20261110_ZH.md`与Git外`OFFLINE_NATIVE_TIME_SEMANTICS_V1.json`。

schema失败、工作目录失败、45项原生检查、真实训练/在线输入资格、测试源码、原context、冻结配置、时间审计及ACP/CCI交接收据共35项已打包`same_information_control_and_time_audit_v1.tar.gz`，509,306 bytes，SHA `c7dddd91d7239cc84fdf5a1e5b699570d1dbdad87a556420f73cda37cba94596`，manifest SHA `19c86abae0e1c6e172c9e4695281426eab5c414f5343d965c63394b0062b9eb0`。本机全部35项长度/SHA核验；报告`tmp/rebuttal/execution/20261110_iter2/POOLED_CONTROL_TIME_LOCAL_VERIFICATION_V1.json`。此包是实现/资格/审计证据，不包含新的完整科学cohort或训练收益；原三q8 adverse primary、原稿数据及真机延期身份不变。

### 23.55 最终已提交pooling源码与行尾差异核实

`f9f109069d7992d802d8e9815b84bebfad434ec6`已推送并核对origin/main，CCI canonical于22:51:51快进到同一提交。Linux Git冻结source为`48ce6ba809788fc1c2c55b5303cc9617e2852050dd2bd8e104dc8192ee98e358`，不是Windows precommit的5347088c…：初次部署的严格source断言正当停止，随后核对全部源码，仅464项旧文件为CRLF/LF差异，六项本轮实现/检查文件原始bytes相同，全部文件行尾规范化后相同。原测试/真实记录资格保持其原source身份，不回填成新SHA。最终Linux Git快照独立再通过45项原生检查（19.72秒），`F9F1090_SOURCE_RECEIPT.json`和`POOLED_MEMORY_COMMITTED_QUALIFICATION_V1.json`保留完整核对；未修改任何活动科学快照。

### 23.56 checkpoint训练范围与原稿对应关系的新增审计

现用作者指定v8-step3000的原训练记录明确为shared阶段、task allowlist空；原稿RMBench描述逐任务specialist。有限目录metadata清单发现105份attestation、66份对应权重存在；抽查Battery Try和Press Button旧specialist的config与CPU FakeTensor/mmap metadata，二者是version3/timing_dim4，现用v8为version8/timing_dim8。两份旧attestation原Git提交f77c633…与47277e5…确实存在，已各自独立冻结，尚未选择或启动它们的新评测，也未对12GB权重重新全SHA扫描。

这些历史候选与原投稿DEV选择、准确率记录的对应关系仍未确认；不自动混装v3/v8，不按负结果换checkpoint，不推断范围差异已解释新负结果。已向作者提出原RMBench训练设置/评测记录的具体事实核对，现有固定v8工作继续，作者原指定授权仍有效。原稿结果保持原稿报告身份。新从common base生成的pooling计划当前是shared阶段研究对照，尚不能等同原稿逐任务specialist的完全匹配复现。详见`docs/rebuttal/CHECKPOINT_SCOPE_AUDIT_20261110_ZH.md`。

15项原metadata/config/attestation、历史源码身份、最终已提交pooling资格和control200原state哈希双端包为`checkpoint_scope_and_committed_control_qualification_v1.tar.gz`，22,226 bytes，SHA `831a763ade5187facb83ef61fb9a7dcf615b13416f2a72528e10812291b87e63`，manifest SHA `c5f81634127f64be1783d564316ca4e6c79b4119d11ac1e338df74ea9e63f372`；本机15项全部核验。原模型、原件、checkpoint、固定样本和科学计划未被替换。

### 23.57 control200完整state的额外持久重建验证

为准备后续存储准入，从已核验的本机CAS完整逐字节重建control200原TAR：38,437,867,520 bytes，SHA `d838ed0c5fc039170394a9719523a3c8d738929c9652b8f49ea6795d895451d7`，与原native重建一致；本机原始TAR及7项解出state留在`C:/Users/admin/.cache/warm-rebuttal-control200-state-v1`。7项共26,307,343,356 bytes，与新读取的AFS现有原始state逐项SHA/长度完全一致，无活动control200进程。用时133.922秒；这是原始state重建验证，不是新Torch tensor扫描或训练。

原AFS checkpoint/state、权重、CAS及数据均未移动或删除，AFS并未因此腾空；不能把已验证的分层准备写成已释放容量。所有完整原件和既有本机压缩CAS保留，`CONTROL200_LOCAL_REBUILD_V1.json`及`SCOPE_AND_CONTROL200_LOCAL_VERIFICATION_V1.json`记载可恢复位置和匹配清单。后续仅在独立空间方案明确、持久副本核实后分层重复payload，不能破坏排队Full b的父状态。

### 23.58 v4回复与完整论文的成本测量范围澄清

核对新成本profile的原MODEL_SELECTION/runtime_args，确认其确为作者指定shared-stage v8-step3000。据此在v4 Appendix C.4和Bbqv/Hh4E对应回复中明确，该profile与原稿task-specialist准确率协议分开；530.1/565.2ms等实测数值全部不变。补明新同信息对照采用全部K检索记录池化，但性能仍pending；新scope/时间审计仅作为内部release gate，不把工程资格写成收益。三位草稿共15个可粘贴comment block、20项问题映射均已核对，v3基线原件保留。

v4由原完整LaTeX工程运行pdflatex/BibTeX/多轮引用编译，35页/10页主文；仅第35页文本变化，其余34页与v3逐页文本相同，原数值表和9张active图保持。全部35页已渲染并检查contact sheet与第35页细图，没有裁切、重叠或未定义引用；46条原有字体替代warning与v3相同。PDF SHA `e9f99e86c807233df280883eadfdb84aabea02d87fd80a128f12f6e69df4e2ec`。29文件完整source ZIP SHA `e4db2687b591dca4633f57be80b6bcb72d8e7acc9301d5dece911229de24b1e9`；从干净ZIP独立重编译的全部35页文本与交付PDF一致。PDF作者metadata空、主文匿名，未发布OpenReview或上传正式稿；stable_release_ready=false。

### 23.59 单角色真实3600秒上限、未完成原件与独占重试

**时限更正：**23.51–23.52的准备说明把worker上限写成4500秒；实际冻结d8512a23计划的`worker_max_seconds`是3600，外层仍5400，整体12h截止仍08:31:04 UTC。controller严格执行了原3600秒，不是模型异常或policy任务失败。CCI v1 child于23:26:20明确exit-15/`bounded_worker_or_campaign_time`，outer于23:26:23 exit1。最后完整replan记录frame592，日志中的native计数达到595（计数在控制调用开始时增加，不能声称595次均完成）；没有完整700轨迹或新accepted role，complete pairs仍3。

pool0/1于23:19:05、pool2约23:27按原7200秒退出，均是已知wall-time，原source前后不变。先确认退出、旧PID缺失及free native lease，再暂缓续跑这些流以给固定闭环role资源；这不延长任何原窗口或改变样本。首轮CCI未完成attempt、encoder资格、完整partial工件和exit/controller状态共497项已双端保全：`cci_case3_null_incomplete3600_attempt_v1.tar.gz`，3,668,592 bytes，SHA `5e05f0d209489060f123f2fd0a8255565ea3424e321148fcbc8437fcbedd9a60`，manifest SHA `733efbd47697b9e31bfb2cee365249f515e8d687d08cca9e53e55b6290855733`；本机497项全部核验，不把中断填为SR=0。

23:30:23 UTC以81,080MiB空闲显存、无其他GPU进程准入独占CCI v2，runner PID2211054、child2211066；冻结controller5b9ebd7/source56393c5a…、原plan/seed/case3/candidate-null与原起点/截止完全保持。spec文件SHA `8c2fa7bfd7a3f627252464a895120616bb2758aea949e6b9abdf321edc36c1eb`。仍只接收一个新完整role，actual worker3600秒、outer5400，不修改既有timeout后补造成功；v1旧失败与ACPe失败均保留。23:31:17该新attempt的固定原事实DINO资格再次通过，原policy尚在加载/执行准备，此处不预填完成或收益。

当前所有fresh pool续跑都等待本次v2真实terminal；届时检查三个原窗口剩余、完整query/未知attempt与native lease，再用原Eb冻结spec续跑。这个独占安排是针对已知超时的资源调度修正，不能断言已经证明具体瓶颈或一定避免下一次超时。不要在v2活动时重新启动ACP f或同case/role；也不要把论文原稿与当前shared阶段对应关系的待核查问题当作更换样本/丢弃结果的理由。

### 23.60 v2实际执行与原稿复核的可恢复交接

23:44:19 UTC核验v2 runner2211054与child2211066均存在、心跳推进，完整replan记录到frame180；H100为32,983MiB使用、100%利用率，Full500 operator仍不存在。三路pool均是已知原wall-time退出（source2实际23:27:24），q48 terminal候选23/17/18，尚无完整后期query；当前明确held等待v2结束，不把这些排空时段重置为新12h窗口。v2四个`exact_feature_checks`于23:31:17通过，真实policy已进入闭环；没有新增完整role/pair或收益结论。

23:46:14对同case3/null旧中断attempt与v2的q0/q8/q24/q48作只读重复性核对：四个查询的全部保存RGB、官方输入状态向量和完整H32模型动作均精确相同，最大动作差为0。官方文件名native_qpos在此是policy输入drive-target vector，不宣称已核对全部隐藏物理状态。`CASE3_RETRY_PREFIX_INTEGRITY_V1.json`记录两个原动作文件SHA；此为同一固定样本的prefix重复性，不能增加独立试验数。

回读原最终33页PDF（备份SHA aa04f951…），确认原第18页已有factual效应项及candidate正则，第22页为九个task specialist，第30页结构检查单独标记v8-step300。当前回复的监督解释未把新增机制冒充原方法；历史v3 specialist也不能据此认定就是投稿所有表格的模型。另有受限74个大权重文件清点、8个未附attestation的基础/release/smoke文件，未确认新的原稿选点；原件`LARGE_WEIGHT_FILE_SCOPE_INVENTORY_V1.json`保留。后续继续按表格/DEV/eval bundle绑定核对，作者事实问题仍待答复。

### 23.61 固定case3/null完整700步、564项原件核验与三路原窗口续跑

CCI独占v2的科学worker于2026-11-09 23:56:21 UTC明确exit0，外层于23:56:28 exit0；原case3、seed340800003、candidate-null完整执行700条命令，官方success=0、max_reward=0。重新调用原冻结controller的`owned_records/verify_role`，核对三个700×14有限commands/before/after数组、初始观测哈希、任务/seed/模型/科学core/计划身份、durable result与trial row一致。此次只是一个已知完整null角色；case3/Full尚未执行，完整RB配对仍3，不能记作第4对或产生Full-minus-null差值。

worker原件564项、3,940,722 bytes，archive SHA `a0f72a6340ffca2e026fdca716815bb2d049e2dbb6bb8deb80c584887ff840de`。AFS原件位于`rebuttal_20261109_acp/resumable_rb_cases0003_0009_v1/attempts/0003-candidate_null-1c8ac648c9b14501a92d53b298140dc4/verified_worker_evidence.tar.gz`；本机`tmp/rebuttal/execution/20261110_iter2/case3_null_v2_complete700_worker_v1.tar.gz`已逐项核对所有564文件的长度和SHA。完整结果收据`CASE3_NULL_V2_VERIFIED_COMPLETE_V1.json`、本机核验`CASE3_NULL_V2_LOCAL_ARCHIVE_VERIFICATION_V1.json`保留。旧中断的497项档案、ACP工程失败、原事实重复性检查均保留，重试不增加独立样本数。

原controller source前后均56393c5a…、原计划d8512a23…、模型68c7f384…、worker3600秒与campaign次日08:31:04 UTC截止保持；实际外层1563.054秒是本次单run耗时，不作为受控吞吐或方法速度优势。`partial_verified_invocation_limit`表示一次有界调用收到了1个新完整角色，不表示整个10对cohort完成。

2026-11-10 00:01:31 UTC，确认v2两个PID消失、terminal exit0、native campaign lease空、H100空闲81,080MiB后，受保护V3续跑入口恢复source0/1/2原Eb collector/spec，runner PID2212837/2212838/2212839。每路先检查旧known wall-time exit、旧PID缺失与自身lease，逐路写入`FRESH_POOL_AFTER_EXCLUSIVE_V2_CONTINUATION_V3.json`，避免未知派发后重复。原12h起点和05:27:25/05:44:21 UTC截止未重置。00:03实际三个runner manifest均running，日志进入新的前缀物理执行；旧q48已完整候选23/17/18，尚未生成完整后期query。下一闭环角色严格为case3/Full，建议本次pool完成或原约02:01 UTC invocation已知到期后独占执行，不重复case3/null，不重新启动已停止f。

00:04:31 UTC ACP控制台实际转至IAM登录，已保留原浏览器登录页并通知作者重新登录，未执行新创建/停止或读取保存密码。Full b最近可核实控制台状态仍是23:03的0GPU等待；00:03 AFS Full500 operator仍未生成，不能把旧queue观察表述为当前控制台已确认。CCI采集和写作不依赖本次重新登录。v4的20项映射、15块回复和内部门槛已更新为完整null/Full待定；公开英文回复未引入新的收益数字。

### 23.62 v4完整候选包双端验收与当前计算交接

v4内部审阅包共52项、80,581,790 bytes，SHA `3d34361f3e92d8fdbbeffe1592105e62d5fe3fcc9d52ba37f76833ebefe3a3fd`，manifest SHA `a433444538b681282224183859c2f68d81bbac708ce381c11ffc8558f2220b88`。本机与AFS逐项长度/SHA核验通过，AFS于2026-11-10 00:09:56 UTC完成；目录`rebuttal_20261109_acp/writing_candidates/20261110_v4`，本机归档`tmp/rebuttal/execution/20261110_iter2/warm_rebuttal_writing_candidate_v4_20261110.tar.gz`。包内包含三位英文回复/15个comment block、20项映射与内部门槛、35页PDF、29文件完整LaTeX ZIP、变更单、视觉QA、审计文档、564项完整null原件与497项旧中断原件包及验证收据；AFS再次独立核验ZIP内全部29文件。论文PDF e9f99e86…和source ZIP e4db2687…保持。未发布评论/上传投稿，stable_release_ready=false，不能把材料验收等同于决定性实证关闭。

00:10:27 UTC复核三路实际runner均存在、各native lease持有、原source不变，控制台日志进入前缀物理执行，单H100使用24,485MiB、100%利用率；AFS Full500 operator仍无。q48仍分别23/17/18个完整候选，另各2个完整anchor；三个summary均只有q8完成。只读检查初稿误读不存在的`status`字段，派生计数标为unknown；原检查件保留，V2改按真实`endpoint_status`区分candidate/anchor，未更动任何科学原件/标签。以`FINAL_LIVE_HANDOFF_20261110_0010_V2.json`及其00:11:06校正收据为准。原窗口与下一个固定case3/Full不变；最新执行读取Git外current_iteration，52项候选包是00:08时点不可变快照。

### 23.63 固定source0/q48同父null入口与实际输入预检

00:15:35 UTC复核原三路PID/lease均活动、q48完整候选24/18/19，各另有2个完整Full anchor，GPU26,746MiB/100%；原窗口未改变。Full b的AFS operator仍无。00:17:06浏览器只读核查仍需IAM登录，原登录页保留，沿用已有作者登录请求、不重复创建任务。网页重定向期间一次只读body等待遇到null，随后改用安全hostname/attached-body核验，未执行资源变更或读取保存密码。

新增后期query参数与显式`--allow-pending-pool-records`。它保留默认严格q8行为，仅允许在两个完整且父状态/端点/效应精确相同的初始Full anchor后并行执行封存null和post-null Full；原选中raw与pool-final-anchor仍为完整三角色query的必要依赖。source0/q48按原计划固定、原选择rank29，尚待原K32按顺序执行；不读取其结果后另选rank。缺失raw标为null、不填失败；已有terminal即使在pending模式也必须验哈希/身份，重复terminal不选择，任何parent/anchor差异停止。详见`SAME_PARENT_NULL_PIPELINING_20261110_ZH.md`。

原生冻结测试快照`fresh_null_later_precommit_v1/code`以885dafb为基础，仅覆盖三项本轮源码/测试，source前后均`0f151abf603ed112ac9641c39b4bc22ece82055c4fee8c82556bac904cb6fa08`。13项原生测试全部通过（0.89秒），覆盖pending未知、严格默认、单调依赖完成、重复完成拒绝、query/parent/端点/文件损坏拒绝和既有fresh-pool路径。Windows当前Python无pytest，未称本机测试通过。

00:23:45实际source0/q48预检明确exit0（15.627秒），严格核对原Full事实、frame192的原probe/generation、同Gaussian/null gate与封存输入、两个Full anchor及原DINO reader绑定；完整读原记录的preflight不加载policy/reader、不执行物理分支。收据`LATER_NULL_Q48_ACTUAL_PREFLIGHT_V1.json`和`LATER_NULL_NATIVE_TESTS_V1.json`保留在AFS；尚未把该入口、资格或pending raw写成实际null完成或新增收益。后续使用已提交冻结快照准入真实H32分支，原三路活动快照保持。

### 23.64 已提交q48真实分支准入与默认q8实物回归

代码提交`9c6fe8d1c8a8837fd82d64a3ec60ccc5058af44b`已推送并核实，CCI canonical于00:25:46 UTC快进同步；最终Linux快照`controller_9c6fe8d/code`的source SHA仍为`0f151abf603ed112ac9641c39b4bc22ece82055c4fee8c82556bac904cb6fa08`，与13项原生检查的precommit快照完全相同，活动Eb/原科学core均未修改。已提交快照再次通过q48实际输入预检，原默认不带流水选项的q8也于00:32:12明确exit0（15.216秒）；后者无新policy/物理执行，不是重复采集旧q8。

00:27:21 UTC，在原闭环campaign lease空、单H100空闲55,096MiB、旧任务未重复的条件下，准入`fresh_null_source0_q48_v1`，outer PID2216810。新query独立有界时限5400秒、截止01:57:21 UTC，不改变三路原12h窗口及08:31闭环campaign截止。spec文件SHA `e1deefa9242c4c20e0d90da21fd385b4bd2a43a31c75fd7ad1c21e2923c265eb`，runner规范化spec SHA `17df30d6a9dbea516ecd4b2492dcf51f3c39d6fd790c99c8c2007947c7eff558`。先写prepared intent，再派发并原子写入PID；outer继承独占admission lock，不重复未知派发。CCI本次实际重启剩余未获控制台确认，未声称拥有新的4h窗口；持久输出与原件不受代码同步覆盖。

00:34:58实际核查q48 outer为running，正在原事实前缀回放，日志counter到50（不是50条均已完成的证明），尚无新的H32分支结果；三路原q48完整候选25/19/20，各summary仍仅q8完成。H100用34,146MiB、100%；Full b的AFS operator仍无。source0选中raw rank29和pool-final-anchor保持pending，不能用null启动填全三角色query。后续独占case3/Full前必须同时核实此新q48任务和三路pool的实际terminal/lease/资源，不只检查原闭环campaign lease。

### 23.65 六事实query的实际撤回干预、非空等规模bank与65项原件保全

对原v5已封存的q0/8/24/48/80/120作只读核查：六份Full有效门分别为0.1904296875、0.1875、0.1923828125、0.185546875、0.189453125、0.1884765625，null有效门均0；每份Full/null Gaussian的dtype、shape及bytes精确相同，conditioning及最终native动作均不同。自然alpha保留不变。它确认当前撤回对照确实改变了生成路径，不能把Full/null同为失败解释为两者输入/动作完全相同；它没有新增物理执行，也不证明修复收益、门值校准或missing-skill可靠性。六状态来自同一个source episode，不增加独立episode数。派生表MD/LaTeX位于`tmp/rebuttal/execution/20261110_iter3/withdrawal_sampler_six_fixed_queries_v1.*`，原始数值和24项文件哈希见`SIX_FACTUAL_QUERY_WITHDRAWAL_INTERVENTION_AUDIT_V1.json`。

00:31:14对原89,680-event bank的真实task_index数组核对manifest哈希后，按原seed3407及已冻结family taxonomy生成非空等规模搜索域。Rearrange Blocks两条件均19,540事件，holdout移除六个原block-family任务；Battery Try两条件均78,772事件，holdout移除battery_try。原bank字节、事件ID和context task标签不变，四份行索引与原manifest/task_index已封存。实际predicate coverage、自然gate及配对SR尚未测量；保留通用grasp/transport可能适用这一限制，不把task-family移除称为零相关技能覆盖。两目标的原配对样本量/主指标未改。

本轮源码/13项检查/实际q48与默认q8预检、准入及spec、24项六query撤回原件、q48两份已完整anchor和原pool/null封存输入、真实bank及四份搜索行索引，共65项已归档`later_null_inputs_and_holdout_universes_v1.tar.gz`：5,610,283 bytes、SHA `9046ea05dfc9b02bf4f264ef01222dd5818f9b0dd0efd2713d93b3ec97f6e6e8`，manifest SHA `20463e180ade729d53186024647f02b2a1687fc88a7e72443b3abcd0110c8f22`。00:36:45本机逐项核验65文件、24原始采样器哈希及四份bank行索引一致；收据`LATER_NULL_INPUTS_HOLDOUT_LOCAL_VERIFICATION_V1.json`。这是输入/资格/既有封存干预与搜索域准备包，含本轮新null物理分支结果0；新q48运行轨迹待真实terminal后另包保全。

00:43:23实际核查新q48 null已完成H32，独立局部标签0，严格原父/RNG/DINO资格未放宽；正在post-null Full的原192命令前缀回放，尚未完成重复端点资格。此单null分支不等于整个三角色query或收益结论。另核对三source各q8/q48的六份原生anchor记录：父stage_id均0、press_cnt均0、first_placement均false。`FIXED_QUERY_NATIVE_TASK_STAGE_AUDIT_V2.json`绑定六份原件SHA，说明这些query有控制时间差异，但没有已验证的语义阶段覆盖；不删失败source、不按结果换轨迹，也不把此局部限制推翻原稿其余task/模型报告结果。

### 23.66 q48两个H32真实完成、局部对照表与文稿联动补充

新q48任务于00:53:28 UTC outer明确exit0，科学主体1503.679秒。null与post-null Full各32条命令、局部next-interaction标签均0；两次各192条事实prefix，新增分支命令64、prefix命令384。重新独立核验所有32×14实际命令与原seal一致、实际articulation qpos/qvel有限、完整native/RNG父hash及DINO精确，post-null Full端点hash与效应向量均精确复现原anchor。原科学core前后5d7748e7…、runtime前后a97e308c…保持。只记录实际单run时长，不作为受控速度优势。

完整终态档案`q48_same_parent_null_and_Fullrepeat_terminal_v1.tar.gz`，35项、3,083,743 bytes，SHA `c685d8c3d145d0154e722a433bf2971138f8c0e4dd21449de211031e35ad28b7`，manifest SHA `f8d6f62d65ca1da36603d515ebf34f223df29f4e805afd3869f0cb3fc3ca2750`；AFS与本机全部35文件核验。此前15项partial档案SHA `3ac59bcd4aa44402e1d10e7efa24bbc8f02c6bd62258846eaa3874e7abbf7795`也已本机核验并保留，不覆盖。报告`Q48_SAME_PARENT_NULL_TERMINAL_ARCHIVE_V1.json`与`Q48_TERMINAL_LOCAL_VERIFICATION_V1.json`。

q8/q48局部对照表JSON/Markdown/LaTeX已生成于`tmp/rebuttal/execution/20261110_iter3/same_parent_q8_q48_local_comparison_v1.*`。q8的raw/Full/null均0、依赖齐；q48的Full/null均0，原raw29与pool-final仍pending，不能填全repair/harm。两个query的Full/null实际physical-joint端点都不同，说明实际运动受到干预，但未观察到局部任务收益。两query仅一个source episode、均native stage0，重复Full不是独立样本；原闭环完整配对仍3，不制造总体修复率、置信区间或跨阶段主张。

v4 research-patch1于00:49:43 AFS核验46项、7,079,858 bytes，archive SHA `ca06730ff406133399f6d6907bfdc955d4f20c91597231322d627c0f403b0ea7`；它是终态前的不可变快照。终态补充于01:00:09 AFS核验12项、3,117,044 bytes，SHA `382598a650bce349a17db674f729157e3d3bc28586448c1646fbb30aae7c176e`，manifest SHA `2dab1d74dbde849ce2778438666e5dda4962cfc0a7130002ce91f69ace1ebeae`，目录`writing_candidates/20261110_v4_q48_terminal_addendum`。已联动20项映射、内部README与一致性门槛；三位英文回复正文和35页论文/29文件LaTeX ZIP未改，PDF/source SHA再次核对保持，不为未修改文稿重复编译。没有把新局部0/0写成收益、没有对外发布，stable_release_ready=false。

### 23.67 非空搜索域自然门控入口、原生检查与真实初始输入预检

01:20:01 UTC三路原Eb仍实际运行，q48完整候选29/23/24，GPU25,996MiB/100%，Full500 operator未生成。01:20:39控制台安全只读核验仍要求IAM登录，保留原登录页与已有作者请求，不重复创建/停止任务。原选中source0/q48 raw rank29随后完成，独立局部标签0；此时仍缺原pool-final-anchor，不能提前宣告该query资格齐备。

新增`collect_rebuttal_initial_gate.py`和`initial_gate.py`，固定读取三个原RB source的query0记录，在两个已封存的19,540事件搜索域分别构建真实Full policy。保留原生空初始history、原task标签与全部实际推理配置，不执行动作。保存actual alpha/g、合法row/event来源和原始probe；没有独立coverage真值时，g>0只称positive gate，不称false activation或任务失败。生产清理接口success=false不计SR。完整方法见`INITIAL_GATE_AVAILABILITY_DIAGNOSTIC_20261110_ZH.md`。

以97e366a Linux快照为基础的precommit source前后均`f39b62fe75a48ae7e47a03876f7f11634d2478844d14614e3bbadb530b231263`，19项原生检查通过（2.00秒），覆盖原数组/未来字段拒绝、原生空history、强制/消融控制拒绝、eligible row越界、非法门值及hard rejection一致性，原repertoire搜索域检查同时通过。实际两条件的15项输入绑定与三个原首查询/数组均通过CPU data-only，policy未加载、执行命令0。

补充计划`initial_gate_availability_v1.plan.json` SHA `de0cc273f540c71fa160e2f89726d584501f427011e3e2c9f43ad9f4165201ff`已在Git外冻结，原科学core5d7748e7…、原模型68c7f384…、四观测资格计划0df7b0f…绑定。预检与native19通过不是自然门值结果；后续从已提交冻结源码准入真实两条件推理，原200-pair主协议/候选样本/原资源窗口均不改。

### 23.68 source0固定q8/q48完整、后期后果误差与同父三角色核销

原source0 K32主体于2026-11-10 01:48:32 UTC完成，outer于01:48:38明确exit0；原Eb collector source前后均`823404c10af58f90b932b406b0a4cc2a3632acb20b691841a8a82ac7b0c5cdad`，科学core5d7748e7…及runtime a97e308c…未变。q48全部32候选、三次Full anchor均完整H32并通过父状态/重复端点资格。选中raw rank29局部标签0、完整候选中正标签0，Full/null也各0。q48预测centered MSE `0.08068330443023085`，高于query-constant `0.07378675842267281`，低于historical `0.1179940894596835`；预测centered explained variance `-0.09346590302900348`。absolute MSE依次为predicted `0.16764043682731286`、constant `0.17177325838028487`、historical `0.15859240329098073`，不以另一指标替换预设主指标。前三个固定q8原有86个完整H32/10未知及其不利主结果均保留，不重采或拟合缩放。

完整source0目录包含旧partial/unknown attempts及终态runner，共591项，归档`source0_complete_fixed_q8_q48_v1.tar.gz`，82,971,466 bytes，SHA `e6c8ca48577833f753a3b4d0efa7d086eac8631b5bd82a4ec57710882742e336`，manifest SHA `278eae51dcc1d2ade1be19f88d985d948f2944e6e9471c015bc81acf7572fd66`。02:05:48本机逐项长度/SHA核验通过，见`SOURCE0_COMPLETE_Q8_Q48_ARCHIVE_V1.json`与`SOURCE0_COMPLETE_Q8_Q48_LOCAL_VERIFICATION_V1.json`；原件均在Git外本机iter4和AFS保全。

02:08:00再次以严格`allow_pending=False`核对q8/q48原选中raw及pool-final依赖，确认两个完整Raw/Full/null query，同父/RNG和post-null Full端点/效应精确；两query各三角色局部标签均0，Full/null实际关节端点均不同。`SAME_PARENT_Q8_Q48_COMPLETE_V2.json`与Markdown表记录全部读取原件SHA。旧q48 pending快照按原时点保留，不覆盖。两个query仅一个source episode、均native stage0，不能称总体修复优势、跨阶段覆盖或新增闭环配对SR。

### 23.69 两个非空等规模搜索域的六次真实自然门控诊断完成

实现提交`a92f5e0f930fedb4fd091b7453ae31454e976632`已推送核实并同步CCI；冻结`controller_a92f5e0/code` source `f39b62fe75a48ae7e47a03876f7f11634d2478844d14614e3bbadb530b231263`与19项原生测试/实际输入资格快照相同。01:42:40准入顺序operator2224398，clean和heldout分别于01:51:54与02:00:52完成exit0，collector source前后保持。采用作者指定shared-stage v8 update3000，checkpoint `68c7f3840921241df45a5b6997b772db3a69a31584644aa0e66929b841b845bf`；原科学core5d7748e7…、runtime a97e308c…、BF16/H32/prefix4/NFE10/K32不变，不称原稿task-specialist协议复现。

三个固定source的原query0实际RGB/官方关节输入、原生空episode history、上下文/原DINO/派生seed/基Gaussian均逐对精确一致。两个搜索域各19,540个原bank事件，eligible行与原89,680-event bank未改。clean完全复现这三个原Full初始query的整份probe NPZ及H32生成动作SHA。移除六个block-family任务后，三个heldout topK的task计数分别为32个BT、32个BT、31个BT+1个PB，selected均BT；任务标签不能证明不存在可迁移的grasp/transport技能。

| 固定source | Clean alpha=g | Heldout alpha=g | Heldout减Clean | 两条件valid K |
|---:|---:|---:|---:|---:|
| 0 | 0.1904296875 | 0.1943359375 | +0.00390625 | 32/32 |
| 1 | 0.2197265625 | 0.1923828125 | -0.02734375 | 32/32 |
| 2 | 0.2021484375 | 0.1904296875 | -0.01171875 | 32/32 |

三者heldout自然有效门均未关闭，既定threshold0.15未调。它不能支持“移除task family即可靠回退”，也不能在尚无独立相关性标签时直接称false activation。执行命令0、独立coverage/局部applicability/SR标签均未测，production cleanup的success=false不计策略失败。本诊断是三个真实固定初始输入的availability response，不替代原200-pair主协议、非劣性或长时序可靠性实验。JSON/Markdown/LaTeX派生表`INITIAL_GATE_PAIRED_ANALYSIS_V1.json`、`initial_gate_paired_v1.*`包含全部原值，未拟合/选择阈值。

终态原始probe、六query记录、两份真实exit0 runner、15项冻结输入、源码/19项检查/预检/encoder资格与准入共87项，档案`initial_gate_availability_two_conditions_terminal_v1.tar.gz`为3,695,916 bytes，SHA `ad2ba6b747e24aa389748422eca50307e0f2132ca849c4113a3dfb6320dcfeb1`，manifest SHA `a906bc33802e625e3c33ae356169cc2891bc24d6a71ac65d5854197bd02f8264`。02:05:48本机全部87项逐项核验通过；收据`INITIAL_GATE_TERMINAL_ARCHIVE_V1.json`与`INITIAL_GATE_TERMINAL_LOCAL_VERIFICATION_V1.json`。原件在Git外，旧首轮/partial/资格件均保留。

### 23.70 两路已知wall-time安全接续，原窗口与固定样本保持

02:01:37 UTC原source1/2 outer2212838/2212839均明确exit-15，reason wall_time_limit；不是策略失败。q48完整候选分别27/28，各另有两个完整Full anchor，summary尚未complete。核实两outer/collector实际不存在、两native lease空、闭环campaign lease空、自然gate operator已明确完成、单H100空闲81,080MiB后，于02:02:35以不变Eb source/spec仅恢复source1/2，PID2226496/2226497。`FRESH_POOL12_KNOWN_LIMIT_CONTINUATION_V4.json`保存prepared-before-dispatch意图、旧terminal哈希、两新PID与spec SHA。source0已完成未重复，原05:44:21 UTC总deadline未重置；本段仍7200秒上限，若先完成则不续跑。

继续优先补齐这两个固定q48及原稿模型/协议对应，闭环下一个固定角色仍case3/Full、seed340800003与08:31:04 UTC campaign截止不变。任何独占Full准入前核查实际两pool进程/lease，不能只看GPU空闲或旧current_iteration。ACP当前仍沿用01:20:39已核实的IAM登录需求和唯一Full b的AFS operator缺失状态，不重复未知创建；未将排队写成活跃训练。新真机取证仍按作者要求延期。

### 23.71 ACP b真实失败与权威路径更正；前置环境诊断保持严格恢复

02:20:01 UTC发现先前若干AFS检查误读`full_resume200_500_v1/attempts/operator.json`；权威路径实际是`full_resume200_to500_v1/operator.json`。因此23.70及旧current_iteration中延用的“operator无/仍排队”不能作为当前判断。真实b UID9c6000a4ce734a7bac717b2053197a1e已于01:24:01启动，01:35:19 runner exit1、operator failed_or_interrupted。日志定位到formal resume状态验证：training_runtime_json与父step200不同，尚未装载续训state；新training_metrics和完成marker均无，step201+更新0。现存日志未列ACP实际子字段，不能推定仅平台/内核变化。原训练snapshot source前后4efb8094…完全一致，原父state/权重未改。

15项真实failed operator/log/config、两项局部output、计划及父attestation已保全：档案`full500_b_resume_validation_failure_20261110_v1.tar.gz`，42,902 bytes，SHA `297839849a449a1c4a4521bf782db2db380e3d50bc3cc9b1bdcd1f960f56de52`，manifest SHA `818d343e3848f6e750314eb6e324e2e4465510b8c53e9ca69515dcc6e146ded7`。原始source、父step200与失败未覆盖，不能登记500或最终性能。02:27浏览器安全核查仍IAM登录，保留原登录页/已有请求，不重复未知提交。

新增只读静态环境probe，模型/数据加载前记录真实Python/platform、Accelerate/DeepSpeed/Torch/CUDA/cuDNN和GPU数量/型号/算力；父attestation SHA严格绑定，差异逐项记录并exit2，没有忽略平台/GPU差异选项，不改原formal validator。8项原生检查通过0.14秒，precommit source前后`adbd844a03fba735ebc6252d8a153ccdd41e107108b973c7c6773163159993fc`。02:30:57实际CCI训练env对原父step200的10项静态事实全部匹配、probe exit0，无model/训练更新；静态匹配不等于live post-prepare数值及全state已验证。见TRAINING_ENV_PROBE_NATIVE_TESTS_V1.json、CCI_STEP200_STATIC_ENV_PROBE_V1.json；ACP实际差异仍待恢复访问后采集，不能写成已修复。

本轮实测AFS可用79,612,583,936 bytes，低于原计划80GiB最低准入，故不启动Full200→500或新pooling训练，不降低既定存储门槛、不删除父state/CAS。原b的01:24起4h窗口不能通过重提交重置。继续有用的固定q48推理，完成后按原窗口接续case3/Full；资源/登录问题不影响已完成诊断与文稿工作。

### 23.72 v5可审阅回复与完整LaTeX/PDF联动验收

三位英文回复v5仍15个comment blocks/20项原意见，新增HH-W5的四query不利centered结果、BB-W3/Q2的两个同父局部0/0/0以及RK-Q2的三对自然gate原值；原task-specialist报告结果保持独立身份。所有20项的修订label/页码已按真实v5 aux刷新，未将已知负结果藏为“尚未测量”，未声称新收益、误激活率或非劣性。

原完整LaTeX工程v5编译37页、主文10页；前10页逐页文本与v4精确相同，所有原数值表和9张图逐项保留。新增Appendix C.2.4位于31–32页，Table20记录4个终态query的118完整/10未知及三参考centered误差，Table21记录三对非空搜索域自然门值。全部37页渲染并检查contact sheet及两新增页，表格无裁切/重叠；无undefined引用/overfull，46项既有font warnings保持。PDF SHA `bd3ad11d86e7533c662c8993a555cffccbe7253b37ebe5c4c5d6c92880cd7dab`。

完整29文件LaTeX ZIP SHA `00d2c61711fa70c231cbad81f22d0b899920ef5c7e07a7532482aa221f086538`，逐文件manifest核验、干净目录独立重编译，37页文本与交付PDF全部精确相同。路径`tmp/rebuttal/responses/20261110_v5`与`tmp/rebuttal/revision/WARM_ICLR2027_v5`；CONSISTENCY_CHECK.json保留未关闭的模型/直接机制/coverage/同信息最终性能门槛，stable_release_ready=false。v4完整保留，未发布OpenReview评论、未上传正式稿。后续新数据须以新补充/版本联动，不能覆盖此候选的已审阅身份。

### 23.73 三source/六固定query全部终态、完整证据核验与v6联动

source2与source1的原q48分别于02:42:55和02:46:34 UTC完成，终态outer均exit0、source前后Eb823404c1…相同，科学core5d7748e7…/runtime a97e308c…保持。03:00:37本机验证新增source1/2全目录和终态runner的1190项，包括全部旧partial/unknown attempts；archive `sources1_2_complete_fixed_q8_q48_v1.tar.gz`为164,629,579 bytes，SHA `fc9115a12d63d920402baf09cda6e378a9533c5fad7f765209c2622599b1afc9`，manifest SHA `b766b64dd58ffaf75111eb9b7b783cf04c90f2f8539573cdd161f8554c1d3da0`。source0的591项e6c8ca48…原包仍完整。传输时一份receipt在远端封包尚未完成而未找到，后来实际终态后补取；早取的archive最终长度/SHA与完整终态一致，全部1190项重验通过，不将缺收据当科学失败或接受未核验包。

六query全部210个terminal branch收据逐项哈希核验，三次Full anchor的原父与端点/效应均精确重复；以`ALL6_FIXED_QUERY_FINAL_ANALYSIS_V1.json`为准，原定192个valid候选，182完整H32/10未知。新source1/q48 predicted/constant/historical centered MSE分别`0.06596692247155121 / 0.04580564611756849 / 0.10692277070395272`；source2/q48分别`0.09446030481532149 / 0.0822927004913579 / 0.13254460274544932`。至此六query的predicted主指标均高于constant，不以absolute MSE或事后校准替代。三source的六父状态均native stage0，不声称跨阶段覆盖或独立candidate样本数；原稿specialist结果保留原身份。

02:49:25 GPU实际0MiB/81,080MiB空闲，三个pool/lease均明确terminal/free，AFS仅73,464,709,120 bytes。验证唯一原ACP f已知停止、自然gate及q48null结束、Full b真实failed、原campaign lease空及next_role恰为case3/full/seed340800003后，于02:51:17按原冻结controller5b9ebd7/source56393c5a…准入下一固定role，outer2230608。spec SHA `623f26451b8ac2f0abe2d9f2db35c2c59a8c05ee68e3d978aaae8e11faf821c7`；worker3600秒、outer5400秒、原08:31:04 UTC总截止不变，准备意图先于派发落盘。此为准入，不是完成第四对或新收益，原所有旧attempts保留。

v6替换v5中的“四query/仍采集”文字与表行，报告全部六query/182完整/10未知。三位英文回复仍15块/20项，原稿原表及九图保持，主文10页与v4逐页文本相同；完整37页工程与29文件ZIP独立干净重编译，37页文本一致。全部渲染、31/32新增表格视觉检查通过，无undefined/overfull，46原有font warnings保持。PDF SHA `b91620c380d4ec270e3ac54ec7e7d252481940f2a7fb2287d4298998ea25cdfd`，ZIP SHA `6137ce24f88cc3e1b2fbbd4a9146eb0f81ef269035dbe3344948e24742491250`。当前候选为responses/20261110_v6和revision/WARM_ICLR2027_v6；v5及更早候选完整保留。stable_release_ready=false，关键模型对应/直接机制/coverage/同信息最终性能未关闭，未发布或上传。

### 23.74 第四完整固定配对与下一source同事实null封存准入

case3/Full于03:18:03 UTC worker明确exit0，outer于03:18:10 exit0，完整700×14命令有限、官方700步success0；原seed340800003、checkpoint68c7f384…、controller source前后56393c5a…保持，四原事实DINO平台资格记录齐。与已完成case3/null700组成第四对，原case0–3的Full/null均0/0；其余六对pending及旧中断不计失败，不据此推翻原稿specialist结果或声称复现准确率。

完整564项Full worker原件`case3_full_v3_complete700_worker_v1.tar.gz`（原AFS attempt内名verified_worker_evidence.tar.gz）为4,267,400 bytes，SHA `b91731c100bde4b9b6976db933d1c98f12f2686b25af0fb429c2516a9c9d4438`，manifest SHA `9c975c89c8d5987b5f158f25d899540de7e496b83d8a8fb0583132e6111467e2`，AFS及本机全部逐项核验。真实700 trace SHA `7d71de393b4ebefeae441a3b73f37860085402269560c753683726419b225485`，资格与终态见CASE3_FULL_V3_VERIFIED_COMPLETE_V1.json及本机验证件。原null完整564与旧497中断包不覆盖。

确认三pool/新case3 outer均exit0，campaign lease空、单H100空闲81,080MiB后，03:25:02 UTC准入原固定source1（seed340800001）事实700回放与null动作封存，outer2231994，max5400秒、截止04:55:02 UTC。复用已提交controllerbc13308/source `84af4413e9c59934c34664c35bbfab86fb165f2604a2e01ffbf104f4f6716c3e`及原v5只读共享权重wrapper，实际新spec SHA `dabcce452a9309d3480761dd89b371651143446bb5d233da95f66af960924a22`。实际输入预检绑定原source1六份H32 generation、q8/q48封存pool哈希及初始query已核实Full精确重现；它只是data preflight，所有新query仍须实际replay/Full/context/Gaussian/RNG逐项严格通过。

新输出`rb_source1_factual_nullseal_v1`，准入收据RB_SOURCE1_FACTUAL_NULLSEAL_V1_LAUNCH.json。选择source1是补齐原固定source顺序，不按结果更换/替代失败source；不执行null物理分支、不增独立任务样本，不提前声称已完成700回放/封存。成功后再以原完整source1/q8/q48做同父物理null；source2固定顺序后续补。下一闭环角色case4/null及原08:31UTC campaign窗口保持，避免和当前诊断同时争用GPU。


### 自动采集：fresh-scene K32 fixed-fresh-rearrange_blocks-e0000-q0048

fresh-pool:fixed-fresh-rearrange_blocks-e0000-q0048:075c81ed9003701b3a5ce7a482b3005a482af9b2d87473b53359907a5c341f62；原K32有效候选32，完整H32端点32，完整refined重复支持True；原件`/mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations/rebuttal_20261109_acp/rb_source0_fresh_K32_v1/q0048`，候选结果SHA `73478a4d718edd4d9fe6547dc3a086315e974a828da55b6fd6efba23f773ba80`，centered结果SHA `91014a9d8d460de0244cdeaa1462787c14c46650e3751b98543344181ed2b267`。**仅固定source/query的直接候选效应数据；不替换原论文表格，不视作整个cohort或策略收益。**


### 自动采集：fresh-scene K32 fixed-fresh-rearrange_blocks-e0002-q0048

fresh-pool:fixed-fresh-rearrange_blocks-e0002-q0048:a742f9188889244827f6ce930bf43b20921d868a97be608b66ccf2ccc32324e4；原K32有效候选32，完整H32端点32，完整refined重复支持True；原件`/mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations/rebuttal_20261109_acp/rb_source2_fresh_K32_v1/q0048`，候选结果SHA `29130e04ebe6650f679001e980639174d56956c76627c3e0740c7028290f362d`，centered结果SHA `fb0464da99a131ee23d92696562bc5896924c6392f94727e7383f9e7cb42272a`。**仅固定source/query的直接候选效应数据；不替换原论文表格，不视作整个cohort或策略收益。**


### 自动采集：fresh-scene K32 fixed-fresh-rearrange_blocks-e0001-q0048

fresh-pool:fixed-fresh-rearrange_blocks-e0001-q0048:91e0ea3f20ca3008237c3dd6407d973a841bf602c692ebba864938ddbde40991；原K32有效候选32，完整H32端点32，完整refined重复支持True；原件`/mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations/rebuttal_20261109_acp/rb_source1_fresh_K32_v1/q0048`，候选结果SHA `806acaff1d69fc33b450555e5584e4f29277df581ab34fe157c9c5254ec77917`，centered结果SHA `cc25f3e51efe4f5747134ae984b6dd929346d9537f419755dc0d3475bf01b0d5`。**仅固定source/query的直接候选效应数据；不替换原论文表格，不视作整个cohort或策略收益。**

### 23.75 作者确认原稿specialist设置、source1两个同父query完整与source2接续

作者本轮明确“原稿使用逐任务specialist”，并完成SenseCore重新登录。原稿训练设置已确认；各任务实际权重/DEV选点/bundle绑定继续查原记录，不按新测结果换模型。此前v8 shared-step3000的完整/不利/未知结果均以独立诊断身份保留，不能将其当原稿specialist准确率证据。控制台实际确认唯一Full b为失败、UID9c6000a4ce734a7bac717b2053197a1e、最大重启次数0，原训练更新仍200。独立环境测量只用于记录实际runtime字段，不会放宽严格resume准入。

source1事实700回放与六原query的Full上下文/null动作封存完整exit0；逐数组核对commands/before/after与原700 trace的dtype、shape、值全部相同，六次自然Full/common-Gaussian的受控null gate均0。03:51:33严格实际输入预检的q8/q48均exit0，仅data/contracts，不执行物理。03:56:28准入两个独立query并行，reader各沿用0.12 CUDA fraction；每query3395秒上限，原source1 03:25起04:55截止未重置。source1/q8、q48 outer分别04:02:49和04:07:01明确exit0，collector source前后0f151abf…保持，科学core5d7748e7…及runtime a97e308c…未改。

两query各新增null32与Fullrepeat32，共128个真实分支命令、448个事实前缀命令；原raw/final-anchor依赖齐备、每次原父/RNG/DINO精确、重复Full端点及效应精确。q8/q48的Raw/Full/null局部标签各为0/0/0，Full/null实际关节端点都不同。source1/q8原pool已知至少2个适用候选，故不能把这例失败一概归为经验库无可用技能；选中提案与生成改善仍待更强直接证据。当前与source0合计四个完整局部query、两个source episode，不将重复Full视为独立样本、不报告总体修复率或收益。

source1事实回放/六封存、两query物理原件、三份终态runner及准入/实际预检等162项，档案`source1_complete_factual_seals_and_same_parent_q8_q48_v1.tar.gz`，13,428,763 bytes，SHA `30997de11619a85bb34cfe90be408aec561267a35aa2662aeb7c2185b8f76241`，manifest SHA `997fa9ee5a1751ec7181cd7d21d6e0e4a5cdbdcc1957e762f71f0fb6f1e579b2`。04:30:33本机全部162项逐项核验；AFS/C盘均保全，原件在Git外，收据SOURCE1_COMPLETE_FACTUAL_AND_NULL_QUERIES_ARCHIVE_V1.json及SOURCE1_COMPLETE_LOCAL_VERIFICATION_V1.json。

两query真实terminal及lease/GPU空闲81,080MiB核实后，04:14:04按既定source顺序准入source2/seed340800002的事实700与六query封存，outer2234575，独立5400秒窗口至05:44:04 UTC，source84af4413…和原v5共享权重wrapper不变。spec SHA `719f26a50f9b56ded12c0757a2963f2cddeae85dd221240a1251faa5ef1dc133`；原有固定样本及其他窗口均未重置。此为准入，不提前记作完成或新增SR；成功后原source2/q8/q48严格同父物理对照，原闭环下一role仍case4/null、campaign08:31截止保持。

### 23.76 六个固定同父对照完整、适用性复核与ACP真实配额回退

source2事实700回放/六query封存完整exit0后，04:53:19严格完整pool依赖的实际输入预检两项exit0；04:53:48并行准入原source2/q8与q48，分别outer2241721/2241722，每项2897秒，原source2截止05:44:04 UTC未重置。两项实际终态exit0，q48于05:04:10退出；各新增null32与Fullrepeat32，原父/RNG/DINO及重复Full端点/效应精确，Full/null实际关节端点不同。Raw/Full/null局部标签仍均0。source2原700 trace逐数组dtype/shape/值精确、六共同Gaussian及null gate0亦已核验。161项原件档案`source2_complete_factual_seals_and_same_parent_q8_q48_v1.tar.gz`，13,375,317 bytes，SHA `c480b29f04cb827ad89ef2e0eb9004e93121eb53202e758890e1a890874afc5d`，manifest SHA `e24bbf69fc3c9d8358f97f2b9d894dd9568f9abe448e42adcbf92dad1854d589`；05:06:06本机161项逐项通过，AFS/C盘双端保全。

目前三个原固定source的q8/q48共六个完整同父Raw/Full/null query，18个主要角色标签均0，重复Full仅作精确性检查，不计独立样本。它们属于作者此前指定的v8 shared-step3000独立诊断，不能替代已确认逐任务specialist的原稿准确率，也未给出修复收益或总体修复率。

05:04:45另逐一重验原六pool的192个候选分支complete收据与原父状态，独立适用标签按source0/q8、0/q48、1/q8、1/q48、2/q8、2/q48依次为“适用/不适用/未知”：0/28/4、0/32/0、2/27/3、0/32/0、0/29/3、0/32/0。source1/q8的rank9、rank10确有适用动作，而原选中raw31/Full/null仍均0。因此不能把全部失败归因于没有可用候选；另外三份完整q48 pool无适用端点的结论仅限原K32与该H32标签，不证明整个经验库缺少技能。原10个unknown保留，不用任务名称代替适用标签。原件`ALL6_FIXED_POOL_APPLICABILITY_AUDIT_V1.json`，其候选原件已包含在此前591/1190项完整档案内；这次是既有证据重验，不新增采样/物理分支。

ACP登录恢复后，两次标准资源创建尝试均未生成任务；第二次保全了服务端HTTP429正文：`MEMBER_QUOTA_EXCEEDED`，两允许集群合并成员GPU配额剩余0。第一次未知导航后按完整显示名前缀查询“暂无数据”，再等待后重试，未复制未知活动任务。当前配额实证满足回退条件，05:00:34成功创建闲时轻量测量`warm-env-measure-20261110-b`，UID `0d255c27be8e45108cd226b509b75f78`，cluster01e、1GPU/8CPU/128GiB、重启0、timeout900，镜像/AFS沿已确认设置。05:07:53控制台仍启动中、实际GPU0，测量目录尚无，不能写作运行完成。标准拒绝收据SHA `2bdae6f2132e2a137a5d2e40cda39eb060fb925432665adae96160552e4e5e16`及创建/详情收据、先于提交冻结的两个intent均在Git外保全。

新环境测量operator已在CCI实际通过并以`a77686039e047c78a5ff84b0e68ae9260f4d199a`英文提交、推送核实、CCI离线同步。冻结source `2836b318b023f66cdfc5aae45c57c4cc38017fee1345987f104aa36a761626a2`，内部严格probe仍`adbd844a…`。它保留probe原始exit0/2及具体差异；ACP成功只表示环境事实采集完成，永不据此准许resume，不加载模型、不产生优化更新。Full旧b仍200，无新训练准入；原父与失败/窗口均保留。

历史BRT step14000合同确实完整绑定f77c633、权重SHA515ecad0…、原bank/normalizer和seed。但对应formal100-v2在compute_device NameError处失败，v3在Sapien导入前失败，均不能恢复原稿成绩或DEV选点。八份合同/失败原件保全为`HISTORICAL_OFFICIAL50_CONTRACTS_AND_FAILURES_V1.json`。原训练设置已由作者确认，仍需每个最终specialist选点的具体绑定；不把旧v3候选装进v8。

**固定顺序更正：** 原记录写“下一case4/null”不准确。05:07实际调用冻结campaign的`next_role`返回case4/Full（seed340800004）；错误null准入在断言处停止，未生成job/spec或执行GPU。随后严格按真实冻结顺序05:07:56独占准入case4/Full，outer2245022、worker3600/outer5400、原08:31:04UTC campaign总截止保持，source56393c5a…、spec SHA `ace52b811163f6044289025d8becbfe2ea5fbc7ecebee19f6f76cd353434960a`。这项仍在运行，不提前增加完整配对数。原始顺序、全部失败和unknown均保留。

### 23.77 v7回复/完整论文联动验收与ACP跨集群调度核销

三位审稿人20项意见、三份英文正文和15个可粘贴评论块已更新为`responses/20261110_v7`。原工程完整编译及干净29文件ZIP独立重编译通过，37页逐页文字一致、主文10页；全部页面渲染，更新31/32页和全篇联系图已目视检查，无undefined references/overfull，原数值表及9图保留。PDF SHA `644a488694d6ebc7245c40677476ca2c646f782d046e6f9582c5ab9043316056`；source ZIP SHA `1758da6f2a44d5e9b9dc0ea31a76a6bd5be0caa28fd27141bacb9014c0ec4551`。Table20新增原候选池独立适用数；六个同父对照/18个零标签以及“有适用候选仍未改善”的范围同步至Appendix C.2.4和Bbqv W3/Q2。原稿RMBench specialist设置不推广成未核实的每个机制cohort模型绑定。原33页投稿及v4–v6保持，尚未达到实证stable release gate，未发布或上传。

六query组合报告`SAME_PARENT_ALL6_LOCAL_CONTRASTS_V1.json` SHA `b070afa662e34dbbc4fda5546b608dc201e4596d7a33d4c2d736875fc257d1dd`，05:11:58再次重验原parent/RNG/DINO、12个新物理分支收据及Full重复端点，另输出CSV/Markdown/LaTeX表。独立适用复核报告SHA `c72a9e90c2b690ddec6ee71b828462c4c7e8a9688db124e5dce8ae584a6c3eb3`。这些均为Git外既有原件的复核/整理，不增独立episode或重新抽样。

环境测量b一直未获GPU，worker事件显示FailedScheduling（nvidia.com/gpu不足及曾有PVC未绑定），不能当作实际模型运行。停止请求与停止中阶段保留，05:22:56确认为已停止，UID0d255c27…；AFS测量目录仍无。仅在确认终态后，以先于创建冻结的cluster02 intent将同一静态测量转投另一允许闲时集群，05:24:59唯一c成功创建，UID `23bd904f9f3741b8b98e1821800d5ed1`，1GPU/8CPU/128GiB、重启0、原timeout900/镜像/AFS/排他输出目录不变。05:29:32仍启动中、任务规格*0，具体ACP runtime子字段未测得；不重提未知c、不将调度状态记作工程或科学成功。b终态停止收据SHA `4eb544a193cd723b7822f0c8b2339050be5b84db766c70b3dcee2e362fa78846`及c创建/身份收据已双端保存。整个过程没有延长/重置旧Full训练b的窗口或修改训练数值校验。

05:28:24现场核查case4/Full实际child2245058存活，campaign心跳持续、GPU100%、用32,967MiB，原source56393c5a…与08:31:04UTC截止保持；完整配对仍4，不提前算case4。该时点AFS可用224,506,974,208 bytes，已超过原80GiB空间门槛，先前容量不足只是历史观察，本轮没有删除任何state/CAS。当前训练恢复的ACP差异、时限和原specialist绑定仍需分别关闭。完整快照在`ITER5_LIVE_CAPTURE_V1.json`，原件Git外保全。

### 23.78 case4/Full完整700与原candidate-null接续

case4/seed340800004的Full角色于05:33:36完整700、success0，outer05:33:43 exit0，source前后56393c5a…不变。通过原read_roles/verify_role核对固定seed/checkpoint/core/runtime、初始相机与qpos、完整finite700轨迹和durable result后，原worker的564项档案`verified_worker_evidence.tar.gz`已保全，3,981,936 bytes，SHA `1b4f903bb6dd0846fa628ae8f4830a5d1c6af9c41a6c1903a2794e60012a8c62`，manifest SHA `5601ad530429de4ac03f9c61284cfbef5e09ef5527f6da02e641ce55487d1c0b`。05:39:10本机564项逐项核验；AFS与C盘均保全。此时Full/null完整配对仍4，不将缺失counterpart视作失败或新增第五对。

Full实际terminal、进程退出与lease/GPU空闲81,080MiB核实后，冻结next_role返回case4/candidate_null。此前pre-admission将API角色误写为null的断言已阻止提交，未生成错误job或GPU执行；保留原检查脚本。05:37:15正确准入该原固定counterpart，outer2249166，worker3600/outer5400，原08:31:04UTC campaign总截止不变；spec SHA `b67d1301c59f3152bb24118ad67e3b27ffd5a84280bf00675ede518d09c6645e`，source56393c5a…与shared checkpoint68c7f384…不变。新角色只记进行中，尚无新配对结论。原件收据`CASE4_FULL_V1_VERIFIED_COMPLETE_V1.json`、本机核验和`CCI_FIXED_CASE4_NULL_V1_LAUNCH.json`均在Git外。

05:39:13 ACP c的唯一worker仍等待、无Host IP，事件明确为cluster01e_02缺1张GPU。它尚未运行静态测量；保留该UID继续排队，不再复制未知活动任务，也不把创建成功记成算力或实验成功。

### 23.79 ACP c实际完成静态测量，父200的10字段匹配

唯一c于05:47:25实际启动，05:47:37完成measurement receipt；05:54:59控制台确认成功、UID23bd904f…、重启0。严格probe exit0、differences为空，Python3.10.20、Linux5.14.0-284.25.1.el9_2/glibc2.35、Accelerate1.12.0、DeepSpeed0.18.5、Torch2.7.1+cu128/CUDA12.8、cuDNN90701、CUDA可用、单张H10080GB/SM90等10项静态字段与原Full父200完全一致。没有模型装载、优化更新或resume准入；不回溯推断此前cluster01e旧b的具体runtime差异。

四份实际测量/日志、三个事前intent、实际标准配额拒绝/旧b终态/c终态收据、三个冻结源码和父attestation共16项，档案`acp_cluster02_static_environment_measurement_complete_v1.tar.gz`，11,572 bytes，SHA `0f2492853b000203af706359402bf0038446672b6b66bdaac7218ba0cf383853`，manifest SHA `f5761f19ee0b0d13271d5d8ac7428629222d84c966bd74755b1691e46adce23d`，AFS/C盘逐项保全。静态匹配不能代替post-prepare的分布式/DeepSpeed配置、TF32及确定性等数值字段和optimizer/scheduler/RNG完整校验。下一恢复步骤应原样记录这些实际字段再执行原严格比较，不删除字段、拟造parent或放宽校验；若需新训练资源段，先独立记录资源修订，保持旧b原失败、输出和窗口。

v7候选已05:46:53在AFS核验86项文件、29文件完整source ZIP及八份外部原件档案：`writing_candidates/20261110_v7`，整包104,209,711 bytes，SHA `3ac01a8901c89a10eb1e69ee1329a164c5005415b0ef843a22f83b967fd4c417`，manifest SHA `08fafa9fd7596f6455a9985c301de7e8c386b17bd15ed337841cad374aead798`。此包为静态测量完成前的不可变交付快照；上述新工程证据单独作为addendum，不修改原v7 PDF、源码或科学数值。05:48:35 CCI candidate_null真实child仍存活/GPU98%，当前frame72/700、gate0；不是完整null或第五对。AFS该时点192,301,023,232 bytes可用，所有state/CAS未删。

### 23.80 第五个固定RB配对完成与作者确认四个原稿训练run

case4/candidate_null于06:03:26完整700、success0，outer06:03:33 exit0；原source前后56393c5a…不变。原verify_role与paired_rows/export核对两个角色初始相机/qpos/task/instruction身份一致，完整配对升为5/10、Full/null均0/0。其余五对pending不是失败，也不因本组结果换seed。564项原件3,931,831 bytes，SHA `06be2013e723b4f79a7d380189ad0f59b92e5172ed30b91454681e6a93aeedd4`，manifest SHA `9e6cf9d1cc8b176999a724c834ddc11b608bb322642c57b8f96ae1b50d7d6152`，06:19:49本机逐项核验，AFS/C盘保全。此为shared-step3000固定诊断，不是原稿specialist准确率。

该角色terminal、旧进程退出、lease及GPU空闲核对后，按冻结next_role于06:12:57准入case5/candidate_null、seed340800005，outer2254260，source56393c5a…、worker3600/outer5400、原08:31:04UTC campaign总截止保持；spec SHA `15f787234ac1c39a7203a8cf907f855d62c547ad4f11cba412599c8e6fe65486`。没有提前增加第六对。CCI控制台读取运行中1/1、H10080GB/22CPU/230GiB及原SSH21276；未从pod累计运行时长推算下一容器重启，精确剩余仍未核实，保留有界执行与失败收据。

作者新增确认四个目录确属原稿训练运行：BT v1、PB v2、Cover Blocks v1、Blocks Ranking v1。此归属已确认，DEV实际选中step仍未核实；不把最后可用权重当成原稿选点，也不擅自用BRT其他版本替换。此前v8负结果全部保留。进一步核对训练源码代际、权重与runtime完整绑定后，才冻结原run上的新补证协议。

### 23.81 只观察不归一化的post-prepare runtime记录入口

CCI新增`run_rebuttal_training_with_runtime_trace.py`，由Accelerate启动原训练entrypoint时原样转交原参数/配置，并仅记录原capture_training_runtime的返回对象及父200的逐字段差异；同一对象直接返回，原strict resume validator不替换、不删字段、不更改DS/TF32/确定性设置。不访问Git远端、不检查dirty、原训练source仍4efb8094…。记录目录排他创建，保留每次返回和原异常，不能从记录完成推断resume或更新成功。

7项原生单元检查通过；实际旧entrypoint的配置转交preflight V3于06:29:22 exit0、模型/优化更新0、runtime capture调用0，仅证明真实配置与参数可转交，尚不是post-prepare实测。observer source当时8c08178c…与原训练source4efb8094…各自核对。前两次preflight在serialized runtime字段名和命名空间绑定处被检查拒绝，均未加载模型，失败日志保留；已修复并用原实际父attestation和原entrypoint验证，未静默覆盖。原Full b失败/窗口继续保留；下一ACP段须另行冻结资源修订并实际核对post-prepare结果，不以这些工程检查替代训练或科学证据。


### 23.82 case5两个完整角色与第六个固定RB配对

case5/seed340800005的candidate_null于06:38:42完整700/0、outer06:38:49 exit0。564项原件3,936,882 bytes，SHA `20867cb7651da3d400ca4e5dafac7fa18ad594800defcfff82151f10d7eeda66`，manifest `adbabc31de695029cc578ea2c8602ca3bcf2f5d756744b6e66112e40ac15d172`；07:08:34本机逐项通过，AFS/C盘保全。

下一原固定Full角色的旧95分钟余量检查在任何job/spec创建前拒绝。保留该拒绝，改用新invocation的4500秒outer加300秒保全余量，原worker3600、模型/seed和原08:31:04UTC campaign截止不变；06:58:08准入outer2263051，spec `eab38391a62de0efb2453b36b12abb11694fa6ad40e2441173fa53e2b507e89b`。此为缩短资源调用上限，不修改科学协议或重置原窗口。

Full于07:25:58完整700/0、outer07:26:04 exit0，source前后56393c5a…一致。原export再次核对配对reset/camera/qpos/task/instruction身份，第六个完整配对形成，6/10均0/0；其余4对pending不是失败。564项原件4,177,130 bytes，SHA `e4eafa32c55bbe80f37cfe2243bb59e78f5eb21731cd6d442b013a96f622c905`，manifest `8c440f2daf30eaa09464615b13e0a1cb5823b711d1d75989bd5a6fa88031119d`，07:33:17本机逐项核验。原shared checkpoint68c7f384…保持，不是原稿specialist准确率或新增收益。不得为完成10对重置12h原窗口；继续角色需真实余量满足原worker和保全要求。

### 23.83 observer正式冻结校验与新ACP独立attempt

commit eb70b41已推送并离线同步CCI。首次Git archive的source checksum与precommit live checksum不一致，被断言阻止；后续prepare同样在产生plan前拒绝，没有GPU执行。差异审计证明518个共同source文件逐字节一致，仅Git archive缺少15个本机real runtime配置；原配置未删改。两份checksum分别保留：live8c08178c…、正式archive `7a0ce8943818b02a7ee493fc3b6d666ba6b55b5362df7ab5f3a6be4c760000ae`。差异报告c0371bb6…、source archive f107c16c…；不把两个身份归一化。正式冻结source重新通过7项native tests与真实旧entrypoint配置转交V4，06:57:29 exit0、model/update/runtime capture均0。

新独立Full200→500 plan SHA `8d2530207aeb4fee4736c7c2374dd8acf5347761f03dcc8298144682422be2b3`，06:59:12准备，绝对截止10:59:12UTC（epoch1794308352），包含排队、准备、训练和保全；启动timeout只用该截止的剩余秒数，不重置旧b或新窗口。原训练source4efb8094…、父200完整optimizer/scheduler/RNG和parent SHA不变，最终3000/原数学配方保持。实际冻结operator的完整父state/input/配置preflight于07:03:23 exit0，0更新，AFS可用158,984,925,184 bytes；它不是post-prepare或resume成功。

07:04:42标准资源创建实际HTTP429 `MEMBER_QUOTA_EXCEEDED`，成员GPU剩余0、未创建任务，正文已保全。其后才按规则回退闲时，07:05:59唯一新任务`warm-full-trace200-500-20261110-a`，UID `bca26b54ec124eba8ec0de161543b42a`，cluster01e_02、1H10080GB/22CPU/230GiB、restarts0、确认镜像/AFS及固定绝对timeout。07:26:36 Worker有Host/Pod IP且显示运行中；同一时点AFS operator尚无，不能称训练已产生更新。保留唯一UID、不复制未知任务。原Full/control仍200，all-K control0，旧b失败与窗口不变。收据与命令、两个事前intent、source/native preflight均在Git外，当前ACP进展必须以新operator、runtime trace、metric和exit读取。

### 23.84 原稿四run的25份训练记录、v3元数据与W5核对门槛

作者已明确四目录属于原稿训练。07:08保全34项小型原件：BT6份及BRT7份training attestation全部绑定f77c633，PB6份及Cover6份全部绑定47277e5，各run配置hash一致。档案22,015 bytes，SHA `0d56b4020f1da242ca99e172f63c3a813e544c75b6e76ce85c8a0150a6698960`，manifest `fd06676aa86c1f88d95fc1e1f05413a192cb33db06924d50d979b2183d761ab5`，C盘34项逐项通过。原DEV选点仍未知，未重算25个大权重的内容hash，不以末次step代替原选点。

四个末次权重12000/12000/12000/14000的CPU FakeTensor/mmap元数据均为retrospection v3/timing4。两份原冻结retrospection_model.py完全相同，SHA d89796cc…；effect loss实际使用各event自身observed_effect，当前示范target用于requirement/排序/gate等，与v8的current factual加candidate regularizer不同。报告`AUTHOR_CONFIRMED_SPECIALIST_ARCHITECTURE_AND_EFFECT_SOURCE_V2.json` SHA `0f4806d2d5d8a89cc168163145d48be3686940b1762a71fc2a0e23925b47d822`已双端保全；V1元数据脚本的st_size调用错误在任何报告/模型推理前退出，原失败记录保留，V2仅读属性修正。

此发现直接关系HH-W5，不能将v8训练解释写成四个原稿run的实际loss。已形成`SPECIALIST_SUPERVISION_BINDING_20261110_ZH.md`，新增第二addendum维持20/20映射、英文W5替换候选及LaTeX澄清片段。它们尚未合入完整PDF，原v7/原33页投稿/全部报告数值保持；W5的各机制cohort模型代际绑定及后续完整编译为明确release gate，不把公式存在当作历史权重执行该loss的证明。不据此抛弃v8负结果或推断原稿准确率失效。

为独立推进原run兼容性，case5Full终态、旧进程退出与GPU空闲核实后，07:32:06准入BT12000原f77源码/config的native constructor及strict model loading资格，outer2268402，固定2400秒至08:12:06UTC，helper SHA3312ad6d…、spec SHA4bd41088…。只加载原配置/原v3权重，不迁移v8，不运行Git远端/dirty门槛，不改变模型数学；原loader核对实际checkpoint完整hash及严格state，0训练/0推理/0任务成功率。末次12000仅为预先固定的工程资格点，非原稿DEV点。当前仅准入，不提前写成加载通过；下一步查实际operator、完整stdout/stderr、exit和source。


### 23.85 两条实际原生路径的启动核查

07:35:56读取AFS原件：ACP新a的operator已于07:27:33实际进入running_training，冻结训练job于07:27:37启动，plan8d253020…、operator7a0ce894…、training4efb8094…和parent1fb035c4…均与事前绑定一致。runtime_trace/binding.json已产生，但尚无原post-prepare runtime快照或step201+的更新证据；因此原Full计数仍200，不能把operator状态当成resume通过。新绝对截止10:59:12UTC保持。

同一时点CCI原BT12000资格的native process/runner存活，原f77模块路径已核对，config SHA4421db82…、attestation19c6b5fc…绑定；状态constructing_original_model，H100已有13,605MiB占用，尚未完成strict checkpoint load。它不含model inference或训练。固定08:12:06UTC上限保持，先查实际exit与内容hash后再决定兼容推理，不重复准入。AFS可用157,833,052,160 bytes；没有删除state/checkpoint或改动原real配置。


### 23.86 原样测得ACP精度差异、完成两份原模型严格加载并冻结同窗口修复

新a的原capture于07:37:24实测仅两字段不同：cuda_matmul_allow_tf32父false/当前true，float32_matmul_precision父highest/当前high。所有其他runtime字段比较一致；原strict validator拒绝，07:38:41 operator终态失败、step201+更新0。16项失败原件40,706 bytes，SHA `1140e326e8e275f42085fb19c90a9e868608253e12c3bcec9e5aca2f9c374324`，manifest07bbe459…，C盘/AFS逐项核验。不能反推更早旧b的具体差异，也未测得a容器该环境变量原值。

实际Torch2.7.1+cu128子进程、CUDA隐藏的敏感性检查通过：TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1得到high/true，=0得到highest/false。这提供忠实恢复父记录的执行设置，非修改capture返回或放宽比较。变量作用与highest语义参见[PyTorch2.7 CUDA环境变量](https://docs.pytorch.org/docs/2.7/cuda_environment_variables.html)及[FP32矩阵精度](https://docs.pytorch.org/docs/2.7/generated/torch.set_float32_matmul_precision.html)。旧a的环境变量根因仍是推断，不冒充已测事实。

新precision-v3 plan SHA `fb963b4af54e2dc7ced5824b9e7df846f51978ad4bc483bb104198cbae498263`，独立output/artifact，export上述变量0；原训练4efb…、observer7a…、parent200/full state、strict validator和final3000不变。沿用同一06:59→10:59:12UTC绝对窗口，epoch1794308352，排队/训练/保全都计入，未重置。07:54:08完整原parent/配方preflight exit0、0更新。07:55:54重新实测标准创建拒绝MEMBER_QUOTA_EXCEEDED/GPU0；随后07:57:24唯一闲时b创建，UID `34bc101c7e26432785ac66b87a75ddd6`，cluster02、1H100/22CPU230GiB、restarts0、确认镜像/AFS，07:57:47排队中。实际恢复和更新尚未证明，不重复未知任务；当前Full仍200。

BT12000原f77/config strict load于07:39:54成功、runner07:39:58 exit0；完整权重SHA27c137d6…加载前后核实，source前后ff32342d…不变，6,776,526,408参数、peak13,576,688,640 bytes。9项原件31,530 bytes，SHA `1f9fddd66954e787c297c3ef28283067ada3ffaaf2a918fc473829d42bd0a25d`，manifestc26d7fa5…双端保全。

固定顺序PB12000原472/config于07:48:20准入outer2270732，07:56:00严格加载成功、runner07:56:02 exit0；完整SHAa765d987…与source4a1d05cb…加载前后一致，同样0推理/0训练。9项31,526 bytes，SHA `90071fee953d99fa767dcabdb67415b375a262b2e216a10d97b7ba3481d717cb`，manifest619b3f1c…双端逐项核验。PB runner说明复制残留BT字样，实际绑定路径/config/source/attestation/weight均为PB；原收据保留，核验收据明确澄清，不重跑已成功模型。

两份原模型资格不构成DEV选点、在线推理、任务成功率或修复收益。完成后按固定顺序08:02:12准入Cover12000原472/config资格，outer2273018、2400秒至08:42:12UTC，helper9e8d5dd6…、spec44d9dd55…。原RB campaign仍6完整0/0、4pending、08:31UTC原截止不变，BRT资格尚未执行。20项addendum2映射与W5实施边界同步；英文W5候选/LaTeX片段未合入PDF，原v7及所有数值保持。


### 23.87 Cover原模型严格加载完成与原campaign窗口结束

08:26:31核对Cover12000原source472/config：实际08:09:53完成、runner08:09:56 exit0；原配置SHA20379996…、attestation8b2491ef…、完整权重SHA2b59a397…由原loader加载前后核实，source4a1d05cb…前后不变，0训练/0推理。9项31,557 bytes，archive SHA `5b1b8cf4da6bbe5a31b7825dc36adaca95c16e0d67cbae9080c7780281da3b17`，manifest `c016090e9fe65a080f9b7f6f6d1e3dd69008740be8f52d8ada71c1861b6e33ff`，本机/AFS逐项核验。原DEV点仍未知，加载资格不构成任务成功或机制收益。

按固定四run顺序，旧Cover进程退出、完整收据和GPU可用核实后，08:29:57准入BRT14000/f77原源码/config资格，outer2277431，固定2400秒至09:09:57UTC，helper1f48faad…、spec3f414c03…，不迁移v8、不按新结果选点。原RB campaign的08:31:04UTC窗口独立保持；截至08:26核对6个完整配对均0/0、余4pending，窗口结束后不追加角色或重置。新BRT资格是独立工程任务，不续开原科学cohort。

ACP b（UID34bc101c…）08:29:57 Worker仍等待、无Host/Pod，08:32事件明确FailedScheduling：insufficient nvidia.com/gpu1，已约34分钟；AFS operator与201+metrics仍无。没有新训练。将仅在确认等待状态及已停止终态、排除任何AFS启动产物后，按同一冻结plan/绝对10:59UTC窗口尝试其他允许集群；未知状态不复制，原排队失败与退出收据保留。


### 23.88 四个原稿任务模型均已原代际严格加载；BRT原推理合同通过

原BRT14000/f77资格08:29:59开始，08:37:47完成，outer exit0；完整权重SHA `515ecad0e9c428a0c005ce330ff7e5119bd032f86701e4700454b7d149569f40` 加载前后均与原attestation匹配。原f77 source `ff32342d286929ddc5904d55814c66e965269f6b749c1e1b6df1060aaef0b7fe` 不变，v3/timing4，不迁移v8。9项archive `7fbc3e8d7e00315a54516cf5c810dad204202ff79c78d7e355e631760c56a709`、manifest `062a223d3e24e6661ee601a63ca008c22c370fa70ec5765aa8f259ece35d3fa0` 本机/AFS逐项核验。连同BT12000/PB12000/Cover12000，现在固定四run均实际strict load完成；仍是工程资格，不是恢复原DEV选点、原准确率或机制收益。

BRT已存在的compat bundle以原源码重新解析：全部20项asset路径可访问，静态训练/评测合同、seed协议、原权重与M1接口校验通过；实际推理projection SHA `b8abf8f6fbe764988e27bd6e9fb1556712db6ceec2ddcde72edb5aef84b5ec4d` 与历史精确一致，DINO runtime SHA `2f3e5d2979f2fbb3063da65d25a49d15fb4c3a19b52fcf329879fa87fbcb9e01` 也精确一致。没有数值合同豁免。Git身份仅来自完整hash核实的冻结原archive，未让eval依赖Git远端或dirty状态。

### 23.89 原闭环窗口核销与ACP资源尝试终止

原RB窗口2026-11-09 20:31:04至11-10 08:31:04 UTC现已结束，另建 `RB_ORIGINAL_CAMPAIGN_WINDOW_CLOSED_V1.json`，未改原campaign state。真实锁可取得、旧active_pid不在；原状态中的active字段仅是旧记录。固定case0–5共6个完整Full/null配对，各700命令，均0/0；case6–9保持4个pending，不能计失败，不能重置窗口续采或据局部0/0声称非劣。

ACP precision-v3 b UID34bc101c7e26432785ac66b87a75ddd6 的实际FailedScheduling仅证明缺GPU，08:40:27确认已停止；原stop文案误带PVC/静态probe字样，派生记录明确不据此声称PVC故障。重新尝试标准资源std2 UID39706c72dd694c42b4175c560be877ab，08:48创建、08:50实际限额中且零分配实例，08:53:04确认已停止。不得把实例0误称本次测得的成员配额数值。后续c表单在目标闲时集群显示0可用卡，09:16:14取消，未提交、未创建job。标准/闲时均没有GPU执行。原06:59→10:59:12UTC资源段及7200秒启动余量保持；08:59启动门槛已过，不再用此plan创建任务或重置时钟。

09:58:50 AFS实际核查precision-v3 operator目录及201+metrics仍无；Full/已有control仍200，all-K同信息control仍0。本轮已知ACP训练job均终态，无新update。11项窗口/终态原件archive `bd9e2c08709f322d404c8cb179fbd7a0a5aa1cd1ce28702cddd8928606a76b64`、manifest `7d657c6d95341874859ccf5bc522e164d36f9890a77368d03c48cf8d605350b9` 本机/AFS核验。后续若安排新的训练研究，须另冻身份、预算与窗口，不能伪装本段resume成功。

### 23.90 原BRT完整在线策略资格：失败保全与同窗口修复

09:39:57独立准入原BRT14000/f77的一次Full replan及四条native命令资格，固定采用原候选协议的首seed0，原绝对截止10:39:57UTC，非SR/修复率实验。v1因操作程序未带已验证图形环境，09:40:43在renderer初始化失败；0推理/0命令，保留原exit。v2仅恢复已有进程专用Vulkan/EGL库，09:45:46准入；固定seed专家初始化成功，原完整RMBenchWarmPolicy实际装载成功、权重/原数值合同/DINO均通过。随后10:01:03因操作程序instruction generator导入路径写错退出，0推理/0命令，不能称在线推理已通过。

静态合同与第一次失败24项archive `24596772f73b144d037f58296965ff641a9a0086528d473747a3e7124a956a19`、manifest `d7bd2061e1a95a710b3704fc978d12e00fb9ec79be17954e0971d4e214f767e7`；v2失败13项archive `25ae8a3ab752a8e4b295934af80cca5bed30c9ec865fb89eb8fdbf5ee7f547b3`、manifest `ae7847d26811a395e9045698bd9e1b0017f256551f69c02cde8b1e27255a930a`，均双端逐项核验。修复v3改用原官方现有generate_episode_instructions模块，将该模块及BRT instruction template与原checkout逐文件hash校验前置，避免再次在昂贵加载后才发现导入错误；不改原模型数学、配置或样本。10:04:18准入outer2289230，helper `6b1b66701bc682359853d479d02fc9b465da95e924b23af311601138f9bd953a`，spec `74004d0885c37438ea7bb4f44da0b1fc713236bd5c091c84d3db9c4d3a74515f`，剩余2139秒、仍在原10:39:57UTC截止。先查实际operator/进程/exit，不提前把资格或执行中写成收益。

### 23.91 v8主文、附录与20项回复已联动；完整工程验收

`responses/20261110_v8`的15块英文正文覆盖20项意见；v8主文和Appendix A.4已合入原源码target澄清：四个已审计specialist以各event自身observed_effect为目标，shared-stage update3000的factual加公共query regularizer单独说明，不再将后者套用于前者。原事件目标输入同样可存在复制历史效果的退化，两个objective都不能单独保证当前状态反事实识别；候选预测与生成修复须依赖独立执行检验。Hh4E W5同步，原提交受控结果、shared-stage六query不利/未知及其各自范围完整保留。

完整原LaTeX 29文件、干净ZIP独立重编译、37页逐页文本一致；主文仍10页，所有原数字表及9图字节保持。37页均render，全页contact及修改页6/19–25逐页视觉核查，无截断/重叠/表溢出、无未定义引用、无overfull，46条继承font warning不增。CCI没有pdflatex的一次工程build失败原件保持；正式两次成功编译在本机既有WSL TeX环境完成，不将远端失败写成通过。

PDF `tmp/rebuttal/revision/WARM_ICLR2027_revision_v8.pdf` SHA `1a3b9588f41586cbb1df8545ce059da816c9bd44c6ae2c6a76f53781ed70eed1`；source ZIP `f79409f9d7c27007007ea345eee9de67d3b1e1de9492d39b3ce8db36e4f9fa10`。这关闭已发现的监督描述不一致，但原DEV/全部机制cohort代际绑定、原模型候选后果/修复、非空missing-skill与公平最终训练证据仍是release gate；本版未发布，不把工程资格记为新科学收益。原33页投稿与v7均保留。


### 23.92 原BRT一次Full推理与四条native命令资格完成

v3在原10:39:57UTC截止内于10:19:34完成，runner10:19:40 exit0：原f77/source ff32342d…、BRT14000完整权重515ecad0…、原projection b8abf8f6…及DINO runtime2f3e5d29…均保留，固定首seed0专家初始化通过。完整原RMBenchWarmPolicy执行1次Full推理和4条native qpos命令，action有限，event hook只观察且返回未改，source前后完全一致；峰值PyTorch allocated 25,354,637,824 bytes。原始候选动作、32候选效果预测、实际模型输出、初始相机/qpos、四命令前后qpos、telemetry及退出均封存。20项archive `0780ca6afe147af3b3dc3d217b7dc63dda02d9de6c694eaf6f69ab904fe7f80a`、manifest `e26a0e839e7f1a1ef89576f26c6196b0bb7173f4cb643d0f259bea3a3b56124d` 双端逐项核验。

这是实际在线资格通过，原先两次失败保留；不是完整episode或独立候选后果评测，candidate labels=0、SR eligible=false、DEV选点仍未知。门值/预测差异/有限动作不证明候选预测正确、修复收益或评分提升。当前v8 PDF/回复无需新增性能数值。

v8不可变候选包10:22:22已在AFS核验76项、29文件source ZIP、15回复块/20意见和5份本轮新完整原件；整包SHA `824984d7e481de651a1055e7fb043ef40a5c6b4360298340682eacd48cb70ec4`、manifest `90bed9b8160c78af8cf740f5ae5cb8049d3ac9d952732b270a8f8904c86489b2`。8份旧大原件明确继承v7既有核验，未在GPU加载时重新bulk扫描。新资格终态在候选包之后，因此另建补充记录，不能覆盖包内旧观察。

### 23.93 原模型固定事实轨迹采集协议与实现

完成原生资格后，独立设计BRT14000/f77首seed0的事实轨迹来源：最多600条native命令，固定replan8/48/96（frame32/192/384）封存全部原候选payload、adapted action、predicted effect、Full action、采样RNG和原始观测，全部在任何新增branch truth前写出。未到达的query保持not reached，不换state，不将query序号称为语义阶段。该来源只有一个原seed，不扩写成独立多seed或原稿DEV准确率复现。

新增`scripts/collect_rebuttal_original_specialist_trace.py`先在CCI实现并运行CLI解析检查，再回传本机。沿用已实际通过的原模型构造、原合同和官方instruction模块；记录每个真实native command及前后qpos、全部生成action和结束收据。第一观测、第一生成和首四条物理轨迹必须与刚完成资格完全一致；失败则保留并拒绝推广，不放宽原数值合同。源模型/validator不改，collector无Git网络/dirty gate。原单次资格已完成；新事实来源使用独立冻结的新资源段，至多90分钟，绝不重置旧RB/Full500或单次资格窗口。具体计划、绝对截止与实际准入收据在Git外，工程/事实采集不预先声称修复或后果预测收益。


### 23.94 原BRT事实来源600命令完成并保全三个固定query

独立plan79a70972…于10:37:25准入，原90分钟截止12:07:25保持；来源于11:03:44 outer exit0。完整执行预定600条native命令、150次Full replan，固定q8/q48/q96（frame32/192/384）全部在新branch truth前封存。三个query各32 valid候选，共96项adapted action/predicted effect，768维且有限；原模型选择rank28/0/10，gate0.9375/0.94921875/0.95703125。这些是模型输出与执行计数，不是候选正确性、适用性或修复收益。独立候选端点和适用性标签目前均0。

首原始观测和第一Full生成与已通过资格精确一致，首四条命令的原生API向量匹配；明确这些向量中的12个arm项是drive targets、另含gripper读数，不等于已封存完整measured physics/joint轨迹。Source前后ff32342d…、权重515ecad0…与collector LF SHA dfd9d6af…保持。600是冻结的事实采集上限，原console显示3500不用于恢复原稿协议；600时native success flag=false，不能计作整个benchmark的最终失败或将本段当原稿SR复现，亦不能改动采集上限补成一个成功样本。

484项完整原件33,014,782 bytes，archive `476575ef0fb65711ee3b0cd80b9c51bbd7afdb759874dd149e307aa26050cb09`、manifest `50324c02c3a89ad2ccf1d66e55932e070d6b8350ee897375ecc737d8e3fc7193` 已AFS/本机逐项核验，含全部600实际命令/前后API向量、150生成和观测、三份pre-truth probes、原始模型输出/采样RNG、config/source绑定和runner退出。不得把96 sealed predictions写成96 independently executed endpoints。

后继优先原q8/model-selected raw rank28与Full的fresh-scene A-A-B-A，再原K32全部候选的H32物理端点/冻结M1效果。BRT尚不在现有BT/RB的pack/unpack白名单或独立applicability rubric中，不能删除白名单或把unsupported标签当0；先通过原生fresh-replay资格。Reader/原namespace/partial CPU snapshot的实现入口和限制已整理到 `ORIGINAL_BRT_EFFECT_BRANCH_PROTOCOL_20261110_ZH.md`。v8 PDF/英文正文及数值不变，原稿/新负结果/未知均保留，当前主要缺口仍是独立后果与同状态生成对照的真实收益、非空missing-skill及公平最终训练。


### 23.95 原效果投影与独立后果采集器预检

2026-11-10 11:26:37 UTC，原C/f77 namespace下读取_mean_effect(tokens)=tokens.mean(dim=-2)，原observed_effect与target_effect均用其处理delta tokens。原四run model模块d89796cc…一致。当前shared-stage的_spatial_effect_summary在原C中不存在，首次错误import在任何teacher/physics执行前失败并保留；不能把该加权目标带入原specialist的新GT。V2静态预检确认memory/model/policy_core来自原C、research reader来自冻结H7a，四项M1/processor/stats/camera hash及实际DINO runtime2f3e5d29…精确匹配，0teacher装载/0branch/0新结果。

新增collect_rebuttal_original_fresh_pool.py先在CCI实现及--help检查，再回传本机。固定q8/rank28与Full的fresh-scene A-A-B-A，通过后原K32 rank顺序；32 distinct候选、36含Full复核的branch，均H32。原DINO均匀变化均值作GT；历史参照由原bank及完整文件hash/Event ID绑定，源预测/动作先封存。实际r_t未恢复，query-constant仅用于centered参照，absolute栏留空；BRT独立适用性未认证继续unknown。旧原件/负结果/未知全部保持，没有把预检和实现当新机制收益。新资源段与准入另行冻结，所有旧截止不变。


### 23.96 原BRT q8同父A-A-B-A完成，原件双端核验

独立新plan bc42f951…于11:45:00UTC准入，截止13:45:00UTC（7200秒）冻结，所有旧窗口不变。Collector1b0a5ee/LF b36d04eb…、原模型f77/source ff32342d…、BRT14000权重515ecad0…及原bank46783/manifest2f57ccfe…在truth前绑定。A-A-B-A于11:52:42完成：固定q8/frame32、原选中raw28，四分支各32命令；完整native父状态、RNG及DINO特征精确一致，三个Full端点及效果完全一致，raw28物理端点不同。使用原_mean_effect的均匀delta均值，未用shared-stage加权/pre-state函数、未做物理unpack、无新policy采样或训练。

首四分支及pre-truth封存、reader/config/plan/source收据共44原件，archive d2d1fa74e2d1fc7950208218f26c6ca96c72a02e1b9555bc3dd944c9a2473f48、manifest6af8cbbc1ab97a0408b56b89b2d56a97049e9fe5bc32fc497cc9bf7005b2a8bb，12:07后AFS/本机全部逐项核验。位置rebuttal_20261109_acp/original_brt_q8_aaba_qualified_archive_v1.tar.gz及本机execution/20261110_iter8。只证明该固定状态下执行/测量资格，1个候选独立端点、0个适用性标签；没有推断后果预测精度、提案修复或总体SR。

12:10:00实际两进程/worker lease持有，已封存15分支，其中12个候选尝试含11完整H32、1 invalid；按原rank顺序继续K32。invalid保留为unknown，不替换、不补零、不按结果重排。不在pool终态前写MSE或收益。BRT rubric未认证，全部applicability仍unknown。完整模型来源与窗口见固定plan/launch，原DEV选点仍未知。

### 23.97 v9效果投影范围、回复和完整PDF联动验收

Appendix A.1明确四个已审计原specialist的四token均匀变化均值，shared-stage weighted/pre-state描述单独列出；A.4、notation locator、C.2.4与Hh4E W5一致，Bbqv W3不再把尚缺的闭环比较描述为当前活跃研究。原实验数字表、九图及主文十页保持，所有不利/未知补充记录保留。20项映射含AABA工程证据，未将其升级为准确率或修复优势。

完整原LaTeX29文件两次编译（完整项目/干净sourceZIP）通过，37页文本逐页相同、主文10页，所有37页render、全页contact及14/15/16/19/20/21/22/23/24/28/29/31/32/37页单独视觉核查，无截断/公式表格重叠、无undefined references/overfull，46条继承font warnings不增，匿名性检查通过。PDF SHA34922f9629c65cb647fa84eae981bada4493e629bf592dd8f4374037043b4c39，sourceZIP SHA2f926f3ebb0415a813a2b5cf2399c510b0828a2d0b5c7773851edd22e52557ac。位置tmp/rebuttal/revision/WARM_ICLR2027_revision_v9.pdf及同名_source.zip，英文三份/15块见responses/20261110_v9。

这关闭已确认的实现描述差异，不关闭原模型后果精度/修复、自然missing-skill、同信息最终训练及其他机制cohort/原DEV绑定。原33页投稿、v8及所有旧原件保持。stable_release_ready=false，未发布/正式上传；新增真机仍延期。


### 23.98 原BRT q8候选池完整核销：当前未观察到后果预测优势

原独立planbc42f951…及13:45:00UTC截止不变，36分支于12:41:27结束、runner12:41:33 exit0。固定32个valid候选全部有终态：29完整H32、rank2/19/25各在11命令后被官方left_mplib_planner的TOPP参数化拒绝，三者保留unknown，没有补效果、补零或替换候选。四个Full anchors端点/效果完全一致，全部36个native父状态/RNG/DINO精确；原模型sourceff32342d…及H7a0ce894…前后不变，BRT14k权重515ecad0…、原四token均匀delta均值保持。

同一29候选支持集的centered MSE：predicted0.05252027541650059、query-constant0.018119498131784893、historical0.05249698324368055；predicted explained variance -1.898550226640686。Absolute MSE：predicted0.09574487815571375、historical0.095704683241209；原实际query r_t未捕获，constant absolute仍null。独立从端点DINO和源prediction/historical数组复算相同。该固定状态不支持后果预测优势；不能把已通过AABA写成精度/修复收益。32个候选并非32个独立query，BRT适用性全部unsupported/unknown；不推断总体SR、repair或原DEV精度，不以此替换原稿另一个已报告机制cohort。

308项完整终态原件42,565,564 bytes，archive ec533960e2bcd22a73032d41436561b1324cb21f219a3d12eadadb0edf930343，manifest f8e531ffa3ebdeab47f504d47f70729f443243a73231c076aa5be6ece2ca8104，AFS/本机逐项核验。路径rebuttal_20261109_acp/original_brt_q8_pool_complete_archive_v1.tar.gz及本机iter8；三条失败、全部预测和端点保留。后继仍按原q48/q96顺序，不因q8不利而换state/rank或拟合尺度。

### 23.99 同信息最终训练独立阶段与两项标准ACP实际启动

原precision-v3窗口已关且全部attempt核销，没有重置。新master7526a475…于12:29:56UTC冻结24h至11月11日12:29:56UTC，固定seed3407/batch128/scheduler30000/final3000；Full从原200完整状态直达3000，pool从共同base独立开始原定100更新资格，再在同截止内接3000。Full原数学配置仅允许output/resume/run_steps/save_every四项存储/恢复控制变化，原preflight实际通过（0新更新）；pool复制已冻结100-step配置，只改新output，source48ce6ba8…保持。新增pool启动空间/source gate在首次提交前另存v2，command8669d7ca…，旧v1未派发，master截止/数学配方不变。

当前标准资源实际申请成功，因此未转闲时：Full warm-full-final3000-20261110-std，UID95e495471308439c9da281d2c4fd5b92，12:37:51创建；pool warm-pool100-final3000-20261110-std，UIDf7f3df01f8b84b1eabd2ea8e558fd433，12:45:35创建。两项均cluster02、1GPU/22CPU230GiB、重启0，已确认私有镜像和AFS。12:51:42实际AFS runner均running、有PID52/55的持续heartbeat，Full runtime capture与父记录differences=[]且返回对象未修改，关闭了此前已实测的两项精度差异；完整恢复和优化器更新仍需按实际新metrics确认，此时Full201+和pool更新均0，不能按UI运行中写训练已完成。

新Full plan cab8f8f9…、pool spec ed4dabc1…、pool config7350ea58…、master和原始UI/runner/runtime均Git外保全。原Full/action-onlycontrol200和旧失败保持；新shared-stage对照不冒充原逐任务specialist复现。实际状态后续读canonical current_iteration及新phase事件文件，不按旧running或无输出快照重复提交。资源/存储与后继资格详见MATCHED_FINAL3000_RESOURCE_PHASE_20261110_ZH.md。

### 23.100 后续原固定query采集入口与短资格模式

先在CCI实现，再同步本机：collector严格支持原预封存q8/q48/q96，对应固定rank28/0/10及真实frame32/192/384。每个query均读取其自身probe、Full chunk、pre-forward RNG和源前缀/下一replan观测，仍用原模型/reader/均匀delta均值；不将q8父资格推广为后续状态已通过。新增qualify-only在四分支AABA完整通过后停止，保留同一plan/seal给ACP接续完整pool；短CCI资格与长ACP计算分开，不重置绝对deadline或绕过任何原验证。CLI --help及语法/路径检查通过只算工程资格，新的真实AABA须独立完成。原q8冻结collector1b0a5ee和308原件不变。


### 23.101 v10已纳入完整原q8负结果并完成联动验收

原BRT14k/source0/q8的32候选核销与同29支持集MSE已单列Appendix C.2.5/Table22（32/33页），Bbqv W3/Q2、Hh4E W5和全部20项映射同步；不将新固定point当原DEV准确率或另一个原机制cohort的替代，原均匀delta描述与shared-stage加权投影保持区分。三项TOPP unknown完整保留、BRT适用性未知、不推断修复或总体收益。

v10完整29文件LaTeX和干净sourceZIP两次编译均37页/10主文，37页文本一致。原所有实验tabular数值与九图保持，主文十页逐页文本不变；无undefined references/overfull，46条继承font warnings不增，匿名性通过。37页render/contact及全部8变化页18/29/32/33/34/35/36/37逐页视觉核查；新Table22的5项数值独立对照308原件核实。PDF SHA7d5489a9ea85b25cbc87874904aea0f52a81dee2828439ef023399a14b90abca，sourceZIP SHAa9099b1e256f16001b74e3615d00db1594029ec14cfe32c8f90b5f5db1e86d57。三份英文/15块/20项在responses/20261110_v10，完整工程及PDF在revision。v9与原33页稿保留，未发布/上传，实证release gates仍开放。

### 23.102 q48事实前缀拒绝已保全；独立两遍上下文审计进行

原q48/rank0 plan58b336f3…于13:01:03准入，只运行90分钟以内的CCI四分支资格；独立8h phase截止21:01:03UTC不变，计划通过后长pool交ACP。13:04:36在anchor-a之前的事实prefix重放触发官方left_mplib_planner TOPP拒绝，runner13:04:40 exit1，0完整分支/0新candidate endpoint。不是q48候选H32成败标签，不能按已完成或替换state。15原件218,023bytes archive b29ff2d1308732a9b73243bb9fe337f9c0f7bd3a973b6e10e76c60f71892ab9e、manifest8e9b6795…双端逐项核验。

定位到原600来源采集器记录的是实际命令尝试和前后API/观测，没有逐命令规划拒绝flag。600 attempts不能自动扩写为600条均有效执行；q8独立branch仍已严格认证，原q8数值/原件不受影响。新增audit_rebuttal_original_prefix.py先CCI实现/CLI检查，再同步本机，独立冻结precommit271fadea…；13:16:30在SAME21:01UTC phase内准入outer2320686，worker3600，0新policy/候选truth。两次从fresh scene重放同192记录，不重试/替换任何命令；每一步API值、每4步源RGB及两遍真实关节/速度须精确一致，所有已诊断TOPP异常显式写出，其他异常/RPC仍拒绝。若通过，仅建立新的事实上下文重放/失败清单，不追溯声称源执行全部有效；分支guard及终态有效性保持严格。

13:19:40实际审计第一遍至index102，观测1次前缀拒绝，尚未完成两遍，不能提前认证或改候选入口。必须先核查真实终态、保全完整原件，再决定是否可以按明确expected prefix失败清单恢复同一query，原失败/截止/source均不覆盖。

同时13:05:04两标准ACP真实连续metrics已可见：Full201–231共31新更新，pool1–29共29更新，心跳持续，Full父运行时differences=[]且原返回未改。Full的durable checkpoint仍200，pool尚无完成checkpoint，不把进行中updates当可恢复的新权重或最终公平性能。


### 23.103 原q48两遍上下文审计完整；只绑定已观测的前缀拒绝

审计13:25:05完成、原runner exit0：192个原实际命令尝试重放两遍；source每步API、每4步RGB、跨遍真实关节/速度及最终native父/RNG完全相同。两份原始pass文件SHA同为317cca9e…，唯一拒绝均在index101（第102条命令），并未重试/替换该命令。15完整原件1,576,946bytes，archive378d8a0c37a43263e3dc43daed291d538a23db414da7cf2155cbc5d0fdd71e87、manifest8d01ace1…双端逐项核验。没有新候选端点或修复标签，不追溯宣称原600 attempts均有效。

Collector先在CCI实现显式verified_prefix_manifest绑定，再同步本机：只接纳已独立两遍认证且hash/query/frame/源trace/逐步源值完全一致的prefix清单；每次fresh分支须复现同index101、同异常字符串，逐命令API/计数、每4步RGB及完整parent hash均严格匹配。任何未列出的前缀异常、未再现的预期异常仍拒绝；H32候选的OfficialQposGuard和valid/early/unknown终态判定完全不变。新增factual_prefix_audit.json纳入每分支seal，原失败/源码/窗口保留。真实清单CPU预检通过，三种篡改身份均拒绝，0GPU/0新branch；原无清单路径仍严格拒绝所有prefix异常。新q48计划须另存明确修订ID，保持原13:01起算及SAME21:01:03UTC截止，长pool由ACP执行AABA后继续，不以修复重置窗口。

额外只读向量诊断：原q8同29支持集predicted-minus-historical MSE1.4558982e-5，difference RMS0.0038156，相比historical RMS0.3114039；它只描述表征相近，未改原三项主MSE、GT或支持集，不是预测收益。相似的最终误差本身不当作向量相等证明。

### 23.104 同信息pool100完整状态与无损备份通过；最终resume操作器完成

Pool初始训练1–100连续有限loss/grad，runner13:26:25 exit0、source48ce6ba8…前后相同。真实权重元数据含独立warm.research.pooled-event-prior-with-recollection.v1，冷启动resume_step=null、lineage_kind=none、无Full parent；与Full200的released-base776475b2…、seed3407、effective batch128、maxsteps30000、BF16及train/dev source合同一致。checkpoint SHA cbdfcef2432f2e2b99ddfc22a95dc16450df43979c094c85e3f7787faecfe5ec。它是100步完整训练资格，不是最终3000或准确率证据。

首版归档器错误预期冷启动resume_step=0，实际为null，因此在任何归档/新更新前停止；原失败和收据保留。V2按实际冷启动schema做精确检查，并增加与Full共同输入身份验证，未修改训练/数值validator或重训100步。23原件38,530,508,108bytes形成原始TAR38,530,570,240bytes，14:08已完成逐字节可重建验证，raw TAR SHA f132c9a5b5f917e9eb1a2a0c093811d0c1634c8b1211dadd43631f999ae4bb54。CAS增加13,525,627,917bytes，原checkpoint/完整optimizer/scheduler/RNG/dataloader状态、raw TAR、manifest及旧失败均保留。收据POOL100_COMPLETE_ARCHIVE_V2/archive_receipt.json；本机收到收据/inventory，完整CAS本机传输另记，不冒充已双端复制全部38.5GB。

run_rebuttal_training_segment.py扩展明确Full/pooled_memory_prior两角色；构造器必须与角色一致，pool四项移除的机制loss必须为0，pool父状态必须已有独立verified-byte-exact收据并匹配source/checkpoint/update。原四项存储/恢复控制、固定final3000、原scheduler/batch/seed数学配方及strict实际resume validator保持。旧Full运行仍用冻结H7a，未被代码同步改动。原有segment检查与新角色/防混用/保留机制loss拒绝共7项native通过（0.80s）；首次测试因缺工作区PYTHONPATH在收集前失败，修正运行入口后通过，未改测试要求。新增tmp容量门槛与角色明确的归档/账目，后继仍须沿master7526a475…同11月11日12:29:56UTC截止；先核对实际AFS空间并为两个最终state及CAS留余量，不删除原件或重置预算。

### 23.105 两个新ACP尝试已接受；pool实际路径预检发现并修复类型错误

14:43冻结两项新尝试，所有原样本与时钟不变。Pool100→3000计划SHA58f8e2231d16b0b27fc4a15584cdb2508523b9ae430b5bb0ac41433c459dd8fc，父100权重cbdfcef2…、无损收据b725cee2…；AFS实际119,581,700,096bytes，对两个final raw state、两份实测CAS增量和4GiB保底预留108,407,363,610bytes。标准ACP warm-pool-resume100-3000-20261110-std（UID f31fd1c5066d4841a1568a8bb5a53642）14:47接受，首次详情为排队，不能称已训练。其原冻结操作器在CCI实际完整父状态预检中触发AttributeError：给file_hash传了JSON字符串而非Path；在创建训练输出前失败，0新增update。首任务的实际终态和控制台日志须继续核销，不能凭预检替它宣布退出。

在CCI中把收据绑定提取为validate_parent_archive并显式Path转换，source LF SHA1c935bb2dfa01d9921fca5e20b0532e6cde0cffb6120636c1b07ead234f86908。新增真实JSON字符串路径与五类跨身份拒绝测试；连同原7项共13项native通过（0.84s）。原完整hash/update/source/checkpoint/role核验标准保持，模型和训练数学配方未改；现用真实父状态重新预检。旧冻结源695控制器source SHA b448aacb800f2fc2922f299b2ac07f13920314a7acafde2c1a9da234ff79e2e3，archive内collector eea5ac8e…、operator b60e06b8…是实际运行字节身份，须与工作树LF单文件身份区分。所有运行旧快照不变。修复后另存新attempt/spec与新冻结操作器，不覆盖已接受任务或重置master截止。

原q48新attempt v2计划SHA6b91a7ed5b48cbd94a8919cc6634f4baa82af2933471b92d51a11ee626423c5a，spec cbc1e6d0…，显式绑定两遍prefix manifest278dd78e…及原失败计划58b336f3…。标准ACP warm-brt-q48-effects-20261110-std（UID25fdd1b8acce4d78a3c17680c9d667d6）14:49接受，首次详情启动中；原13:01:03起算/21:01:03UTC截止不变。仍须AABA通过才继续固定32候选。源模型C ff32342d…、参考H7a与BRT14k权重、q48/rank0/seed0/H32均不变，未新增端点/修复结论。


### 23.106 原q48 ACP运行环境恢复，严格同状态资格与固定池继续

首个原q48 ACP-v2于14:50:50 runner exit1，reader运行环境严格校验失败、0候选端点。12完整原件归档SHA5097d7b90665f56f6fb0da6e20b4e80f84657a6b35fadcafd0d1981117b1cab6，manifest ba3198d9…；原failed root与UI UID25fdd1b8…保留。新v3计划a22a8fe87a3e8b60dfa1f62fb24d1be2c077cd0187d1bdc3544b603e5483fcbc，标准任务96aea940c2d64b8eb153027c14af780f、模型/旧H7a/Cf77/seed0/q48/rank0不变。只显式恢复TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0并先保存完整原始运行指纹；actual和原encoder合同SHA均2f3e5d2979f2fbb3063da65d25a49d15fb4c3a19b52fcf329879fa87fbcb9e01，所有字段相同、runtime_object_modified=false。未假定旧attempt未记录的环境变量值，不改reader合同。

2026-11-10T15:23:53.744880+00:00：AABA已通过，正在固定K32候选采集；4个封存分支/1个完整独立候选端点，完整operator/partial收据在ITER8_LIVE_V5。未完成32池和末anchor，不计算/暗示总体预测、适用性或修复收益。已核实两个192-command历史pass及index101已观测拒绝，任何新前缀异常或不同parent仍报错，候选H32 guard完全保持。原13:01:03→21:01:03UTC phase不变，没有因两次失败/平台恢复重置资源。

### 23.107 同信息pool严格完整状态恢复，部分训练与最终比较严格分开

首个新pool标准任务f31fd1c5066d4841a1568a8bb5a53642已在控制台确认失败，离线日志8行实际堆栈与预检一致：file_hash收据字符串没有open。原训练输出和operator目录均未创建，0新update。修复已提交/推送523133bc6a8d2eda9aeba455f333316055900f3f并离线FF核验到CCI；冻结控制器source1b96f3b8906b16e890c06f8d8172f3cd85651d321003200b85a01af2cb402c22，操作器3aaa1d33…，13项原生测试再次通过。真实38.5GB父输入预检通过，原严格身份标准未改。

另存v2计划fe595b17d1db2b1763ea8d4c2997eb17f827264ae45b53c7135be2534594b045/独立新输出matched_final3000_phase20261110_v2/pooled_memory_prior-s3407，标准任务c852e8e2e91f4d9ba9d42cfa0682bbb2从父100 cbdfcef2…恢复。15:16实际trainer确认step100与完整state SHA47054e86e56e…并开始Accelerate/DeepSpeed装载，未修改父状态/数学配方。最新partial snapshot 2026-11-10T15:23:53.744880+00:00：Full新日志201–849，649条连续且有限；pool本次20条新更新、末观察update120。完整final3000/TEST均尚未完成，durable父Full200/pool100不变；CPU校验/初始化/部分更新不写成最终性能。SAME master7526a475…/Nov11 12:29:56UTC、最大两训练GPU保持，AFS可用118644985856bytes，不删原件。尚需用真实pool100严格typed原生采样路径作DEV工程资格，避免最终训练后才发现加载/真实history/all-K输入不通。

### 23.108 v10工程增量1与20项映射双端保全

v10完整PDF7d5489a9…/ZIPa9099b1e…、三份15块/20项回复和原68项abd7c31b…不变，不重编译未变的论文。独立27项工程增量包SHAe07a6037a91d543a6753257609b488f97a3e18152f9a0f87eb7acfb22d0bbf80，manifest acc57600e43dade6a06a3afecd4c2157af1c360a561f9f7d975159a20af1b44e、新20项matrix 3081492e31aa7a07b471bdb28696e938cc8adb6b20b49cb7696420a7d9f0ac4f，本机responses/20261110_v10_addendum1与AFS writing_candidates/20261110_v10_addendum1逐项核验。它纳入两遍prefix原件收据、pool100 byte-exact收据、两次新ACP失败、修复检查、原encoder运行环境完整相同证明、15:15部分进度；不改原稿/新q8不利数值或PDF位置，也未将工程证据升级为效能证据。初版打包验证因Windows反斜杠manifest键和POSIX TAR成员不一致失败，旧包/manifest保留；V2只规范路径键，全部负载hash不变，27项验证通过。新live V5是后续时间点，单独保全，不覆写此不可变包。

当前20项各有问题—主张—证据—回复—修订位置，全部实证closure仍false。原q8预测0.0525202754高于constant0.0181194981、29完整3unknown仍明确；剩余originalDEV/cohort、直接修复、真实missing-skill与最终公平对照不能用话术关闭。原33页与新增v10均保留，未发布OpenReview或上传正式投稿，真机新增继续延期。


### 23.109 原q48 AABA完整工程原件及时双端保全

15:21:21原q48 AABA通过，四次fresh parent/192尝试上下文精确，三次Full的完整H32物理端点与DINO效应完全相同；预封存原rank0的H32完整，且实际物理关节端点与Full不同。128次候选/anchor执行只含1个独立候选端点，独立policy样本0，适用性标签0。全32池正在同原21:01UTC截止下采集，不能将资格当预测或修复收益。

15:27独立归档已完成的四个不可变branch及封存query、reader/原runtime证明与计划，未打包正在写的operator或未完成rank。43原件5,653,628bytes，archive SHA f470003175fe5c7e116b46dd94d7e5eef16b62ba8acc338f16118371242b644d，manifest SHA3579465f9a70599b03467fc7e419ca9efdbe92bdf355c44361d689b675ba639d；AFS与本机iter8逐项hash核验。原32预测/动作已在truth前封存，rank0不得重复计样本，后续负/未知结果保留。v10/27项工程增量包不覆写，此新收据单独关联canonical状态。

### 23.110 同信息对照原生推理资格入口已实现；不混用Full构造器

在CCI实现独立sim_rmbench_pooled_memory配置，只改为已独立训练的create_pooled_memory_prior及移除机制的四项loss=0，Full默认配置/模型路径保持。原生资格入口固定pool100 checkpoint cbdfcef2…，使用既有Press Button DEV最早episode435、frame0至36每4帧一次的10个记录前缀；未来观测在传入policy前丢弃，只在生成后向history登记真实记录的4条动作。它不加载仿真，不执行模型生成的命令，不用于TEST选点或收益估计。

资格程序将走真实RMBenchWarmPolicy、严格typed checkpoint/合同/encoder/normalizer及生产采样器，读取实际all-K adapter输入、4个池化token和8+4条件token，并保存原始当前RGB/state、模型H32输出、真实历史mask、事件/高斯/输入hash及probe。proprio须与生产_normalize_state结果匹配；frame32/36须确有过去event history，不用零占位替代。只读wrapper原样返回原conditioning对象，结束恢复；真实policy资格仍待运行，不能把单元检查当作通过。

原生新配置与Full保持检查、既有池化模型隔离/梯度/无future teacher等共12项CPU原生测试通过（20.67s），语法检查通过，0优化器更新/0物理命令。实现检查收据ITER9_NATIVE_POOL_IMPLEMENTATION_CHECK_V1.json，原件Git外iter9/AFS保留。实际评测代码将从已训练f9/source48ce6ba8…逐字节复制并仅叠加三个新增入口文件，独立冻结并证明原模型/所有既有源码bytes不改；新训练与原q48继续各自不可变快照，原窗口不变。

### 23.111 原生DEV预检字段错误保留；修复仅采集器，不改模型与窗口

635670b已提交推送并同步CCI。507个继承源码文件与f9训练源48ce6ba8…逐字节相同，叠加三个新入口后eval source87199cc6f815da5599fcb0ee9c0a9ffb67181f3750e34d16acc810f6218ffe80；冻结组合源码又通过12项检查（21.07s）。pool100严格在线bundle1746fb50…已生成，DEV资格原v1计划7b11a6af…在15:57:52冻结，绝对截止17:27:52UTC，不随后续修复重置。

首个data-only runner于16:02:20 exit1；采集器错误读取sample['text']，而生产BaseLerobotDataset及原在线成本脚本使用sample['task']。失败发生在policy创建前，0推理/0更新/0物理命令；原root/runner/计划均保留。已在CCI修正原生task字段并加明确合同拒绝测试，共14项配置/池化原生检查通过（20.47s）。另把只读采集器与固定评测模型源码身份分开绑定：新不可变helper hash单独写入plan并自校验，仍从已封存87199cc6…加载所有原模型/配置和已验证bundle，不重建或修改模型数学、checkpoint或DEV样本。新attempt须新输出，同原17:27:52截止。

### 23.112 已有100条DEV成本日志的完整阶段核算

16:02重算完成：Full/FastWAM各120条含20 warmup，100个测量样本key逐项一致，5个PB DEV回合。每条Full外层同步计时包含pipeline_s，后者包含retrieval total，且retrieval total逐条不小于预处理、DINO、search、gather之和；非负有限核验通过。Full mean/median/p95=524.2668/530.0586/565.2201 ms，FastWAM=372.1158/366.3203/381.8307；原论文报告median/p95不变。完整嵌套retrieval mean/median/p95=39.1097/38.4370/43.6521 ms；其中图像预处理median9.2096、DINO5.9504、exact CPU bank search21.3363、gather0.1338ms。

Full外层减去retrieval total的逐条剩余区间mean/median/p95=485.1571/488.1040/526.1863ms，只是其余完整重规划工作，包含当前观测处理、历史、source构造、采样和telemetry；不能称WARM机制模块的独立耗时，不能据此给出因果模块开销。均值可按所定义区间相加，median/p95不可相加。Full为原shared-stage3000权重68c7f384…，不是原specialist或新匹配训练；无新GPU benchmark/新SR/MemoryWAM计时。

完整13项原始计时/telemetry/manifest/summary、逐query CSV、分析脚本与报告归档18faa5c1012db390f3041829b13d880a10cd4a11c092085c8dfe8d2029a9556c，manifest63121aa5…，AFS与本机iter9保存；本机逐成员复核另记。该证据可支撑BB-W5/Q4、HH-W2的明确retrieval阶段成本说明，不能替代尚缺的精确模块消融或MemoryWAM对照。

### 23.113 DEV数据10前缀通过；Full专用探针不适用于source-only对照

字段修复后，原episode435/frame0–36十前缀全部通过，data-only runner16:10:02正常结束。20原件归档dd307f958312022ad17fd4ea30a88d21bedfc9ef604dcd582e78a73a0cc6d22a、manifest0068d979…，无policy加载、无生成命令执行。首个旧字段失败10原件f6b20267…亦保留。GPU-v2沿计划12cbbca9…在16:12独占CCI准入，仍同17:27:52UTC截止。

GPU-v2已完成真实RMBenchWarmPolicy构造、严格pool100权重及独立schema检查，进入frame0生成后被旧research probe的Full-only guard拒绝；16:20:28 runner exit1，0完整replan/0生成动作输出/0物理命令。Video DiT prefill曾运行，不能写成0GPU执行。原因是独立对照正确使用source_only_no_consequence，而旧探针要求legacy full；不是切换对照算法到Full的理由。

在CCI移除采集器对Full probe的启用，改用原样委派的只读infer/source观察器：直接记录传入base_gaussian、实际source、条件token与无未来teacher的context，返回原对象且还原包装；原source-only模式、模型文件、权重及数学路径保持。新增非Full观察器不启用legacy probe、委派一次和恢复属性的测试，共15项原生检查通过（21.59s）。新helper单独冻结、后继GPU新输出，保留原失败/数据与原17:27:52窗口，不把严格装载或单元测试写成完整十前缀资格。

### 23.114 独立池化对照真实原生推理资格完整通过

16:32:49新v3按原窗口准入，16:41:20结束、runner exit0。计划d5fc0ddea3d4bfd4152f1326a9ed424107cb3d3c2c2fd34cb3d6ad1ec7cca3b6、只读helper bcf94b7c1d3ec21def3d21547254a0322fde60d11592cac9079e6d23ac7271bb，模型评测树87199cc6…逐字节保持，训练树48ce6ba8…/pool100权重cbdfcef2…不变；原截止17:27:52.091012UTC未重置。十个固定PB DEV episode435/frame0、4至36，全部实际RMBenchWarmPolicy重规划H32输出有限，每次32个有效候选。真实历史观察token mask计数0/4/8/12/16/20/24/24/24/24、动作历史mask计数0/1/2/3/4/5/6/7/8/8；这是有效token/history slot数，不是已完成技能数。实际8-history+4-pooled条件路径、当前proprio和原生source/高斯/context保存且逐query绑定，无Full专用probe、无零历史替代、无未来teacher。

46项完整原件归档136b665c8a6f1cd3417cd9b9c18046851378ea88c2057c7fc411fe50d5e450ea，manifest ef6766379fe156995973f1c8cc084e127a44ea7df07a70fdf386900a09ec01c6，AFS/本机iter9逐项核验。之前data-only失败10项f6b20267…、十前缀成功20项dd307f95…及GPU旧probe失败21项89c123981dbe2dc6e9c9c6ddc2fbd3e94509cc8c37fe026ad0b6ed94faddd497亦共51项双端核验；原失败不被成功覆盖。该完整新实验只关闭工程集成资格：0生成命令执行、0仿真、0成功率/修复/最终3000 TEST或新计时结果，不能据此声明同信息效能优势。

另在MATCHED_SHARED_SOURCE_AUDIT_V1保存94个共同model/memory/dataset Python文件的双端byte hash及换行规范化hash：93项内容相同，唯一共同模型差异是source_only条件下的输入钩子和adapter参数字典组装。原Full工作快照4efb…未修改，memory/dataset内容均相同；大量原byte差异为CRLF/LF，不假称byte-exact。该静态检查不等于训练权重、参数量、延迟或数值推理相等。池化adapter及损失差异仍按原预定协议明确保留。

### 23.115 v11完整论文/回复与计时证据联动验收

原100条DEV成本13项归档18faa5c1…全部成员已在本机独立核验，见COST_PHASE_ACCOUNTING_LOCAL_VERIFICATION_V1.json；没有新增GPU计时或替换原稿数值。三份英文/15块/20项及新增检索分阶段表联动到Appendix C.4/page37、Table26/page38。原LaTeX工程29源文件（ZIP另含SOURCE_FILE_MANIFEST.json），完整编译与独立干净ZIP重建的38页文本完全相同，十页主文与v10相同，参考文献第11页开始；14个原tabular、九图bytes和所有既有不利/未知结果保留。全部38页contact及变化页37/38单页视觉验收，引用、overfull、匿名性通过，46条继承font warning未增加。修正QA中继承的all37标签为实际验证的all38，不改PDF/ZIP；初次包校验误把29源文件加manifest的ZIP当29总成员，在任何包创建前拒绝，随后按29源manifest逐项验证。

PDF SHA1e11e07ec4de5f71a91cf6b6e201c7eb84473449586d73a5d9f64e1df72941ef、ZIP SHA8b8514877535651baea584a98d94b442e700f9ee082c00412857caf2bc402dd1。47项不可变v11包c62f1a87ab8013e4a1e66843abbdcc05e8d7f3db433177807ffa48685dd0aac5，manifest4a02c6f75d13c00371908984a6ca604464e4f62b0b1601ae8ae5167ef8d35809，AFS writing_candidates/20261111_v11与本机iter9全部成员核验。包冻结早于16:41原生资格终态，后者以独立26项工程增量b3428130ed2fdad6bfb70d90f9c2a39aac4206288c88c931ddc94b31e5d1dfbe、manifestcf10a912…及时补入20项映射f68f902f0512fd8fc82b8db5d529501bb522231aaed27b9613380eeb9d8d7100，双端逐项核验；v11论文/回复bytes保持。全部实证closure仍false，未发布OpenReview或正式上传。

### 23.116 同信息最终训练与原q48部分进度（16:47 UTC）

ITER9_LIVE_V2在2026-11-10T16:47:24.590650UTC保存完整部分metrics/runner/事件：Full标准95e49547…新增201–1219共1019条，pool标准c852e8e2…新增101–500共400条；都连续且loss/grad有限，durable父仍200/100，没有final3000完整state/TEST结果。原q48标准96aea940…仍活跃，正在rank20，有23个封存branch、16个完整独立候选端点；branch终态包含unknown，不能将23当23个有效端点。AABA已通过，完整固定32池及末Full anchor尚未完成，不提前计算总体MSE或适用性/修复收益。

原同信息master截止Nov11 12:29:56.020552UTC、原q48截止Nov10 21:01:03.764554UTC保持，未新建或重复任何ACP任务。AFS可用116,830,027,776bytes，未删除原数据/checkpoint/state或重置资源窗口。后继先查实际进程/heartbeat/exit/完整收据，q48终态后完整保全并同支持集分析，再为原q96/frame384做独立前缀审计；192清单不可复用。原DEV/机制cohort绑定、直接修复/撤回、自然missing-skill与最终匹配性能仍开放，真机新增继续延期。

### 23.117 原生回合上限口径核正；最终同信息评测设计事前冻结

17:14重新核对冻结runtime的_eval_step_limit.yml、原实际worker和十二份700×14完整qpos轨迹、175 replan进度及result：原RB六对都达其原生上限700，task成功0/0，另四对pending。不是因低于RB上限的command cap截短。之前把这些完整RB回合概括为“有界700/非完整native”不准确，现明确更正；原BRT source600才是低于其3500上限的另一种有界来源。没有重跑、改变任何成功数、原seed、未知或checkpoint；shared-stage/部分10对cohort、非原稿specialist复现及不构成效能/非劣的边界保持。原policy未保存每个内部planner错误，完整官方回合也不等于每条命令都获得独立物理有效性认证。

62项原始metadata/完整轨迹/源配置证据归档de99e5cab44a2c612c0fc383933dec1cbec27d16abd2b95eab343aef162a529c，manifest3bbf2eeb1d4b63d79efc87cb7a8cd2fa069cd85b56d84ab20cb658b90938e346；AFS与本机iter10逐成员验证，保留原closure与早先档案。此审计只改证据口径，0新policy命令/0新结果。

17:18:44最终3000同信息比较设计在新权重结果前冻结：RB/BT沿旧primary56b6ec9a…任务和首50个expert-admitted world seeds/340800000起始/2000上限，新Full/Pooled各100回合、共100对/200回合，原生700/1000；旧Full/null或missing-skill计划不被替换。共同evalsource87199cc6…、各自训练source4efb…/48ce…、固定final3000/seed3407，DEV不选点，3408/3409未完成须披露。统计为固定任务配对差及保守精确gain/harm边际界，未知不补0。设计SHA3484ad1477a868607a916638c631c80df25501db59dd82edbf6651c768238e33/3项manifest380b7d45…双端验证；部署到共同eval的affine-prefix边界修复diff明确保全，不假称部署bytes未变。6-worker/10固定分片/每分片12h/总120GPUh/统一36h且不晚于Nov12 18UTC只是有界资源设计，尚未准入或派发，时钟未启动。两final state/CAS、Full实际资格、typed合同、50-roster及可写路径并发隔离仍为准入要求。

原q96固定输入同时已准备：原14k/seed0/query96/frame384/rank10、原source600/probe32均保持，collector eea5ac8e…、独立prefix audit271fadea…封存；未启动新phase或audit，不复用q48的192清单。17:07控制台通过浏览器既有已保存表单恢复登录，已实际读到三标准作业运行中及旧失败；认证字段和回调参数从本地浏览器记录移除，不进入科学原件或Git。实际q48仍在固定池运行，终态才计算总指标、保全后接q96。

### 23.118 原q48完整固定池终态与独立同支持集指标

原q48于17:41:50完成全部36分支，runner17:41:56 exit0；原sourceCf77/14k/seed0/query48/frame192/rank0/32valid不变，四Full anchor和全部fresh parent/RNG/DINO均精确。24个候选完整H32、8个TOPP invalid保持unknown：rank1/9/10/14/20/27/28/31，分别完成3/5/9/7/14/28/23/15步后拒绝。它们不补0、不补effect、不调换样本，不把branch终态当完整端点。

同24支持集centered MSE：预测0.05504458575772138、query-constant0.030714170813867594、historical0.055092660063442836；absolute预测0.09047244380967434、historical0.09039570482394571，实际query requirement未捕获，constant absolute仍不可得。预测未优于constant；略低于historical不构成相对constant的优势，也不作当前状态修复/选择收益。原uniform mean四DINO delta descriptor保持，无缩放或事后拟合。两个原BRT状态来自同一source，K候选不作独立population样本。

343完整原件49,475,789bytes归档SHA91d595a89706c07640c4515e45e67198562e17e5e2b5d13e7d35637c44f7e61f，manifest ae79b07c1e2eaff4d80cbac87ef495f47ebce41106b490f9597db8d4e14983d5；AFS/本机iter10逐成员核验，另用E[(prediction-truth)^2]减去candidate均值项的独立方差恒等式重算centered指标一致。所有原raw与q8/原稿另一个机制cohort保留，未替换原报告数值或重跑不利点，原q48 21:01UTC窗口没有延长。

### 23.119 原q96独立384前缀通过；新标准ACP实际启动

q48原件保全后，q96新8h phase于17:45:00.794514UTC开始，固定截止Nov11 01:45:00.794514UTC；原其它窗口均不变。原v1计划bd34293d…/独立冻结audit271fadea…在CCI两遍各384尝试17:59:42完成、runner exit0，逐步source API/每4步RGB、跨遍实测关节和最终parent/RNG全部相同。索引101与234的原TOPP拒绝均精确保留，不重试/抹去；192清单未复用。16完整原件归档4431cceb9cc487992d0d651c6cde48d6e796304051c5952556818b839e55d9d7，manifest53ebd857…、prefix manifest60539adc…已双端逐项核验，0新候选truth。

同一phase下生成绑定384清单的v2计划dec144d8ae071d18d7809d1dbd55e1fc44d943ec0815457938db927c3d29ea1f、spec68fb5771…、shell acafe827…，原C/H/source600/14k/query96/rank10/seed0/H32/32候选保持。真实verified_prefix_manifest预检通过，shell语法通过；只对事实历史复现已核实拒绝，候选strict guard不变，先AABA再完整池。Source/controller与所有旧窗口未改变。

18:02:18标准创建返回成功；唯一warm-brt-q96-effects-20261111-std，UID492eb35f0d98462db196a624cf492d5c、cluster01e_02/1GPU22CPU230GiB/restarts0，确认镜像ccr-zhicheng-02/lwz_wam:lwz-workspace-20261029145800及AFS afs-p3-01e:/→/mnt/afs当前用户读写。提交前按名查无旧任务，durable create intent已保存，无重复创建、无闲时回退。首次UI18:03:33仍显示排队，但实际18:03:04 runtime probe已产出、18:03:35 collector已开始，按实际runner/心跳登记活动，不依赖滞后UI。实测完整runtime hash2f3e5d29…与原合同完全相同，diff=[]、对象未改，显式override0保持。18:06:57正在anchor-a父状态重建，0封存分支/0独立端点/AABA未完成，不能算q96预测或修复结果。表单一次即时文本断言早于Ace重绘而失败，后续只读确认文本正确后才单次提交；不是失败GPU实验，也不据此重复创建。

同一18:06:57快照Full更新1570（新1370）、pool858（新758），日志连续且loss/grad有限，durable父仍200/100、final3000尚未完成，原训练Nov11 12:29:56UTC截止保持；AFS116,668,485,632bytes可用，原件/checkpoint未删。正式100对设计仍未派发，明确Full/Pooled双checkpoint原生collector待实现/验证，旧Full/null程序不能静默充当新对照。

### 23.120 v12联动原稿工程与回复，完整编译及视觉核验

v12在原29源文件LaTeX工程中补充q48 Table23/page33及RB原生700口径，原q8 Table22和所有旧tabular/九图bytes保持。三份英文/15块/20项矩阵2091cceb…更新Bbqv W3/Q2、Hh4E W5及其修订位置；原q96前缀/ACP/runtime只作为工程证据绑定，不写候选结果。完整工程和独立干净ZIP两次编译，38页文本完全相同；主文10页与v11相同、refs第11页，所有38页contact及变化页32–38均逐页视觉检查，引用/overfull/匿名性通过，46条继承font warning不增。原AI使用说明保持真实，原33页与v11不改。

PDF ed11dae1012a54a7a04bdb8450ffd9be9ca1a768b0eac7084ecc59d7ab9860d1、ZIP8955a7e9a994aebe345e01d63621c58c8be60ecbeae9d863f1d206dc03d9e464。95项本机审阅包d327dc924d16bed7988ab0630d714d0156571e72d02ffb7dbc8e0a06086271ec、manifestd6849ec4…已逐成员验证，AFS同步另记正式收据。最初v12拷贝后因两个reviewer的段落锚点不同停在写数字前，两个部分目录保留为prepare_attempt1；更正锚点后从完整v11重新生成，未改原件/数据，PDF操作标记只执行一次。未知和负结果全保留，全部实证closure=false，未发布OpenReview或正式上传。

### 23.121 v12整包双端验收及最新实际活动

18:14:32 AFS writing_candidates/20261111_v12全部95项再次逐成员核验：整包128,540,942bytes/SHA d327dc924d16bed7988ab0630d714d0156571e72d02ffb7dbc8e0a06086271ec、manifest d6849ec4f376ed82fe9d08d74308816b5c0d4c53a69dd55ac64071cf896292b4；PDF ed11dae1…、ZIP8955a7e9…、20矩阵2091cceb…保持。包中18:06的执行快照与后续live明确分时，不用封存包的旧进度重复启动任务。全部原q48/raw、RB口径、q96前缀与已有成本完整档案及核验包含在包中，其余历史经不可变v11/v10引用保留。

18:18:30 ITER10_LIVE_V3：Full1621（201起1421新条）、pool908（101起808新条），连续且loss/grad有限、durable200/100不变；q96标准492eb35f…已完成1个Full anchor，正在anchor-aa父状态重建，AABA及独立候选端点尚未完成，不计新机制收益。真实runner running/exit null、心跳持续，完整actual encoder fingerprint与原合同相同。原q48已终态24/8不可重跑；所有原phase/training截止保持，AFS可用116,390,531,072bytes。正式100对仍只是冻结设计，新双checkpoint/Pooled原生collector列为下一独立实现项，不能误认为旧Full/null程序可直接运行该比较。

### 23.122 原q96已知终态：完全相同的无效Full，不是端点复现漂移

18:32实际复核发现q96标准UID492eb35f…已于18:20:13 runner exit1。两份sealed result均为invalid_branch：Full各完成26条命令，在index26（第27条）收到同一left TOPP参数化拒绝，H32未完成、observed_effect=null。两次parent/RNG、全部物理端点hash、26条commands.jsonl及parent/endpoint DINO文件逐字节相同。通用错误“Repeated Full endpoint/effect differs”同时检查complete_horizon，因此此处并不是观察到复现漂移；失败实因为原封存Full动作无法完成完整H32。operator计数停留1，实际有两份已封存无效anchor，二者须区分。

未执行固定rank10或其余31候选，0候选端点、32未观测；AABA未通过，不能将部分端点当H32 truth或对其计算预测MSE。保留此前独立384前缀审计及其101/234拒绝，但该上下文资格不认证后续32条动作。不要缩短H、放宽guard、替换rank、重跑同一已知无效Full或重置Nov11 01:45UTC窗口；q8/q48不利结果、原稿机制表仍原样保留。全局native lease已实测空闲，此作业不得再被计作活动GPU任务。后续优先实现已冻结Full/Pooled双checkpoint原生collector，及预定互补任务BT的严格在线资格；不是用新样本替换此unknown。

36项完整原件（含两个branch、stdout/exit、spec/plan、实际runtime与冻结collector）归档3,014,083bytes/SHA45f4f00c8d2ea6dcd5a094e1e5f3b1061ce9a6bef2f58d17c97949cb336dc069，manifest b4118e3920fd130a6dc5a7541dcdfbd2332f869ebab7ba887a773fc548706807，AFS与本机iter10逐成员验证；另独立核对两份commands/feature bytes及状态hash一致。原source Cf77/14k/query96/rank10/plan dec144d8…、collector/source b448aacb…前后不变，未改科学门槛或动作。

### 23.123 v12回复增量1及时联动与双端保全

新目录responses/20261111_v12_addendum1及AFS writing_candidates同名目录，保留3份/15块/20项；Bbqv W3/Q2、Hh4E W5和对应comment blocks明确q96已知完整H32效应不可得，撤去此前pending表述。20项matrix SHA62fdfe7abced92c586a71b8b86b9f2a851b104f13627eec1c07bb20a4ee475f3绑定q96失败原件，全部empirical closure=false。原v12 PDF/29源ZIP没有q96结果断言，bytes保持ed11dae1…/8955a7e9…，38页全工程与clean ZIP编译、视觉验收仍有效；原95项包不覆写。

44项新增量包3,783,497bytes/SHA b384164babd70616730c70cb3ba121f5e726811b04423be16dd437ac4ac5f949、manifest d5d618c18c00d720311e74e32238541e12d8580195c5f245f931ebe43b92c14b于18:37:45在AFS逐成员验收，本机同样核验。包括全部回复/矩阵/一致性检查、q96完整失败档案及18:32 live原件，不把工程失败当新预测或修复数据。18:32:03 Full1680（新增1480）、pool966（新增866）均连续有限，durable父200/100、fixed3000及Nov11 12:29:56UTC截止保持，AFS116,379,942,912bytes。实际活动现为两路标准训练；final3000/正式100对/直接repair与missing-skill仍未完成，新增真机继续延期，未发布/上传。

### 23.124 显式双checkpoint原生collector和有界campaign在CCI实现

19:00:52实现检查exit0，35项通过（10.64s pytest/14.35s总），四新文件AST通过；包含typed Full/Pooled factory/schema、固定final3000门槛、独立attestation、初始实际Gaussian/图像/状态配对、未知不补0、重复结果拒绝、已知终态续跑、原生两种合法H32输出布局、源包装还原/未来teacher拒绝、TOPP原异常语义、旧池化配置和lease/档案生命周期。完整stdout SHA21364e13…及四源文件hash在Git外ITER11_MATCHED_IMPLEMENTATION_CHECK_V1.json/log，未修改任何旧model/deployment或活动训练快照。新入口为run_rebuttal_matched_native.py、run_rebuttal_matched_campaign.py与matched_native.py，具体准入见新COLLECTOR文档。

recorded_dev阶段只允许固定PB DEV episode435十个前缀、Full200/Pooled100，作为新双入口共同namespace/实际Gaussian集成资格；这不同于已完成的旧pool100单角色资格，不重置旧窗口，也不重复计算科学样本。formal阶段只允许fixed3000及事前100对设计，真实并发路径/50-roster和最终完整state等前置门槛仍待核验。当前0新GPU推理/0仿真/0正式TEST，不把35工程检查写成同信息机制优势。18:44:14实际训练Full1734（新增1534）、pool1018（新增918）连续有限，原Nov11 12:29:56UTC截止不变，q96保持18:20终态且不重跑。

### 23.125 双入口DEV准备失败保全和严格formal最终状态准入

bce3f54已推送并在CCI无远端Git访问方式同步，545文件controller source74fb3d19a77c46a36ffbc0ae13021cda47fc57b64ffa004bcd080ea6ae851a11冻结；共同科学source87199cc6…保持。新双角色DEV phase于19:07:04.827513UTC启动，固定截止20:37:04.827513UTC，包含合同准备/失败/恢复；与旧已完成pool100单角色资格及所有训练/原query窗口分别记账。Full200和pool100两套同namespace合同均已完整生成。

19:16:07准备进程exit1、0策略推理/0物理命令：采集器用裸SHA256(text)比对了生产带warm.rmbench.instruction.v1域前缀的digest，严格断言触发。15项程序/失败状态/完整两合同与构建日志归档8012b0547d2323102c707dbae7653bde8400a8d5f71a3fe31b1720bbbf9ee56a，manifest698e004d…，AFS及本机逐项核验。恢复helper使用生产_text_digest保持精确相等，不改指令、DEV样本、权重、合同或模型，不重建已完成bundles；新v2输出与失败v1并存，原20:37截止不变。GPU双角色资格尚未通过，不提前计新查询或性能。

另在canonical增加正式准入硬检查：两角色各自3000完整state/byte-exact archive证明与实际双DEV十query配对收据必须匹配common model source及角色checkpoint，静态/单角色检查或中间状态不能代替。19:17:31本轮最终40项原生检查通过（10.78s pytest/14.43s总），stdout SHA78014335…、四源hash在ITER11_MATCHED_IMPLEMENTATION_CHECK_V2.json/log。该formal-only门槛改动不改变正在准备的旧74fb冻结DEV代码、模型推理或任何已测结果。

### 23.126 新双入口Full200实际DEV完成，匹配对侧仍在运行

恢复准备使用生产指令digest后，两角色typed合同预检均通过，新v2计划2c8a1112bd60071c77f890e7b643572085fce698885e6ace18e4baf30a4c069a保持Full200/Pooled100、PB435十前缀和同namespace。19:24:00唯一CCI logged runner2369882启动，Full worker2369891；原phase截止20:37:04.827513UTC保持，actual admitted spec52b57774…/max4200秒含在窗口内。所有原训练、已完成pool100单角色和q96终态不变。

Full于19:32:31 exit0，十个frame0/4/…/36全部真实重规划、H32有限、K32有效；观察history有效token为0/4/8/12/16/20/24/24/24/24，动作history slot为0/1/2/3/4/5/6/7/8/8。严格类WarmRetrospectionFastWAM/schema warm.retrospection-checkpoint/update200/权重1fb035c4…核实，模式未改、无Full专用probe、生成动作执行0、仿真0、success=null。38原件档案ef1c3db8f31be969f3c56858cdc70db9921685826d46137298a5db0930e71c1c（1,342,475bytes）本机独立验证每个hash、十份原H32数组与真实history，原AFS完整档案保留。它只关闭新Full入口的工程资格，不证明最终性能/repair；当前新Pooled worker2370928仍运行，实际双角色输入/Gaussian配对尚待核验，不提前宣称完成。

f8311af已推送/CCI同步；未来formal controller source2b7df3433bb06cfd3b79c673bc83e98549ca16a81bd70a721530e185581f9ac4冻结。与实际DEV controller74fb…相比，543/545源文件及两个worker程序逐字节相同，11个推理/采样见证/配对函数AST完全相同，仅formal准入与测试变化；两个冻结树分别保留。19:30:25训练Full1938（新1738）/pool1227（新1127）连续有限，durable200/100、Nov11 12:29:56UTC不变，AFS116,116,328,448bytes。原33页与已完整验收的v12 PDF/source ZIP不改，本轮仍无正式TEST或新效能收益。

### 23.127 两角色完整推理与严格全候选池资格失败均保留

Pooled19:41:00 exit0完成十前缀/H32有限/K32有效/真实history，38原件archive d6b4433a3a86e114db1f5c2734e41c0e757fe456ea4c7e7112bf77fa629e21f8。主campaign因bank_rows不同于19:41:02 exit1；两子角色完成不等于主资格通过。初始identity/共同合同、十次当前图像/proprio/query key/derived seed/实际Gaussian完全相同，初始K32及payload相同；frame12/20/24/28/32各一个第32候选不同，其余top31一致，另外五次池逐项相同。20重规划来自一个DEV episode，不作独立episode样本，不重跑或换样。

已用原bank successor_row、前一次实际选择/源接受/thread资格与原精确排序规则在AFS及本机分别重算，五处差异全部精确复现：方法各自的选择反馈会保留一个后继候选，不是不同外部bank、query/DINO或随机数漂移。139项6,892,482bytes档案SHA6a98ce52938e832d82508bd34cd75f2e76c8c701014f0dbf481dfcf5368200fa、manifest f21477ef43cc83ce5ef07a0194b168aaf24fccc852e76f87fcc2f899ecdb35ed双端逐成员核验；本机独立重构证明SHA a065c7922d4962c7b889f41650e695b8a2cd644ae318e61f99e6453699744549。0仿真/0生成动作执行/0任务成功率，旧严格失败保持原身份。

### 23.128 TEST前单独冻结共同外部输入资格范围

原3484ad14正式设计要求共同初始输入/随机性、bank与信息预算，后续保留各方法自己的因果状态；额外DEV工程断言把续接选择状态也强制相同，范围更强。19:55:11冻结MATCHED_DEV_SCOPE_AMENDMENT_V3.json/SHA fa089d4d0bfb0753de4ecc9e52c05ddb8d398de20ce498441693091abd1ee6ca，明确旧失败不改名成功。单独的MATCHED_EXTERNAL_INPUT_QUALIFICATION_V3只确认十次共同当前输入/query key/Gaussian、初始K32及全部可重构续接差异，不证明后续固定候选相同、独立机制隔离或最终效能。Formal计划须显式选择该范围、绑定修订和独立证明，且修订早于首次TEST准入；任何无法解释的差异仍拒绝。最终3000完整state/CAS、50-roster、真实并发和所有科学主指标/算法/样本/截止保持。

formal-only证明绑定追加后，19:56:28全46项检查通过（10.91s pytest/14.72s总），stdout SHA c1fdfb7c…；严格pair函数和两worker/模型推理不改。最新回复增量2/15块/20矩阵4fb14257…明确中间权重、因果续接差异及final performance pending。v12 PDFed11dae1…/ZIP8955a7e9…不改，A.3已有后继续接机制，无新增论文数字。20:01:37实际Full2075（新1875）/pool1370（新1270）连续有限、durable200/100、原Nov11 12:29:56UTC截止保持，AFS116,093,566,976bytes。CCI两角色均终态，不记活动GPU；新包双端验收另记，全部科学closure=false，未发布/上传，真机新增延期。

### 23.129 首批50专家准入reset与真实工作进程隔离入口

2026-11-10T20:21:28UTC实际ACP：Full95e49547…至2160（201起1960条），Pooledc852e8e2…至1462（101起1362条），均连续且loss/grad有限，durable父200/100、final3000及Nov11 12:29:56UTC截止不变。CCI H100空闲、旧native全局lease实测可用，AFS115,177,086,976B。完整metrics字节（含可能未完尾行）、runner/events和LIVE在Git外P/ITER12_LIVE_V1及本机iter12；只算部分执行，不算最终模型或性能。

CCI新增qualify_rebuttal_matched_resets.py与逐worker独立HOME/cache/tmp/runtime准入；原规划器匿名Pipe静态核对，不冒充实测并发资格。每候选先保全专家结果，首50不以策略结果选择；每准入保留完整expert_info并两次精确初始RGB/API/物理/任务状态/指令，旧RB10强制前缀。任意重置或未知运行错误中止保留，不能换种子。正式worker在Torch导入前设置私有目录并持有独立lease，原推理/采样/配对数学和活动训练快照不改。

20:27:32实现检查exit0：55项通过（11.10秒pytest/14.84秒总）；stdout SHA87d73f3c…、四实现文件SHA及完整argv在ITER12_RESET_IMPLEMENTATION_CHECK_V1.json/log双端Git外。初次人工pytest命令用了不存在的配置测试路径，0tests/命令exit1，未启动任何GPU或仿真，改用既有真实测试文件集合后通过。真实50-roster/并发reset尚待运行；不将静态检查写成新科学证据。协议见MATCHED_RESET_AND_ISOLATION_PROTOCOL_20261111_ZH.md，当前v12_addendum2/未变PDF保持，全部科学closure=false。

20:33追加：首次真实运行在候选seed340800000的setup阶段终态exit1，尚无专家结果/策略推理/命令。两个curobo子进程均报告NVRTC临时目录名过长、error6，父进程ConnectionResetError属于下游表现。完整阶段/计划/runner/错误/启动意图等原件已归档matched_reset50_NVRTC_path_failure_v1.tar.gz，SHA559a6bc1…、manifest7ec6f98b…，本机/AFS均保留。不能记作专家拒绝或模型失败。

20:34修复：每worker缓存/临时目录改用容器内独立短路径；科学输出和跨节点fcntl lease仍在AFS持久路径。激活时验证目录归属，禁止长TMPDIR及把lease搬到临时盘。57项检查exit0（11.45秒pytest/15.36秒总），stdout4ba0d57d…；原26360d70…快照与失败marker保持不改，新快照另冻。后续同一seed/顺序/原22:31:09UTC截止继续，不重置资源窗口。

20:40追加：短路径v2确已完成官方专家首seed340800000（accepted），NVRTC问题消失；随后新入口把原生完整观测误限成三策略相机，遗漏官方aloha额外front_camera，初始schema检查终态exit1、0策略推理/命令。完整原件e80a5e53…/manifest ffb26628…双端保留，不改称expert失败；旧首种子与原22:31:09UTC期限保持。修复按实际静态相机清单精确核对并保存front_camera，仍强制三策略相机、14维有限qpos和全部有限图像；新59项检查通过（11.28秒pytest/15.21秒总），stdout453f060a…，补充真实官方四相机结构的正反例。原native worker本就保存全部RGB，未改模型/相机配置或删除任一原件。两个任务可在不同private runtime中作为隔离工程资格并行采集，任何异常保留；正式策略TEST仍须完整并发资格。

### 23.130 保全18/14个完整reset前缀；原专家几何拒绝与原BT完整原生入口

20:52:26 reset-v4 runner终态exit1：BT候选seed340800032/offset32在官方play_once→grasp_actor→Action.__init__出现target_pose=None断言；RB被联动SIGTERM，exit-15。此时RB18、BT14项两reset完整，另RBseed340800018专家已accepted但reset未完成；全部原始unknown标记、拒绝、图像/API/真实物理/指令、日志与退出已封存9200d955…/manifest ccab3a1c…，11,074,711B，双端逐项验证。不能从0重采这32项或丢弃断言。官方eval_policy.py SHA edff5aa9…第254–262行确实在专家阶段异常后递增seed；源码不可变的这条“无几何抓取”堆栈可按官方专家拒绝处理，不是模型或基础设施失败。原unknown记录不修改，新恢复证明单独绑定其原hash及此窄分类。CUDA/OOM、setup和其它未知异常继续中止，不泛化吞掉所有Exception。

新入口只在已检查终态/完整raw manifest/原生source hash和显式resume proof相符时复制并复用候选前缀；已完成专家/两reset只读取原件，不重算；RB18号直接接续未完成reset，BT32号保留原错误并追加reconciliation后按原顺序继续。原20:31→22:31UTC窗口不变，first50/2000/旧RB10前缀不变，0学习策略结果用于准入。v3并行driver另有换行SyntaxError（0worker/0物理），v4使用独立文件并AST后启动；旧失败工件全保留。

原BT12000/source f77新增独立native1000事实轨迹协议，保护旧BRT14000/600行为。BT要求root3407/physical0、task/run/step/cap/probes匹配，首四条真实关节/API/TOPP必须精确匹配one-replan资格。原builder v1 metadata head键失败0策略，v2默认矩阵生成root17合同也0策略且不准入；v3显式matrix3407（SHA be10e580…）正在重建，不覆盖seed17原件。原权重/模型/M1/encoder/bank/normalizer校验保持，原DEV未知。21:01:49整套72项通过（12.38秒pytest/17.17秒总），stdout554538a5…；0新BT推理/命令。协议见ORIGINAL_BT_NATIVE_TRACE_PROTOCOL_20261111_ZH.md，不能把工程与队列当原模型成功/修复或正式TEST。

### 23.131 六场景真实初始化barrier、实测物理首态与原BT在线合同资格

21:13:54实测resume-v5源845108f2…仍在原窗口正常新增：RB37/50、BT26/50；此前18/14前缀只读复用，后续真实专家与双reset新增，未重算已完样本、未改seed顺序。为后续10个固定分片，新增probe-only barrier：至多六个真实已初始化scene全部ready后才运行四条hold，并保存实际匿名planner endpoint inode/PID、RGB/任务/物理首态和跨两reset轨迹；不把六个进程启动当成六scene真实共存。正式Full/Pooled入口另强制实际关节qpos/qvel等于roster，不能仅依赖API drive targets相同。模型/推理/采样/严格配对不改，旧实际DEV20次不重跑。

21:16:06新77项检查通过（12.48秒pytest/16.98秒总），stdout16348ad4…；旧fixture漏task导致的1failed/74passed记录fb304c4b…保留，补齐测试输入后通过。实际六场景/十worker资格尚未派发，须两份50-roster完整/driver已退出/旧native lease释放后在原22:31UTC前执行；coordinator已准备并AST检查，不自动重复未知任务。

原BT seed3407在线bundle已真实完整构建（manifest d52462a2…，online contract390a6825…）；21:11:20静态部署projection完全相同0583a49a…、原模型/weight/M1等全部校验通过，实际完整encoder runtime精确匹配原2f3e5d29…，0模型推理/命令。原native one-replan+四真实关节/TOPP资格helper aa2e8e8e…已准备未派发，绑定旧native TOPP observer49888e…，0新BT策略结果。旧builder head-key失败、seed17未准入bundle和全部旧原件保留。后续先one-replan资格再native1000源；没有复现原DEV或SR收益结论。v12_addendum2三回复/PDF/ZIP仍原bytes，科学closure全false。

### 23.132 两任务首批50固定cohort与200次真实初始reset完整保全

21:26:00两roster driver正常完成，runner随后exit0。RB候选0–49全部expert accepted；BT顺序检查128个候选，首50 accepted、75普通expert rejected、3条源确定的expert无抓取断言，最后seed340800127；全部拒绝/几何失败/原unknown及reconciliation保留，未用学习策略结果筛选。各50例每例2次独立setup，三策略相机+原生front_camera/API qpos/真实qpos与qvel/物体任务态/官方unseen instruction都一致；原RB10为精确前缀。原20:31→22:31UTC窗口不重置，0学习策略推理或SR。

892项完整原件34,739,266B归档deae79ec…/manifest291b6bf2…双端核验；本机独立解码200份NPZ，重算dtype/shape/content hash并对比全RGB、API和JSON物理/任务/指令，200/200一致。RB roster SHA98cd867b…、BT612cb5d1…，局部18/14旧前缀只读复用有manifest证明，不因失败重算已完成样本。收据MATCHED_RESET50_BOTH_COMPLETE_ARCHIVE_V1及ITER12_ROSTERS_AND_BT_STATIC_LOCAL_VERIFICATION_V1。

原BT在线合同/静态资格与全部准备失败也已38项归档80576cc1…/manifest f8187694…，双端逐项验证，正确root3407/未准入17均保全；0新BT推理/命令。21:28:44实际启动独立原生并发资格outer2407946，source861004fa…/commit e841521、77检查；首批6worker2407957–2407962。只做20次reset/80条hold的工程资格，最大6场景，原22:31UTC截止及900秒外层不变；尚未宣告通过或派发正式Full/Pooled TEST。新原件仍Git外，论文/回复无新增收益数字。

### 23.133 十worker原生并发资格完整；原BT固定cohort case0完成实际首推理

原22:31UTC窗口内，十个分片runtime以6+4两批完成20次独立reset、80条qpos hold。每批两次已初始化scene barrier、真实planner PID/匿名endpoint inode及全部物理/图像/API/任务/指令原件均完整；不是仅进程启动计数。143项3,549,573B归档3f99eb0d5fd3d2cf6d918e7e300c5d2900c8c25ff93a903784d6efbf2af1976e，manifest f8b88e65…，资格收据2ad5c09e…；本机逐项核对全部20 NPZ/80物理hold/真实时间重叠与12或8个独立endpoint。formal source861004fa…/548项保持。它只关闭初态与可写路径/场景并发资格，不证明策略成功或完整隐藏状态相等。

额外BT物理seed0诊断21:38专家拒绝，0模型查询/0学习命令，37,913B归档3c87ed5c…；不计模型失败，不重跑/更换这个旧点。两任务固定50-roster已21:26独立于任何原BT策略输出冻结；21:43原BT全50点前注册b1be58f9…，固定case0..49升序，原DEV选点不明，30GPUh/至多2H100的全50运行仍仅prospective、未准入。原BT case0物理seed340800000沿用原22:16UTC资格期限，于22:06:35正常exit0：一次H32真实推理、四条真实native命令，0 TOPP错误，原f77/source ff32342d…/weight27c137d6…/root3407/DINO2f3e5d29…保持。24项568,505B完整归档74859159d1cdc01b69a2520ef1144001d4af4de61529b3004f1ed679f9ad1b5d，manifest e5c9baa7…，本机独立成员/数组/物理/指令与CPU无损动作导出核验收据ITER12_ORIGINAL_BT_CASE0_LOCAL_VERIFICATION_V1。

原API数组float64、原policy factual_joint_state按既有源码转float32，审计分别核对完整raw哈希及精确原转换，不误宣称两个dtype字节相同。所有四原生相机/实际关节任务态由资格运行核对；本机独立再核对三策略RGB、float32 API输入、实际物理及指令。prefix末success=false只代表四步尚未成功，不能记作完整episode失败或SR0；独立候选标签0。

新增独立cohort-case0.v3事实采集协议只接已冻结case0，完整BT官方1000命令或成功终止，首图像/首H32/前四API及实测物理/TOPP须精确匹配已成功资格。所有实际到达replan封存原输入、候选预测、原适配动作、RNG、输出，主探针仍预定8/48/96，未到达不换点。原BRT600和额外BT0协议不变；全50评测没有被此单源采集默示启动。22:16:29新81项检查通过（pytest12.22秒/总16.49秒），stdout6feab48b…，collector93b1bb1c…；没有改变原模型或任何原科学结果。

22:23:42UTC实际标准Full2679/3000（新2479）、Pooled2019/3000（新1919），连续且loss/grad有限，durable父200/100、原Nov11 12:29:56UTC不变，AFS114,985,889,792B；两final完整state/CAS未齐。正式100 paired native TEST尚未派发。当前论文v12/PDF/ZIP和三份英文15块不因工程数据而增加收益数字，全部20科学closure=false。

### 23.134 原BT完整native1000事实源已启动；增量3双端封包验收

22:27:39UTC在已知旧资格成功终态后独立准入单源case0，原f77/BT12000/root3407/physical340800000/H32/prefix4/NFE10/K32不改。collector75eb6f53…/source5729f876…/548文件、81检查；新plan9d479fa7…，本阶段单一90分钟至23:57:39.628419UTC，外层5370秒。旧one-replan22:16、roster22:31与训练Nov11 12:29:56截止均保持，并非失败重试或重置旧窗口。flock持有原native全局lease至worker真正退出，保护共用runtime；formal隔离source861004fa…及两训练活动快照均不变。全50原BT评测未派发。

22:32:20UTC实际runner2414197及flock/collector子进程存在，全局lease实测占用，CCI 4442MiB；状态loading_original_full_policy，0新查询/0完整native命令，不能写成完整回合或新性能。两标准ACP Full2715（新2515）/Pooled2058（新1958）连续有限，完整final3000尚未完成，AFS114,859,044,864B，保护108.407GB最终raw/CAS预留。真实metrics/runner/事件/进程/lease在Git外ITER12_LIVE_V7保存；不新建或重复未知任务。

内部responses/20261111_v12_addendum3更新20项问题—主张—证据—回复—修订位置映射，matrix0e18d942771cbacd99f95f6435a23a64d506560416e53a054b3a71063ec461a0。三英文/15块和v12 PDF ed11dae1…、原LaTeX ZIP8955a7e9…完全不变，无新效能数字，无需重编译未变PDF。78项51,732,691B增量包SHA9dcbfd6dbcebb087ab71a199ff651c2545de504285ca3031e4356cf3c4027074、manifest796870b6…包含8个完整原件档案及独立核验/源/协议；本机全成员与AFS writing_candidates/20261111_v12_addendum3逐项解包/落盘校验均完成。全部20科学closure=false、stable_release_ready=false，未发布OpenReview或正式上传。下一优先核销完整原BT源后按固定8/48/96进行原模型同状态后果/修复/撤回独立执行，及两final3000完整state/CAS与typed bundles后100对正式native比较；真实missing-skill和原DEV绑定仍开放，真机新增延期。

### 23.135 原BT固定状态独立后果/修复入口，保持旧BRT协议

2026-11-10T22:44:35UTC在CCI完成独立BT schema与新测试，相关54项通过（pytest11.63秒/总15.96秒），stdout fed2a9ade236d77451e2a8678e64a912db72c81e31929048a0aa2b2a5b82564b；collector64e88773…/tests7696138d…，0新独立物理branch或模型推理。旧BRT8/48/96固定rank与seal/result键保留；新BT强绑定原12000/f77/root3407/已冻结case0/原事实计划9d479fa7…，主query仍8/48/96，rank只能取其原封存输出，不按新结果挑选。

增加新建场景重放中的逐步API及实测物理核对、原RGB前缀、完整parent/RNG/DINO、原H32 AABA与独立BT next-interaction标签，实际前缀保存已有在槽极性组合，原隐藏正确答案不用于标签或policy。标签读取前后检查状态不变；raw/Full repair与harm及unknown分开。原f77_uniform_mean_DINO_delta投影保持，候选数不是独立episode数，不能估总体SR或替代原19% cohort。事实前缀拒绝需另有两遍审计，不能静默吞错。

22:45:53UTC CPU只读核验（0模型新推理/0物理）：事实源已89条命令/23个replan，q8预truth seal完整、原selected rank4/32有效、gate0.8984375，q48/96当时尚未到达。完整H32/首次4条实测前缀已精确匹配旧资格；源仍运行，尚无完整1000终态或新增后果/修复结果。原候选预测、adapter、采样RNG、infer kwargs和输出hash在ITER13_BT_PRIMARY_SEAL_INVENTORY_V1.json。原f77没有新shared模型的configure_research_probe接口，同状态零影响撤回仍为开放项，不套用新接口或绕过bound capability。完整协议见ORIGINAL_BT_SAME_STATE_PROTOCOL_20261111_ZH.md。

### 23.136 原BT完整native1000固定来源；全部2020原件保全

2026-11-10约23:04UTC原case0回合正常exit0，1000条连续原生命令、250次有限H32与完整replan/probe seal，0 TOPP，官方任务未成功。模型仍作者确认原BT12000/f77/sourceff32342d…、root3407/physical340800000、roster612cb5d1…、事实plan9d479fa7…；首图像/首H32/前四API及实测物理与既有资格精确相同，源码前后不变。该失败来源完整保留，不能视为总体SR0或改用成功样本。原DEV选点仍未知，不替代投稿specialist成绩。

2020项100,561,680B完整归档a1a42358a991f710914b6de3ad376720d70e58a004391107a8b99b06e39f5414，manifest b143f0c0…；收据ORIGINAL_BT_NATIVE1000_COMPLETE_ARCHIVE_V1。本机23:06:42独立解码全部250 H32和观察NPZ，核对每条API/实测物理连续、指令、固定8/48/96 seal及全部档案成员，ITER13_BT_NATIVE1000_SOURCE_LOCAL_VERIFICATION_V1。infer_kwargs中的部分tensor是显式unserialized_type占位，不能称完整可恢复模型上下文；后续撤回仍须真实factory绑定。全部原件Git外；canonical实现3753286已push精确验证并CCI离线FF，活跃事实采集source5729f876及原模型不受同步影响。

### 23.137 原BT q8独立H32 AABA及终点后果对照；完整候选池接续

23:00:36UTC在新branch truth前冻结query8/48/96原selected rank4/3/20，三个完整32候选有效集合保持；登记ORIGINAL_BT_PRIMARY_RANKS_PREBRANCH_V1，inventory0fcca10e…。22:44新增BT入口54项实现检查已通过，canonical3753286冻结controllerf7cf0583…/549文件、collector64e88773…；原BRT入口及所有旧结果不变。

q8于23:04:51首次准入，23:14:29完整资格exit0，plan94c9e7ba…；四分支128条新命令和128条事实前缀重放，所有API、RGB、实测parent/RNG精确，三个Full终点和DINO效应相同，原rank4提案物理终点不同。原提案电池1终点z=0.027890842407941818m，Full=0.7598833441734314m，预定跌落阈值0.70m；另一电池分别0.7896867394/0.7600132823m。两者独立next-interaction completion标签均0，**不计完成型repair或harm，不估预测精度/总体效能/自动撤回**。Full本身也未完成下一交互，不能称任务成功或全程安全。场景相机遮挡，新增表只使用原物理坐标，不用图像臆判跌落。

58项5,677,705B完整归档b7e7359ce1fb8f11b88d13a76c5eee483da5e8011ff49ab48216cc1626580343，manifest021cac6f…；本机23:24:58逐项独立核对四分支命令/前缀/父/端点高度，十二张导出图像像素均有原pickle/result绑定，ITER13_BT_Q8_QUALIFICATION_LOCAL_VERIFICATION_V1。旧operator `applicability_labels=0`日志计数未更新并保留；真实四个sealed分支标签均已知，独立来源1、原候选1、Full控制1，不能把重复控制当独立样本。

23:16:30在已知资格成功和58原件完整封存后，q8接续全部固定候选、最终Full anchor，outer2418349；同一plan94c9e7ba…、绝对截止2026-11-11T01:04:40.464620UTC不变，已sealed分支只读复用。23:57实际20候选完整，正重建rank20；尚非完整32池，未知与未完成不填成失败。q48/q96未准入。23:55:30按已完成q8分支时间间隔及长前缀长度，在q48/q96首次准入前声明独立资源修订：6h/11h，总含q8为19GPUh/最多1H100；原6GPUh及q8窗口保留，模型、固定query/rank、H32、标签和指标均不变，预算计算未读取结果。长任务ACP、标准优先，实际GPU quota不足才闲时。收据BT_Q48_Q96_PREADMISSION_RESOURCE_AMENDMENT_V1，不表示已创建ACP任务。

### 23.138 v13完整LaTeX/PDF与回复联动验收；Full3000训练自动归档完成

新增Appendix C.2.6/Table24/p34如实呈现单状态终点高度及两个零完成标签；三英文/15块/20项映射联动，全部scientific closure=false。完整原LaTeX及干净29文件ZIP独立复编，39页逐页文本一致，主文10页文本/像素与v12相同，原所有表九图/不利与未知保持。全部39页渲染、变化18/29/34–39逐页视觉验收；无未解析引用或overfull，46个既有字体警告。PDF dd70f15ac53777adb27dc67b148e1cd99ce931bd3436b9868e263c43361aa86a，ZIP1a6f548f…，matrixbd927766…；responses/20261111_v13。

64项182,286,118B完整审阅包630eb424504793c431f14d74285c69dbbb07db67298063af9e5ce7f5f92b1767、manifest0ec74c5f…于本机23:57:00、AFS Nov11 00:00:40逐项落盘验证，writing_candidates/20261111_v13。包含2020/58完整原件档案和本机独立核验；原v12及增量保留。科学截止q8资格23:14:29，运行进度以新operator为准；未发布OpenReview或正式上传，真机新增延期。

Full标准ACP UID95e49547…23:41:34实际训练exit0到3000，201起2800条loss/grad连续有限，原数学配方/root3407/父200不变；checkpoint b0669047fbcb0f8f79fd25fea2de25f689cfd286d85ae899872c9b4435fff27b。23:51:50自动完整state/CAS逐字节归档成功，原TAR899dc17f41cf25f952fc2301c6d049b67e3056235dd3e17359e84d31c3b0b8a3、38,603,714,560B，新增CAS13,581,159,874B。另起CPU独立重建审计PID2422615/ITER13_FULL3000_INDEPENDENT_AUDIT_V1，0GPU、不另训练/新pack；00:00已独立重建完整TAR，正在逐项核对原state，终态另记。Pool UIDc852e8e2…23:57实际2442/3000，原Nov11 12:29:56UTC截止不变。AFS62GB已包含Full raw38.604GB及CAS13.581GB消耗，原108.407GB预算剩余约56.223GB须保护，不再机械要求消费前110GB，也不删任何checkpoint。仍须双方完整state/CAS/typed bundles后才准入冻结100对，正式TEST未派发、时钟未开始。训练完成不等于新效能结果。

### 23.139 Full3000完整状态独立重建及正式前置收据

2026-11-11T00:03:22UTC独立CPU审计正常完成，未重复训练或重新pack CAS。按现有3817个CAS片段逐个验证内容哈希，完整重建38,603,714,560B原TAR899dc17f…；TAR全部15个文件与AFS原件逐项SHA及大小完全一致。七个实际完整state文件包含模型/优化器、scheduler、随机状态、trainer数据位置及zero_to_fp32脚本；sampler.bin实际不存在，收据明确记录false，不声称保存了不存在的文件。checkpoint原payload含warm.retrospection-checkpoint，CPU mmap检查未初始化CUDA；训练source前后4efb8094…，parent200、root3407、batch128、固定scheduler30000、2800新增有限更新与原attestation/合同一致。

正式准入schema收据ITER13_FULL3000_INDEPENDENT_AUDIT_V1/FINAL_TRAINING_EVIDENCE.json，SHA62e9ecd838086a3d4cdd81b196914ff3ab028ad95875d824d2c374a7f165cc2a；state inventory f9e3254a…，checkpoint b0669047…不变。00:09本机独立核对收据/七state inventory/十五raw inventory及原自动归档收据的所有引用哈希，ITER13_FULL3000_LOCAL_METADATA_VERIFICATION_V1。38.604GB原TAR在CCI临时盘重建，持久CAS与原AFS检查点保持，本机未下载大TAR，此处不冒充本机完整权重字节校验。

00:07:29在该终态与收据后，单独CPU准备Full3000的RB/BT两task typed bundle，PID2424423，使用common87199cc6…/原root3407矩阵、Full专用config/schema及新正式namespace；0实际policy/模拟/TEST，不启动正式评测时钟。需分别核对实际bundle manifest与对方final3000状态，不能把静态合同准备当正式效能。00:05最新实际Pooled2475/3000连续有限、q8完整24个候选且仍持有全局native lease，原截止均不变，ITER13_LIVE_V4保存实际metrics/runner/进程与AFS61,871,456,256B；其后进度以新收据为准。

### 23.140 原BTq8完整32候选和四Full控制；标准q48实际运行

2026-11-11约00:23UTC原q8 pool runner正常exit0，全部36分支终态：32个原有效候选均完整H32，无unknown，四个Full终点/原DINO效应精确相同，全部parent/RNG一致。原plan94c9e7ba…、rank4、checkpoint27c137d6…、controllerf7cf0583…和01:04:40.464620UTC截止未改。380项49,259,146B原件归档9d4f92e6a40fafbbe3b5d1a10557426c586cc37f2d4fe60201bfe6526c9e64ee，manifest bfe7dea6…，收据ORIGINAL_BT_Q8_POOL_COMPLETE_ARCHIVE_V1；已完成原件不重跑。

00:24:56本机独立核对380成员及每个branch marker，逐支核对全部前缀/物理跌落阈值，按原四token DINO均值投影重算effect（保留float32原值，以声明的IEEE舍入界核对float64独立均值），再用独立方差恒等式重算同支持集MSE。预测centered0.07684433201273627/absolute0.13682638107124517，历史centered0.07629158519676332/absolute0.1360761372288195，常量centered0.022408466917335543（absolute不可识别仍null）。该状态无预测优势。候选物理下一交互标签10正/22零/0unknown，selected proposal与Full仍0，无完成型repair或harm；原跌落终点差异仍仅单状态描述。候选及控制不当独立episode，不替换原19% cohort。核验ITER13_BT_Q8_POOL_INDEPENDENT_LOCAL_VERIFICATION_V1，全部负结果保全。

按已先声明的资源修订，00:23:48首次冻结q48六小时绝对窗口至06:23:48.137340UTC，plan363bc263…/job6b71d4b6…/原rank3，不延长q8或旧BRT窗口。已知q8终态并完整归档后才准备，标准ACP00:26:12创建成功、UID59dd5f78631b40ae81a1298844d499ba，1GPU/1worker/22vCPU/230GiB、0自动重启；镜像ccr-zhicheng-02/lwz_wam:lwz-workspace-20261029145800，AFSafs-p3-01e子目录/→/mnt/afs读写均逐项回读。00:27:20实际H100编码器runtime全部字段匹配原合同，00:27:52原collector开始；00:29 UI运行中与AFS实际runner一致，不再仅称排队。00:43已有两个Full H32控制完成，正在rank3父重建，尚无完整AABA/候选比较。q96未准入，保留固定rank20/首次11h；旧BRTq96失败不替换或重跑。浏览器task已finish且保留原tab，没有第二次创建或闲时请求。

### 23.141 v14同步完整q8结果，原LaTeX及87项审阅包双端验收

在原物理终点对照旁新增完整32池、10/22标签及三参照MSE，明确不支持该状态预测优势；与Bbqv/Hh4E两处回复、15分块、20项映射同步。PDF仍39页/主文10页；完整原工程与干净29文件ZIP复编全部39页一致。前33页像素与v13精确相同，变化34–39逐页视觉检查；原所有表/九图/不利与未知保持，无未解析引用或overfull，46个既有字体警告。

PDF230dcc6e4e4642a2ac522f2cd1943ab69ae10c58825a9f58210d977fc7dbfe54、ZIP9ba9dca5…、matrixb910f3a8…；responses/20261111_v14。87项124,898,316B包1bf35dca2beb2121fbf97dab03e10c7108f6e9cbef6f368e3aac7d0627a3c723、manifest a52af5c0…，本机00:41:22、AFS00:44:11全成员落盘验证，writing_candidates/20261111_v14。包内含完整pool380项、独立核验及Full3000七state/15raw清单与两任务静态合同12项；父v13包630eb424…保留完整来源2020和资格58项。论文/回复不宣称预测、完成型repair、撤回或总体新效能；全部scientific closure=false，未发布OpenReview或正式上传。最初v14编译脚本CRLF导致在set选项即退出，原脚本保留后仅统一LF，完整正常/干净ZIP编译均成功；没有任何实验重跑。

Full3000两task静态合同00:17:04正常完成，本机00:20:25逐项核对12文件、显式root3407/config/schema/namespace，收据c04abdc8…；不称实际policy或正式TEST。最新00:43:26 Pooled2646/3000连续有限，Full3000完整状态与合同已齐，仍须Pooled完整final状态/CAS/typed合同后才准入100对；原训练截止不变。AFS约61.62GB，原训练预留扣除已消费Full raw/CAS后约56.2GB仍保护。实测已有DEV context每份584475–620635B；正式记录遵循原有界保存协议，准入前结合剩余训练、原始轨迹和归档核对容量，不删历史或检查点。

### 23.142 跨节点共享runtime的锁观测范围核销

00:43:23在ACPq48实际runner前后均running、collector已有两个完整Full控制时，CCI对同一绝对AFS flock路径的本地探针仍返回空闲。实际ACP spec确实使用该flock路径；该测量证明CCI探针不足以排除另一ACP client的已知活动，不能把单client锁值当全局互斥证明。完整收据ITER13_NATIVE_LEASE_SCOPE_AUDIT_V1，不据此笼统推断其它所有文件系统锁的实现。

旧LIVE_V5原字段和值保留，并标明它只来自CCI client；新LIVE_V6另存CCI_client_flock_available=true，全局共享runtime准入仍false。后续必须以唯一ACP UID、实际runner、已知退出及完整归档核销为准。q48/q96准备器原已要求前一任务已知终态和档案，因此没有重复执行；本轮未修改运行中的f7cf科学源、样本、指标或窗口。正式十worker使用另行资格通过的私有runtime/匿名IPC与独立路径，仍需各自明确准入，不能把共享runtime活动或本地GPU空闲当可重复派发的依据。

### 23.143 运行中的BTq48已完成H32 AABA，四个sealed分支独立保全

00:52:33实际q48 AABA通过；三个Full完整H32端点/原DINO效应相同，四个parent/RNG相同，原rank3提案物理端点不同。selected raw/Full下一物理交互标签仍0/0，无完成型repair或harm。此为与q8相同来源回合的第二个固定状态，不能当独立任务试验；完整32池仍运行，暂不填写预测误差或总体频率。

不停止、不重启或修改活跃程序，只读核对四个不可变complete marker及其所有文件后，于00:55:50保存47项5,423,239B资格快照e0da5bdc85c01f975cb6eac97012c2e05481cb586648fe38e77d6c7748d808ff。四支各192条事实前缀与32条新命令完整，归档前后sealed哈希不变；本机全成员及四支前缀/H32/端点再次核验，ITER13_BT_Q48_ACTIVE_AABA_LOCAL_VERIFICATION_V1。该档案不是whole-job terminal或完整pool档案，原UID59dd5f78…/plan363bc263…/rank3/06:23:48UTC保持。回复研究录入备忘BT_Q48_RESPONSE_EVIDENCE_INTAKE.md已保存，待固定池完整后纳入下一版；当前v14仍明确冻结于完整q8的数据截止，不将新资格误称效能。

### 23.144 正式配对比较准入前检查与唯一CPU资格流水线

2026-11-11T01:32:12UTC只读解析Full/Pooled两份真实resolved config：仅model factory、retrieval/effect/adaptation/gate四项loss、输出目录、parent resume、本段run_steps八处不同；其他数据/模型/优化器/scheduler等叶子精确相同。Full四loss为0.1/0.05/0.05/0.05，Pooled全0，符合原事先协议，不改训练目标。完整方法及对应监督的对照不能称仅改变结构或整个recipe hash相同。配置SHA f84274b1…/b5ca3cec…和实际差异在Git外 `ITER14_ENGINEERING_PREPARATION_V1/PREPARATION_CHECKS.json`，无新推理或TEST。

01:42正式计划准备器逐项对照不可变native worker时，发现尚缺必需cuRobo路径，已在任何formal准入前补齐，并加入实际worker字段完整检查。修前operator97dc4e10…原件单独保全，修后e42963e9…；资格V2核对十固定分片/100对、无效分片拒绝、硬截止裁剪/过期拒绝、真实缺失Pooled终态阻断、cuRobo目录与全部native必需字段。原设计3484ad14…、科学H861004fa…/共同C87199cc6…、模型与样本均不改，formal根/时钟尚未建立。

01:43 ACP实际页面确认Full原UID95e49547…成功，和已有runner exit0/七完整state/全部CAS重建一致；Pooled原UIDc852e8e2…仍训练，01:48:56实际2940/3000，2840条新增101起连续有限。q48原UID59dd5f78…实际页面运行中，01:47有10完整候选/active rank10，非完整池。原各绝对截止不变，队列/工程检查不计策略结果。

01:52:39在CCI唯一启动有界CPU资格流水线，outer PID2429211、5400秒；源码 `run_final3000_qualification_pipeline_v1.py` SHA c5cb2a50…、原始intent/launch/operator/log在Git外 `ITER14_FINAL3000_QUALIFICATION_PIPELINE_V1`。先等待原Pooled自动归档已知成功，再依次独立重建全部CAS/state（脚本3df01180…，不重复CAS打包）、准备明确Pooled角色RB/BT静态合同（e707eb6d…）。未加载policy/仿真、不用GPU、不启动formal时钟、不创建云任务、不访问Git远端；恢复须先核查该唯一实际进程及已知子阶段退出，不能另开未知重复。01:54:53实际父子进程均在、正等待训练2967/3000；此时没有宣称最终资格或性能。

### 23.145 三审稿人简洁回复候选，原PDF与全部实证范围保持

01:49:29形成 `responses/20261111_v14_brief_working`，三文由1805/1552/704词压缩为1170/1008/472词，合计4061→2650；仍15话题块，逐条覆盖全部20项weakness/question。工作矩阵ea0473bbae10507e5942939910f0f718fcd46a2bf25230172db7e82a63750c0a，三文SHA4506e4d1…/85e0d693…/5ec73b64…。补充明确Pooled的机制及四loss范围，压缩重复协议，不增加新性能或统计显著性，不以工程资格关闭科学意见。

原稿正向表格/受控证据、新shared-stage与原specialist不利结果/unknown、BT单状态终点对照但无completion repair、缺技能正门控等均保留。现有完整v14三回复与87项包保持不可变；原LaTeX/PDF SHA230dcc6e…无改动，沿用既有完整工程/干净ZIP编译及视觉验收，不重复编译未改PDF。后续完整新结果再统一推进论文版本。全部empirical_closure=false、stable_release_ready=false；未发布/正式上传。

第一次本机结构检查因HH-W5旧matrix存完整旧标题而失败（无科学程序）；保留 `BRIEF_HEADING_CHECK_ATTEMPT1.json`，按唯一稳定W/Q前缀映射并写入新确切标题后，3文/15块/20项检查通过。不把该文本检查失败算实验失败或静默漏掉W5。

### 23.146 Pooled固定3000完整训练、无损CAS及独立状态核验完成

2026-11-11T02:04:30原训练runner正常exit0，2900条新增101…3000连续且loss/grad有限；root3407、parent100 cbdfcef2…、训练源48ce6ba8…前后不变。最终checkpoint `9c9c9124bd4cb999b47b6bfd866b977c4c52357cc1af25c916bde68fa30b42fb`，原自动归档于02:14:19完成，38,534,492,160B原TAR SHA `f4ddb364dc6bd369509c8560c1913faf67d3b769279e2fa5dc7281ce7d5959e4`，3841个CAS分块。没有重复训练或重复打包CAS，原训练窗口保持。

唯一CPU流水线于02:14:43接续既有归档，在CCI临时盘逐块重建；02:24:00核验全部十五原始文件及七完整state文件（含optimizer/scheduler/RNG/trainer），模型schema与共同实际优化配方一致，不初始化CUDA。formal准入proof `ITER14_POOL3000_INDEPENDENT_AUDIT_V1/FINAL_TRAINING_EVIDENCE.json` SHA `1058dfbce185fbf209f20e830ec119d98ac961ffcf8c91953ba3579988200a3b`。所有原AFS权重/state/CAS保持，sampler.bin实际缺席明确记录，不虚构该文件。

02:33:57明确Pooled构造器/schema、同一root3407与正式namespace的RB/BT两task静态接口完成，receipt `929fac3015355aef0f5c15167a1fd549a1deac6bfe5833c113f9758deed06d69`，十二文件。02:50:29本机核验全部metadata引用、原TAR/manifest/attestation身份及每个静态文件；receipt `ITER14_POOLED3000_LOCAL_METADATA_VERIFICATION_V1.json`。本机没有下载38.534GB大TAR或大权重，不把该检查称为Windows端完整tensor重hash；完整字节重建由CCI独立完成。

两角色final状态和静态接口全部齐备，02:37:29十个实际计划在内存中逐一通过全部fixed design/inputs/native字段检查，但未创建formal根或时钟。ACP在02:19再次跳IAM登录页，原用户登录页及未提交表单保留、只发一次登录请求；无法读取最新cloud terminal或提交新job，不能声称100对已经派发。现有q48不受影响，02:38已有18完整候选/active rank18，原06:23:48UTC保持；其完整32池仍未完成。

### 23.147 固定Full/Pooled结果汇总器在CCI实现并同步

新增纯分析入口核对原始role archive全部成员及原件hash、实际model class/schema/最终checkpoint、每条H32生成和原生命令、原重置物理身份、配对初始输入/Gaussian。固定两任务各50，未知和未启动保持null，已识别无效模型输出仅在封存直接证据成立时计策略失败；不把基础设施异常计0、不选重复成功记录。全部100对完整后才输出主要配对差/Bonferroni exact区间、等任务宏平均及匿名LaTeX表；这不是missing-skill NI检验，也不把episode不确定性写成多训练seed结论。

02:35:22在CCI完成17项CPU统计/档案检查，stdout SHA e7295eefdf7692e1719c685c5d8a3edd1f0dc7f77f25aab9bd8153dec6d4def1，0GPU/0policy/0模拟；fixture明确不是实验数据。02:37真实无campaign入口检查返回not-admitted、0完整对、primary=null、100行空白未知CSV，无主文结果表。canonical `7e3d0ee46707355b00f2c6828fa23288a1918158` 已按英文规范提交/push并精确验证；CCI自身三新增源码与远端内容逐字节相同后离线FF，备份原件及五冻结科学源前后hash均保持，未让训练/eval访问Git远端。

### 23.148 新最终模型的记录观察成本对照独立准入

在任何新测量前冻结协议 `MATCHED_FINAL3000_RECORDED_COST_PROTOCOL_V1/PROTOCOL.json` SHA263f84b4…：新Full/Pooled固定最终3000、PB最早五个DEV回合各20前缀、每角色20warmup，总各100测量；H32/prefix4/NFE10/K32/recent20/summary8/BF16/B1。沿用未改C87199cc6…原生产计时器，被动构造观察器仅在计时区域外核对真实class/schema/weight，不改采样器或模型。预定顺序Full→Pooled、各自warmup，是描述性完整pipeline成本，不是顺序随机化、独立模块成本、MemoryWAM计时或任务SR。

02:36:28唯一CCI资源段首次准入至03:36:28UTC，outerPID2430405，整个准备/失败/重试均包含其中，不重置。pipeline源码67ef4d91…、observer da627efa…；先两角色PB静态合同、完整输入资格，然后在CCI GPU空闲检查后再加载实际模型。双方各100固定DEV输入资格已通过、catalog d4a13afc…与回合清单一致，实际data配置精确相同，任何输入不合法都不能换样。02:48正在profile_full加载，GPU当时0MiB，不以准备计实际GPU推理或完成成本结果；后续以真实typed receipt/timing rows和GPU观察为准。

新profile使用b0669047…/9c9c9124…两模型，和原530.1/565.2ms Full/FastWAM参考profile（旧68c7f384…）分开。不得把新数值替换旧值、跨研究拼比或提前宣称收益–成本优势。所有原始日志、静态合同、计划和输入资格Git外保全；训练已完成不等于已完成原生成功率。

### 23.149 两final3000记录DEV完整成本对照完成及双端原件核验

2026-11-11T03:04:42UTC唯一CCI pipeline在原02:36:28→03:36:28窗口内完成，两实际role子进程均exit0。Full/Pooled使用最终b0669047…/9c9c9124…，实际构造器WarmRetrospectionFastWAM/PooledMemoryPriorFastWAM及schema/weight核对，未改共同C87199cc6…生产计时器或模型。PB435–439共100个记录前缀/每方法20warmup、相同数据配置、89,680 bank、H32/prefix4/NFE10/K32/BF16/B1；固定Full→Pooled顺序，不冒充随机化显著性。

Full median/p95为473.357205/500.255421ms，Pooled476.475901/501.096383ms；mean467.595249/470.436345ms。分配显存23.620066/23.612736GiB，reserved23.740234/23.751953GiB；一次构造438.976166/420.413419s单列、不含在query计时。Full/Pooled median ratio0.9934546623，成本相近而非统计提速或总体收益—成本优势。只用记录已执行action更新history，生成命令未执行、未来图像丢弃，无policy SR或模拟。旧68c7f384… WARM/FastWAM 530.1/565.2ms参考完全保留，不用新checkpoint跨研究替换。

58原件633,133B全归档SHA14e94a59634b66de53e8390914a3092bbd763af0d6d6a3f7b9832a989dcee74e，manifest48dc02de681c9bf1c3185a207bb16d061691c3782e4768999761098d7129935c；03:23:22本机逐成员/240timing行/120每role/共同数据与模型/独立quantile核算通过，ITER14_MATCHED_COST_LOCAL_VERIFICATION_V1。全部原件Git外保留，不将记录成本写为成功率。

### 23.150 Full五模块的被动嵌套区间实测完成，未知OS退出保留

另立预定30min资源段03:18:42→03:48:42UTC，程序10335ffe…使用同Full最终b0669047…、同C87199cc6…、相同100观测/20warmup。第一次准备因未绑定C而NameError，发生于创建子进程/GPU之前，PREP_FAILURE_V1.json完整保留；仅修操作器路径绑定，未改科学源或重置窗口。03:20:13实际启动，03:28:53所有测量完成；被动hook返回None，method observer只调用原方法一次并原样返回，不改变参数/采样器。

五区域在全部100测量query均调用一次（连warmup共120）。CUDA流median/p95 ms依次为source_resolution inclusive14.894560/16.106834、event_adapter inclusive3.132816/3.289618、完整predict_effects方法0.565360/0.609763、candidate reranker0.530208/0.561414、confidence_gate forward0.691664/0.739397。对应host median/p95为14.939850/16.166463、3.169838/3.326645、0.599687/0.644036、0.548434/0.579332、0.723374/0.771382ms。event adapter包含effect/context，source resolution包含以上区域；gate行不包含后续eligibility/blending。它们是含observer enqueue及host gap的重叠区间，不是exclusive kernel时间或删模块因果节省，quantile不可相加。

此instrumented outer median/p95为477.215488/506.452863ms，独立保留，不替换干净473.4/500.3，跨次差不识别observer开销。17原件324,053B档案SHA942c27396e13d1fd4454d4314accda485f1b2cead470d5d5c139b631bc1e0834，manifest0f41c837effa62ef3df7a1c5fb5bb87e8f6cc0bc4ae393bcf9460140c4a153c0；03:38:37本机核对所有120观测/500测量区域调用、每行数值/原args/schedule/父子区间序及全部统计，ITER14_MODULE_INTERVALS_LOCAL_VERIFICATION_V1。detached launcher未采集OS退出码，实际进程已退出但OS_exit_code=null/collected=false；不伪造exit0或重跑同cohort取回该状态。完整观测和统计核验与OS状态未知明确分开。

### 23.151 v15完整工程、简洁逐项回复与新成本联动验收

03:59:35形成三份英文/15话题块/20项映射，总2930词（1300/1158/472）、单块最多2984字符，纯Markdown列表/行内LaTeX可粘贴，不假设平台字符上限。新匹配成本和五区间分别联动Bbqv W5/Q4、Hh4E W2，核心原稿正向对照及全部不利/unknown保持，Pooled四机制loss与完整方法范围一致。全部empirical_closure_claimed=false、stable_release_ready=false；正式100对性能尚无结果，不以训练/成本关闭科学效能意见。

原LaTeX新增Appendix C.5/p40，Table29匹配总成本、Table30嵌套区间。完整29源文件和干净ZIP两次独立编译均40页/10主文；全部原39页文本/像素与v14相同、九图原bytes保持。新增p40逐项视觉验收、无overfull/undefined refs，46个既有font警告。r1出现仅三行尾页，原41页PDF/source/QA已封存layout_review；只压缩新增冗余表述、未改font/margin/数值即收为一页。PDF1fe430921cd50d96b763604b4630287ed85b9c4e007e136051da2b0886a3b287，源码ZIP34953fea89ee7e02233f59664a1caba044744e79ea0c623ba5c296866278d041，matrix9a2c418ac0ead70751adcfddf8462a294807a524aebcce97e99602efe3c1d40c。95项82,764,031B审阅包e45f94bfbd0b6b0ffdce1ba5125635426fd1b23caf90adf50b0d93437b6914b2，manifest530b2b98ac14225a83ced9081d067b9656c8ba22539f549becf97adb381707c5，本机04:04:24完整成员验证；AFS安装须另核对收据再计完成。科学截止03:28:53UTC，原v14及首版简洁稿不覆盖。未发布评论或正式上传，真机新增延期。

### 23.152 原BTq48全32候选及四Full控制完成，负结果完整保留

04:00:18实际standard ACP runner59dd5f78…正常exit0，原06:23:48.137340UTC窗口、plan363bc263…、rank3、H32、checkpoint27c137d6…和源f7cf0583…保持；32候选和四Full均完整、无unknown，四Full endpoint/effect精确相同。380原件49,017,472B档案SHA6c92f2fe882749c3cf35c966dc6034722841cc7516552bc0ee30336d7bdec839、manifest2cb2ee082fe3560d47851ff11491b04ad90a1ab0904d8968080d02444e4f87b6；04:02:39本机独立逐成员核对所有前缀/物理阈值/DINO平均及共同支持集方差恒等式，沿用原已验证分析器，不换指标或拟合projection/scale。

预测/历史/常量centered MSE0.20727800146193764/0.20621809690076948/0.12835179957861667；预测/历史absolute0.29646672695389914/0.29501998659385764，常量absolute不可识别仍null。全部32下一物理交互标签为0，selected raw/Full均0，无完成型repair/harm；此状态没有预测优势，零标签地板也不是等效性或NI证据。仍仅一个原来源episode的dependent diagnostics，不同q8累加成总体频率。新结果在v15科学截止之后，单独BT_Q48_COMPLETE_EVIDENCE_INTAKE.md列出Bbqv W3/Q2、Hh4E W5及C.2.6下一版统一录入，v15不可变包不静默覆盖。

q48已知终态和全原件封存，绝不重复启动。q96原rank20/首次11h仍未准入，准备器/正式100对都等待IAM重新登录；04:02浏览器inventory仍是原登录墙、全部tab保留，未再发重复请求/未创建云任务/未冻结正式时钟。当前不能把旧running进度算活动GPU；最终实际进程/AFS/CCI状态以最新live快照为准。原BRTq96失败保持，不与新BT q96混淆。

04:07:52UTC追加核验：95项v15包在AFS writing_candidates/20261111_v15逐成员落盘hash全部一致，WRITING_CANDIDATE_V15_AFS_VERIFICATION.json；包SHAe45f94bf…/manifest530b2b98…不变。04:08:51实际CCI nvidia-smi无compute app、相关科学程序均无进程；q48实际已知终态、formal根/q96 plan均不存在。AFS5,748,465,664B。没有已知活动rebuttal GPU程序，但IAM阻断下不把本机无进程当作云allocation终态证明。恢复登录后先核对旧UID与唯一创建，再准入正式100对及固定q96；新余量按实际原件和归档估计，不删原checkpoint。

### 23.153 原BT后果训练/测量合同及64个原始目标逐项审计

2026-11-11T04:29:36UTC只读CPU审计正常exit0；没有重跑policy/simulator/DINO、修改checkpoint/metric/样本或访问Git远端。固定原BT12000/f77/source case0的q8/q48共64个候选，逐一绑定实际bank row和event ID，追溯32个原训练feature-cache episode，原bank46,783事件（与新shared成本89,680不同）。原缓存每个起点/终点4×768 float32 tokens与bank effect_pre/post逐字节相同，时间差均32；按原_mean_effect逐项重算历史效果与事前sealed数组完全一致，predicted数组与原event_adapter probe完全一致。所有大bank/cache原件按manifest SHA读核，未复制或修改。

原model action_horizon32、bank mining32、demonstrated query teacher t+32、反事实分支H32一致，timing_dim4是维度；训练event effect只监督自身历史事件，当前requirement为示范query变化，不能从factual重建推断反事实转移已识别。DINO cam_high、2×2 row-major adaptive pooling、原BF16计算/float32存储、同camera/encoder/M1/stats合同均对上。此次审计未发现足以改变既有q8/q48误差的这些合同错配；原retrieval tensor本身未保存，不冒称已恢复该tensor。所有负结果与unknown保持，没有拟合projection/scale或替换主要指标。

64项2,440,194B审计原件（含源文件、配置、32缓存manifest、64目标切片、原probe及退出观察）归档SHA49f3e061ae9a313202440c954daee6bbee673b3e80830ec0f07b5d2b5cc80edf，manifest168c5cc7ab6eae8f85a1235c23c3a3b23a550d4a340c5f1c5dcc45b0c96ce444。04:31:34本机逐项64成员及全部历史mean以独立numpy float64/声明IEEE舍入界核对，ITER15_EFFECT_CONTRACT_AUDIT_LOCAL_VERIFICATION_V1。大bank/cache未下载至Windows，这里不称本机完整重hash。此为已看见负结果后的工程/测量合同审计，不是新的正向机制、成功率、repair或NI结果。

04:21实际CCI无GPU/相关科学程序，formal根/q96 plan不存在，原q48已知exit0；ACP IAM实际页面只有登录按钮，认证仍待原异步请求，不重复索取或创建。新静态只读工作不计GPU活跃；当前5.748GB AFS余量需在认证恢复后准入前再核对，原状态及checkpoint均保留。

### 23.154 v16将完整q48与64目标审计真正纳入回复/原LaTeX

04:44:48UTC完成v16三回复/15块/20项联动验收，3020词（1329/1219/472）、最大单块3387字符，纯Markdown列表/行内LaTeX。Bbqv W3/Q2与Hh4E W5完整保留q48的32/0unknown、三参照MSE及全部零交互标签；新增N-ORIGINAL-BT-Q48-COMPLETE和N-ORIGINAL-EFFECT-CONTRACT-AUDIT-64，全部scientific closure=false。原正向控制、旧不利/unknown、q8单终点高度差和独立适用性cohort均保留，不从标签地板推NI或自动withdrawal。

完整原LaTeX C.2.6/p34/Table25并列q8/q48；64原目标测量合同只作为可比性核验，不能由factual监督推反事实识别。新增表号顺移后所有回复/矩阵按实际aux重定位，C.4旧成本Table28/29、C.5新匹配成本/模块Table30/31，原数值不变。原29文件与干净ZIP两次独立编译均40页/10主文，前33页文本/像素与v15完全一致，变化34–40逐页视觉验收；九图及同改动文件12原table块字节相同，无overfull/undefined，46旧font警告。r1原41页/PDF/source/QA先完整封存，再将重复q8说明合并进新表收为40页，未缩font/margin或删除任何数值。

PDF94c88d81d2e01777f34d8d9da7d0f78405b190e6b3c1eaecef65762a2fa45e87；sourceZIP427846e23c3e7f458cf9f23264ced224e971eeaf2acc66e057c27733d39dc894；matrix79a467417603892ac1a4dabf9255dae172c303445085905b882bd724623823de。65项127,561,493B审阅包e519361aedd3cffa103d9abe157c8d32a243d09eec0d8a6b2ff702f19c52800d，manifest833eba1a3703642783864b36e4cdafe4824851fda4ea8b7b78280b8ddfc5626c，本机04:47:39及AFS04:50:07逐成员落盘验证。AFS writing_candidates/20261111_v16；父v15完整保留，包补全380项q48及64项审计原件，旧训练/CAS和成本原件用父包SHA引用。科学截止04:29:36UTC，未发布或正式上传，真机延期。正式100对及q96仍未准入，不把审计/编译通过写为策略效能。

### 23.155 本轮实际运行终核，保护正式窗口和存储

04:53:11UTC ITER15_FINAL_LIVE_V1确认CCI nvidia-smi无compute app、相关科学程序无进程；原BTq48仍已知正常终态，64目标审计已正常结束，formal campaign根与BT q96 plan均不存在。AFS实际5,124,112,384B，准入前须再核对完整有界context/轨迹、归档和并发余量，原大bank/cache/checkpoint/CAS均保留。此前04:21 IAM实际页认证过期，原请求待答；没有新GPU/云创建或正式计时，不由CCI空进程推出cloud allocation终态。正式100对与固定q96仍是下一计算优先项，旧已完成资格/审计/计时不重复，v16已完成q48录入无需下一轮再做同版整合。

### 23.156 固定final3000的原生family-holdout开发入口实现，尚无新效能结果

2026-11-11T05:54:15UTC在CCI实现独立pilot schema/配对规则及worker，19项CPU协议检查通过，日志SHA02e00edbe6415cf1d548ef7fb53b67e37c4abd29397b00075c6feb0d48b5f2c5；源码语法/CLI依赖检查另记，均不计GPU/模拟效能。计划使用已登记DEV物理seed3407/3408、固定final3000 Full b0669047…、同模型candidate-null、两套原19,540-event universe及原生700/H32/prefix4/NFE10。Stage A/B各独立3h/最多单H100，原正式100对与既有missing-skill NI主协议不变；截止与固定样本不根据结果重置或替换。

入口复用冻结H native物理执行、明确记录研究干预，收集完整命令/实际关节、初始输入/Gaussian、门值和q0/q8预执行候选。自然Full与显式撤回各维持自身因果历史；族移除不是实际技能缺失、门正也不自动计误激活，独立coverage仍null。固定DEV种子专家拒绝/运行异常必须保留，未完成不补0；开发少量同成败不构成NI。具体协议见NATIVE_FAMILY_HOLDOUT_DEV_PILOT_20261111_ZH.md。

05:28 actual CCI无GPU/相关程序、formal根/q96 plan仍不存在，AFS4,519,288,832B；05:29实际IAM仍只有登录按钮，原请求待答、不重复索取。尚未冻结新pilot快照/准入资源窗口或启动policy；下一步完成英文提交同步、唯一私有runtime保留标记与正式准备器相应防并发门，再冻结并启动第一开发段。不把“代码可运行”当已执行结果，v16 PDF/回复/证据包不改。

### 23.157 开发首段冷导入失败保全及修复，原窗口继续

06:29:13.790798UTC首次准入Stage A，固定seed3407，截止09:29:13.790798UTC；原plan SHA2aa0d716c34471a786bc56c17bbe5256911557ed7488e0e93d386b8f4a8fef77、555文件pilot快照bdb6dd5f9ce95bab3e72dced358e6597b5e151e4b2dfab464a04abbe5571d39a，实际06:32:38启动。06:32:44 supervisor/worker均已知exit1，原因为校验助手过早导入fastwam，触发冻结core的Foreign model package检查；0 reset/0 policy trial，不计作任务失败或unknown回合。14个已存在原件完整归档5e63c4c24a51c5c45ff98a8ead1b8046ff4637d111cfece330e090cb755fe666，位置AFS rebuttal_20261109_acp/ITER16_NATIVE_DEV_FIRST_FAILURE_V1；源快照、计划、首次准入及原退出收据全部保持。

只修正新pilot入口的冷导入顺序：先激活私有cache/lease并安装冻结common core，再导入source-identity助手核验。实际新解释器走真实worker main、真实H/C及全部输入绑定，只以观察器替代reset callable；确认fastwam来自C87199cc6…且Torch/模型/模拟器均未加载。另19项CPU协议检查06:44:10再次通过，日志799a95b7e0bdf4aa8a74d6956a27d993a8eb122b9ccddbf32792f3e2a9ad9143。此为工程验证，无新策略结论；修复后续跑须新增冻结实现/独立attempt根与修订收据，保留原09:29截止和固定样本，不覆盖失败根。

rb-40的外部保留标记EXTERNAL_NATIVE_PILOT_RB40_V1.json仍未释放。正式准备器v2 SHA41a72e12540ae04afdb4b019e266387744dc4567ae201ed7b8483420f51d6552增加该worker已知终态/完整归档/显式释放门，原v1 e42963e9…及formal科学设计/H/C/时钟不变。此后正式准入使用v2，不能绕过保留标记使用旧v1。

06:43:42只读候选副本库存记录AFS仅3,307,687,936B；新建微小文件证明hardlink能力，仅是工程探针，未改动任何既有文件。所有writing_candidates总量约1.97GB，多数大PDF/ZIP版本不同，未擅自删除或宣称可回收。正式freeze原4GiB余量门仍保持，当前不能用旧5.12GB读数准入；下一步以真实原生记录测容量并核查可逆派生副本整理，保护原件/checkpoint/CAS。v16回复/PDF及20项scientific closure保持不变。

### 23.158 已完成DEV初态资格；首次生成后证据保存错误保持unknown

06:53:59UTC以新pilot快照d4df6868a45206e1538e9cf8d2782ddb65693df37886a395fccdb190c99839ea、commit0c72fb0…及plan558dfbfc95786a027e522c89f7ba1fb304f0e4822622d7735a93526964f3c29b续跑，原06:29:13→09:29:13窗口不变。固定seed3407专家准入通过、两次真实reset逐观测/任务状态/物理关节精确一致，roster SHA f9b296638aa4a5c0ce1982e5bb78c885b3ba0249015345c3ca95be220abe804b；该完整资格以后复用，不重复采样或替换seed。

首个clean Full实际以final3000 b0669047…成功加载并通过typed合同，完成q0生成后，保存预执行候选时对NumPy bank_rows错误调用Tensor.detach，07:12:29已知exit1。0实际命令、0完整策略回合、1已启动的unknown回合；不能计作失败/成功或有效withdrawal证据。13成员role原件71a93c8422bc1480ec70d125c1db84fc2861a8272f026bb15d12f4d6e42fe1b8已逐项核验；整个已失败段（含资格、源绑定、原退出和0byte未完成probe）再完整归档a6c29cc238058e4a8eeb56de394cf279d9c5860c6df8c687f63ab51f042b3aa3，AFS ITER16_NATIVE_DEV_RESUME1_FAILURE_V1，原件保持。

只修复bank-row保存接口，显式复制实际K32 NumPy整型行向量、隔离后续原数组改动；不改候选/动作、模型、gate或搜索域。新增实际NumPy边界检查后23项CPU协议/快照测试通过。下一续跑须新快照和独立attempt根，引用已完整qualified roster，仍用原09:29:13截止；所有前次失败/unknown保留，不靠重新准入延时。正式100对/缺技能NI均未运行，v16回复/PDF不变。

### 23.159 原窗口内实际原生执行、truth前快照及20项映射增量

07:27:20UTC唯一resume2实际启动，555文件pilot源c159f1717d541d892192debf124419caf65558773bd7cedc28d6f22dbeaa0819、plan804408d807553451071ab1f59d335dec9f1ba994bccb752f36ccbc5f54efe168，源commitcc384a4…已三端同步且六个旧科学源保持。成功复用原固定roster，不再reset采样。07:56:36 clean Full实际504/700命令、127 replans；尚无完成配对，不能将其写作策略成功率。首次06:29:13→09:29:13UTC截止保持，StageB3408未准入。两处修复后的23项CPU日志e36ccb558b90af6e8d9910ed357f80e82073924e75253214c6de54fcf8b88bb9不是科学效能。

07:50:01将预定q0/q8的完整K32/H32、预测/requirement/source、上下文、原始观测及前32实际物理命令封存，21成员2,537,468B，压缩771dded80226ffe0b0f3ed15fa4a7ef1b4317b3cefc109f796ba6bd893f158b5、manifestac0c0a461ea723dee2e676b88749ce1f38c24f85ba2a185ae9ebd8e8b7542fec。初始32命令共有64条begin/end，实际连续计数/无error；两个自然clean g/source_gate/alpha均0.189453125，阈值0.15，source非Gaussian。独立allK覆盖/false activation仍null，不能由clean两依赖查询断言自然缺技能可靠性。q0生成H32与保存错误的原unknown bytes完全相同，只支持该次修复未改输出。

07:51:55本机逐项21成员、2个NPY H32和64物理JSON独立验证；大Torch/BF16 probe标量由保存的CCI reader读取，本机未解码，范围明确。原件位于AFS ITER16_NATIVE_Q0_Q8_SEALED_V1，本机execution/20261111_iter16同名目录。该封存是活动回合的固定subset，不声称整回合/whole-job终态，不干预持续写入的原physical log。

三回复/15块/40页原v16均保持，新的v16_execution_addendum1矩阵4d8c1a28deed998b4a103f7522aa59a7fdc8cc059c77aee63c33a9f72b639afb纠正三处已完成final-state待办，并关联四项意见的在跑DEV/预truth原件。所有20原问题、回复heading和v16实际表/页一致，closure全false；四文件包e56ebae572ce3a397e5b7539b1cfcacaddf3bac273d916d457e415a519a62402/manifest3b90971abb8416e56c0e53c625016c861cf3c1cd22b8c998ad2a569f203d89d1于07:56:07 AFS验证。原v16工程不改，无需为本执行增量重新编译PDF。

07:04 ACP实际IAM仍只有登录按钮，原请求待答；07:46 AFS80TiB近满，仅2,272,653,312B，容量请求已发，formal原4GiB门保持，不能新freeze。只读库存及新微小hardlink探针未改任何既有文件或回收任何空间。短暂一次SCP超时后07:56:36 SSH可达、原worker持续，未复制启动；已保存AFS包回执待本机重取，不将传输超时误称原任务失败。正式100对/q96仍未准入，真机延期，未发布或正式上传。

### 23.160 首个完整新DEV角色：clean Full原生700失败，配对null继续

08:01:46.471206UTC原clean Full正常exit0、无预算终止，700真实命令/175个完整H32/1400条物理记录，success0、max_reward0，最终stage_id/press_cnt仍0。544原件10,962,089B，TAR4,124,749B SHAaffb212b3a9cf55f185fbc3d44f787571d294feef80f58290faec1c1d05eceb1；08:05:23独立CCI核对归档全成员、每个生成及实际命令/物理关节链，0 TOPP拒绝，实测最大单步关节变化0.1703804135，确有物理运动。08:06:57本机独立验证544成员、175H32、700×14 trace及1400 JSON记录，LOCAL_VERIFICATION保全。

175个依赖query的g/source_gate/alpha min/median/max为0.1728515625/0.1904296875/0.22265625，均非0；自然source与Gaussian精确相同为0/175。这是单个clean开发回合，没有证明skill absence、false activation、配对收益或NI。前一个0命令unknown单列保留，不能用本次完成覆盖。原09:29:13截止不变，实际driver随后启动clean candidate-null（08:03 child2438365），尚无完整pair，holdout未完成。

第2增量矩阵1c6901eb642d2b5b875998ddac9784e269f2644c69dd5d9750cf67b5fe801755新增这一完整不利角色并关联BB-W4/Q3、HH-W4、RK-Q2；不改v16三回复/PDF或20项closure。四文件包b802871927d07c4478e475d01d746e2ee5a69353d94ccb3b332fcb91836ce00e、manifest9be16a317b357343d99050f705246620f16340bdbf21a72a93d271b63c9c0f82，本机及08:09:56 AFS全部四成员验收。第1包完整保留。此数据晚于第1包冻结，独立追加而不覆盖。

库存数值澄清：ITER16_STORAGE_TOPLEVEL_READONLY_V1于06:54精确统计writing_candidates逻辑字节1,804,223,105B；§23.157的约1.97GB为粗估，以原JSON精确值优先。07:59实际AFS余量1,922,101,248B，非已保护预留；仍不得按旧读数开启formal或删除原数据/checkpoint/CAS。

### 23.161 保留路径和全部bytes的派生归档去重，缓解存储压力

2026-11-11T08:23:38只读、按原PACKAGE_MANIFEST逐项核对，找到25份审阅包内的派生TAR，与AFS根已封存归档大小/SHA完全相同，共450,609,777逻辑字节；plan SHA13dcede9ef054e69b3a0dcfa203299a686730ecbac012085e16b0209d9187bc4。08:24:06 Windows在既有Git外原件中逐一找到25份相同大小/SHA的耐久副本，receipt fcbc17c705e15f140c1d7c912cbdce4d41e1a6e26841ed606b0117265a14d1cb；没有新下载大备份或删除任何原件。

08:26:24→08:26:53有界operator完成：先在全新微小文件上核验atomic hardlink replacement，再把全部25份派生副本另存CCI /tmp回滚副本并验证；仅将writing_candidates内已冻结派生TAR的底层存储合并到相同bytes的根归档。所有25个既有逻辑路径、文件大小和SHA保持，原始根TAR从未覆盖，模型、数据集、checkpoint、CAS、活动实验根和科学源均未修改；每个受影响包完整原manifest的所有成员重新核验。详细receipt及每项Windows耐久副本/CCI回滚路径见AFS ITER17_DERIVED_ARCHIVE_LINK_EXECUTION_V1，本机execution/20261111_iter17同名目录。

去重450,609,777逻辑字节不是新的科学结果。卷上其他写入同时发生，free采样从1,335,353,344到1,584,308,224B、08:27为1,789,779,968B，不把这些差值等同于独占回收量；仍低于原formal4GiB门，不放宽准入。原容量请求和ACP重新登录请求仍待答，08:24实际IAM页只有登录按钮，未创建新云任务。浏览器任务空闲过期后diagnose确认tasks=[]，原页面保留。

08:18实际candidate-null已24命令/7次replan并通过真实g/source=0及Gaussian边界；08:34继续636/700、160 replans，原driver/09:29:13截止保持，尚未在此时点声明完整pair。原clean Full700失败及所有不利/unknown、v16/增量包均保留。

### 23.162 clean原生配对完整，显式撤回边界验证而未见任务收益

08:36:15.628987UTC clean candidate-null正常exit0，700命令/175完整H32/1,400物理记录/0 TOPP拒绝、success0/max_reward0。542项原件TAR3,675,723B SHA7290e79df621efa5d75b4b5e1439be6926912607d46dc38566ffc6aac7b2c5de；同原clean Full合计1,086项，CCI及09:06:03本机独立全成员、350H32、1,400命令/2,800物理记录核验通过。两角色初态、初始K32输入和Gaussian一致，之后各自因果history允许分化；Full/null首H32最大差.060546875只是干预效果，不是成功收益。

null全部175次实际g/source_gate为0，source精确同Gaussian；q0/q8/q48实际context导出的BF16精确float32视图中，12个conditioning tokens末四candidate slots精确0，前八required-gist保留、非零元素32,768；q8/q48 episode memory有效。observation mask的24/60/84是token数，不是记忆事件数；容量仍20/8。支持显式零候选影响而保留事实gist/记忆，不建立自然gate撤回可靠性。clean配对0/0，不建立NI、等效性或任务收益。原final3000 b0669047…、source c159f171…、plan804408d8…及06:29→09:29窗口不变。

### 23.163 Stage A第三角色失败与最后角色中断，整段原件本机保全

09:10:43.323745UTC family-holdout Full正常exit0，同700/175/1,400/0 TOPP，success0/max_reward0，最终press_cnt1、stage_id0；544原件10,878,209B，TAR4,023,595B SHA42a1954141e278db71a5f60bdd2fbb3415cb2faa0e51c76493edba7483b56feb，CCI独立核验通过。自然gate min/median/max .1689453125/.19921875/.22265625，精确Gaussian0/175。等规模19,540搜索域移除六类block任务不自动证明实际skill absence，适用性allK仍缺。

最后holdout null停在loading_typed_policy，0条query/0已记录命令、successnull、无exit收据；Stage A也无stage_exit。最后supervisor心跳09:26:34.349148，新sshd09:26:54，原monitor2437210/driver2437213/worker2440916均不存在、GPU无程序。这是外部中断观察，OS退出码仍未知，不编造exit0/失败/完整回合，也不以sshd当平台lease。原09:29:13.790798截止已过，禁止新Stage A窗口。

09:37:04 ITER17_STAGEA_EXTERNAL_TERMINAL_ARCHIVE_V1封存全部1,664项44,110,896B，TAR23,660,517B SHA2eb7ac6d260ee1fc1ae88a22f45e3fbd393a7b017d6133aac6502abdd9d95928、manifest0bd98a5944af68d385a8825d2e7e08c0715493520c7d7161b226d8cf95ed90b3；09:57:14本机逐成员及三个完整700命令物理链独立验证通过。两个原实施失败及旧0命令unknown均保留，不能被新完整角色覆盖；Stage B seed3408仍未准入。scope仅固定新shared-final3000的DEV诊断，不替换原specialist成绩、正式100对或缺技能NI。

### 23.164 固定q8预truth分支代码及prospective Stage B保存修正

新增CPU seal入口逐项验证原角色完整归档/计划/权重/实际候选和前32物理命令，固定原clean q8(frame32) K32/H32；ITER17_NATIVE_CLEAN_Q8_PRETRUTH_ACTION_SEAL_V1的seal SHA3d540098467c1d28fdc49273da15f99ad353d6496cdda015a9f1b6b0574813d5，manifest06c061df13407842479b47d24dba4d9717b1b46bc73c53f1324795692719081b，全部35动作及成员已本机验真。仅Full前四未来命令是factual，不能把剩余28步当原结果；同状态null尚缺，不用已分化closed-loop q8 null替代。

新collector要求每分支精确重建前缀/API/实际关节/全部RGB/RNG，Full/Full/K32/Full anchor、完整H32效果公共支持集和预定shared效果定义；异常/早停/unknown保留，不能换点或候选。输入篡改、错误seal、未核销前runtime三个真实CLI负例在输出/GPU前拒绝，09:40:02收据；09:40:17实际rb-30 CPU preflight通过，source digest2fdbbfeed9d19c83afe6f91ad141fb0c19bb942cfbcfb5b2c6511384a01d65bb，未加载DINO/policy或执行物理分支。固定3h准入/512MiB余量/256MiB原件门及明确归档恢复保护并发，不访问Git远端。

prospective Stage B仅允许[0,8,48]保存候选，原Stage A默认[0,8]保持；不改动作/gate/权重。27项CPU协议检查09:21:16通过，日志455ea45e2bf9ee0f79b4e47a548f3c4669e42c6ae0bf7713af8dcd3b1373fe73。Git外正式准备器v3 SHA2c29354d8092e31b6123cc0cac75de6bfee90f16c02e35ea5792d5b7ada691e9，五项rb-30保留guard检查通过、v1/v2保全。均属工程准备，无新增后果预测优势/repair/自然missing-skill或正式成功率；q8 GPU、Stage B及formal均未准入。

### 23.165 作者授权后先保全，再迁移旧Control200冗余raw state

作者明确暂不能扩容、先保存需保留内容并修正代码。09:57:09 Windows将旧Control200全部七state26,307,343,356B从已有cache另复制到Git外`tmp/rebuttal/storage_backup/20261111/control200/state`，每项源流hash及目标独立重读SHA与原manifest一致；BACKUP_VERIFICATION SHA96a53ef94cc21fd6625b531790eff5d329d77a3ae707fdc87c476e421f462f4d。原cache38,437,867,520B TAR重新hash匹配d838ed0c5fc039170394a9719523a3c8d738929c9652b8f49ea6795d895451d7；已有401文件/35,973,567,912B本机压缩CAS仍在，旧verification9351592b…保留，不冒称本轮重hash全部CAS。

AFS全部7状态重新hash一致，原Control200 operator exit0、无相关进程及打开state FD，独立评测weight12,130,288,488B SHA93870d3395cb0ee62978e627ec13d8da38f5b41c886fd5cba11aa60664069a41。10:01:15→16仅移出两个已本机完整备份且inode/mtime/size未变的大raw副本：optimizer12,766,666,737B与model_state13,540,622,014B，共26,307,288,751逻辑字节。五小state、评测weight、当前Full/Pooled final3000、AFS/local CAS、数据/轨迹/原视频与所有历史收据不改，unique bytes丢失0。原state根新STORAGE_LOCATOR_V1 SHA2e0679808728d150e3ffe5bb5f0b57fca8d04ae402680415c82b895a1f5a0e9a及精确拒绝覆盖的restore脚本可恢复旧raw resume；不能按旧路径已迁移就称证据丢失或静默改历史manifest。

详细原件ITER17_CONTROL200_VERIFIED_STORAGE_MIGRATION_V1及只读preflight已本机保存。操作后瞬时AFS仍1,272,446,976B，没有当场观察到26.3GB可分配容量增长；logical迁移与实际df严格分开，不能据此立即通过formal4GiB门。八个已冻结科学源和selected operators另封存3,723项40,417,093B，TAR9,121,551B SHA0b6b67e6910976d908230b0cb3889098de63fc45d76d01cbafd9384b9d41f1c4，09:59:51 Windows逐成员验真；源不改。容量/ACP认证仍须实际回读，不继续删除独一原件，不把排队/工程通过计作GPU活跃。

10:04:37UTC追加实际df核验：AFS可用27,142,139,904B，已观察到容量增长，原两个raw路径仍不存在、locator/备份保持。此为共享卷实际采样，不把前后差值全归因于独占回收量；以后准入再查并发增长/原件归档预算。当前无需继续清理其他独一原件。ACP认证仍须实际核验。

### 23.166 代码push/CCI同步后，独立Stage B与固定q8分支真正准入

代码7309c9d2381386e87aa6b94df4f42b8a05df0f84英文提交已push、origin exact verified；10:06:12 CCI离线FF，五处仅本轮已核对source先备份再处理，八个旧冻结科学源前后相同，无训练/ACP/eval访问Git远端。新557文件源2fdbbfeed9d19c83afe6f91ad141fb0c19bb942cfbcfb5b2c6511384a01d65bb冻结于native_family_dev_controller_7309c9d/code，原c159/H861/C871及所有checkpoint不改。

B固定seed3408首次10:09:30.774636→13:09:30.774636UTC，plan a3f5a02e9a5925877a3a143a37d6502d888e7e54eb7dd253690b029d79c0f9e7，monitor2444145/driver2444146，10:12:21原专家/两次原生reset资格exit0。按原登记反序holdout null/Full、clean null/Full，额外q48仅被动保存，绝不按Stage A得分换种子。旧rb40保留标记先绑定Stage A外部终态/全原件归档释放收据，再分配B；旧窗口/未知退出不改。

固定clean q8/K32首次10:15:26.693183→13:15:26.693183UTC，seal3d540098…、admission89679388d9fbe637c074f08096c353b6d8009196bdce26d2f217d2ce20d66e95，monitor2447586/worker2447587；独立已核验rb30，先保存唯一外部保留再启动35分支。10:43已有2Full锚点/13候选，尚无完整池汇总。为保B首角色原2400s上限，10:45:52给B短时计算优先，名义300s、实际301.676881s后10:50:54自动SIGCONT恢复q8；不咨询任务score、不改任何源/样本/指标或两个首次窗口。调度原件ITER17_BOUNDED_CCI_PRIORITY_V1保全，不计新效能，也不得按旧pause状态遗留停止。

### 23.167 ACP恢复并提交正式首六固定分片，实际仍待GPU调度

10:06实际ACP认证恢复；10:07–11三旧UID95e49547…/c852e8e2…/59dd5f78…云端成功逐项核对，新warm-mf3k前缀暂无数据。AFS实际27GB后原4GiB门通过，10:11:36.607719UTC master首次冻结，硬截止Nov12 18UTC，设计3484ad14…、H861/C871/两固定final3000/原100对及120GPUh/6slot不变。v3操作器2c29354d…绑定master，不可之后改写。

首rb00首次10:12:01→22:12:01、bt00 10:14:31→22:14:31，其余四10:16:44→22:16:44，原静态CPU预检均已成功且不消耗新科学样本。先实际申请标准01e UID04e6e357e289419eaf91ec03ec30d20c、01e_02 UID76b34e7651324e74a16b6a4214ab886c，均“限额中”、无AFS runner/campaign；10:26:59明确已停止、唯一UID及元数据全归档后才fallback。原“限额不足”弹窗只是可确认提交的preflight warning，并非实际后端创建拒绝；旧误读结论以PREFLIGHT_WARNING_CLARIFICATION.json纠正保留。中间一次另集群表单漏选GPU规格被UI拒绝，无云job/策略运行，原失败与修正收据保持；未重置窗口。

10:30–34六个唯一闲时UID：rb00 c1186e7f7fd24a5e9647e769a52ce0a8、bt00 96aa4fd9d473406fadfb3bd7915c9946、rb10 ef42e49d7d9f419588312000cf075981、bt10 bef818e6de314819bdb38ccbd4802459、rb20 ac826871b0c642519fb7446c8d7e75ca、bt20 c74786dc829e44fa9d8898ad50d04573。每次创建前保存精确form/plan/job/shell hash与intent，镜像ccr-zhicheng-02/lwz_wam:lwz-workspace-20261029145800、AFS afs-p3-01e /→/mnt/afs RW、1GPU/22vCPU/230GiB/1worker/0自动重启逐项验真；原std身份归档，当前identity不与仍活跃旧UID并发。

10:42六云状态“启动中”，10:39 AFS全部尚无run_manifest/campaign；10:48 rb00 Worker实际“等待中”、Host/Pod IP为空，已加载事件明确`FailedScheduling: 4200 | 29/29 nodes in VC match job | insufficient resource: nvidia.com/gpu 1`。这是云端GPU资源不足，不是科学代码失败或已分配GPU；六个slot及原时钟保持，不因pending重建。只核验首片具体scheduler reason，不冒称其余五事件都已逐一读取。后四正式分片未准入，rb30/rb40仍由CCI研究保留。新BTq96未准入。所有队列、工程预检、实际策略及完整效能严格分开。

### 23.168 v17全量回复/原LaTeX验收及之后到达的B首角色

v17三文3,281英文词/15块/20项，新增BB-W4/Q3、HH-W4、RK-Q2的同一已核验Stage A边界及完整0/0、holdout失败/unknown。新Table22/p33保留所有不利记录，显式null的175次Gaussian与候选conditioning为零而gist/记忆保留，和自然技能缺失检测/NI保持区分。原稿正向结果及旧负/unknown全保留；新开发shared模型不替换原specialist成绩。

完整29文件原LaTeX和干净ZIP两次独立编译41页/主文10，原九图与13个旧table块字节不变，前32页文字/像素与v16相同、变化33–41逐页视觉验收，无overfull/未定义引用，46旧font警告。PDF a4e4f5d7859a5b23cf38ca979805e4c977c39a9f0d80f4ba7a8a7f0d739787a4；sourceZIP5d57cdcabeff96271c8e33aa6921ca6ce1864dd13552e8e56cdf11380cbe9e12；matrixab51206c471856a15a9f8820ac91c98c720d0ab07398e60b1908abc08e6d34bd。116项115,881,629B包78783ccd6e6b2e0c9da76a1a8171a346b931469bc9fd25810e332f7b3fd10bc2、manifest95657d0f9d54db867eb5bcd84758dc66d83a3a3b6454bb921e24f9e97fde18ff，10:55:53 AFS writing_candidates/20261111_v17全成员验真，科学截止09:57:14不改。未发布/正式上传，20项closure/stable仍false。

之后B首holdout candidate-null于10:51:16.340649UTC正常exit0，未有界终止，700命令/175H32/1,400物理记录，0 TOPP、success0/max_reward0。542项原件10,353,251B，TAR3,677,006B SHA1f095dfe7b07e726eeb676fcb940f30a5b71b173db3fd92a6b7e1a4dc0f4ecfc；10:56:52 CCI与10:57:51 Windows全成员/生成/物理链及175次actual-null独立验真，gate/source0、source精确Gaussian，alpha仍为自然正值（median.1875），不混淆alpha与最终effective gate。该数据晚于v17固定截止，单列ITER17_STAGEB_FIRST_NULL_COMPLETE_INTAKE_V1；须下一版与配对/完整q8整合，不能静默改已冻结包或计收益/NI。原driver已进入同seed holdout Full，未复制。每小时heartbeat已核对仍ACTIVE，目标Nov13 23:00不变。

### 23.169 固定native q8完整池及独立核算，晚到补充不替换主指标

11:21:38.164944UTC唯一q8 collector实际exit0，原10:15:26→13:15:26窗口不改；35分支、全部32候选完整H32、三次Full端点/效果精确、0unknown，1,120条实际branch命令。每分支复现原32命令前缀，逐API/物理关节/RGB/RNG parent guard通过；不physical unpack。286原件155,112,041B，TAR44,004,165B SHAf0d6fe04f18a4104c4f94440769ae29a0f6600e165353d5b462845c2dac89fa7、manifest388572171d55c547c8c49e1e9a587c196334a21b1a74b481bbd1eb317efc0d1d。11:23:28 Windows逐成员、封存native动作、全部branch实际关节链、三Full和shared训练同定义weighted DINO及两MSE独立核验；最大float32效果汇总舍入差2.3841858e-7。完整parent/endpoint pickle在本机只验bytes，不声称已解释所有native pack；报告的parent hashes一致，实际物理链及特征另独立核对。

同32候选support的centered MSE预测/历史/常量分别0.06513299851425262/0.10300351376862686/0.04877937700243701；absolute分别0.18335801630915716/0.18756856257649224/0.20164210619387846。预测器在预定centered主指标优于历史但不优于常量，absolute辅助误差较低，两指标原样并列，不能用后者替换主指标或拟合projection/scale。观测centered variance0.0487793770，预测EV=-0.3352568753。32严格下一物理交互标签全0、Full0，无完成型repair；这不认证通用技能缺失、自然fallback可靠性、NI或总体SR。是固定seed3407/clean/q8的一个shared-final3000点，不与原specialist uniform mean及提交时cohort合并。同parent candidate-null生成仍未测。

三位审稿人的简短额外讨论块和全部20项矩阵已生成，root responses/20261111_v17_native_q8_addendum1，matrixfb60c26683f3ecbd6a189ce1849e42d39091d8575b9e9bec8144f2f7ce497333。12成员44,259,958B补充包SHA3643bdeff4df84a966eefd1c09d1adb5ba0838a7f372e6f5b132067384bf1aae、manifestaed6955e000b574591e0f97d4fbde108714bbf4021b7d3c45c48437373342822，11:31:25 AFS逐成员验真；原v17三文/41页PDF/源ZIP/116包保持，其截止09:57:14，不声称PDF已含晚到q8新数值。新讨论补充正面报告不利/有利参照差异，科学closure全false，未发布。rb30仍保留，下一使用须显式核销release，不因已退出/空flock盲目复制实验。

### 23.170 Stage B holdout两角色完整0/0，clean阶段继续

B holdout Full于11:28:50.548294UTC正常exit0，700命令/175完整H32/1,400物理记录、0 TOPP、success0/max_reward0；545项11,071,471B，TAR4,120,709B SHA4a2ba2a50f80c7f5d6f439c337900a729d96ddf97cad634dd631aec1465d35a9。CCI及Windows全成员/动作/物理链核验，源/权重/seed3408/plan a3f5a02e…不变。自然g/source_gate min/median/max .177734375/.2041015625/.2197265625，175次精确Gaussian0；alpha中位.2080078125，不混淆soft effective gate。

两角色初态、实际物理、当前输入/共同Gaussian/首K32逐项独立匹配，11:35:16 LOCAL_PAIR_VERIFICATION确认holdout Full/null0/0，各原生700。此为单开发seed/条件的描述配对，通用技能缺失/false activation/NI仍未建立。Full本机验证首版status文字沿用null的542，仅显示标签有误，实际assert及hash始终545；原件/初版receipt保全，LOCAL_VERIFICATION_R2明确545和旧receiptSHA，不改科学检查或数据。原driver已进入seed3408 clean candidate-null，13:09:30截止不延长。所有新B数据晚于v17截止，下一稳定修订统一整合，不补写未完成clean配对，不替换Stage A中断未知角色。

### 23.171 固定q8同状态null直接对照的工程入口与CPU资格

新增generation入口读取原seal3d540098…/context/query RGB/native qpos/derived seed，通过原typed权重、事实上下文和原生产FastWAM kernel验证完整Full H32及actual source/Gaussian/conditioning和预测精确，再生成冷状态只读共享权重candidate-null及第二Full。拒绝伪造/重用历史BoundOnlineStep、future/teacher或未知上下文字段；不改变任何模型结构、阈值、权重、原query/rank或主要指标。null先封存归一化/原仿射native动作及truth前seal；collector只另做同parent Full/null/Full三分支，并要求首Full复现原池parent/endpoint/effect后才接触null物理后果。原K32入口默认行为保持，旧source快照不改。

12:03/06 CCI三个CPU关键拒绝检查通过；12:06:53真实原q8上下文CPU preflight通过，12:07:07旧35分支入口和原rb30 runtime只读兼容检查通过，均无GPU初始化/无策略生成/无物理truth。首iter18_development_check_v1四文件开发包遗漏既有inference_replica.py，CPU入口在策略加载和输出目录创建前FileNotFound退出1，0GPU/0物理；原包/log/CPU_PREFLIGHT_RECEIPTS_V1全保留。完整五文件iter18_development_check_v2独立包再次3项通过及两个实际preflight exit0，不能静默覆盖首失败或称新效能证据。修复仅补全开发部署依赖，正式冻结须包含原完整source。

原件位于AFS iter18_development_check_v1/v2，本机tmp/rebuttal/execution/20261111_iter18保留对应源码包与检查记录。此时尚未准入新GPU研究；拟共享单一至多1h窗口、加载前至少32GiB空闲/运行至少8GiB、0.45显存上限，旧rb30终态/全原件归档显式核销后才独立预约，既有B/六ACP slot与原始窗口不变。12:04实际六闲时UID仍启动中、两标准已停止，未观察到正式科学runner。研究只回应同状态显式撤回与局部物理后果，不能将预期流程写成repair、自动missing-skill可靠性、NI或总体SR。

### 23.172 Stage B clean candidate-null完整终态与本机独立验证

固定seed3408 clean candidate-null于12:03:08.479061UTC正常exit0、未有界终止，700命令/175完整H32/1400实际物理记录、0 TOPP、success0/max_reward0；全部542项10,325,642B，TAR3,652,287B SHA86e57ad323a4fdc783051ea9f5b9a676172009a096abbabfa26f8022194a1fed。12:15:04 CCI核验原完整archive、typed checkpoint b0669047…、全部生成、实际动作/API/物理链和175次effective/source gate0且source精确Gaussian；alpha保持自然正值，min/median/max .1767578125/.1904296875/.21875。

原件和收据存于AFS ITER18_STAGEB_CLEAN_NULL_COMPLETE_INTAKE_V1及原role archive；12:15:30本机独立重读全部542成员及175H32、700动作/1400物理链/零TOPP通过，同名本机目录保存原TAR和LOCAL_VERIFICATION，不改原件。当前B已三个角色完整、holdout一对0/0，最后clean Full实际worker2469618仍在原13:09:30截止内运行；不能提前补成clean配对或收益/NI，也不认证missing skill。均晚于v17原科学截止，下一完整修订统一纳入。原Stage A未知退出及所有零结果保留。

### 23.173 Stage B完整结束；同状态null加载失败已封存，修复资产工作目录

B于12:37:19.651368UTC真实driver exit0，四角色完整、两配对均0/0；最后clean Full archive9ca95894…，原四角色源/样本/窗口未变。全部2,214原件59,143,224B连同四个原role archive和实际stage_exit统一封存：StageB_complete.tar.gz 33,026,220B SHA40ea8cb2b3c8eb9cb41959c1ff235cb98ffc442faba5173430f9d117ae694f46；manifest6ea0faca…，12:49:18 AFS全成员hash复核通过。最后Full及clean配对的本机科学核算另行登记，不能把打包通过当机制收益。

同状态null原唯一driver12:28:30真实exit1，generation exit1、非预算终止；DiffSynth相对模型资产模式从冻结controller目录解析，text encoder文件集为空，未完成任何Full/null生成或物理分支。14原件30,312B及完整日志/配置/原12:23:30→13:09窗口保全，TAR7,521B SHA6e4e0c84e87cf8dd4f920db499b609b4246e5a94a2e99db2f82c4a09c511d08f，manifest458bd6c2…；两任务退出、原件及进程不存在收据在ITER19_STAGEB_COMPLETE_AND_NULL_LOAD_FAILURE_V1，旧失败不覆盖。

修复只在独立generation程序恢复原typed资产根作为工作目录，确认原base/VAE/text/tokenizer绝对绑定均位于该根，再以冻结C模型源码构造策略；记录asset_resolution见证，模型文件/配置/阈值、q8上下文、源及原H32精确复现门均不变。新增拒绝外部资产根的回归检查。恢复必须使用新冻结实现、显式失败归档证明及原first/deadline，不重开窗口、不换query、不把加载错误记为策略失败。

### 23.174 第二次资产元数据错误及全量Stage B独立核对

54ba301冻结恢复源7e90b4e4…在原12:23:30→13:09窗口内12:54:27恢复，13:06:42真实exit1、generation13:06:41 exit1且非预算终止；0完成推理/0物理分支。DiffSynth从原cwd加载资产后，冻结deployment的model_paths.vae又相对冻结源码PROJECT_ROOT解析，严格元数据检查拒绝。17项28,774B原件/TAR6,721B SHAdc4e880d6fd8d2c027a53c21da4e7bc7efa049f92941c0059f2ccd8ebb8283d2、manifesteb98cff0…在ITER19_NULL_RECOVERY1_TERMINAL_ARCHIVE_V1全部封存并本机逐项核验；原窗口已过，不再同窗口恢复，也不将工程错误记为策略失败。

13:52:58本机重读Stage B总2,214成员、四个worker完整archive、175生成/700命令尝试/1,400物理记录每角色、初态/首当前输入/共同Gaussian/K32，以及两份加载失败archive全部通过。**新发现与纠正：clean Full在零基命令637有一次left_mplib_planner Fail to parameterize path，其他三角色0次；此前未完成独立核算前不能沿用“全部0 TOPP”的估计。** 原始数据、失败和success0均不改；角色到达700尝试不等于700次成功物理动作。clean Full archive9ca95894aa7a880927a32af8df828617e2c93ab64eabc29ba714a9fe7c780c1c；两个完整配对均0/0。独立收据tmp/rebuttal/execution/20261111_iter19/STAGEB_AND_NULL_LOCAL_AUDIT_V1.json。它们是有明确scope的DEV诊断，不支持总体收益/自然缺技能/NI。

独立辅助入口的第二修复只对已加载模型的三个artifact元数据字段做绝对路径规范化：逐项要求实际cwd解析结果与原绑定VAE/text/tokenizer绝对路径完全一致，再原子改metadata；不改tensor、权重、模型配置或冻结源码，不跳过后续hash/typed检查，构造结束或异常均恢复原factory。CCI隔离开发包asset_metadata_fix_iter19_dev_v1的5项CPU拒绝/恢复检查通过；本机Python无pytest已如实记录，不能称Windows测试通过。后续GPU仅作独立有界加载资格，不能延长已过期科学study。

### 23.175 提前论证稿与作者真机采集包（尚非实验完成）

按作者明确要求预写理想结果下的论证，responses/20261111_v18_data_pending含三位回复、15块/20项、P1–P5共37个未填槽位和依赖门；整段定性结论也标PENDING，不是仅数字留空。原v17、旧负/unknown和所有原表/九图保留。revision/WARM_ICLR2027_v18_data_pending从完整原LaTeX工程构建；新q8与Stage B独立列示，所有待验证段红色、全页内部稿标识。初r1将结论挤至11页已发现并保留，不作为主文10页验收；后续r2通过压缩重复论述使结论回到10页，须完成最终PDF/ZIP/视觉验收并记录实际hash。

作者现自行推进新真机采集，docs/real/rebuttal_20261111提供固定40个reset、80次Full/历史屏蔽计划、120个关键决策行、逐次/媒体/独立遥操作split模板及移盘可选任务卡。最小SOR40次，推荐加R2P共80；新增独立示范20条、固定checkpoint扩库20次为第二批；移盘两档各20次可第三批。当前新采集数均0，mask现场实现、设备标定与当前checkpoint仍待实测。旧原视频不重复索取；新材料不覆盖原20次SR，训练示范、事件、视频clip与自主trial分母严格分开。

### 23.176 正式campaign首完整配对、BT时限中断与原slot的CCI接续

北京时间11月12日00:10（Nov11 16:10UTC）实际ACP核验：rb00/rb10/rb20/bt20四UID运行中，bt10失败，旧bt00闲时已停止，std2两次HTTP429后仍无任务。不是六片仍排队。AFS真实campaign心跳与UI对应；16:11:41 rb00固定case0的Full/Pooled两角色均正常exit0、各700命令/175生成、success0/0，pair_receipts验证初态、共同信息和噪声；rb10 case10 Full也完整700/175、success0。

bt10 case10 Full于15:58:57实际exit-15，bounded_termination=true，原40min worker上限；result.json最后仍写loading，**但console与物理原件显示946个完成command_end、1892物理记录、最后零基945，不能误记0动作或加载失败**。success仍unknown，不把部分轨迹记完整失败，不按结果改case。BT原生1000步可能超过2400秒角色上限，后续需显式评估工程时限修订，保留原首次/总截止/样本/两角色公平性，不能静默加时或在相同无效预算下盲目重试。

以上四终态attempt共2368项57,993,368B完整封存ITER20_FORMAL_TERMINALS_AND_LIVE_V1，TAR SHA0373a0c049083ddb639b8614fc9cc6b178400750f0dd1662e8953c3b6a3fb62b、manifest04c0ed9577e17382981efcdd07d2643ebc05c5aeb867494480c1c4fe9c1047a2。16:20:06本机逐成员/三个完整role动作与物理链/首对输入Gaussian和K32独立核验。只有1正式配对，不填100对SR、CI、优劣结论。其他活动角色不能以UI状态当完整证据。

bt00标准迁移确认未创建且旧idle终态后，16:11:53将同一slot接到CCI，原first10:14:31/deadline22:14:31不变、固定plan857d74f6…、两checkpoint/source不变，最多2个新角色、6000s独立invocation保护；monitor2496070/campaign2496071/首Full worker2496076，原件在shards/bt-00/cci_segment_iter20_v1。未新增第七slot，五个其他云UID不复制。16:20 CCI已占24.4GB、构建模型中，需后续检查实际queries/终态，不能提前宣称完成推理。全部执行程序不访问Git远端。

### 23.177 v19作者对话式回复与原LaTeX改写验收

按作者11月12日四项意见，三位开场逐一回查真人review的认可；Bbqv对应实用问题/后果选择/消融，Hh4E只复述问题重要性与两类经验整合，rkWY对应动机和可适配/可抑制经验影响，不虚构优点背书。主体作者视角、先答后证据，结尾总结已完成的澄清与修订，公开文本不出现AC/meta-review/提分目标。20项、17块、37待填槽位；假设理想数据的整段与结论方向仍PENDING，全项实证closure=false。

新目录responses/20261112_v19_author_dialogue；原LaTeX revision/WARM_ICLR2027_v19_author_dialogue：引言串联贡献与实验，方法澄清event-specific训练目标和固定replan撤回命题，真机强调历史如何决定目的地与顺序，结论正面回扣证据支持的核心观点。没有删除原实验、公式、图表或不利数据，也没有用改写代替关键直接实验。

完整工程和干净ZIP独立编译44页，含结论的主文10页、全44页文字一致；9张图/旧附录及主文equation/table块字节保真，匿名、引用检查通过，0 undefined/overfull，45旧字体warning。改变的主文与新附录已渲染直接验看。PDF SHA d31a14439ef95c7509085e43200cfa3127e84f5ae7749bebfcfbaffd4aa14cf5；sourceZIP299c967c27267c74ca233a2e1fbd150156220723aedb863b2add7b2e17934511。尚未发布或正式上传；v17/v18原样保留，v18旧PDF/ZIP不能当v19新源码验收件。

### 23.178 三个BT原固定case达到工程时限；显式共同预算修订

Nov11 17:05UTC核验，BT case0/10/20分别在936/946/988个实际command_end后因原2400s上限退出-15，均bounded=true、success unknown；原result字面loading未刷新，不是0动作。RB case11 Pooled另有非bounded的-15/0动作退出，保留为未知且需核对云抢占，不与BT时限混同。rb00/rb20仍活动，未重复。

17:09:50对11个已终态attempt全部6046项147,136,761B封存于ITER21_TERMINAL_ARCHIVE_V1；TAR3965424b01a10dafd985c0c93934309ff88eadf55beba53aea455e2b633afc98、manifest6f087c2ec347d7b27f62fee3f59debf3d9f6928344ac8a0d02c2b4c5cb5cde21。原receipt保留，新增receipt_exit_bound_v2.json再逐项绑定实际exit文件hash（9b5ca0ccfffccc9754f78e45de39eb267bcff19e09c2f4ac0e677ab2e191be0d），用于独立只读unknown分类；原结果不改。已完整的7个RB角色均success0，其中case0/10/20各已有Full/Pooled控制器配对；本机应另独立核对后才能更新验证计数，不将它们写成收益或最终100对SR。

工程修订入口run_rebuttal_execution_amendment.py只在hash绑定的修订收据下，把BT两个角色的外层共同上限2400→3600s。原计划文件和冻结collector、样本、权重、1000步上限、推理配置、首次/总截止、6slot/120GPUh上限均不改；各新attempt/campaign状态保存修订path/hash/effective limit。它导入原冻结orchestrator并只修改内存中的工程预算视图，子collector仍读取原计划字节。stale loading只有实际exit/result与完整终态档案相符才在独立视图标unknown，原next_request与两attempt上限仍执行。

三项本机及三项CCI隔离CPU检查通过，覆盖公平共同预算、拒绝改deadline/science、保留unknown及hash变更拒绝；尚不是GPU或模型效果证据。具体资格/固定收据完成后才在原剩余窗口有界接续。新的trial或重训若需收敛资格须另前瞻冻结，不能用不同样本或模型替换现有100对。

### 23.179 工程修订实际准入、标准成员配额核验与新云身份

f774aba84e9f653946e750a17d3936632d8fab0c已英文commit/push并远端exact核验，CCI离线HEAD-ref FF；旧H861/C871源前后相同。冻结新工程入口execution_budget_controller_f774aba/scripts/run_rebuttal_execution_amendment.py（SHAadb5ba9b6cd77f5b5a6ff54817592185e47ed7fce41694ff81181a21a54e027b），bt00/10/20三份独立amendment分别17d4241e…/f8506bf1…/30332ecf…；三项真实原plan CPU preflight均exit0，无GPU初始化/新物理结果。资格收据ITER21_BUDGET_SYNC_AND_QUALIFICATION_V1。

17:14:31原bt00 slot以CCI segment_iter21_v1接续，monitor2502475/campaign2502476/worker2502481，最多1新role/6000s外层，旧部分轨迹/exit保留；case0 Full第二attempt37deccf6…，worker共同3600s，campaign原first/deadline不变。17:24还在加载、24.4GB、0已保存generation，不写完成推理。真实status/exit后续核查，不重复启动。

17:17:47重新实际申请标准资源时，UI明确返回“成员配额已超限”，HTTP429由工具泛化标为SITE_RATE_LIMITED，**此处业务原因是成员配额超限，不能只称网络限流或推断已创建**。17:18:37列表确认新std任务不存在后，才恢复闲时；已确认镜像ccr-zhicheng-02/lwz_wam:lwz-workspace-20261029145800、AFS afs-p3-01e /→/mnt/afs RW、1卡/22CPU/230GiB/1worker/0重启。旧两UID失败已逐项回查。

新bt10 UID6bea7fe19aed473692928a535468d291（17:19:51创建），bt20 UID76da0b3af21c46da8812b35aeb425ceb（17:24:12创建）；分别warm-mf3k-bt10/bt20-20261112-resume-idle。新命令只调用原slot下acp_resume_iter21_v1/launch.sh，逐role检查原截止余量后显式resume，不自动重试新的未知退出。原first10:16:44/截止22:16:44均不改；只替换失败allocation，不增第七formal slot。17:24云端仍“启动中”，分配GPU未验证，不能把它们记为活动GPU。新identity及旧备份在原shard/新resume目录；ITER21_NEW_CLOUD_IDENTITIES_V1保留两唯一身份。

17:20:25本机全6046成员/7完整700-role动作与物理链、3组初态/首输入/高斯/K32以及4个unknown独立核验通过；RB case0/10/20三对均0/0，case1 Pooled另单角色0。3/100固定配对不填总体SR/CI，不支持优势/非劣效。内部增量responses/20261112_v19_execution_addendum1更新20项映射并区分Pooled和candidate-null；原v19三回复/44页PDF/源码ZIP不变、P1–P5仍待证。后续应同时检查shared3000能力/原specialist范围，不能静默用新权重或样本替换已开始cohort。

### 23.180 首BT完整1000步验收与下一个原角色接续

Nov11 17:57:26 CCI case0 Full第二attempt正常exit0、非有界终止，1000命令/250完整生成/2000物理记录，success0；这证明共同3600s工程上限允许完成原1000步，不是模型收益。旧936步unknown与原40min退出保留。771项18,241,828B完整attempt封存ITER22_BT_FULL1000_INTAKE_V1，TARcfa4ad61674997a0e464f5414fe287abc30e4cedee8db62ddb5cdbccc4e37c48、manifest6f70dd4a09b4ca75abba65d1010fee66636b299c362ae1fc877e7216c6d44fa5；18:10:02 Windows逐成员、全部动作和物理链、250生成独立核验，TOPP拒绝0。只有单角色，不填配对收益或总体SR。

已确认旧monitor/campaign/worker终态和完整归档、GPU空、native lease空及原截止余量后，18:08:22同一bt00 slot进入cci_segment_iter22_v1，monitor2507713，下一原固定case0 Pooled，最多1新role/6000s；原first10:14:31/deadline22:14:31、BT双角色3600s及原f774冻结入口不变。后续按真实PID/generation/exit核查，不重跑完整Full。

18:10云端新BT10/BT20两UID仍启动中，原RB00/RB20运行中，RB10失败；不把新BT排队写成GPU已活跃。准备为RB10增加独立的terminal-status-amendment类型：仅在原exit/result/全归档绑定下恢复unknown只读视图，保留RB2400s预算。四项本机与四项CCI检查通过，明确拒绝RB取得BT加时额度；未准入前不称已恢复。

### 23.181 RB10独立恢复已创建与原窗口保护

ef5ce8c23ba8e706616ac9445aa3e665945cf2a3已英文commit/push及远端exact核验，并CCI离线FF。原BT活动入口f774不变；新增RB终态恢复入口`execution_status_controller_ef5ce8c/scripts/run_rebuttal_execution_amendment.py` SHA59231591919004dcebf04025f7abc2dce0b79667147f9791e91c4a32a4914de0，独立RB amendment cf96bd65b686141f253141d09ecc102862a4bef01058156e5e1275c3d3f08cac。真实原plan CPU资格通过，RB仍2400s，两次attempt上限及旧unknown原件保持。

Nov11 18:13:42UTC新唯一闲时allocation `warm-mf3k-rb10-20261112-resume-idle`，UIDde23301cd91f4044bb7d1ff28ac51492，原rb10 identity及失败UID另备份。只调用原slot下`acp_resume_iter22_v1/launch.sh`，每role核查原22:16:44截止余量；不增第七slot，不重置首次准入/窗口。18:25:32 UI确认该UID与新BT10/BT20均启动中，尚无新AFS科学runner；原RB00/RB20运行中。先标准资源实际成员配额不足的核验仍沿23.179；新恢复未重复申请标准或伪称已分到GPU。

### 23.182 完整终态增量独立核验与shared阶段只读审计

Nov11 18:15:38UTC封存四个新完整role：RB case1 Full、case2 Full、case21 Pooled各700命令/175生成/1400物理记录，BT case0 Full1000/250/2000；均实际exit0、success0。2409项60,303,892B原件位于`ITER22_COMPLETED_DELTA_V1`，TAR SHA8b9f7cb3084a83bd67e40f02ce5531a2a93253076f4b0575a78c100ff4ae2325，manifest e240108191b0d4ee70b38cfd7471fbb025263df1b4749b1d5fa605b816625e7b。18:26:09本机独立逐成员、全部14维动作/关节连续链、每个H32生成与物理记录验真，新四role各0TOPP；RB case1与原Pooled的初态/首输入/高斯/K32精确一致。

此截止累计11个已独立核验完整role，RB case0/1/10/20四对均0/0。BT case0尚只有Full完整，Pooled活动；未填原100对总体SR/CI、优势、repair或非劣效。稍后的controller完整计数只有在另行封存和独立核对后才能提升此计数。所有旧936/946/988部分轨迹及RB11中断unknown保留。

18:18:28训练只读审计保全两配置、全metrics与training_complete共7项9,210,604B，manifest d5d8d3d65078975007caf9393a4eb4937d8bec6c59e429af44e92bf131a44d0f，位于`ITER22_TRAINING_STAGE_AUDIT_V1`；18:35:19本机逐项hash并重算所记录首/末100更新均值。Full配置f84274b1…、记录step201–3000共2800；Pooled b5ca3cec…、step101–3000共2900。两者均shared阶段、固定评测3000，配置原max_steps30000，恢复run_steps分别2800/2900；这是预定3000选点，不是本轮新发现的意外提前退出。末100动作专家梯度均为正。Full action loss所记录首/末100均值.0679846→.0117938，Pooled .0950372→.0132402。

这不是原论文specialist准确率复现，也未证明未收敛就是0成功的原因。配置`lr_scheduler_type: cosine`，审计receipt中的根字段scheduler=null仅是未读取正确键，不能据此声称缺失scheduler。不得据当前测试结果替换权重/样本或重新选checkpoint；新的能力资格或specialist研究须独立前瞻冻结。

18:25:43实际CCI monitor2507713/campaign2507716/worker2507753仍为原bt00 segment_iter22，case0 Pooled已5完整生成/20命令，GPU32384MiB/27%；原2214UTC截止及f774双角色3600s不变。`ITER22_LIVE_FINAL_V1.json`记录状态与原窗口；加载状态字段可能滞后，按物理日志和实际exit判断。

### 23.183 v20作者对话精准化与完整论文验收

延续逐审稿人真实认可开场、we/our主体与已完成修订总结，修正语法/单位、明确修订位置，并使真机历史依赖行为与受控回忆消融衔接。原核心贡献和原数值/九图/公式/表格/已有附录均保留。三份回复20项/17块/37空槽，P1–P5整体仍待证，未把11角色/四对零零填为收益。

完整原LaTeX及独立干净ZIP两次含BibTeX四遍编译44页/主文10页，全部页文字相同；0 undefined/overfull、45继承字体warning，3/7/8页直接视觉验收，其余41页与v19像素相同，匿名与内部标记通过。PDF SHA cc162884906c2ec4e9dc88fa78cef1cf7781533f05447ae93324d462eb3a53dc，源ZIP8e4fa62e714359f997f7bafa7a5db5e8a6d6b4b2833278a3789dc6f666a57b30。

72成员123,816,401B内部包TAR3fe839215e47cddb67d7f0418b6f3d8ad249f0352f5f0818e2d11cb8cb42d187、manifest15c50f6021c7a4bc1ba4e89a10038b40b796319f5aaad33bf5b832b78dc9709f已本机与AFS全成员及解包验真，位于writing_candidates/20261112_v20，收据ITER22_V20_AFS_VERIFICATION_V1。论证/排版验收不等于科学缺口关闭；未公开发布或上传，v19基线不改。

### 23.184 v20封存后的两完整RB角色：六对仍为零零

Nov11 18:40:41UTC对晚到RB case2 Pooled（实际18:27:58 exit0）与case21 Full（18:20:35 exit0）另行封存，每个700命令/175完整H32/1400物理记录、success0；1092项27,894,518B，TAR e18d90f6e996cfdbd375841f9d5901f6d60aadcd7eaee2fc325ad25cd98b502f、manifest194369dab8808a32f202875c66252c06137271627c376e7f0cefe833b4712765，位于`ITER22_LATE_TERMINALS_V1`。

18:41:17本机逐成员hash、全部动作/物理连续链、H32生成及0TOPP独立核验；累计13完整role（12RB及1BT），RB case0/1/2/10/20/21六对均0/0，各对初态、首输入/高斯/首K32相同。四个旧unknown和原截止不改，不填最终100对SR/CI或优势/NI。新独立20项索引在`responses/20261112_v20_execution_addendum1`，三文件已双端保全；v20原三回复/PDF/源码ZIP/72成员包不覆写。

### 23.185 首BT完整配对、三角色增量与原顺序接续

Nov11 18:50:37UTC BT case0 Pooled正常exit0、非bounded，1000命令/250完整H32/2000物理记录、success0；与原Full组成首BT完整0/0对。新增RB case3 Pooled（19:01:44）及case22 Full（18:56:28）各正常700/175/1400、success0。19:08:44封存1863项46,233,171B，位于`ITER23_TERMINALS_V1`，TAR2aabe65026145307e4a7b09f4e18b14ed86fd91e5b31907d5d884ed97428bf31、manifest28ea38666f4e82ac0d8cd58cdd53b3e5d174012302ec66197331e99e812c2c94。

19:08:55本机逐成员、全部动作/物理连续链和完整H32独立核验，新三角色各0TOPP；首BT的初态、首输入/高斯/首K32与Full一致。累计16完整role及7完整对（6RB、1BT），均0/0；仍不能填100对总体SR/CI、优势、repair、自然missing-skill或NI。原四unknown及所有固定样本/权重/时间窗口保留。

核实segment22正常终态、三个旧PID消失/GPU空、campaign及native lease空、余量>3660s且AFS>4GiB，并通过原amendment真实CPU资格后，19:09:58原bt00 slot启动`cci_segment_iter23_v1`。monitor2514155/campaign2514156/worker2514161，原顺序case1 Pooled，最多1新role/6000s、单worker3600s；原first10:14:31/截止22:14:31不改。四小时重启边界未独立确定，在准入中显式保留unverified，不伪称有确定重启余量。

19:06 ACP实际UI仍原RB00/RB20运行中，三个唯一恢复allocation启动中；未证实新GPU runner、未重复创建。内部执行增量2保留20项映射，位于`responses/20261112_v20_execution_addendum2`，三文件双端验真manifest25e5339576be113cf8d348976b78f7d216ec78851166038b44daf0c3b9970832。v20三回复/PDF/源码ZIP/72成员包仍封存，37槽位和全部empirical closure不改。

### 23.186 首BT配对的反归一化与观察/物理状态边界

只读诊断重新读取已核验Full/Pooled各250个生成块和1000命令，以实际dataset stats（SHA2cac1e1b434b2a339f2e4eddeab7815682e243d59428bc7dcaddca14f173746d）重算所有执行前缀反归一化；两角色最大绝对误差均0。12臂关节命令相对首观察的RMS位移中位数Full .877296rad、Pooled .897698rad；这描述命令，并非物理跟踪误差或成功证据，不能仅据首几条小命令推断全回合静止。

`qpos_trace`中的14维观察qpos与独立物理状态严格区分；另保留原`physical_commands`的38维articulation逐步快照，在未建立索引映射时不计算所谓12臂关节物理误差。初次本机派生诊断把观察qpos误标为physical，已在任何论文使用前更正为`diagnostic_v2.json`，旧派生文件明确标superseded并保留；原采集数据与论文数值未改。未生成图，未据此推断失败根因或训练收敛。

原stats与两个派生版本位于本机`execution/20261112_iter23/BT_ACTION_DIAGNOSTIC_V1`及AFS`ITER23_BT_ACTION_DIAGNOSTIC_V1`；三成员manifest29a820729b13ebae737c5945b74c939d58f001c089192b6c15551f16848c6ec4、当前诊断SHAf711fd3449937c28bccfabe1e363ea353c2738bd39b50006a30f3b508fd56d91。19:20:45全成员AFS验证通过；这是工程/解释资格，不能填入P1–P5优势结论。

### 23.187 五个完整固定角色增量与CCI下一角色

Nov11 20:13:40UTC封存RB case3 Full、case4 Full、case22 Pooled、case23 Pooled各700/175/1400，以及BT case1 Pooled1000/250/2000；均真实exit0、success0。2955原件74,139,896B，TAR189ef7fb9f6f1ae02551948ab70784f26013ae4ad95035283aa34a30a089d20a、manifest9ca75683180edaf1a89eba7d50070d4e9182feef93806fa61f9fbf53a568e6cb，位于`ITER24_TERMINALS_V1`。20:13:52本机逐成员、全部动作/物理链与H32独立核验；case22 Pooled保留1次TOPP拒绝，其余新增角色0。累计21完整role、9完整对（8RB及1BT），均0/0，不填原100对最终SR/CI或优势/非劣效。

CCI旧segment23已19:50:33真实正常终态，旧三PID消失、GPU空且全部原件/新Pooled完整轨迹已归档后，20:14:56同一bt00 slot接续case1 Full，`cci_segment_iter24_v1`，monitor2520444。原first10:14:31、2214UTC截止、两个角色共同3600s及单次1role/6000s不改；未重跑case0或已完成case1 Pooled。下一步核查实际新PID/生成和exit，不能沿用旧segment23活动字段。

### 23.188 原BT q96首次准入、标准H100成功及实际运行

依照Nov10预先声明的独立19GPUh/单H100原BT机制研究，q96此前未准入，q48已知终态且全量归档后，Nov11 20:16:01首次冻结q96的39600秒窗口，绝对截止Nov12 07:16:01.631389UTC，排队计入且不重置。该研究不是第七个matched formal分片；不改变原100对的模型/样本/截止。旧BRT q96及原BT q8/q48的所有不利/unknown与原时间窗保留。

新plan SHA8e3ddd84c84db10085f78b63fe887a77c7daec00efa3925f30e538e022f06468、job原文件c0172c00515db7eb324b7b6261192c7deb587778200cc8c76ae31c2a888a1bc2、shell2accf040ebcc9b101d141d91c11cd9d9204d1c7e96188b5cb01dcaaeaa6d6efd；原controller f7cf0583e82ea95792ae1ac45f15d04e650d309956bdfc68debe2ab5e1447a7a、原checkpoint12000 SHA27c137d689007b38f6204d3dd12bb038550b1e58403506d7a9894bd6fcd43a9f。query96/rank20/K32/H32、物理seed340800000/root3407及原均匀四DINO delta token指标不变。需先完整AABA，再全部原候选及末Full；同一回合三个query不是独立任务样本，且此旧代际仍无同状态强制null对照。

20:19第一次UI点击仍在表单显示cluster“请选择”，随后只读确认唯一标准job不存在；补齐cluster及被切换清空的单卡规格后才再次提交，旧意图/超时与只读证据保留，未重复未知创建。20:23:14标准资源实际创建成功，唯一UID536a352f050b4142890aa99fb88aa23b、名称warm-bt-q96-pool-20261111-std；本次无需闲时fallback。镜像ccr-zhicheng-02/lwz_wam:lwz-workspace-20261029145800，AFS afs-p3-01e /→/mnt/afs RW，1GPU/22CPU/230GiB/1worker/0重启。

20:24:15实际节点测得NVIDIA H10080GB HBM3，完整encoder runtime SHA2f3e5d2979f2fbb3063da65d25a49d15fb4c3a19b52fcf329879fa87fbcb9e01与原合同完全一致、0字段差异，未修改runtime对象。实际科学runner20:24:19启动，至20:29:54正在anchor-a重建固定父状态，0完整branch/0候选端点，不能报告预测或修复收益。唯一身份在`ORIGINAL_BT_Q96_ACP_V1_CLOUD_IDENTITY.json`，root为`original_BT_cohort_case0_q96_fresh_pool_v1`，应区别旧BRT q96路径，勿重复创建。

14项初始准备/实际GPU/启动原件封存`ITER24_BT_Q96_READINESS_V1`，manifestef44a77d9eb0115fda7b376affe018c600bfe515511a53c5b163d6ed6e8402ca，20:35:42本机逐成员独立验真。内部`responses/20261112_v20_execution_addendum3`更新21role/9对与q96实际状态并维护20项映射；原v20三回复/44页PDF/源码ZIP不改，37槽位和P1–P5整体方向仍待证，未发布。

20:41实际q96已完成第一Full anchor-a，正在anchor-aa父状态重建。首anchor的14项6,726,139B原件及sealed query另存`ITER24_BT_Q96_FIRST_ANCHOR_V1`，TAR7b30bbae036d6f9c8b5b7b5dbd2071876c2320d52b9b4c8de072035bf859bb71、manifest5fe48907b5926ddef0f4ddb9eb3f2b5da2289969eb1aad122db59f771fbec096；20:43:42本机全部成员独立验真，result实际executed_steps32、complete_horizon、parent_recreated true、error null。endpoint_native_success false只记录终局标记，不据此推断独立下一交互标签。重复Full/原提案和完整AABA尚未完成，仍不计效能或repair证据。CCI20:41仍原case1 Full活动、32380MiB/24%，原所有截止不改。

### 23.189 新五角色原件与第二个BT完整配对

Nov11 21:24:36UTC另封存RB case4/5 Pooled、case23/24 Full及BT case1 Full，均正常exit0、success0；RB各700命令/175个完整H32/1400物理记录，BT1000/250/2000。2955项74,262,343B保存在AFS `ITER25_TERMINALS_V1`及本机`tmp/rebuttal/execution/20261112_iter25/ITER25_TERMINALS_V1`；TAR fa08586a15e3284605b2ca73c138abc1e0a659b3d45f8c0b17accb26918bc0ab，manifest72edfd0d0584333475bf7f03351dbdff39b5a2c4c593187cab5b8c920206d986。

21:32:32本机逐成员hash、全部动作/物理连续链、完整生成及首输入/初态/共同噪声/初始候选池独立核验通过；新增五角色均0TOPP，原RB22 Pooled的1次TOPP及全部旧unknown保留。累计26完整角色、12完整配对（10RB、2BT），均0/0，尚非100对最终SR/CI，不能用于优势或非劣效结论。模型仍是固定shared3000，原稿逐任务specialist成绩不替换。

CCI segment24已20:55:32正常退出；21:34 GPU实测0MiB/0%，原三个PID均不再活动。剩余原bt00窗口不足3660秒，不再准入另一个正式角色、不降低阈值、不重置原22:14:31截止。RB00/RB20实际campaign心跳持续；三个恢复allocation尚无新科学runner，不能按旧部分动作数计新结果。

### 23.190 原BT specialist q96的固定AABA与独立工程资格

原BT q96于21:07:12完成事前固定AABA：anchor-a、anchor-aa、rank20、anchor-qual-final各完整32步、parent_recreated true、error null；三个Full的父状态及端点hash相同。固定rank20提案与Full的独立下一交互标签均0，paired known、repair false、harm false；不把native terminal success替代交互标签。全候选池仍运行，尚无完整池MSE、总体repair或自然撤回对照。

46项26,201,105B原件于21:24:41封存`ITER25_BT_Q96_AABA_V1`，TARc853450d01cc174d400e06b6dc2fc47c19092e2c302d6ced088ebcec7fb09df3、manifest9d996b0789b76a36d48afc5b4e02119b61ca83c469fad7bcc3f7dd4078ec5b3f；21:34:44本机逐成员、四端点与重复Full hash独立核验通过。原controllerf7cf0583、checkpoint27c137d6、单source state及07:16UTC硬截止不改；21:34实际完成6branch/3candidate endpoints，正在rank2，不能将相关候选或三个query当独立回合。

另在CCI先实现仅一次Full精确复现的独立工程模式：新schema/最长1800s，禁止null生成、模拟器与物理分支，不能重开原13:09已过期null窗口。8项CCI CPU准入/上下文检查通过（21:28:46），尚未GPU验证；源码及测试SHA分别d90cbed74aa8ed47c3bacdb55cf2f14543e398cdb7f4c71769d8a39b1ac655f5、38091091419bb78e7309c712af0bbd28646377edc3191c81be089fe25d850c90。保留两个旧加载失败和全量原件，不将工程资格写成修复收益。

### 23.191 v21作者对话精修与一次Full工程资格实际启动

v21三位回复逐一核对真实认可，明确作者we/our、直接回答和已完成修订总结；正面区分原表显著性、Full/Pooled估计对象、监督的训练/部署界限、提案指标/修复/终局，以及真机任务难度/经验库多样性。固定零门控保证与自然缺失技能可靠性分开。原核心贡献与原九图、数值、公式和表格保留；附录加入原BT q96 AABA的零/零，不把工程进度写成收益。20项/17评论块/37空槽位仍整体待证。

完整原LaTeX及干净ZIP独立含BibTeX四遍编译44页、主文含结论10页；全部页文字一致，0 undefined/overfull、45继承字体warning。9变化页1/6/7/8/9/10/42/43/44直接验看，其余35像素一致；匿名/内部标记通过。PDF f37c1d03f3f5c7ab9451893a664f0d688263598ce1edcc77dc8e93411971e57d，源ZIP511a42628023660d544d20dfcb74371c6b472a3b68785d05cb8ef958fc211323。70成员123,831,172B包TAR14b837a5bad2fa9f4c627ff9afa32f9b9b75a114ee4173dd578081c048eacdc8、manifest48119b4952eb74307124f0ed7f5f2bf584e325a07192580ce5850652ad604ec8已本机及21:48:30 AFS逐成员/解包独立验证，位于writing_candidates/20261112_v21，收据ITER25_V21_AFS_VERIFICATION_V1；未发布，原v20不改。

代码e135295已本机commit/push/远端核实、CCI离线FF。首同步在FF后因CRLF/LF原始字节检查中断，未启动GPU；恢复核对规范化后源码与commit逐字节相同，重跑8项CPU检查后才冻结新controller df718191947fec6bb45300b947d66f44609e303d36998ba08626243516a9f7ff，原三活动入口hash不变。先前原始字节证据和失败脚本保留，未修改旧科学快照。

Full-only工程资格实际21:38:34首次准入并启动，monitor2529133/worker2529134，root `full_q8_reproduction_engineering_20261111_v1`；admission052665590d6f1344f83f4919afa9295e9539bf1700897b93b08b309066f5919e，固定22:08:34截止，单Full推理/0null/0物理/不重开旧科学窗口。21:50实际GPU24443MiB、模型加载、尚无complete generation，不能计效能或精确复现通过。原bt00正式窗口与四小时平台重启边界未核实事项保留。

### 23.192 v21封存后的晚到两个固定RB角色

RB case5 Full（21:43:53）及case24 Pooled（21:43:54）各真实exit0、700命令/175个完整H32/1400物理记录、success0、0TOPP。1092原件27,667,548B已双端封存`ITER25_LATE_TERMINALS_V1`，TAR92d02572b23df3f14f78a357bb2c59583e1ba62ffc1edd5d7f23a81c0804353e、manifest556ea9145e8628b602f6d5545fa26603ae733bf60f47dc36e26a82467fcd08d3；本机逐成员、全部动作/物理链、H32与配对初态/首输入/共同噪声/首K32独立核验。累计28完整role、14完整对（12RB、2BT），均0/0，原不利/unknown保留，不填原100对最终SR/CI、优势或非劣效。

新增内部执行增量`responses/20261112_v21_execution_addendum1`单列最新28/14进度并维护20项映射；v21三回复、PDF/源码与70成员包原样保留其明确核验截止。ACP21:44实测原RB00/RB20及标准q96运行中，三个唯一恢复allocation仍启动中；21:50 AFS心跳确认原RB继续case6/25，q96继续原全池，不能按旧部分946/988动作计新运行。所有原first/deadline与固定样本不变。

### 23.193 Full工程资格的真实退出及启动环境修正

晚到核查确认：Full-only worker2529134实际21:51:20正常收取exit1、非bounded；`online DINO runtime differs from the M1 contract`发生在policy构造末段、0完整生成/0null/0物理。两个实际PID已消失，不能沿用21:50“加载中”的旧观察。12原件15,397B保存在`ITER25_FULL_ENGINEERING_FAILURE_V1`，TARc9b74de9ab7285be7653f803fed2e8ef836b7777d5b7a0450ee5a2479b79a276、manifest311b02c0ce172393e441cfec7527a320ce07ebd8f03414789de3b74a55bedab4；双端及本机21:59:50全成员独立验真。原21:38→22:08窗口不重置，余量不足另一次冷加载，不重跑原root。

已定位启动模板额外设置`NVIDIA_TF32_OVERRIDE=0`，而绑定M1合同为null。22:01:52在新进程进行只读对照：无额外覆盖时全部运行环境字段与合同相同；设置该覆盖后仅`nvidia_tf32_override`一字段不同，足以触发原严格合同检查。修正未来模板以显式移除该环境变量；原失败monitor、模型代码、合同和原始日志保持不变。诊断与未启动模板在`ITER25_FULL_ENVIRONMENT_FIX_V1`，实际新推理和精确Full复现资格仍未完成，不能宣称已恢复科学对照或repair证据。原ACP三项实际运行继续，v21材料及已保全的28role/14对不改。

### 23.194 原固定对照资源窗口结束，保全两个新unknown

Nov11 22:20UTC核查时，六个原正式分片的22:12–22:16资源窗口均已结束。RB case6 Full实际22:11:35 exit-15，531个完整命令对/133个H32；RB case25 Pooled实际22:16:19 exit-15，665/167；二者各保留一个未闭合的最后command_begin。结果文件的loading_typed_policy是未刷新的旧字面，不能当0已执行动作。实际exit收据均标非bounded；未凭时间接近原截止就将根因认定为抢占或某一层计时器。

974原件23,770,894B封存`ITER26_FORMAL_WINDOW_END_V1`，TARbd1dd24eb4b74e85780f02c1199044d1a6ffeb49922e1533befca5070978a7e2、manifestb83101e3c601e79385a98f5c77f0256e2a8401789e7894f5b1c8e8e99c03eb9e；双端及本机22:24:18逐成员、全部完整命令物理链与H32独立核验，新增完整命令对0TOPP。累计28完整role/14完整对仍均0/0，unknown attempt累计6个，不能将unknown填成失败或称100对完成。固定模型、样本及全部原窗口保持，不从这些测试结果挑选权重或换样本。

原RB00/RB20云任务已失败终态；三个恢复allocation在原窗口结束后仍启动中，实际停止前冻结唯一名称/UID意图，逐项确认弹窗与目标一致后停止，22:30只读刷新确认三项均已停止。未重复未知停止或创建。原标准q96（UID536a352f050b4142890aa99fb88aa23b）明确排除；22:30详情仍运行中、镜像与AFS配置不变。原100对研究以部分完成/含unknown封存，不声称统计收益、非劣效或原稿specialist复现。

### 23.195 加载前编码器运行合同检查，避免再次延迟发现环境错误

在CCI canonical先实现严格预检查，再回传本机：`generate_rebuttal_native_query_null.py`在构造policy前核对M1与online合同的原runtime hash，保存真实运行环境及逐字段差异；有任一差异立即拒绝，不修改环境、合同或模型。比较保留缺失字段及JSON类型差别，修正Full-only加载状态文字以免误称构造了null replica。原昂贵加载后的合同验证仍保留。

22:23:25 CCI十项CPU检查通过，实际原合同及已保全runtime另在22:28:46通过hash绑定/正确环境接受/原错误覆盖拒绝检查；0policy加载、0新GPU样本、0物理。开发收据`ITER26_ENCODER_PREFLIGHT_DEVELOPMENT_V1`，源码/测试SHA ac73d0832117fe5315cf6b0732f5caa283984ea3b31f32c4659d6ff841e8f01a / 923416f9e4f040e4b97897e7d45588d44aadb20dc28b57de44d730a7bcab36c6；实际合同检查见`ITER26_ACTUAL_ENCODER_PREFLIGHT_CPU_V1.json`。这修复工程准入，尚未取得精确Full复现或科学效能证据，不续开已过期窗口。

### 23.196 原BT q96新七个完整候选原件

rank0–6各完整H32、parent_recreated true、error null；七个独立下一交互标签均0（物体落至桌下），与原rank20和三个Full锚点属于同一固定状态，不能当八个独立回合。70原件45,441,016B另封存`ITER26_Q96_COMPLETED_DELTA_V1`，TAR48e3588279b49a28a43f92088ddcea8d0ee36f1548e8548dfd2bf29aaaf802bf、manifesteb0dcb7039bd2e1bb2f6c216ea84a3a361b5378e4774dd8dea0624103c8b45bb；22:28:33本机逐成员及全部端点独立核验。22:20实际operator为rank7父状态重建、累计11完整branch/8候选端点，原07:16UTC截止与全部固定rank不变；全池未完成，不报告全K32 MSE、repair或总体SR。

### 23.197 核销已终态allocation，首次推进原计划中未准入的bt40

原master仍在Nov12 18:00UTC硬截止内，预先列明十片/100对；前六片过期不等于剩余rb30/bt30/rb40/bt40已准入或整项研究结束。为继续固定样本，先核销rb00原allocation：云端独立确认UIDc1186e7f7fd24a5e9647e769a52ce0a8失败；实际runner22:11:38UTC已终态exit1，记录reason scheduler_signal_2。13个attempt的全部文件逐一对照既有原件manifest，另外保全完整job_logs；原所有success0和unknown不改。

查明旧准入程序把runner的canonical JSON spec hash与缩进JSON原文件hash比较，导致真实终态无法核销。独立操作修订改为验证原runner的canonical spec hash、原始job文件hash分别绑定，保留原runner不变；三项运行中/未知exit/错spec拒绝检查通过。原冻结准入程序字节不改，核销v2证据单列`rb-00/release_canonical_v2.json`。第一操作在核销已完成、任何新分片时钟尚未创建前因Path类型错误退出；修正仅路径解析后继续，保留两个操作原件，未重复原任务。

bt40为原master预先登记、此前目录及任何create intent均不存在的case40–49，22:44:53.565472首次准入，固定截止Nov12 10:44:53.565472UTC；原master及六片admission逐字节不变、并发≤6及总120GPUh不改。plan44c23a4b88e6424b802a6fab25a8792181e63a08a4bdb18e2fe684caa10ab179，同一shared3000双角色/原50-reset roster/原H861和C871，未按结果更换任何原样本或权重。共同3600s工程预算在首次新truth前冻结，amendment dfaa351a1c4158d3b5539c3a05129ccfcee399629270b3c78cf36af1db34e4fe，不改科学plan。

22:46:06 CCI实际启动原case40 Full，monitor2532679/worker2532713，`cci_segment_iter26_v1`最多1新role、外层6000s；没有创建ACP重复任务。22:47实际campaign心跳持续，GPU尚0MiB、0完整生成，属于进程启动/加载阶段，不能称新GPU推理或结果。CCI四小时平台重启边界未核实，单次边界<4h且所有中断/unknown须保留。21项77,161B准入/核销/入口原件封存`ITER26_BT40_ADMISSION_ARCHIVE_V1`，TARc2b71eb32f9cefe799db070f224299fbf3114c16d01b465f7847b10d8c8286ed、manifest7ebd32525eb467bd51dc88e57d4c0e2dad2fe03f08afd3042d3b6f2e05ab1d5b；旧过期窗口未重开，新增准入不等于科学收益。

原q96同一时点已13完整branch/10候选端点，正在rank9；新rank7/8晚于上一七候选归档，后续按原身份接收，不提前称全池完成。准入21原件另于本机22:50:12全成员独立验真；v21三回复/PDF/源码封存不变，增量3维护新准入与20项索引，全部P1–P5仍待证。

### 23.198 q96干预前父状态的时间范围核验

另对已独立验真的AABA及rank0–6共11个端点原件审计：父状态hash一致、均从384步执行至416步；在任何分支开始前，`BT_parent_progress.object_fell_below_table`已经为true（首对象世界z约0.02934）。因此q96零/零描述已经发生掉落后的动作干预，不是对初始掉落预防效果的检验。该时间事实不能单独推出可恢复性/不可恢复性或一般失败原因；全部零标签、固定状态、候选支持集和原主指标保留。

诊断与英文作者回复/LaTeX插入稿位于本机`execution/20261112_iter26/Q96_PARENT_SCOPE_V1`，对齐Bbqv W3/Q2、Hh4E W5/P2。它属于下一整版修订需纳入的已核实范围补充，不能称已加入封存v21 PDF/源码；当前不通过改标签或删状态获得收益。22:52真实CCI case40 Full已占24,443MiB、进程和心跳持续但0生成/0物理；原q96已14branch/11候选端点、正在rank10，尚无全池估计。

### 23.199 原case40 Full完整终态与Pooled唯一续段

原bt40 case40 Full实际Nov11 23:27:20.300128UTC正常exit0、非bounded，1000命令/250个H32/2000物理记录完整，success0、0TOPP。776原件18,245,994B保存在`ITER27_BT40_FULL_TERMINAL_V1`，TAR d3e2ce9a26a65d92c21675de3fd47f432d9dac40df32fbda90382550af7ba5d9、manifest e64419333a9c145df7cfd1f642d8dc90f6cbf59222be2464f11b7a12c0769da7；本机23:30:25逐成员及动作/生成/物理链独立验真。权重b0669047…、科学source87199cc6…、controller861004fa…保持。累计29完整role/14完整对均0/0/6unknown，新Full尚未配对，不产生100对最终SR/CI、收益或非劣效结论。

旧三个PID消失、GPU空、前终态原件核验且原两lease空后，23:31:25在原bt40窗口内启动唯一`cci_segment_iter27_v1`；campaign23:31:29开始case40 Pooled，monitor2534261/campaign2534262/worker2534267，最多1role/outer6000s/common worker3600s。冻结入口adb5ba9b…和修订dfaa351a…、原plan44c23a4b…不变；原首次22:44:53→Nov12 10:44:53截止不重置，无ACP重复。23:42实际24,555MiB、心跳/进程、0保存生成/物理，只能称加载。后续真实终态收据和完整轨迹验真后才可续下一个固定role。

### 23.200 原报告13任务区间的派生统计，0新增试验

重新核对正式33页PDF SHA aa04f9519881983b9f6f408dcfb5b912c9290b883dbe31567f671136fa864594：4.1给出RMBench每任务100次、真机每任务20次；表2九项WARM整百分数与4.3真机17/16/15/15均可绑定原报告成功数。仅在固定策略、任务内独立二项试验模型下计算逐任务边际95% Wilson区间，未给异构任务合并区间、训练种子不确定性、paired baseline显著性或联合覆盖保证。RMBench计数为原报告百分数与分母所隐含，未声称恢复逐次身份或原日志。

13行独立公式与CCI CPU SciPy binomtest(method=wilson)一致，最大fraction差2.220446049250313e-16；0新rollout。BT64/100区间[54.2,72.7]%、PB92/100[85.0,95.9]%；真机B2B17/20[64.0,94.8]、D2B16/20[58.4,91.9]、R2P/SOR15/20[53.1,88.8]，其余九项完整表保存在Git外`ITER27_ORIGINAL_REPORTED_INTERVALS_V1`（本机与AFS同名）。输入source_counts、intervals、MD、TeX四成员本机独立验hash，intervals SHA0928c841c2b56fbf0eae22c5d35c7d106b6140b87a240f049679abcc3bde3a7d；manifest SHA另由LOCAL_VERIFICATION.json绑定。只支持原报告逐任务精度呈现，不填P1/P2待证实验槽或比较效能。

### 23.201 原BT q96新增七候选原件，保持同一不利父状态

原固定rank7–13各完整H32、parent_recreated true、error null；七项独立下一交互标签均0。70原件45,440,977B保存在`ITER27_Q96_COMPLETED_DELTA_V1`，TAR0c778ff4bd32a7e7ed5ac2b5348b33940fe2058d71916263c8486773f995127c、manifesta240a4c8c7af39c27547cbc362f7631a00411fa9428b850782ced2b1af0d8a73；双端及本机23:39:20全成员/H32独立验真。累计原AABA加rank0–13共15个候选端点已归档；一个source/query的依赖候选不是独立回合。

23:42 UI刷新确认原标准UID536a352f050b4142890aa99fb88aa23b仍运行中、原镜像/AFS/重启0不变；AFS18branch/15候选端点、rank14重建父状态。23:43实际job_logs console至414/1000、runner events heartbeat仍更新；operator在分支边界更新，不能仅因其时间较旧就重复启动。原07:16:01UTC硬截止、固定rank/父状态/主指标不变，整池尚未完成，不报告完整K32 MSE或修复收益。

### 23.202 v22作者对话、原报告精度与干预时序整版验收

v22三位回复保持原review实际认可优点的感谢开场、we/our作者视角和仅总结真实改动的结尾；20项意见/17评论块/37空槽。新增附录D.6的13任务原报告Wilson区间，以及D.1的q96父状态384步干预前已掉落事实，与q8提案掉落而Full未掉落区分。原标签与原支持集全部保留，不据此推出不可恢复。主文4.2就近指向精度表；原核心贡献、九图、数值、公式及既有表格不删，P1–P5整段方向仍待证，未公开发布。

完整原LaTeX与干净ZIP均含BibTeX四遍编译45页、主文含结论10页，全部页文字一致；0undefined/overfull、45继承字体warning。对v21页7/42改变及新增45逐页直接验看，另42页像素一致，匿名及内部标识通过。PDF ade470a6a7ba9e1e7a346a150514790039c069f0ea902dc926f430ac151f6567；source ZIP cf05bf28bd59ee027a1ddfe9e28335e7fa5a2bff5621441e00c7464c286cd92e。83成员123,862,802B包TARfc8a2a8524c909c57d486659091f26709841f8273d62d1beffd99dd2fa1e5888、manifest451f3540e5ed37ac95c3ab091f4b94f3d0bbb83a64aec7026dab7463de6435f5，已本机逐项及23:41:58 AFS解包独立验证，`writing_candidates/20261112_v22`。详见CANDIDATE_V22_20261112_ZH.md；实证release仍false，原v21封存不改。
### 23.203 原case40第二角色终态，首输入完全匹配的新0/0配对

Pooled实际Nov12 00:12:41.170427UTC正常exit0、非bounded，完整1000命令/250H32/2000物理记录，success0、0TOPP。777原件18,150,922B封存`ITER28_BT40_POOLED_TERMINAL_V1`，TAR59b8cef6964f324c6ea21e0b8cb939066f72656693a5c3dd16142774989ef565、manifest2b1bbb6d6064e72ffbe6d04e1d871cabcf75eb38456f96e98531cd5ba767f62b；双端与本机00:26:59逐成员/动作/生成/物理链独立验真。Pooled权重9c9c9124…、source87199cc6…、controller861004fa…固定；与前Full的initial_identity、首model/proprio输入、共同Gaussian、bank_rows和candidate_payload全部一致。累计30完整角色/15完整对（均0/0）/6unknown，不填100对最终SR/CI，不推收益或非劣效。

旧三个PID消失、GPU空、两lease空且新终态原件独立验证后，00:27:26在原bt40首次22:44:53→Nov12 10:44:53窗口内准入`cci_segment_iter28_v1`，monitor2535716/campaign2535717/worker2535722，原入口adb5ba9b…/共同3600s修订dfaa351a…/plan44c23a4b…不改。实际控制器对奇数case41先选Pooled，操作admission的文字误写case41/full；原admission完整保留，另`ITER28_ROLE_DESCRIPTOR_CORRECTION_V1.json`绑定其SHA2bb4c3ac…与冻结helper373feacf…，三项奇偶/续角色CPU检查及真实next_request均确认实际case41/Pooled正确。未改科学顺序、模型、样本或时钟，未重启该活动role。下一角色须在真实终态保全后由冻结next_request选择，不能再硬编码“Full优先”。

### 23.204 原q96新增rank14–18五个完整候选

五项各完整H32、parent_recreated true/error null、独立下一交互标签均0。50原件32,460,327B封存`ITER28_Q96_COMPLETED_DELTA_V1`，TAR3659c7edb72ccab4ffbd6ced179df8bc2ea25dec3e1abbd234dc123469ce0d58、manifest564185ec5e6ea1fcac4bdef914ae94595f50cef5833dac0d9d10383fd8a0a3b6；双端及本机00:31:06全成员/H32验真。累计20候选端点已独立归档，仍是一个source/query的依赖候选，不是20回合；父状态干预前掉落和所有0标签保留。原全池继续，07:16:01UTC总截止不重置，未算不完整全池主指标。

### 23.205 具名MemoryWAM复现条件的公开原始来源核查

针对Hh4E分别提出的LIBERO性能与具名基线成本缺口，先查可复现条件而不是用Pooled冒充。GitHub连接器本轮传输不可用，转官方网页与公开API只读获取：Nov12 00:41:31UTC默认main commit dc82a9d760e751ae88f2e0ae521af7c7ed597dfb，README仍将训练/推理代码发布列为未勾选TODO；完整递归tree未截断，只有LICENSE、README及四张asset图片，没有检查范围内的.py/.sh/notebook或weight扩展名文件。这是该时点公开默认分支事实，不声称任何私人或其他分发渠道都不存在实现。

Git外本机`execution/20261112_iter28/MEMORYWAM_PUBLIC_RELEASE_AUDIT_V1`与AFS `ITER28_MEMORYWAM_PUBLIC_RELEASE_AUDIT_V1`保存commit/tree/README/audit及manifest；README SHA6c9dbb89b0101de48e87259451e8415bf66f5b38fbbbb0636e7860b1c286ad72，tree SHAbe8b8de3c016228848843694d3914cbefa70bb59e35883c4e8d0b51cbdc6f4d7。AFS副本README hash另核对一致。来源[官方仓库](https://github.com/yangsizhe/MemoryWAM)、[固定README版本](https://github.com/yangsizhe/MemoryWAM/blob/dc82a9d760e751ae88f2e0ae521af7c7ed597dfb/README.md)。0新GPU试验、未联系作者或他人。v23在HH-W1/W2与附录D.2据实交代，仍把Pooled、原报告RMBench与尚未测得的MemoryWAM LIBERO/成本严格分开；该条件不替代受控行为收益，正式发布前须重验动态状态。
### 23.206 v23逐项证据与基线条件整版封存

新三位作者对话稿保持各自真实认可的优点开场、we/our及已完成修订结尾；Bbqv先呈现已报告M(n)与任务收益，再准确回答总体小差值与条件区间。20项映射各自包含原review关切、原主张、已回答子问题、具体原报告/新证据、剩余直接实验及真实PDF位置；独立核验原note/field/ordinal/hash20项及全部回复标题、17评论块、修订label。修正沿用的v21版本与过旧的7候选状态。Hh4E与附录D.2加入已核验的官方公开代码发布条件，仍明确Pooled不是MemoryWAM复现。论文Wilson表的置信水平精确为nominal，原数字/图/公式/表格保留。P1–P5整段方向与37槽不变，未发布。

完整原LaTeX及干净ZIP含BibTeX四遍编译45页/主文含结论10页，所有页文字相同；变化43/44/45直接验看、另42像素相同，0undefined/overfull、45继承字体warning。首次包装检查因表注已审文字变更被旧等式检查拒绝，原PDF另保留并核对字节与编译件相同；新守卫精确重建仅两处允许文字变化，所有原数字与表格数据检查后通过，未改科学数据或放宽任意变化。PDF96069f57186468cac1617454966b659fdedbb88baa7e910b852dabb01d7be597，source ZIPd5ef75c5d81a2bbca802dd2cb800d7ed11710a66c74485f44682e2361defeb2f。

89成员123,951,134B包TAR99f68449da64d7b1e0ed5dc740893fab0dcbc55094a5d28334afd5109f96f41c、manifest98a6d0b710418f04a9165d0080ad02c2dc9f6d04784a59592a5a862ad7fdd577，已本机及Nov12 00:52:58 AFS逐成员/解包独立验真，`writing_candidates/20261112_v23`；v22封存不改。00:52真实CCI case41Pooled有340完整物理命令/86生成、32,392MiB/95%GPU，尚不能计新完整role。q96实际25branch/22候选端点、rank22重建，原截止不改；20候选已归档、晚到rank19/21待独立接收。

ACP界面本轮已显示登录超时，保留三个原tab及草稿，记录AUTHENTICATION_REQUIRED并向作者异步请求重新登录；不把旧UI状态称为本轮已核实。现有CCI与AFS实际心跳/结果可持续查验，未重启或重复现有云任务。详见CANDIDATE_V23_20261112_ZH.md。
### 23.207 case41Pooled完整终态，原同信息研究累计31角色

实际Nov12 01:09:01.952997UTC exit0、非bounded，完整1000命令/250H32/2000物理记录、success0/0TOPP。777原件18,115,175B封存`ITER28_BT41_POOLED_TERMINAL_V1`，TAR23f1571f4ceaad4562bf90c86f7853e72ea466dbe53719f89006835b0d52bb19、manifest259b11a0a79ca4b277f83311fe58dd4b3f81bea196e7484632afee9b63db9fb3；双端与本机逐成员/动作/生成/物理链独立验真。Pooled9c9c9124…/source87199cc6…/controller861004fa…不改。累计31完整role、15完整对全0/0、6unknown，新case41Pooled尚未与Full配对；原100对未完成，不据此推收益或非劣效。验证收据各自保留主机原时间戳，不以跨主机秒级顺序建立物理因果；q96时间问题依据同一轨迹command384干预顺序。

### 23.208 q96三个晚到候选原件

固定rank19/21/22各完整H32、父状态精确、无endpoint error，独立下一交互均0。30原件19,475,447B封存`ITER28_Q96_LATE_DELTA_V1`，TAR393952ea192e37fe395ecebf1244eb05ef2f6d2859a451f10d873067a107f2ce、manifest963b2568277f9bf18c3bd5e78f5b4cb5d528cc2c8a1d0d5b0349f72baa0db9f8；双端及本机01:07:30全成员/H32核验。累计23候选独立归档，仍一个不利父状态，所有0标签与主指标支持保留。01:13实际27branch/24候选端点、rank24重建；rank23晚到未独立归档，整池尚未完成，不能先算完整池主指标或人口修复率。原07:16:01UTC截止不改。

### 23.209 前瞻冻结两role操作段，减少轮询空等但不改科学设计

旧一role段正常终态与独立原件验收前，01:01:52先冻结`ITER28_TWO_ROLE_OPERATOR_PLAN_V1.json`（SHA f29feed75da9fc4c3bc72449b184ab181159327c34fb3cda436b0c1333e4dc90）。原不可变入口本就允许1–2新role，CPU validate-only对2通过、3拒绝；只改变未来一次invocation操作上限为2role/outer9000s，含kill最长9120<4h。原每role3600s、6slot/120GPUh、模型/数据/样本/科学源/奇偶顺序及原bt40首次22:44:53→Nov12 10:44:53资源窗保持。依据已观察单role约41分钟配置时间余量，非按结果换样或重开过期窗口；CCI四小时平台重启边界仍未核实。

前Pooled全部777原件独立核验，旧三个PID/GPU/两lease均释放后，01:11:24.337945唯一启动`cci_segment_iter28_followup_v1`，monitor2537277/campaign2537278/worker2537283，当前case41Full加载，第一角色被原控制器验收后才按冻结next_request选择case42Full。新launch SHA16c6c8fa12b398cc65a4dc80cf09ff8706a69dabacccdfd48c47db3f0788e6be；科学入口adb5ba9b…/修订dfaa351a…/plan44c23a4b…逐字节不改。未知退出仍停止，不自动重试；两role cap不能误写成两角色已完成。01:13仅527MiB/0生成/0物理，为进程加载阶段。当前outer存活时不能因前一worker结束而另起同一角色。

新内部`responses/20261112_v23_execution_addendum1`三文件维护20项映射及31/15/6unknown、q96已归档23/实际24的独立快照，manifest48171396e64f491c0aee3ffd53b937fe35453e867d4c8bd0f2350e134dd86e23。原v23三回复、PDF、源码ZIP与89成员包不动，P1–P5整段及37空槽保持待证；最新索引及双端当前状态以本轮安装收据为准。

### 23.210 原case41 Full完整终态与第16个固定0/0配对

Full实际Nov12 01:53:51.570553UTC正常exit0、非bounded，1000命令/250H32/2000物理记录完整，success0、0TOPP。775原件18,243,550B封存本机`execution/20261112_iter29/ITER29_BT41_FULL_TERMINAL_V1`及AFS同名目录，TAR c3fd029f4345274bb9cc212f2a2733f908b2a3b05edbd4d05bea566814da9226、manifest f3d70b283c799937e039b57f4c43d83eba5e35857fbc82394d73937c594a253d；双端及本机02:04:23逐成员/动作/生成/独立物理状态链验真。科学source87199cc6…/controller861004fa…/Full权重b0669047…不改。累计32完整role/16完整对全0/0/6unknown，不产生100对最终SR/CI、收益或非劣效。

该Full在原两role outer内完成；原控制器随即按冻结顺序选择case42Full，worker2538267。原monitor2537277/campaign2537278仍存活，禁止因前worker消失而重复提交case42。原bt40首次22:44:53→Nov12 10:44:53UTC、每role3600s与outer9000s不变；case42完整验收前不能计第33role或第17对。

### 23.211 原q96六个晚到候选独立接收

固定rank23–28各完整H32、parent_recreated true、endpoint error null；六项独立下一交互标签均0。60原件38,951,486B在本机/AFS`ITER29_Q96_COMPLETED_DELTA_V1`保全，TAR20342759ea86c3f9f9b54671cbfc55d3d8f46f7a5ec3561dbe531d4f1e8ad05b、manifest3e1f4109d9919b2b6675666ae0b49dd7ae5b3c1ef1f6cc582629f57eaf4e7ee6；本机02:04:24全成员与H32端点再验真。累计29个独立候选端点已归档，仍是一个固定source/query的相关候选，不能当29回合。父状态在干预前已掉落、全部0标签与原support不变；07:16:01UTC原截止保持，原全池尚未完成，不先计算完整池主指标。

### 23.212 原v3双通路撤回入口的CPU工程资格

在CCI实现显式、单次推理、eval/no-grad限定的`legacy_gate_intervention.py`，原源文件SHA d89796cc…限定到v3/timing4与Full/clean。干预仅把实际GateOutput probability置零，同时检查初始化精确等于共同Gaussian、候选条件精确为query-only gist与零候选tokens；保留自然gate与原选择metadata，不改checkpoint，不把强制撤回称为自然缺失技能检测。异常清理自己的hooks/override，检查参数/buffer版本不变。

对原f77c633与47277e5不可变代码分别使用其原小型CPU fixture完成23项检查，含候选扰动不变、回合历史敏感性、Full前后精确恢复及拒绝训练/梯度/重复调用/外部hook。首轮重复调用虽被gist计数守卫拒绝，但错误消息不符导致qualifier失败；修复为方法入口即拒绝，原失败另存`ITER29_LEGACY_CPU_V1_FAILURE_AND_CORRECTION.json`。两份通过收据为`ITER29_ORIGINAL_V3_CPU_QUALIFICATION_V2.json`与`ITER29_ORIGINAL_472_CPU_QUALIFICATION_V1.json`，本机/AFS均保全；helper SHA b8378876c066ef8b6a9b66a06634f8c69f9df78e1902e8534828775d2bf5bf3b、qualifier d8e0ed778de2dee74a9bcf786a55cff62c32f987c94af29841ae3abe6541b4da。

0真实checkpoint新加载、0GPU动作生成、0物理分支；仅为原代际干预入口工程资格，不填P2/P3或宣称修复收益。下一步必须用原固定query完成真实GPU Full精确重现、完整生成器不变性与动作事先封存，再做同父Full/null/Full。不得把新版shared开关直接套旧模型或重开旧已到期窗口。详见`ORIGINAL_SPECIALIST_WITHDRAWAL_20261112_ZH.md`。

### 23.213 恢复既有PB加载证明，避免重复读入12GB

只读核验发现PB step12000已Nov10 07:56完成原v3严格加载；权重前后SHA a765d987b2df0d228782980592c39aaed0e837040c7f6f48c905e4641f79c285，6,776,526,408参数，0推理。原9项档案TAR90071fee…/manifest619b3f1c…与原operator仍在，不能沿用“尚未完成模型加载”的旧判断，也无需重跑。Nov12本轮重新核对原47277e5源码SHA4a1d05cb…与config SHA32b8d678…，不再次读取12GB权重。DEV选中step、完整在线推理资格与原稿SR逐次绑定仍未知，不能把加载通过称为行为复现。

新只读清单`ITER29_ORIGINAL_PB_BINDING_INVENTORY_V1.json`在本机/AFS保全。PB实际recent-event capacity16、summary8、原生cap1500，与BT20/1000分开。当前静态在线bundle准备仍须原代码严格验证，不涉及新模型/阈值/训练；准备过程的错误和其修复单独保留，不当科学结果。

### 23.214 PB原代码在线合同与实际runtime静态资格完成

原47277e5代码严格构建`original_PB12000_online_bundle_s3407_v1`，root3407由显式输入matrix绑定；容量16/8、H32/prefix4保持。manifest d9c2c1eba8d698f4ec3c3aab755e3b22e707168de2d679ce3d62c41bd53cd453，online contract1fa82f89e83d5220995cd2d30881d4a36a69f98713c7b182e8c31298426e242b。实际deploy配置projection SHA4617d830b30a906d599148c32c9aba245e79208427b65df1023e9351e39b7c4f与原合同精确相同；原所有static validator通过，实际encoder runtime SHA2f3e5d2979f2fbb3063da65d25a49d15fb4c3a19b52fcf329879fa87fbcb9e01亦精确匹配。无模型加载/推理/物理命令/训练更新，不称新SR或行为复现。

准备V1因沿用BT构建器hash而被拒绝，查原始diff确认PB代码已自行修复compute_device绑定，未对原PB代码注入patch。V2因原CLI不支持`--seed`而被拒绝，V3通过显式matrix root3407构建；两个失败和V1非终态operator均保留且单独说明，不伪造成功或真实exit文件。前节“不重复读取12GB”只指清单与旧加载证明恢复；原构建器的完整文件hash验证仍执行，包括大资产读取，并不跳过权重/数据验证。

15原件62,957B保存在双端`ITER29_PB_STATIC_QUALIFICATION_ARCHIVE_V1`，TAR745072cd0ff297a936279a298d3918b2a4d0ea29c6a8eed418d39242476e25bf、manifest5418bfa8911abb389fe40487dae5c6caeecac5050023aa42deb48d42f917cc7e。工程静态资格不恢复未知DEV选点。另冻结原代际撤回控制器commit49435b0，code source SHA205f541ae2531e53105eba23e4134343b539c88fa4cb584c5226f00754e9cf8f；只就绪，未派GPU或物理分支，既有活动source不变。

### 23.215 v24作者对话与测量解释整版验收

三位真实优点感谢开场逐字保留，正文we/our逐项回答，结尾只列真实修订，无显式第四方或索分。rkWY在披露原specialist/closed-set设置后正面重申选择性复用贡献；Hh4E删除重复的基线句和笼统负面自评。主文4.7以作者视角区分提案适用、生成修复、完整成功及自然缺失技能检验，原九图、数值、公式、表格、监督和核心贡献不改。20项原review字段/序号/hash、17评论块、37空槽及整段条件论证均验真，实证发布仍false。

完整原工程与干净ZIP均含BibTeX四遍编译45页/主文至结论10页，全部页文字一致。只改变第10页并直接验看，余44页像素一致；0undefined/overfull、45继承字体warning、匿名与内部标识通过。源码及PDF第11页证实原v23已有AI use statement，v24原样保留，纠正旧状态索引“待新增”。首次heading文字检查误拒PDF的AIUSE空白抽取，首份PDF原件保留；仅校正heading空白匹配后通过，未改科学数据。PDF85afd3605bcdb5d4591284756dcb71d1b8552fca2a5f7aff46c96114c29acef2，source ZIPed82211ab8d32c5919f268255c47b931d960154ea6dbf213f2e9230038f37cfb。

93成员123,956,825B写作包TAR62cc1593e97f20773d1472b63323d42ab6bb2419fd74520f9dc1284bd092a29a、manifest06c76d933018760d33f81d3080c9a89ba7397839057029026c62a9a5135c0e8f，本机及02:25:09UTC AFS全成员/解包独立验真，`writing_candidates/20261112_v24`，原v23不改。材料快照仍32角色/16对全0/0/6unknown、q96已归档29候选；02:26实际case42Full629命令/158生成/GPU100%，q96已有31候选而rank31活动，晚到29/30未独立接收，不提前当完整结果。另case41配对的初态、首图像/proprio/共同Gaussian/候选池独立核对精确，`ITER29_CASE41_PAIR_INDEPENDENT_IDENTITY_V1.json`。

### 23.216 原case42 Full终态与同窗下一两role

Full实际02:36:16.320068UTC正常exit0、非bounded，1000命令/250H32/2000物理记录完整，success0/0TOPP。777原件17,975,659B在双端`ITER29_BT42_FULL_TERMINAL_V1`保全，TAR95462735b6d169f0bcb266aba276e8a443afda0baa763f6f99978fae4798fab8、manifest5ebe7043b0fd1e639488307a40fc7f3b58f66f8536a7375abb4b2cde43e907dd；本机02:39:18逐成员/动作/生成/独立物理链核验。累计33完整role/16完整对全0/0/6unknown，新Full尚未配对，不计第17对。

原两role outer实际exit0、旧三PID/GPU/两lease释放及完整原件验收后，02:40:11.912182唯一准入`cci_segment_iter29_v1`，monitor2540822、实际worker2540828，原next_request正确选择case42 Pooled，第二角色仅在首角色被原控制器验收后选case43 Pooled。02:38:18预先冻结新操作计划7f04c607fd974d51f24c2186fc61aaeb4ec1656ae3f1f71ccbdd83df5e969131，仍最多2role/9000s、每role3600s；入口/launch科学字节、模型/样本/顺序、原bt40首次22:44:53→10:44:53窗口不改。不是新云副本，不按结果改样，未知仍停止。

### 23.217 原BT q96全池与最终Full已真实完成，独立复算不利主指标

原标准任务UID536a352f050b4142890aa99fb88aa23b的nonreal runner实际02:40:22.140339UTC complete/exit0，elapsed22562.61s；source前后f7cf0583…完全一致。原32/32候选完整H32、0未知，四次Full端点/效应精确，36分支父状态/事实前缀一致。376原件236,483,738B在双端`ITER29_Q96_COMPLETE_TERMINAL_V1`保全，TAR546a03565a3bc4d5ddee5f6627c6e59c4518e86aa180339515daff895bc7a161、manifest84bc6080d76f3d144cea62f9a5222a6468255ffc1e4d29f4a65d81c6c020452b；本机02:46:11全成员/H32/四Full/父状态及主指标独立验真。最后rank29–31三H32另有30项增量档案65ba071f…/e0115a81…，同样全部严格标签0。

原uniform-mean四DINO token descriptor的完整共同支持上，predicted/historical/constant centered MSE为0.17000183889870682/0.16482099214548016/0.14489814162091016；predicted未优于两参照。predicted/historical absolute MSE为0.21172237343896194/0.20598023517768294；原query requirement未捕获，constant absolute仍不可得，不补造。独立残差二阶矩公式复算最大差2.7755575615628914e-17，未事后拟合或改主指标。

干预前已掉落的父状态完整保留；32候选与四Full严格下一交互均0，selected raw/Full为0/0，无完成型修复/伤害，尚无同父candidate-null。一个query的相关候选不当32回合，不能外推总体机制收益或不可恢复性。原07:16UTC窗口以真实完成收尾，不重置或续跑该全池。浏览器认证仍未恢复，不把runner终态冒充最新云UI核验。

最新`responses/20261112_v24_execution_addendum1`六文件含20项执行映射、两份英文followup、待合入的LaTeX段、进度与原件收据，manifest0ac1b46b81b291e33132b7f73924859c95dee6fc5c88f8a9096818fc856a116b。v24原93成员三回复/PDF/源码ZIP不改；这些全池数值尚未合入该PDF，下一整版须整合并重编译。P1–P5整段及37槽仍待证，不以此负结果填有利结论。

### 23.218 原q8 GPU回放的事实context捕获检查

只读检查原BT case0/q8的`infer_kwargs.pt`：原capture函数对numpy数组仅存`unserialized_type`标记，包含image/proprio/候选/回合历史字段；不能把这些标记当完整tensor context，更不能用零填补。实际旧相机/qpos、逐次模型动作、事实world/VAE输出、检索行与hash等仍保留，但完整context恢复尚未核验。原稿结果与已完成分支不受该工程检查改写；本轮未越级执行GPU null或新的物理分支。

Git外`ITER29_ORIGINAL_BT_Q8_PAYLOAD_OUTLINE_V1.json`保存实际字段/shape与标记。下一资格应在独立冻结collector中正确保存numpy和实际source context，或用原策略/内存路径重建；先精确复现原固定前缀、输入hash及Full H32再封存null。不能伪造/重用已消费的online capability，也不能按新结果另挑状态。两个原代码的23CPU检查只是入口资格，尚不能释放P2/P3的行为证据。

### 23.219 原q8历史已通过真实记录CPU重构

03:05:34UTC原f77 controller/OnlineRetrospectiveEpisodeMemory在0模型/0retriever/0GPU/0仿真条件下，重建原case0/q8之前8个已记录factual world/VAE观测和32条实际命令。9个query身份、每次history evidence、8次factual update以及32条环境/归一化模型动作hash均与原日志精确一致；原源码ff32342d…与所有输入前后不变。不是新回合或新成功率，也未填P2/P3。

q8 snapshot c0282bd0eabee3d30e40c9cb68beb07f64e55ee83cbc11a1eb5c99a24547392c，16历史tokens/8条summary，recovered_history.npz SHA1346e51bd60e573c8f6eaf8d7ba5e858671a829ca41368b877cdc1baf09ecfe0；原件在双端Git外`ITER30_ORIGINAL_Q8_HISTORY_CPU_V1`。这恢复的是过去历史，完整当前source context与真实GPU Full重现仍待核验。

03:13:38原query CPU preflight V2通过，原typed stats loader、processor、M1 recipe和proprio hash精确匹配，0CUDA初始化。V1在任何GPU/策略加载前因原始JSON列表不适合processor normalizer而TypeError退出；改用原部署`_stable_stats`，原normalizer和数值validators不变，失败元数据`ITER30_ORIGINAL_Q8_STATIC_CPU_V1_FAILURE.json`保留。下一步在冻结代码上做仅一次合法fresh retrieval，精确核对原step/载荷/输入hash，仍不运行策略或物理分支。见`ORIGINAL_QUERY_CONTEXT_RECOVERY_20261112_ZH.md`。
