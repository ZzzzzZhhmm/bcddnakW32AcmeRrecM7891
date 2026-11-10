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
