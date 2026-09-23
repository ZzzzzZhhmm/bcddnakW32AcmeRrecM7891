# WARM 实验记录与论文回填索引

更新日期：2026-09-22。本文是持续维护的回填入口；原始记录、数组、运行输出保存在仓库外，并在服务器与本地各留一份。每次取得数值，先保存，再核验，再更新本文。失败与不支持假设的结果同样保留。

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
