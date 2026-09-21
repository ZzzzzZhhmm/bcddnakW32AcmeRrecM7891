# WARM 实验记录与论文回填索引

更新日期：2026-09-21。本文是持续维护的回填入口；原始记录、数组、运行输出保存在仓库外，并在服务器与本地各留一份。每次取得数值，先保存，再核验，再更新本文。失败与不支持假设的结果同样保留。

**目前没有新增可直接回填为当前论文正式主实验的结果。** 已有一项完成的实现自检，以及一批原始计数已核验、与稿件数值不一致的历史 LIBERO 结果。下表的“可用范围”是回填限制，不得省略。

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

## 5. 后续记录规则

每项实验记录：ID、claim、训练/推理干预身份、code/config/checkpoint/split/bank哈希、种子、硬件、开始/结束/exit、原始产物、k/n与失败/unknown、统计单位、复算命令、可填论文位置、限制和下一步。运行失败时先保留失败目录，再以新ID重跑。

当前仍未验收：RMBench 755/900原始episode，六行消融训练身份，五轮干扰donor记录，真实snapshot/restore backend与renderer。未拿到这些原始证据前，相应论文行继续标记待核验，不填零或虚构区间。
