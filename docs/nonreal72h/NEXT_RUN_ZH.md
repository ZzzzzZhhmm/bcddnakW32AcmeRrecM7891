# 下一项实验：W02-A 完整 Action DiT 双路零门控检查

**进度更新（2026-09-21）：下述50前缀任务已完成，50/50通过，作业实测610.33秒。下一项W06 source诊断已在单卡启动。结果、备份和论文回填限制统一见 [实验记录](EXPERIMENT_RECORD_ZH.md)。本文下方保留原启动计划与耗时预算，不能将“已启动”当成当前仍在运行。**

2026-09-21 已按授权在 CCI 的 1 张 H100 上直接启动，无需申请多卡 ACP。

## 验证的问题与范围

固定真实 DEV 前缀、factual history、Gaussian seed、NFE=20 和 token/mask 数量，将 source 与 candidate conditioning 两条路径同时强制归零。先对相同输入重复完整推理，冻结 `max(1e-6, identical-call max error)` 容差，再替换候选内容，比较完整 Action DiT 的模型空间输出；不经过动作反归一化或后处理。

这是实现机制自检，使用现有 v8 **shared 300-step smoke checkpoint**。不能据此声称 learned gate 有拒绝能力、方法提高成功率、正式 checkpoint 已验证，或真实仿真状态恢复已通过。调用的是生产 `FastWAM.infer_action` 计算核心；factual context 来自受契约验证的 DEV 数据，未走在线 `BoundOnlineStep`，所以也不等于在线 retrieval parity。

脚本仅向模型提供当前帧、当前 proprio、文本缓存、历史事实和候选。`target_action`、`target_effect`、`current_semantic_teacher` 等教师量均为空；单测覆盖此边界。候选内容替换是有限数值的结构压力测试，不是合法机器人动作或正式 donor 协议。

## 已完成的 pilot 与耗时依据

| 项目 | 实测 |
|---|---:|
| 任务 / 前缀 | Press Button，1 个 DEV episode 的 1 个固定前缀 |
| 检查结果 | passed；候选替换前后完整输出最大误差 0 |
| 前缀检查耗时 | 5.55 秒，包含原输入、重复输入、替换候选三次推理 |
| 推理 median | 0.65 秒；只有 3 次调用，不作为正式性能表 |
| 峰值已分配显存 | 12.83 GiB |
| 子进程总时间 | 500.28 秒，主要为完整权重加载和多次哈希校验 |

基于 pilot，50 前缀约 `494.73 + 50 × 5.55 = 772 秒`，还需计入 launcher 输入校验与共享存储波动，预算 **15–25 分钟**。设置 2 小时硬上限用于异常保护，不代表预计要跑 2 小时。没有把旧 4-GPU 训练速度当单卡速度。

## 已启动的批量任务

- Press Button、Put Back Block 各 5 个 DEV episode，每 episode 5 个内部等分位前缀，共 10 episodes / 50 prefixes。
- 选择规则只看 task/episode/frame 长度与完整 H32，推理前封存；不按 gate、预测分数或结果挑选/替换样本。
- BF16，batch=1，H32，NFE=20；每个前缀固定同一 seed 做三次完整推理。
- 数据目录：`/mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations/nonreal_full_null_20260921/dev50`。
- 启动日志：同父目录下 `dev50.launcher.log`；父进程 PID 写在 `dev50.pid`。
- job 级日志：同父目录下 `job_logs/<run_id>/console.log`、`events.jsonl`、`run_manifest.json`、`source_manifest.json`。
- 输出包含 `frozen_prefixes.json`、解析后配置、模型/数据哈希、逐 query 误差/耗时/显存、原始 action 数组、两组 probe，以及最终 `summary.json`。只有实际最终 exit=0 与 summary passed 才算完成。

已经启动的命令如下，**不要重复提交**（重复运行会拒绝覆盖已有结果）：

```bash
WARM_NONREAL_OUTPUT=/mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations/nonreal_full_null_20260921/job_logs \
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
bash /mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations/nonreal_full_null_20260921/code/scripts/acp_nonreal72h.sh \
/mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations/nonreal_full_null_20260921/dev50.job.json
```

## 后续 GPU 排期

当前没有需要立即提交、且已通过放行检查的多卡任务。W01/W02 分支测量本身可单卡执行；现在的阻塞是正式模型身份、SAPIEN graphics/Vulkan 渲染和真实分支恢复 backend，增加 GPU 不会解除这些阻塞。本次重测 renderer 仍报 `failed to find a rendering device`。

若最终必须续训现有 v8 shared：其训练 attestation 是 4×H100、batch 8/card、accumulation 4、global batch 128；以原记录约 0.274 optimizer steps/s 粗估，300→30000 步约 30.1 小时计算时间，加保存/验证/初始化/排队会更长，需要至少两个 ≤24h ACP 段，并且尚不包括 specialist 与 rollout。该 estimate 不是可运行命令的验收。当前不派发从头训练六种消融或未经验证的多卡恢复命令。
