# 四张 H100：从原完整状态续训 shared WARM

2026-09-22。目的：解除当前只有v8 smoke300、自然gate全零覆盖的前置阻塞，为尚缺的W01/W02/W06提供兼容checkpoint。不是重跑已完成的零门控、source或梯度诊断，也不重训六种消融。

## 第一段：300 → 15000

在同一共享盘、同一训练软件环境的 **4×H100 80GB** ACP上提交一次：

```bash
bash /mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations/nonreal_resume_20260922/code/scripts/acp_nonreal_resume.sh \
  /mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations/nonreal_resume_20260922/stage1/plan.json
```

保持ACP分配的GPU可见性，脚本不覆盖为其他GPU。不要在CCI单卡上运行此命令。已部署的源码快照独立于同学正在修改的主仓库，任务不执行fetch、pull或远端探测。

- 原四卡optimizer、scheduler、RNG、dataloader进度完整恢复；8/card × accumulation4 × 4卡 = global batch128。
- 总训练计划仍为30000步、warmup1500步；本段只新增14700步，不重置LR，也不把300-step weights当作全状态resume。
- 原checkpoint/config保持只读；新产物在 `stage1/training`。每1000步保存完整状态，结束保存15000步状态。
- 原记录约0.274 optimizer steps/s，对应本段纯训练约14.9小时；建议预算 **16–20小时**，训练子进程22小时硬上限。共享存储、初始化、评测及保存开销可能改变耗时；ACP申请24小时。
- 同一plan重复提交会被锁或started标记拒绝。失败/超时后保留目录，从最后一个完整checkpoint新建下一段，不能直接覆盖旧运行。

## 已检查与尚待现场检查

已在CCI完成父weights与attestation核验、完整25GiB ZeRO状态树哈希、四个rank的optimizer/RNG文件及scheduler检查、父配置与attestation匹配、子配置shared recipe不变、Hydra命令解析。服务器新增入口测试3项及原评测字段/训练attestation测试25项通过。

父weights SHA-256：`ab4042b1a761f53ec5f5379ce461f5a2f6f3decf89e3f3a16322b66894918a1d`。

父state tree SHA-256：`48be391eb0c223f862e5197e1fc77455b67225ce1df98b7773b3478ecd5af0cf`。

本段配置SHA-256：`34f5dc0a6f4cc354da61c0ba7bdc394b6c9652c7ef7706e72e8cbb683ec9a00c`。

实际运行源码集合SHA-256：`665d0620a2a8a15e1d3daaa2ddad38e6ca7c383f8af5aa407ef0784aff797bda`。源码manifest是运行的精确身份；快照不依赖Git元数据存在。

**尚未在四卡上执行恢复。** 启动时先核对硬件、Python/PyTorch/CUDA/cuDNN/Accelerate/DeepSpeed及平台；原trainer继续核对真实四卡runtime、optimizer/scheduler和recipe，恢复后验证参数有限且rank一致。若ACP镜像与原环境不匹配，会提前拒绝，而不会关闭校验继续训练。单卡两batch诊断不能替代这一检查。

## 自动记录与结果回收

公共目录：`/mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations/nonreal_resume_20260922/stage1`。

- `launcher.log`、`job/console.log`：完整控制台；`job/events.jsonl`有心跳，`job/run_manifest.json`有开始/结束、exit、源码身份。
- `training/training_metrics.jsonl`：逐记录loss、各分支梯度、gate、LR和速度；保留原始记录，不只保留最后均值。
- `summary.json`：完成性、最后指标、已保存checkpoint及状态路径。
- `EXPERIMENT_RECORD.generated.md`：结束后自动从原始JSONL生成的Markdown指标表，供并入主实验记录。失败也输出不完整状态；汇总器本身异常则保存`summary_error.json`与日志。
- `training/checkpoints/weights/step_015000.pt`及`.training.json`、对应`state/step_015000`：本段成功的验收产物。只有exit0、目标步数记录与完整目标checkpoint同时存在，并且最终weights哈希和resume lineage核验通过，才标complete。

运行完直接把 `summary.json` 路径或最后日志发回；也可只告知任务完成，我会从共享盘读取，备份产物并合入 `EXPERIMENT_RECORD_ZH.md`。ACP不自动修改共享Git文档，以免与另一组提交冲突。

第二段15000→30000的纯训练估计约15.2小时；需第一段真实产物通过后，绑定其新checkpoint/state/config生成下一份plan。15000步是资源分段点，不预设为论文最终checkpoint，也不承诺届时gate或SR达到某个值。
