# 一条ACP命令：续训、验收、新模型门控与条件source诊断

2026-09-23。申请 **4×H100 80GB、24小时**，仅提交下面这一条，不再同时提交旧续训命令：

```bash
bash /mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations/nonreal_bundle_20260923/code/scripts/acp_nonreal_bundle.sh
```

预计 **17–21小时**：原续训16–20小时，新的checkpoint验收和DEV诊断约0.5–1小时，另留初始化余量。新入口设23小时运行预算，训练阶段外层最多22小时；剩余时间不足会保留结果并报告不完整，不超时硬塞下一步。总ACP仍申请24小时。训练需要4卡，随后诊断仅使用分配中可见的第1张卡，不能把其余卡当成新的4卡配额。

## 自动执行内容

1. 核验已冻结源码、原续训plan/config及DEV身份；做一次有日志的renderer检查。renderer失败不阻塞CUDA训练，但绝不放行仿真分支。
2. 调用原不可变四卡入口，从完整step300状态续至预声明step15000。保持30000总schedule及global batch128；不修改旧源码或旧plan。
3. 验证作业状态、最后训练步、metrics哈希、目标weights/attestation、四rank状态文件与scheduler、parent lineage和解析配置身份。CPU读取gate权重，但不把权重变化当成真实接受能力。
4. 在**新15k checkpoint**上运行一次Full自然门控诊断：Press Button/Put Back Block各5个DEV episodes、每episode5个冻结前缀，共50个；NFE20、seed3407，与旧诊断同一bank/query/normalizer/prefix身份。
5. 从保存的数组复算自然g。若50个全为0，停止这批前缀的source对照，不重复三个已知退化分支；若非零，则用两个独立进程追加scale-only与Gaussian，与已保存Full配对。Full不再重跑，固定噪声、候选、选择、conditioning和g的一致性逐项检查。
6. 每一阶段结束更新JSON和Markdown记录；失败保留退出码、控制台、原始数组、身份和未完成原因。

这是新checkpoint的DEV工程诊断，**不是**新版表22的独立训练SR，也没有表18所需的独立适用性标签。它不替换已填表19/21/24，不做第二次smoke300运行，也不因gate高低重新挑选训练checkpoint。DEV50结果不能代表后续S1新cohort的门控覆盖。

## 仍不会自动完成的论文缺口

- 表4/17/18：尚未验收真实RMBench全状态恢复、H32候选执行和独立标签。renderer单独通过也不够。
- 表23：requirement-only语义历史干预尚未实现/验收。
- 表20/22：尚缺匹配独立训练控制器与闭环评测。
- 表11/12/14：旧run身份、donor逐query记录及原始配对结果仍未找齐。

这些条件明确出现在每份汇总的`paper_blockers`中。完成已具备条件的流程时状态为`completed_available_stages`、退出0，但 **`paper_evidence_complete=false`**。不能把ACP绿色完成状态理解为论文全部数据已齐。

## 日志、记录与重启

根目录：`/mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations/nonreal_bundle_20260923`。

- `bundle.launcher.log`：总控制台。
- `run/bundle_summary.json`：阶段状态、验证过的checkpoint、新数值与论文阻塞。
- `run/EXPERIMENT_RECORD.generated.md`：可审计的Markdown实验记录，验收后并入主实验记录。
- `run/jobs/*/console.log`、`events.jsonl`、`run_manifest.json`：各阶段命令、退出、时限、心跳和源码身份。
- `run/checkpoint_inspection/gate_tensors.json`、`run/natural_gate.json`、`run/source/mode-*`、条件生成的`run/paired_source`：实际张量统计和原始推理数组。
- 原训练产物仍在 `nonreal_resume_20260922/stage1`；逐步loss、checkpoint与完整训练汇总不搬走、不覆盖。

同一新命令有独占锁。训练已完整完成时重新验收并复用；诊断阶段只有收据、输出哈希和作业状态都匹配才跳过。已开始但未完成的训练、没有完整收据的中断输出均拒绝覆盖；需检查后绑定最后完整state或新输出目录准备下一尝试。不要同时启动新旧两个入口。进程运行期间不会执行Git同步、安装依赖或下载模型。

## 已验证与限制

本地和CCI均通过19项相关测试，覆盖完整性拒绝、阶段证据改动、失败日志、零/非零gate分支、Full只运行一次、旧训练不重复启动及原resume/source逻辑；shell语法与CCI CPU预检通过。未执行真实四卡恢复，也没有新15k模型可进行完整推理验收；这些检查在任务现场严格执行，不能保证计算任务绝不失败。

新汇总器也读取并核验了已有N06-R2的全部50条原始Full数组，复算得到既有的10 episodes、g非零0/50、alpha均值0.119140625、mean c²=1。记录为`reader_compatibility.json`，只证明读取兼容性；没有重新运行推理，不新增或重复计入实验样本。

新流水线精确源码SHA-256：`70f316e0e2e46718ba33b2dea2d0bb5836ce0fcb446ecf895fb98aa127757265`；旧训练源码仍为`665d0620a2a8a15e1d3daaa2ddad38e6ca7c383f8af5aa407ef0784aff797bda`。部署包SHA-256：`348643a15b022e57c6760230b4bd093ff90e687ab4bc726e041b0832a7f68451`。原始plan、CPU日志和测试日志已下载至仓库外`L/bundle_20260923`，L见主实验记录第3节。

准备阶段共享AFS不允许tar恢复属主，首次解包返回非零，尚未启动任何实验；随后以`--no-same-owner`重新完成相同包的解压并通过身份核验。该部署问题不占用ACP GPU试验，也未触及原训练快照。
