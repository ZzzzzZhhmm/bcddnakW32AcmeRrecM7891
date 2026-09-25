# 表4：LIBERO 同状态候选分支实验（2026-09-25）

## 最新决定与范围

作者已明确撤回使用R2权重，改为LIBERO及对应LIBERO权重。表4的候选效果机制协议
本身不要求RMBench。不要等待RMBench/Vulkan，也不要把R2的参数或模型混入本实验。
本实验不重新训练，不改论文主表，不把旧LIBERO成功率当作本次机制结果。

最新稿件：`WARM_ICLR2027_new (1).pdf` 第9页表4；ZIP中的
`sec/4_experiment.tex`和`sec/appendices/c_additional_results.tex`。
红色草稿数字不是目标值或观测结果。三行必须使用同一checkpoint、同一query、
同一批adapted动作及独立执行端点。

## 已通过与待通过的检查

CCI `slai_debug_warm`，H10080GB，MuJoCo3.3.2；使用 `MUJOCO_GL=egl`、
`LIBGL_ALWAYS_SOFTWARE=1` 已成功渲染LIBERO-10任务0的两路256×256RGB。
这是Mesa软件EGL渲染，模型仍用H100；并非修复SAPIEN/Vulkan。

无模型环境验收：A–A–B–A，每分支32步；物理端点绝对容差1e-10，两路图像要求逐像素相同。
六类父状态恢复检查通过。四次执行分别约3.506、3.485、3.476、3.469秒。
这只是环境/恢复工程验收，不是表4的四项指标。

原件服务器根：
`/mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations/table4_libero_20260925/`
子目录 `branches-egl-v2/environment.json`、两路PNG；本地小文件位于
`F:/WARM/result/table4_libero_20260925/`。
environment.json SHA256为`a4993bd43702ff296a641260bbf6cf5851d223865eae8e7e8dbc0ee90a8d1684`。

完整模型采集器另用任务0、官方initial-state49、frame80进行独立资格验收，
与正式initial-state0–9不重合。正式每个任务首次query也执行恢复/编码重复性检查。
必须在实际模型采集器通过后才放行ACP，不以CPU测试或空环境代替。

完整模型资格验收现已通过，固定代码为同根`release-v3/`，原件在`qualification-v3/`：
任务0/init49/frame80的32个候选均执行32步、取得768维effect、父状态恢复32/32通过。
实际模型下A–A–B–A通过，重复DINO读图最大差0；32候选总计107.72秒，
整个过程798.41秒（其中模型加载561.5秒）。组装器读取真实记录亦通过。
这个保留验收query的32个goal-progress标签均为0；不进入正式统计，也不据此调整采样。
单query不支持三行效能结论。其他任务仍在正式运行时逐任务进行恢复资格检查。

## 模型与架构边界

权重：

```text
/mnt/afs/task3_2/L202500276_lwz/projects/WARM/runs/libero_warm_2cam224_1e-4/warm-full-4xh100-zero1-numerics-20260718-110409/checkpoints/weights/step_019100.pt
```

对应原评测目录：`WARM_evaluations/results/step_019100/libero_10/task_XX/seed_3407/formal-s3407-v2/`。
配套旧版模型代码：`WARM_evaluations/code/c4763a975298de6f00939360551616af7902d57a/`。
加载时检验checkpoint字节、统计量、训练证明、train/dev source contracts和bank/camera/encoder契约；
新research采集层写独立protocol，不能冒称完全复用了原正式SR rollout配置。
启动过程不进行Git联网、同步、cleanliness检查或修改旧模型源码。

**重要：旧LIBERO模型的effect head为historical effect + warped-action/context residual，
实际被执行的proposal还包含action adaptation。** 不把当前新版head套上旧权重，
不重新初始化缺失层，也不将本实验称为新版RMBench架构验收。
可回答“该LIBERO checkpoint的候选预测能否解释实际adapted proposal的结果”，
不能自动证明所有后续架构版本，也不能替代主表98.7%的溯源。

旧正式LIBERO evaluator还依赖`warm-step019100-eval-v2`的运行时兼容修复：
ActionSummary的terminal gripper需要展开到7个动作坐标后，才能与21维preview匹配。
首次新采集入口未带入此层，已在`qualification-v1.console.log`保留15/21维报错；
`research/libero_legacy_compat.py`明确复用该签名语义并恢复encoder数值策略。
不修改旧worktree或checkpoint，不用跳过memory替代修复。

## 预声明正式设计

- LIBERO-Long (`libero_10`) 全部10任务，每任务官方initial states0–9，共100个source episodes。
- seed3407，保留原模型每10步replan；factual policy action frame80、160处采样。
  模型提前完成造成的缺失query另记coverage，不补新reset、不按成败补采。
- 最多200 queries，K32全部合法候选、H32，无分支内重规划；最多6,400分支、204,800命令。
  统计独立单位为source episode，不将candidate对当作独立样本。
- hook读取真实 `mu_i, Ehat_i, E_i`，requirement来自同一次模型diagnostics；先写不可覆盖的
  NPZ/JSON及SHA，再执行分支。H32完整端点才有DINO effect；早停保留真实outcome及缺失状态。
- effect采用契约一致的4个DINO spatial tokens差的平均；候选动作反归一化及gripper变换
  与旧LIBERO部署一致，最终env clipping另记实际命令，禁止换成DiT最终动作充当proposal。
- 三参考的effect/排序指标按candidate→query→source episode→task等权聚合；10000次
  episode-cluster bootstrap；三行同一列使用同一query支持集。
- `delta_obs=delta_pred=top_tolerance=1e-6`，magnitude weight0.25；先固定，不能看测试结果后调阈值。

### 独立标签的精确含义

本次采用可由LIBERO BDDL直接核验的 **32步子目标进展**：分支端点至少新增一个成立的
goal predicate，同时保留query时所有已成立的goal predicates，标签为1；否则0。
前后goal布尔向量、goal定义、实际命令、early termination均归档。标签不读取预测分数或gate。

这是一个窄而明确的“下一子目标完成”判据：仅接近物体、尚未完成子目标的动作记0，
不意味着该动作通常无用，也不等同于整任务成功率。回填时需把附录定义写清楚；
若正例稀少或全部为0，应如实报告，不能将bootstrap的退化区间描述为普适高置信度。

## 开销及执行方式

仅根据环境实测：6400×3.48秒≈6.2 GPU/worker小时，另有模型加载、factual前缀、
特征编码及恢复开销。因此此正式设计不满足作者“单卡<=4h直接完成”的条件。
优先1个4-H100 ACP job，4个独立worker，无DDP；各worker都覆盖10任务，reset按mod4分片：
`0,4,8` / `1,5,9` / `2,6` / `3,7`。最重worker60queries、1920分支。
完整模型资格验收实测107.72秒/32候选，按200query估计仅候选执行约5.98 worker小时；
最重worker60query的候选执行约1.80小时，另有prefix、10个task retriever加载及AFS读取。
4卡建议预留3–5小时，实际由CPU资源和AFS决定；不能保证4小时内完成。

脚本：`scripts/acp_libero_table4.sh`。新输出目录、环境验收、四worker、错误传播、
完整性合并及实测LaTeX三行均串联；不覆盖失败输出，不自动安装包或修改系统。
ACP镜像仍需Mesa EGL/libEGL及同一持久化Python环境；脚本会在加载模型前检查实际渲染。
缺少这些系统库时只报环境失败，不降级伪造图像。

ACP使用4张H10080GB，建议至少16个CPU核；Mesa在CPU上渲染，不能只按显存选择CPU配额。
`WARM_TABLE4_MAX_HOURS`是每worker的安全上限（包括加载时间），不是完成时间承诺。
CCI上的Mesa可用不保证另一ACP镜像也有相同系统库；实际render preflight仍是必需的。

已经完成单query完整模型资格验收的ACP命令（正式全任务结果尚未生成）：

```bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WARM_TABLE4_MAX_HOURS=6
export WARM_TABLE4_OUTPUT="/mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations/table4_libero_20260925/formal-$(date +%Y%m%d-%H%M%S)"
bash /mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations/table4_libero_20260925/release-v4/scripts/acp_libero_table4.sh
```

完成输出 `report/table4.json`（全部coverage、scope、CI、配对差异）和
`report/table4_rows.tex`（实际三行）。原始分支、标签、proposal、timing都在shard目录。
只有所有预声明任务/reset/query均已记录或有明确early termination，且没有未解决分支，
才允许导出完整表格。可先在CPU执行pytest验证统计，但其结果不属于实验效能证据。

## 论文回填时的解释边界

表4是机制测量，不是新的LIBERO整任务成功率评测。论文需要明确suite、checkpoint、
实际query/episode数、完整端点比例和独立标签含义。旧checkpoint的head输入与最新稿
式中`stats(mu_i)`不完全一致：实际head使用warped action。不能只替换数值而保留
不对应的实现声明。建议在附录实验协议中说明：

```latex
We conduct this probe on the LIBERO-Long checkpoint at step 19,100.
In this implementation, the effect head conditions on the aligned retrieved
action and episode context, while the executed proposal includes the learned
action adaptation. We measure its prediction against the resulting adapted-action
endpoint. For the choice metric, a positive label denotes completion of at least
one additional BDDL goal predicate within 32 steps while preserving all predicates
that were already satisfied at the query state.
```

该段仅解释已核验的实现/预声明协议；正式完成前不写成已经得到有利结果。

## 9月25日ACP失败后的续跑（最新命令）

`formal-20260925-041226`仅worker0成功，保留全部10任务的initial states0/4/8，
共30episodes/60queries/1920候选。其余worker因全局EGL0与CUDA可见卡号不匹配在导入时失败。
不使用原release-v3启动器继续运行。

release-v4将Mesa EGL设备选择与CUDA选择分开：保持CUDA_VISIBLE_DEVICES不变，
在进程内使用MuJoCo的独立EGL选择器；不改site-packages、模型参数、动作或统计协议。
不能简单把软件EGL编号改成CUDA1/2/3，因为软件EGL设备命名空间通常只有设备0。
每个worker在实际CUDA可见性下先检查可用CUDA、真实渲染和A-A-B-A，再加载大权重。

以下命令读入并验收旧shard-0，四worker只运行缺失initial states：
`1,6` / `2,7` / `3,9` / `5`，共70episodes/140queries。
新产物使用独立目录，旧目录只读；结束后自动合并旧30+新70episodes。
模型SHA必须等于旧shard，最终合并仍要求完整100episodes采样范围且无重复、无缺失。

```bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WARM_TABLE4_MAX_HOURS=6
export WARM_TABLE4_REUSE_SHARDS="/mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations/table4_libero_20260925/formal-20260925-041226/shard-0"
export WARM_TABLE4_OUTPUT="/mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations/table4_libero_20260925/resume-$(date +%Y%m%d-%H%M%S)"
bash /mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations/table4_libero_20260925/release-v4/scripts/acp_libero_table4.sh
```

4H100、建议32CPU核；根据首轮worker0的2.37小时/30episodes估计，补跑预留2–3小时，
并发CPU/AFS竞争可能增加耗时；6小时是每worker安全上限。没有改seed3407或选择有利样本。
完成后必须出现`TABLE4_COMPLETE=.../report/table4_rows.tex`，仅进程退出不代表汇总成功。
若再次中断，只能显式复用完成并通过验收的shard，不能覆盖或拼接未完成分片的部分query。
