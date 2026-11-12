# 11月12日 ACP CPU/GPU并行执行与恢复记录

本轮按作者要求将纯CPU审计与GPU推理解耦。不能把控制台“启动中”当成已经获得GPU。以下时间均为UTC，换北京时间加8小时；所有实时状态需要重查。

## 已实际执行的CPU任务

均使用标准资源、`computing_cluster_01e_02`、0 GPU / 4 vCPU / 16 GiB、1副本、重启0次。确认镜像为`ccr-zhicheng-02/lwz_wam:lwz-workspace-20261029145800`；AFS为`afs-p3-01e`、子目录`/`、挂载`/mnt/afs`、当前用户读写。

|任务|UID|实际结果|
|---|---|---|
|warm-cpu-q96-20261112-p33|a3946661617843e0a83122efe9035912|08:51:19–08:51:46，exit0，27.42秒|
|warm-cpu-matched-20261112-p33|078a0be9d9294a6e87baae23f07c74d1|09:06:02–09:09:57，exit0，235.87秒|

CPU进程的`CUDA_VISIBLE_DEVICES`为空，入口前后source digest均为`6b64cc5bd9f93b780cee5ffa64f4e77195d6f1363ceec30d43787693111144cb`。代码由CCI canonical `3cdebbee419f72f5130668d1a4657716b784f961`本地Git archive冻结；运行程序未访问Git远端。CPU审计不是新增模型试验。

q96重新检查全部32个候选与36个H32终点、四个Full重复终点及父状态，重新计算MSE误差最大约2.78e-17；保留32个标签全0、干预前父状态已经倒下、预测未优于历史/常量参照的结果。不能把重复审计算成新增32个独立状态。

配对审计逐轨迹核验得到20完整配对、41个已知角色结果，均无成功；另有6次已归档未知尝试。RB为12对、BT为8对。原固定100对未完整，`primary=null`；不报告最终优越性、自然缺失技能非劣效或原稿specialist复现。更正报告成员说明：46Full已于08:58:16结束，包含在09:09报告的41角色中。10:21结束的46Pooled才是此后新增的唯一角色；不得将46Full重复加计。

原件根`/mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations/rebuttal_20261109_acp`（下称P）。22成员CPU包为`P/ITER33_CPU_RESULTS_ARCHIVE_V2/raw_cpu_results.tar.gz`，SHA `b3849cf322aa34eea78c432e2be8bec5f110ed0f451e50292a95e9ac8f5a4e2a`；manifest SHA `5d560fdeab5d874805e6ee5a66af6448335c0e09f47698322151d9fb1e1aa975`。V1包保留；V2额外收录q96独立核验与总结文件。工作站原件在项目`tmp/rebuttal/execution/20261112_iter33`，不入Git。

## 已提交的GPU任务与实际限制

09:19标准1 GPU / 8 vCPU / 128 GiB创建被控制台明确拒绝：“当前工作空间在所选资源的限额不足”。随后才申请闲时资源。以下三项使用相同确认镜像/AFS、cluster02、各1 GPU / 8 vCPU / 128 GiB、1副本、重启0次。

|分片与固定病例|任务名|UID|
|---|---|---|
|BT 30–39|warm-mf3k-bt30-20261112-p33b-idle|1868612385df4b8d85ed85c9866656e7|
|RB 30–39|warm-mf3k-rb30-20261112-p33-idle|64f51463275f4795bab6eff333c4392d|
|RB 40–49|warm-mf3k-rb40-20261112-p33-idle|230090418b9a45e6a3c70c9b52a85210|

均属于原100固定配对设计此前未首次准入的分片。三者首次准入分别为09:14:11、09:38:45、09:38:47；绝对截止仍为原master的11月12日18:00，未重置任何旧时钟。两种角色使用相同3600秒worker上限；固定任务/病例/训练seed/3000更新模型/原生任务步数/成功标准未改变。

09:41控制台为两项启动中、一项排队中；未见新科学runner。09:46对BT30实际Pod事件核查：`FailedScheduling: 4200 | 29/29 nodes in VC match job | insufficient resource: nvidia.com/gpu 1`。这是实际闲时GPU不足；全局列表显示8卡并不证明本工作空间能分配。09:48刷新另一集群01e闲时0卡。保留三项等待，不重复创建同分片，也不宣称已实现四卡推理。已向作者异步询问能否提高标准GPU可用配额，等待期间继续CCI和CPU工作。

首个BT30 allocation `3d061596db444c5e9fdbccfc6840b323`于09:30核实已停止。此前详情AFS名称一度为空，完整加载后确认原配置正确；此次停止是对未加载完成界面的保守处理。其原收据全部保留，09:37未见该分片科学runner。后继p33b使用同一冻结计划与原截止；不是重置实验或替换样本。启动操作曾未改变其“已停止”状态，已独立回读后才创建唯一后继。

旧RB00的canonical spec序列化核验、BT00、RB10/BT10的云端已停止状态、原始轨迹归档与实际进程/lease空闲均独立核实；RB30/RB40旧开发预留也在保全17/2214项原件、保留失败及原窗口后显式核销。全部释放收据在`P/ITER33_BT00_ALLOCATION_RELEASE_V1`和`P/ITER33_NATIVE_AND_OLD_ALLOCATION_RELEASE_V1`。释放的是资源占用，不是将未知试验改成失败或成功。

## CCI继续推进

45Full、46Full分别08:15:55、08:58:16真实exit0，各1000原生命令/250次生成/success0。原两role invocation于08:58:22正常结束。09:38:11只接续原顺序的46Pooled，monitor2561344、worker2561350，目录`bt-40/cci_segment_iter33_v1`；max1角色、outer3600秒加kill120秒，原10:44:53截止不变。

09:50核查worker与心跳存活，GPU约24.6GB、0生成/0完整命令，进程等待AFS读取，仍属加载；不把显存占用称为完成推理。运行中快照`P/ITER33_PARALLEL_EXECUTION_LIVE_V1.json`。每次恢复先查实际PID、心跳、命令链、退出收据及两lease，不因低利用率再派发相同角色。

## 存储、文本与下一步

AFS再次仅余约3.4GB，低于原4GiB门槛。本机从现有压缩CAS完整重建旧Full200 38,599,198,720B TAR及七项state，再独立重读TAR和七文件；与AFS七项及原200→3000计划逐字节一致后，仅迁出两份旧raw大文件26,450,253,965B。五小state、所有评测weight、当前Full/Pooled3000、AFS/local CAS和科学轨迹保留。09:50实际空闲29,768,081,408B；即时删除收据的free仍旧值，保留计量延迟，不把逻辑大小当净空闲。

恢复详见`STORAGE_TIERING_20261112_ZH.md`。这项迁移不会改变现有最终模型评测；旧Full200原路径续训前须先恢复，不能用“路径已迁移”否定原证据。

v27三回复、46页PDF/10页主文及全部原始结论保持封存。本轮没有新证据支持将37个pending有利命题整体放行。20项映射的执行增量保存于`tmp/rebuttal/planning/issue_evidence_revision_matrix_20261112_parallel.json`；新资源、CPU核验、未知结果和写作放行必须分开。

后续按三UID查Pod实际事件；可分配才记GPU活跃。结束后核验完整轨迹并生成新的独立CPU报告，不用旧报告给新角色记数。原固定研究完成度/有利程度分开记录；按现有单角色实测时长，资源等待仍是截止前覆盖全100对的风险。等待时完成原稿specialist PB的直接机制方案和真机作者新数据接收，不能让新增shared-stage对照替代原稿specialist主张。

09:57:18UTC更新：CCI实际GPU100%/32,384MiB、17份生成动作、66条完整物理命令、心跳正常，已开始原生推理；未终态，不计新增完成角色。三新ACP分片job_logs均无runner。原件P/ITER33_PARALLEL_EXECUTION_LIVE_V2.json。
