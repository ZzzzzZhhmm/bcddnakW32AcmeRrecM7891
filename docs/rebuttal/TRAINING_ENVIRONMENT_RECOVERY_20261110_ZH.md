# Full step200续训失败：真实状态、静态环境诊断与恢复边界

**最新状态（2026-11-10 05:30 UTC）：** 作者重新登录后已确认旧训练b失败，重启0，更新仍200。严格probe未放宽；新增独立measurement operator只把“静态数据采集完成”与“resume准入”分开，CCI实际preflight通过并以a776860提交/推送/冻结。标准创建实际HTTP429 MEMBER_QUOTA_EXCEEDED、成员GPU0后才切闲时。环境测量b因GPU/PVC排队未执行，已05:22:56确认终态停止；随后唯一c（warm-env-measure-20261110-c，UID23bd904f9f3741b8b98e1821800d5ed1）在cluster01e_02创建、重启0/timeout900，05:29仍GPU0启动中。排他输出目录ACP_ENVIRONMENT_MEASUREMENT_V1尚无，因此ACP具体子字段仍未测得，不能称修复完成。旧b的原窗口不重置，任何新资源段须独立记录。

05:28 AFS可用224,506,974,208 bytes，已超过80GiB空间准入，旧容量不足不作为当前障碍；本轮未删除state/CAS。作者确认原稿逐任务specialist，当前shared-stage训练/诊断继续独立归档，不冒充原稿准确率或擅自迁移旧v3权重。

ACP唯一任务`warm-full-resume200-500-20261110-b`，UID `9c6000a4ce734a7bac717b2053197a1e`，已于2026-11-10 01:24:01 UTC实际执行训练入口，01:35:19以exit1失败。不能沿用旧0GPU排队状态。权威产物根是`WARM_evaluations/rebuttal_20261109_acp/full_resume200_to500_v1`；先前某些absence检查读成`full_resume200_500_v1/attempts/operator.json`，因此失效，已在台账更正。

失败发生在父step200完整state哈希校验后、正式resume装载前：`TrainingAttestationError: resume parent contradicts the live training recipe/runtime: training_runtime_json`。没有新的training_metrics、完成marker或step201+结果，不能登记500更新成功。旧训练source前后均4efb8094…，父权重/optimizer/scheduler/RNG及失败attempt均保留。15项失败原件档案SHA `297839849a449a1c4a4521bf782db2db380e3d50bc3cc9b1bdcd1f960f56de52`，Git外iter4/AFS双端保全。

当前日志只证明runtime整体不一致，尚未暴露ACP实际哪个子字段不同；不得先断言仅kernel/platform变化，也不得删除/伪造父runtime或放宽数值校验。恢复优先核实实际差异及原资源窗；原b从01:24起的4h训练窗口不能通过重新提交重置。

新增`probe_rebuttal_training_environment.py`在模型和数据集装载前记录10项真实静态事实：Python/platform、Accelerate/DeepSpeed/Torch/CUDA/cuDNN版本、CUDA可用性、GPU数量/名称/算力。以原父attestation哈希封存输入，逐项报告差异，任何差异exit2，不支持忽略字段/迁移override，输出排他创建。该工具没有policy/model/optimizer装载或训练，不访问Git远端。静态匹配不等同post-prepare数值runtime和全state恢复已通过，原严格validator保持。

8项原生测试通过（0.14秒），包含平台差异保留、同时GPU/软件变化不得被隐藏、字段/设备inventory格式拒绝以及“静态匹配≠完整恢复准入”。实际CCI `warm`训练环境与原step200父记录10项完全一致，02:30:57 UTC probe exit0；这仅排除CCI该静态子集的变化，没有测得ACP实际runtime。冻结检查source `adbd844a03fba735ebc6252d8a153ccdd41e107108b973c7c6773163159993fc`。

02:27 UTC浏览器重新核查仍被IAM登录阻断，保留原登录页与已有作者请求，不重复索取登录或创建未知任务。登录恢复后先确认b终态及资源释放，再将轻量probe置于后续任务入口；差异未明确前不将跨节点续训写成已修复。本轮首次容量检查为79,612,583,936 bytes，02:49:25再次实测仅73,464,709,120 bytes，均低于原80GiB最低准入，暂不启动Full续训或新pooling训练；保护既有所有state/CAS，不通过降低门槛或删除原件腾空间。三路固定候选池已全部完成，02:51按原窗口独占接续case3/Full，同时推进同父对照与文稿工作。
