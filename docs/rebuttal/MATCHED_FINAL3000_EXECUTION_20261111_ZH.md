# 固定3000 Full/Pooled 正式配对评测：执行与恢复

**04:04 UTC恢复更新：** 下文01:52的训练pending为历史状态。两角色固定3000完整训练/state/CAS/typed bundles和十计划资格均已完成；Pooled proof1058dfbc…/bundle929fac30…，不重跑训练或CPU独立重建。实际formal campaign根尚不存在、时钟未启动，IAM登录过期阻止新标准ACP创建；原rb00仅保留草稿。准备器SHA e42963e9…不改，认证恢复且可实际提交后才freeze/prepare，以保护固定窗口。BT q48已04:00:18正常退出并380原件完整封存，q96首次11h未开始；独立私有formal worker并发规则保持。现有AFS约5.98GB，首次准入再检查完整轨迹/有界context及其压缩归档的余量，不删除原件。

已完成的记录成本见候选v15 Appendix C.5，不替代正式成功率。新增CPU结果汇总器17检查已通过，固定100对未齐时primary始终null；不重采roster/reset/十worker资格或把DEV输入一致性当性能。以下保留首次准备过程以便恢复。

本页记录当前已授权协议的工程执行，不提供新的策略效果。科学设计仍为 `MATCHED_NATIVE_FINAL3000_EVAL_DESIGN_V1/DESIGN.json`，SHA `3484ad1477a868607a916638c631c80df25501db59dd82edbf6651c768238e33`。RB/BT各50对、总100对/200角色回合，native700/1000，H32/prefix4/NFE10/K32/recent20/summary8，root3407。两方法均固定最终3000，不以DEV/TEST选择权重。此为新的shared-stage补充比较，不替代原稿逐任务specialist成绩或命名baseline复现。

AFS操作根目录：`/mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations/rebuttal_20261109_acp`，下称P。

## 01:52 UTC 的实际状态

- Full已完整3000：原标准ACP UID `95e495471308439c9da281d2c4fd5b92`；完整state/CAS独立重建与七state文件核验通过，proof `ITER13_FULL3000_INDEPENDENT_AUDIT_V1/FINAL_TRAINING_EVIDENCE.json` SHA `62e9ecd838086a3d4cdd81b196914ff3ab028ad95875d824d2c374a7f165cc2a`。RB/BT静态合同已齐。01:43实际ACP页面再次确认为成功。
- Pooled仍为原标准ACP `c852e8e2e91f4d9ba9d42cfa0682bbb2`，01:48:56实际2940/3000；新增101起连续、loss/grad有限。只读观察不能称最终完成。原训练绝对截止Nov11 12:29:56.020552 UTC保持。
- 原BT q48标准ACP `59dd5f78631b40ae81a1298844d499ba`仍实际运行，01:47有10个独立完整候选；原06:23:48.137340 UTC截止不变。共享runtime不得重复准入；正式worker使用另行验证的十个独立runtime/cache/IPC路径。
- `matched_native_final3000_campaign_v1`尚未建立，正式时钟、正式TEST和云任务均未启动。rb00浏览器表单仅为草稿，不存在新云UID。

## 唯一 CPU 终态资格流水线

01:52:39在CCI启动 `P/run_final3000_qualification_pipeline_v1.py`，outer PID2429211，5400秒上限。源码SHA `c5cb2a50723b682f3ab4279e089596c9fce599061161de40ab91bc413dc995c2`，收据根 `P/ITER14_FINAL3000_QUALIFICATION_PIPELINE_V1`。恢复时先看operator、launch.json、launcher.log、真实PID和子进程退出，禁止再开相同流水线。

顺序如下，均不使用GPU、训练、Git远端或云创建：

1. 等待现有Pooled operator确认 `complete_training_and_byte_exact_archive`，不自行重复打包CAS。
2. `verify_pooled3000_completed_archive_v1.py`：读取原已完成archive和runner，核对固定3000、父100、源48ce6ba8…、连续2900新增有限更新及实际优化配方；在CCI临时盘重建原CAS全部part，与AFS每个原件逐字节匹配，核验optimizer/scheduler/RNG/trainer等完整状态。proof目标 `ITER14_POOL3000_INDEPENDENT_AUDIT_V1/FINAL_TRAINING_EVIDENCE.json`。脚本SHA `3df01180ab5b97f85e5efe4ee0ec447507282eb34e718b925adee72562669bc9`。
3. `prepare_pooled3000_typed_bundles_CPU_v1.py`：用明确Pooled构造器/schema和同一root3407/正式namespace，准备RB/BT静态接口。目标 `MATCHED_FINAL3000_POOLED_TYPED_BUNDLES_V1/receipt.json`，脚本SHA `e707eb6daaf621fdd87ef13259e1550c6cbdbfb7321b82568c3ca1d5edef3a9f`。这不是实际policy或仿真性能。

任何阶段错误均保留原失败、输出和窗口。流水线不会启动formal时钟或提交任务；必须在独立终态核验和静态合同均齐后另行准入。

## 正式准入和云提交

准备器 `P/prepare_formal_final3000_v1.py` SHA `e42963e974f15950559a975a064e5d326bf20c5a01fa011dac94805b0622f0c6`。它直接调用已冻结H861004fa…的plan/input验证，不改变科学运行程序。共同推理源仍87199cc6…。01:42发现的cuRobo路径缺失已在任何正式准入前补齐；修前97dc4e10…原脚本单独保全。

`--mode readiness`不会启动时钟。两final完整proof、明确角色的两任务合同、external-input DEV修订fa089d4d…和独立证明a065c792…、50-rosters和十worker并发资格齐后，才使用 `--mode freeze` 固定唯一phase/master。相位最多36h，且不晚Nov12 18UTC。

之后用 `--mode prepare --shard rb-00` 等固定分片，建议rb00、bt00、rb10、bt10、rb20、bt20交错。共10个固定分片，每个10对/12h，最多同时6个单H100，总上限120GPUh。首次准入时钟包含CPU preflight、队列、错误与重试；不能重置。每个计划绑定各自私有runtime和准确role/schema/权重。CPU preflight验证实际静态合同后，才生成launch/job/prepared收据。

每次点击ACP“完成创建”前记录唯一create_intent和plan/job/shell哈希；标准资源优先。只有标准GPU配额实际拒绝后可改闲时；队列、HTTP不明或UI未回读不等于配额不足。镜像/AFS沿用已确认配置并逐项回读。每个任务1GPU/22vCPU/230GiB/1worker/0自动重启。

未核销的已准备、未知创建和活动分片均占用slot。释放须同时有唯一UID、真实云终态、实际runner已知退出、所有raw归档与哈希收据；本机flock空闲不能作为跨节点释放依据。已知失败才允许显式恢复原分片/原计划/原时钟；不得换case、角色、模型、主指标或遗漏unknown。

## 对照范围和写作

实际两训练resolved config仅八处不同：model factory、四个机制loss、输出目录、parent resume和本段run_steps。数据、其他模型设置、优化器、scheduler、总更新数、seed等叶子值相同。四项loss为retrieval ranking/effect/adaptation/gate，Pooled均为0，符合原事先协议。整个recipe hash因此不同，不能要求相等后伪称“只有网络结构不同”。本比较检验完整方法及对应监督，而非孤立同目标结构消融。

01:32/01:42准备检查和实际配置差异保存在 `ITER14_ENGINEERING_PREPARATION_V1`；无策略/仿真或TEST。Pooled正式性能仍待实际运行，不能将排队、完整训练、静态合同或环境资格填为性能。

内部简洁回复工作稿 `tmp/rebuttal/responses/20261111_v14_brief_working` 与未变v14 PDF联动：3文/15块/20项，4061→2650词，矩阵ea0473bb…。压缩重复、明确对照损失范围，不改变任何原稿或新实验数值。全部empirical closure和stable_release_ready保持false；不自动发布或正式上传。
