# Specialist 连续训练与完整恢复副本：11月14日 iter63

本文件记录运行安排和存储保护，不是新的成功率或机制收益结果。阅读最新 `tmp/rebuttal/planning/current_iteration.json` 后，再核对下述独立进程的真实终态；不要按本文的时间截面重复创建任务。

## 已冻结的下一段训练

- 当前唯一 GPU 运行是 CCI 的 `legacy_pb_pooled_s3407_formal_20261114_v7`，从完整640恢复至1280；07:45 UTC 日志已1180，最近完整保存仍640。
- 下一操作端点1280→2000、增加720更新，在1280完成前预先固定。2000与原生产周期存盘点重合，避免额外终点快照；全局12000、seed3407、数据顺序、优化器/调度器、固定100对评测名册及模型源1055文件/manifest `11d283fd…` 保持。
- 实际生产 sampler 与 Accelerate 1.12.0 的0GPU游标检查于07:12完成：从epoch6/b2528/global1280到epoch10/b2080/global2000，四次epoch尾部边界为1309/1496/1683/1870；这只验证控制流程，没有生产模型更新或新SR。
- 接续计划 SHA256 `485d9c1b18775c72236a8f5da92378b5f099e49f18413ec8836f981fc8515236`。下一训练仍需：原段已知complete/0及1280、独立完整state/weight/attestation审计、实际64GiB可用空间、GPU空闲和原campaign余量。单次11400+180+900=12480秒，预约3.5GPUh；192GPUh及11月17日23UTC截止不变，平台剩余寿命仍未知。
- 唯一本机接续器 `execution/20261114_iter63/R7_TO_R8_FOLLOWUP_V1/current.json`，PID64140，07:17 UTC启动，原等待上限7200秒到09:17。最多一次准备/准入调用；未知调用只读核销，不重启原接续器。排队不代表第二个GPU任务已启动。

## 完整恢复副本与可逆冷存

P为 `/mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations/rebuttal_20261109_acp`。

原step66只读完整备份仍沿用05:22 UTC的9000秒stream时限，到07:52:23 UTC。它未完成前不能计“两完整副本”或可释放空间。已知失败时保留原TAR前缀、原退出收据与全部AFS原件；之后另立有界恢复尝试，先验证确定性重建的prefix SHA，再发送缺失尾部，不延长旧时钟。

新增step256完整备份只读取已完成的R5。07:27 UTC已对9件、38,515,456,964字节逐块/完整SHA验证，state-tree `d299e25b…` 与原完整state审计一致，manifest SHA256 `36f807ef31dd7518e23eeb8ab798a206267fc3d4f80ead3dedd3273efee3b237`。本机原step2的9个完整base文件也已重新全SHA验证；精确字节复用20,115,915,267字节，仍需传输18,399,541,697字节。复用是无损传输优化，最后必须形成不再依赖base的完整raw和完整TAR两份；两份仍位于同一物理C盘。

本机 `storage/20261114_STEP256_FULL_STATE_v1/current.json`，PID34832，07:29 UTC启动，7200秒总上限；四个原生reader各有原5400秒上限。07:41的独立带宽协调器只短暂暂停这四个已验身份的只读reader，优先原step66 stream，最多900秒并在finally恢复；不向GPU进程发信号，不重置任何备份时限。读P/`ITER63_STEP66_NETWORK_PRIORITY_V1/current.json`，不要重复暂停或启动第二个协调器。

Windows对仍打开文件的长度曾显示0，但实际原生stdout计数已增长；“传输阻塞”的早期判断已撤销。`OPEN_FILE_PROGRESS_CORRECTION_V1.json` 保存正长度观测，`SSH_PIPE_PROBES_V2.json` 的两种创建方式均精确传完1MiB。缺psutil的V1探针在SSH前失败，保留且不计通过。

唯一冷存协调器 `execution/20261114_iter63/REVERSIBLE_FULL_COLD_COORDINATOR_V1/current.json`，PID56224，07:43 UTC启动，原5400秒上限。每个checkpoint最多一次原生操作。只有两完整副本逐成员SHA及TAR全SHA再核后，才上传证明并执行原生9件全SHA/stat、非当前parent和无FD检查；永久locator、恢复helper、原路径旁pointer和intent均先于三个冗余热存文件的unlink。六件小metadata保持AFS，全部checkpoint内容永久保留，不执行未批准的不可逆保留提案。已知backup失败会跳过该项；未知原生动作必须先读永久intent/receipt核销，不重试。

恢复位置为P/`checkpoint_cold_storage/step{66|256}_full_20261114_v1`，仅在真实冷存成功后才存在完整locator/receipt。将对应完整raw或TAR复制到locator的 `restore_staging_root`，执行永久 `restore_full_checkpoint.py --staging-root ROOT` 验真，再加 `--apply` 原子恢复缺失的大文件；不要替换原六件metadata。

## 材料边界

v40实证回复29表、17评论块、20项映射及两套完整论文工程保持。原37实证字段和新specialist6结果仍未测，无新SR；旧41对shared0–0、13unknown及负结果均保留。新训练、CPU控制流程、备份进度和冷存容量不能代替正式12000及100对评测。ACP07:21实际导航仍为IAM认证页，原登录请求不重复，没有新建或未知ACP任务。
