# 旧写作版本的无损分层

2026年11月12日AFS空闲跌破4GiB；按作者既有清理授权，所有数据先核验可恢复副本，再整理冗余。当前v26、唯一实验原件、模型checkpoint和已有训练CAS未移动，运行任务的输入、代码和冻结窗口不变。

第一阶段核对本机与AFS的19个旧写作封存包，逐一验证1442个解包文件都可由两端封存包恢复，然后清理2,302,804,661B冗余解包内容。完整路径/成员/hash恢复映射在Git外`ITER30_WRITING_TIERING_PLAN_V1.json`，旧目录有`TIERED_EXTRACTION_20261112.json`。

第二阶段将19个旧写作外层TAR.GZ（2,068,396,667B）无损表示为AFS内容寻址块（1,073,509,158B），保留本机全部完整原TAR.GZ。这里保留的是整个原容器的身份：每个AFS块先读回校验，重建完整原TAR字节，再以原GZIP header/trailer及已验证压缩参数重建，原外层SHA256必须完全一致，才分层其冗余物理文件。19个容器还在本机独立验证了所有顺序块hash和GZIP配方。另用独立CLI恢复完整v25包，SHA3757c283…精确匹配。

AFS新存储根为实验根下`writing_archive_cas_20261112_v1`，含`INDEX.json`、`manifests/`、`chunks/`、`RECEIPT.json`和journal。旧版本目录有`TIERED_ARCHIVE_20261112.json`，原容器文件路径现在可以是逻辑地址；缺少其冗余物理TAR或解包副本不表示原件丢失。恢复不需要重跑任何科学实验。

先用本机原PDF/ZIP/LaTeX即可审阅旧版。需要在CCI重建原外层容器时，使用已提交的`scripts/tier_rebuttal_writing_archives.py`的`--materialize-manifest`模式，默认只dry-run：

```bash
python scripts/tier_rebuttal_writing_archives.py \
  --evidence-root /mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations/rebuttal_20261109_acp \
  --materialize-manifest /mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations/rebuttal_20261109_acp/writing_archive_cas_20261112_v1/manifests/20261112_v25.json \
  --output /tmp/warm_v25_restored.tar.gz
```

加`--apply`才写入指定新文件，拒绝覆盖；AFS目标还要求恢复后有4GiB余量。`/tmp`只用于可重建的临时读本，持久原件始终在本机及AFS CAS中。原GZIP压缩环境zlib1.3.2已记录，若后续环境变化导致外层hash不同，程序停止，不接受近似恢复；本机完整原TAR仍可直接使用。原成员/TAR数据始终按hash精确可重建。

单独恢复旧解包成员可用实验根下`restore_tiered_writing_v1.py`，指定`--folder`和可选`--relative-file`，默认dry-run；该工具支持从完整TAR或新CAS读取。恢复前查实时空间，保留新CAS及所有未核销失败。

第二阶段结束空闲快照4,701,954,048B，随后实测约4.80GB；共享卷并发写入会改变净空闲，因此冗余payload/容器的分层大小不能当成卷空闲净增。后续准入继续核查原4GiB门槛。

当前候选已更新为v27。07:15UTC AFS再次低于4GiB，第三阶段只处理已成为旧版的`writing_candidates/20261112_v26/verified`：92项125,784,549B逐一验证能从完整AFS TAR和本机完整TAR恢复，计划SHAc210b205…及独立收据齐全后才按journal移除重复解包文件。完整v26 TAR SHA7e3ca714…、当前v27及全部科学原件/checkpoint/CAS前后不变，v26目录留下`TIERED_EXTRACTION_20261112_V26.json`及恢复映射。需要恢复时，把保留的完整TAR解包到一个新的空目录，并按保留manifest逐成员核验；无需重跑实验。

07:21:25的即时卷快照前后均4,258,127,872B，之后实测4,382,732,288B；保留原快照，不把125,784,549B的重复文件大小当即时卷空闲净增。完整plan/proof/journal/receipt分别为实验根的`ITER32_V26_REDUNDANCY_*_V1`与journal，journal SHA143b8e38fc31a1fa34aa393e7c335fcd6dea45fddd75f9fa8adf1c143189f083。


## 第四阶段：旧Full200 raw副本的独立保全与恢复

11月12日09:20UTC AFS仅3.37GB。本机`C:/Users/admin/.cache/warm-rebuttal-full200-state-iter33-v1/full200_byte_exact.tar`完整38,599,198,720B，原SHA b2a7a0e2b0b4dd441a24e70f9bb6145ac49db3827321c890b672a60a0db035b6；从保留CAS重建并解出7个state，再独立重读TAR和7文件，与AFS及原200→3000 plan逐项一致。本机原件和独立proof位于execution/20261112_iter33/FULL200_LOCAL_{REBUILD,INDEPENDENT_VERIFICATION}_V1.json。

只迁出P/matched_training_update200_v1/full-s3407/checkpoints/state/step_000200/pytorch_model下两旧raw文件：bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt（12,891,263,217B，SHA236c932fca5d005ee9451c86dca47dd83e4b1e3e298dbbb9a646483e38435db4）和mp_rank_00_model_states.pt（13,558,990,748B，SHAa5fdf1c0d9071eb53c887c6a74544fc43117c752717692409ef4877dd07b7322）。7文件均在本机extracted/原名，完整AFS/local压缩CAS不动；五小state、所有评测weight及final3000不动。原state目录先写STORAGE_LOCATOR_ITER33_V1.json，不能把迁移当数据丢失或静默更改历史manifest。

需要旧Full200 raw恢复时，先将上述两文件按原pytorch_model相对路径暂存到有空间的独立目录（例如CCI临时盘），再用P/ITER33_FULL200_VERIFIED_STORAGE_MIGRATION_V1/restore_full200_raw_state_v1.py --staging STAGING_DIRECTORY。该helper逐项核对原SHA/size，拒绝覆盖现存原路径，检查AFS可容纳两文件和1GiB余量，再原子恢复两个原名。恢复helper的--help检查已通过，尚未实际回填，勿宣称重新恢复过原状态。

移出26,450,253,965逻辑字节；即时free3.35GB因共享卷计量延迟未变，09:50df实际29,768,081,408B。保留所有原快照；4GiB后续准入门不变。完整迁移收据P/ITER33_FULL200_VERIFIED_STORAGE_MIGRATION_V1/receipt.json SHA5407991f8dbaf810f4a63ff6286c28b5623ed2e9879cfa35c886c051b666699c。

## 第五阶段：已完成warmup200的旧weight冷存

Nov13 00UTC共享卷降至约1.16GB。先尝试另存Full3000 optimizer，传输较慢后中止；该原件始终在AFS，两份本机partial读本不计完整backup。改从已保留的本机Full200完整38,599,198,720B TAR（原SHAb2a7a0e2…）重新完整重读，提取`training/checkpoints/weights/step_000200.pt`，再独立重读提取文件。12,148,647,926B、SHA`1fb035c41663752d72098196cc04f04ff03cefcfa7db623bc994142d14af0ac4`精确，副本`C:/Users/admin/.cache/warm-rebuttal-full200-cold-weight-iter45-v1/step_000200.pt`、完整原TAR和AFS CAS均保留。

AFS再次完整hash、确认无进程/FD引用和49冻结入口未变，先写locator及恢复helper，才于00:13:42UTC冷存旧冗余物理文件。旧路径P/matched_training_update200_v1/full-s3407/checkpoints/weights/step_000200.pt，现在调用旧200模型必须先恢复；当前Full3000、原稿specialist及科学轨迹未改。旧“所有评测weight常驻”只对应前四阶段，本次有明确可恢复的冷存例外。

locator在原weight同目录`STORAGE_LOCATOR_ITER45_WEIGHT_V1.json`；收据P/ITER45_FULL200_WEIGHT_TIERING_V1/receipt.json，恢复helper同目录`restore_full200_weight_v1.py`（SHA`2b792da0d0750f031c947bece01609584685ab5b894c37b0678b43dddb7947fd`）。先把完整weight暂存到CCI临时盘；`--staging-file PATH`只dry-run，逐项校验hash/size、目标不存在及余量，显式`--apply`才原子恢复。`--help`通过，尚未实际恢复。首个只读尝试因错误runtime参数key停止、未迁移；核实ckpt_setting后执行成功。

即时free1.1656GB保留原快照，稍后df实际13,312,421,888B，写作验真后约13.06GB。共享写入仍改变净空闲；保持实时4GiB任务准入和512MiB运行保护，不按逻辑冷存字节推算可用空间。
