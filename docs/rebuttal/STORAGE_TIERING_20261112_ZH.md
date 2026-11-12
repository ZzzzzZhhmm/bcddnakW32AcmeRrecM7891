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
