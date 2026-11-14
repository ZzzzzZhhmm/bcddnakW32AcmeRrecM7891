# 完整训练checkpoint的永久可恢复冷存

本记录区分两件事：完整内容已保存到两份本机恢复副本后，释放冗余AFS热存；以及未来新快照的不可逆舍弃。当前只执行前者。`CHECKPOINT_RETENTION_PROPOSAL_20261114_ZH.md`中后者仍待作者答复。

## step2：完整内容永久保留，三份热存payload已释放

原运行位于`WARM_evaluations/rebuttal_20261109_acp/legacy_pb_pooled_s3407_formal_20261114_v2`。9份原件共38,515,456,772字节，包括weights、DeepSpeed model/optimizer、scheduler、全部RNG、trainer游标及原训练attestation。

| 对象 | 保存位置或身份 |
|---|---|
| 完整本机raw副本 | `C:\Users\admin\Downloads\slai_download\WARM\tmp\rebuttal\storage\20261114_STEP2_FULL_STATE_v1\payload` |
| 完整本机TAR副本 | 同目录`STEP2_FULL_RECOVERY_V1.tar`，38,515,486,720字节 |
| TAR SHA256 | `7a14b0e5c616c89c99e89df5d174a55db7a778f449058ba53d0994bc2e4e240c` |
| 原weight SHA256 | `566c6170e44e5daec1144e81dd011163b132be56f9f968dd9c799decaa9a4880` |
| 原完整state-tree SHA256 | `a3f3d594a7876240b7354f44d55497427576a03e50e513406d51e07c141485c4` |
| 永久AFS索引 | `rebuttal_20261109_acp/checkpoint_cold_storage/step2_full_20261114_v1/COLD_LOCATOR_V1.json` |
| 永久恢复程序 | 同目录`restore_full_checkpoint.py` |
| 释放收据 | 同目录`REVERSIBLE_COLD_RECEIPT_V1.json`，逐项记录三个热存文件及检查 |

这些AFS相对位置的共同前缀为`/mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations/`。索引绑定9份原件的绝对路径、size/SHA、两份完整本机副本和恢复程序身份；weights/state的原路径旁另有`*.COLD_STORAGE_LOCATOR.json`。两份副本位于同一本机物理盘，数量不表示两个独立失效域。

首次完整备份04:21UTC完成；随后重新全读本机TAR及全部raw文件、重新全读原生9文件并核对stat、确认无活动FD以及step2不是活动训练的parent。索引和恢复程序均先于热存释放写入。05:14UTC仅释放三个大文件，逻辑38,515,399,733字节；六份小原metadata保留，step66/256/640及原specialist末步、shared3000、CAS和失败/unknown均未动。空间以事后实际`statvfs/df`为准。

## 恢复操作

1. 先确保AFS有容纳完整约38.52GB恢复树和当前运行所需余量的真实可用空间。
2. 将完整本机raw树复制到`/mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations/rebuttal_20261109_acp/cold_restoration_staging/step2_full_20261114_v1`，或在该位置解包上述完整TAR；保留`weights/`及`state/step_000002/`层级。
3. 用永久恢复程序加`--staging-root`及上述完整staging路径先验真；该默认模式不改原件。
4. 验真通过后，在同一命令末尾增加`--apply`。程序再次核验全部原件，对三个缺失大文件作同AFS文件系统原子rename，保留已有小metadata，并写恢复收据。原checkpoint完整内容仍可由本机两副本再次恢复。

恢复程序不读取Git远端、不构造模型、不覆盖未知文件。cold locator旁的原始manifest和释放/恢复收据永久保留；不要把缺失热存路径当成丢失checkpoint，也不要把小型实验封存TAR当作完整模型备份。

## step66及后续

step66两份完整恢复副本已开始只读采集，以`tmp/rebuttal/storage/20261114_STEP66_FULL_STATE_v1/current.json`的真实终态为准。未完成全SHA验真前，不计备份完成或可释放容量。新checkpoint的不可逆保留策略没有获准，不能依据本文丢弃内容；已核实的完整checkpoint和原件保护要求保持。
