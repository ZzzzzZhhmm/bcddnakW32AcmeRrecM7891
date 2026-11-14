# 无损独立恢复容器：只读试验与边界

本方案解决本机完整恢复副本持续增长的容量压力，不改变模型、训练配方、评测或任何论文数值。现有raw、TAR、部分传输、失败收据和永久恢复索引均保持；本阶段没有原件释放或索引迁移。

## 可核查的容量依据

`execution/20261114_iter64/LOSSLESS_RECOVERY_STORAGE_PROFILE_V1.json` 汇总17个历史weight及完整step2/256的已验真实际block身份，覆盖逻辑283,482,579,038B。按这些已覆盖条目的精确block集合，未压缩唯一payload为108,613,623,387B；两份自包含容器的payload估计217,227,246,774B，较两份普通payload减少349,737,911,302B，尚未计容器metadata。这是覆盖范围内的存储估计，不是全盘清理成果。

另在原step2三大文件的事前固定24个4MiB窗口上实际做zlib level6压缩/解压验真，100,663,296B压为80,432,143B；0.799只是样本压缩比，不代表完整容器已测大小。没有数据删除、格式晋升或新SR。

## 原字节恢复，而非依赖外部base的delta

`scripts/nonreal/recovery_container.py` 仅用标准库。容器内部保存每个文件的顺序recipe、每个完整原始文件的SHA256及所有所需payload对象；每个容器独立包含全部对象，不依赖另一个checkpoint或外部base。TAR按真实成员payload边界分段，以共享raw与TAR中的相同checkpoint字节，同时逐字节保存原TAR header、padding和metadata。

构建时检查源文件注册身份及前后stat，并核对已知原weight/TAR的完整SHA。验收时每个容器都实际重建所有文件的SHA；第二容器是独立复制、inode不同，仍位于同一物理C盘，不能称独立故障域。恢复命令对已有输出拒绝覆盖，路径recipe不得逃逸，payload损坏、recipe乱序、源变更及资源界限都会拒绝。

7项Windows完整性测试通过：删除测试源后仍能恢复raw/TAR/header/padding/空文件，能拒绝对象损坏、recipe乱序、源变更、越界/重复路径、覆盖已有输出以及无效/耗尽的时限。最初纳秒超时测试受Windows时钟粒度影响的失败保留；修为确定的耗尽deadline并在任何输出操作前检查。测试只证明软件数据完整性，不构成真机或科学结果。

## 当前只读pilot

固定PB2000/PB4000/PB6000三个已完成冷存包，58个文件、89,955,012,703B，包含原raw、完整TAR、历史部分传输及所有小metadata。计划SHA256 `7c34a005ee74d597dc7ac68a0ab6ae9e69cee0275aeb96b2edb8ec9028221d1a`，冻结module SHA256 `2d9c75a3ec015abe80b51f851f83d30b018acd59f4d95913146e4f2f865a7293`。

pilot为0GPU、本机7200秒、至少120GiB余量，创建两份完整容器后逐文件全SHA验真，并从第二容器实际导出原PB2000 checkpoint核对12GB完整SHA。输出 `storage/20261114_PB234_SELF_CONTAINED_PILOT_v1`，实际状态以current.json及终态receipt为准；准备或运行不计成功。旧source、所有原路径、native locator及当前训练依赖都不动。

只有pilot真实通过后，才可另立可恢复迁移步骤：逐项核对两完整容器、全文件原字节身份、实际恢复输出、永久新locator/helper和活动依赖，先完成新恢复索引再考虑移出已确认的冗余本机raw/TAR。它不执行未批准的不可逆checkpoint舍弃策略，不能删除未知或未核销的失败。当前step2原始base仍被step256备份使用过，任何后续处理均需先看该依赖真实终态。

科学材料保持v40；1280仅是正式训练过程端点，完整12000及固定100对评测仍需完成，37+6字段不因存储或CPU检查回填。


## 实际执行截面

pilot于08:46:48 UTC以PID63360在本机唯一启动，原7200秒/120GiB门槛不变；09:01正在逐文件验证第一个完整容器，不计两个容器成功或释放容量。冻结模块与已推送源相同，7项Windows及7项原生Linux软件完整性测试均通过。所有58份旧文件、原生永久恢复索引、当前训练及其依赖保持；后续只读状态以current.json和最终全SHA证明为准。
