# 原生 DEV 终态、固定 q8 后果分支和存储恢复

本记录接续 2026-11-11 09:37–10:01 UTC 的实际结果，后续以 `tmp/rebuttal/planning/current_iteration.json` 及新的带时间戳收据为准。所有运行原件、备份和 operator 在 Git 外；此文不构成正式成功率、NI 或缺失技能的证实。

## 已完成与未知分别保留

原 Stage A 首次窗口为 06:29:13.790798→09:29:13.790798 UTC，固定物理 seed3407、final3000 Full、700 原生命令、H32/prefix4/NFE10/K32。运行源 c159f171…、plan804408d8…及既有两次实施失败不改。

| 搜索域及角色 | 已知退出 | 完整命令 / replan | 任务成功 |
| --- | --- | --- | --- |
| clean Full | exit0 | 700 / 175 | 0 |
| clean candidate-null | exit0 | 700 / 175 | 0 |
| family-holdout Full | exit0 | 700 / 175 | 0 |
| family-holdout candidate-null | 未采集 | 0 条已记录命令 / 0 条 query | unknown |

最后一项在模型加载时失联。旧 monitor/driver/worker 均不存在，最后心跳09:26:34，新sshd09:26:54；这符合外部重启中断，但不是已采集的 OS 退出码，也不是平台 lease 起点证明。原 `stage_exit.json` 与最后角色 `exit.json` 缺失保持，不编造，不用新 Stage A 窗口续跑。

`ITER17_STAGEA_EXTERNAL_TERMINAL_ARCHIVE_V1` 封存全部1,664项、44,110,896B原件；TAR23,660,517B，SHA `2eb7ac6d260ee1fc1ae88a22f45e3fbd393a7b017d6133aac6502abdd9d95928`。Windows逐成员核验及三个完整角色各700命令/1,400物理记录的链检查通过。旧保存异常造成的0命令unknown仍另列；不能被后续完成角色覆盖。

clean两角色初态、初始候选/输入和共同Gaussian一致；null全部175查询实际gate/source_gate为0且source精确等于Gaussian。q0/q8/q48已导出的实际conditioning最后四个candidate slots为0，前八个required-gist slots保留，episode recollection的原mask仍有效。mask中24/60/84计数是观测token数，不是记忆事件容量；容量仍recent20/summary8。该实现边界支持“显式撤回候选影响、保留事实记忆”，不证明自然gate能识别缺技能。

clean配对0/0、holdout Full失败均完整保留。holdout Full仍有正gate（175查询min/median/max .1689453125/.19921875/.22265625），最终press_cnt1而成功0。移除六个block任务族后仍非空且等规模的19,540事件，不能仅凭这一规则宣称剩余技能全不适用；独立allK适用性尚缺。小样本0/0也不是NI、修复收益或等效性。

## 固定 q8 分支实现：CPU已核验，GPU尚未准入

`scripts/seal_rebuttal_native_pilot_query.py` 对完成的自然Full源角色、原plan、类型/固定3000权重及完整归档逐项验真，再导出预定q8(frame32)之前已记录的32候选H32、预测、历史效果、requirement、query观测、context及前32条真实命令/物理链。Full动作只有前四步是原factual未来，余28步仍是反事实，不将其冒充原轨迹。

现有 `ITER17_NATIVE_CLEAN_Q8_PRETRUTH_ACTION_SEAL_V1` 的seal SHA `3d540098467c1d28fdc49273da15f99ad353d6496cdda015a9f1b6b0574813d5`，固定35分支顺序为Full anchor-a、anchor-aa、rank0–31、anchor-final。本机已核验全部seal成员和动作。不得因结果不利换query、rank、seed、模型或主指标。

`scripts/run_rebuttal_native_query_branches.py` 用每分支新建原生scene、同一32步前缀、精确API/实际物理关节/RGB和RNG parent重建；不做physical unpack或RPC规划。Full H32 anchor不可用则停止该固定query，不能改点；全H32公共支持集才比较原shared-stage效果定义下的predicted/history/query-constant误差。中断、无效分支、早停和未知完整保留，不对缺失完整效果补值。零短窗交互标签不等于可靠的技能缺失证书。

GPU入口要求唯一私有runtime外部保留标记、既有Stage A全部原件归档及实际进程消失的hash绑定收据、首次3h窗口、冻结controller身份、至少512MiB可用空间和256MiB原件上限。恢复需要原未知尝试已归档及显式审计，保持首次窗口；完成分支只复用原件，禁止复制。当前CPU实际rb-30 preflight和三个真实CLI负例均通过；27项协议检查通过，均无新policy/物理结果。不要把这些工程检查称为后果预测优势。

prospective Stage B固定seed3408及原先登记的反序保持。只有未准入的Stage B允许额外q48被动候选保存，原Stage A默认q0/q8不变；这是观察保存，不改模型、gate、动作或memory。正式准备改用Git外的 `prepare_formal_final3000_v3.py`（SHA2c29354d…），保留v1/v2，增加rb-30外部保留门，五项实际guard检查通过。未创建rb-30标记前也不能冒称已启动分支实验。

## 已备份的旧 Control200 原始状态迁移

作者已明确无法扩容，授权先保存再清理。迁移只覆盖已结束的旧Control200 raw state中的两个大tensor；当前Full/Pooled final3000、全部评测权重、CAS、数据、原视频/轨迹和历史收据保留。

Windows耐久备份：`C:\Users\admin\Downloads\slai_download\WARM\tmp\rebuttal\storage_backup\20261111\control200\state`。全部7文件26,307,343,356B从已有cache逐字节复制，并独立重读SHA核验；`BACKUP_VERIFICATION.json` SHA `96a53ef94cc21fd6625b531790eff5d329d77a3ae707fdc87c476e421f462f4d`。独立原38,437,867,520B TAR仍在 `C:\Users\admin\.cache\warm-rebuttal-control200-state-v1\control200_byte_exact.tar`，当日重新hash匹配d838ed0c…；另有既有本机压缩CAS，不依赖唯一cache副本。

AFS迁移前重新读取全部7原件SHA，并核查原Control200正常终态、无相关进程/打开的状态文件；原12,130,288,488B评测weight SHA93870d33…单独记录。只移出以下两个原始路径的冗余raw副本，共26,307,288,751逻辑字节：

- `matched_training_update200_v1/control-s3407/checkpoints/state/step_000200/pytorch_model/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt`
- `matched_training_update200_v1/control-s3407/checkpoints/state/step_000200/pytorch_model/mp_rank_00_model_states.pt`

AFS `ITER17_CONTROL200_VERIFIED_STORAGE_MIGRATION_V1` 有精确清单、前后stat和恢复脚本；原state根增加 `STORAGE_LOCATOR_V1.json`，历史收据不改。旧Control200 raw resume需先从上述耐久备份将两文件复制到AFS临时staging目录，再运行目录内 `restore_control200_raw_state_v1.py --staging DIRECTORY`；脚本先核验完整SHA、拒绝已有原路径、精确恢复两个文件。当前final3000评测不用它们。

10:01:16两冗余路径已移出、唯一bytes丢失0，评测weight的inode/mtime/size与较小五state不变。**当时AFS df仍约1.27GB，尚未观察到相应实际可用容量增长；不能把26.3GB逻辑迁移等同于即时可分配容量。** 后续必须以实际新df和并发写入余量准入；不继续删独一原件来追逐df，不放宽原formal4GiB门。

另有25份完全相同派生审阅TAR已保留全部逻辑路径/bytes/SHA，通过hardlink合并450,609,777逻辑字节；15包915成员复核通过。八个历史/冻结科学源和选定operator另封存3,723项40,417,093B，9,121,551B TAR SHA0b6b67e6…已在Windows完整逐成员核验，源未改。

## 后续动作

先完成上述源码英文提交、push及CCI离线FF，保留无关36处既有删除和所有冻结科学源。再以真实AFS余量和CCI剩余可用时长准入已登记Stage B及固定q8分支，不重置Stage A；ACP认证恢复后核对旧UID/未知创建，再标准资源优先提交正式100对。旧training、完整CAS、roster、资格、成本和原BTq8/q48均不得重复。新事实应同步进入20项映射、逐人回复与原LaTeX，完整编译/视觉验收后才成为下一候选；不自行发布评论或上传正式稿。
