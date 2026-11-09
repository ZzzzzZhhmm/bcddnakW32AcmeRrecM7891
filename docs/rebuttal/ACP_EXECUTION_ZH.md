# WARM rebuttal：CCI开发与ACP运行

作者于2026-11-09确认可申请ACP算力，标准资源优先，标准GPU配额不足时再使用闲时资源。CCI用于修改、测试和冻结代码，也可运行适合单卡且小于4小时的任务；ACP只运行准备好的快照。自动填写的默认配置见`configs/nonreal/rebuttal_acp_defaults.json`。

- 私有镜像：`ccr-zhicheng-02/lwz_wam:lwz-workspace-20261029145800`。
- AFS：`afs-p3-01e`，子目录`/`挂载到`/mnt/afs`，以当前用户权限读写。
- 工作空间：`p3-spatial-intelligence`，上海一区`cn-sh-01z`。

首批ACP入口`run_rebuttal_acp_pairs.py`为两个独立worker各分配一张平台可见GPU，分别运行Full和candidate-null的前10个Rearrange Blocks官方回合。使用已绑定的作者v8-step3000模型、原core、原bank/data recipe与固定v3主计划；H32、prefix4、NFE10、K32、官方700步和自然事实历史不变。仅当独立初始图像、qpos、逻辑任务状态、指令和完整角色身份一致后才合并。资源并行度不会变成采样预算改变。

作业启动时记录实际GPU、驱动、Python/Torch以及控制器/core/配置身份。旧图形客户端只适配580.95.05驱动；不同驱动必须留下preflight失败，另行在CCI准备并验证客户端后重提，不能混装或跳过核验。每个批次最多6小时，失败、未完成回合和退出码保留。输出、运行日志与结果均在AFS的Git外目录；作业内不拉取代码、不探测Git远程、不安装依赖。

本轮已完成的Battery Try正式case0为Full0/1、null0/1，两个角色各执行1000步。这只是一个固定完整pair，不替换投稿表格，也不足以建立策略收益。正式机制source0因第一条事实轨迹不匹配停止；原分支与失败保留，在CCI修复后重新资格验证。匹配Full和conditioned-prior各到step100，Full完整state续训和AFS无损重建通过，但3000更新及SR比较仍未完成。

ACP显示“运行中/完成”仅代表平台状态。科学验收须读取原始命令、checkpoint/config/source、完整回合、失败计数、配对身份与保存哈希。控制台创建失败或网络响应不确定时先核对实际作业，避免重复占用算力。用户其他项目的作业保持原样。
