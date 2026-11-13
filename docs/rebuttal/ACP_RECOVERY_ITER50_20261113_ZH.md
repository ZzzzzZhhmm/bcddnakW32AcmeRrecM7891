# iter50：登录恢复后三路ACP运行与证据边界

**后续覆盖，08:40UTC：** 下文08:01三路运行是历史截面。null父程序08:20 exit0，但两子case均exit1、在撤回动作生成前被固定16项history布局检查拒绝；各9次Full查询/32原前缀命令精确，0额外查询、0撤回分支命令。26原件已本机与AFS核验；这是采集器工程失败，不是已完成的Full/null结果。在线history会逐步增长，CCI修复只保留真实长度和值、核对对应mask。新恢复入口要求两个已知错误/已知退出及完整13项原件，严格保留原首准入和11UTC截止；未启动新恢复任务。48原生CPU通过，Windows32通过/16Torch依赖skip；POSIX路径测试修复与首次失败保留。AFS降至约1.46GB，先完成旧PB10000权重的可恢复备份/核验，再满足4GiB准入；当前PB12000、3000及全部活动快照不改。以current_iteration最新覆盖为准。

## 最新恢复入口

本次运行截面为2026-11-13 08:01:45UTC（北京时间16:01:45），不是排队状态。三个独立run_manifest均running、三个父operator心跳距截面不足20秒；08:04控制台也确认第三个Worker运行中。后续恢复先读Git外ITER50_ACP_DISPATCH_V1.json及新的current_iteration.json；下方历史文件中的“仍待登录/尚未创建”不再作为当前状态。

|固定实验|实际ACP名称|唯一活动UID|原生首准入UTC|原截止UTC|
|---|---|---|---|---|
|case0 q96|warm-pb-ns2-q96-case0-20261113-b|34a11139a63744cab8d5868c0ebb150f|07:59:03.381834|10:34:23.203393|
|case1 q96|warm-pb-ns2-q96-case1-20261113-a|68653e49f15f43be86a3af1ef02d126c|07:24:18.631150|10:35:04.911124|
|两个case的q8撤回|warm-pb-ns2-nullq8-20261113-b|96e56a7a25a04df5a59643fd06dfcf4f|07:47:40.544934|11:00:00 hard end|

三路均为computing_cluster_01e、闲时1GPU/8CPU/128GiB、单副本/重启0，已核对私有镜像ccr-zhicheng-02/lwz_wam:lwz-workspace-20261029145800与afs-p3-01e:/挂载/mnt/afs、当前用户读写。q96 wrapper仍为5944997/digest371acb55，原物理collector仍为dc6快照；null仍为d6ce4ab/digestbf544968。49旧入口及三冻结代码树全部保持。

## 先停止确认未运行的排队任务，再转交

原两namespace科学程序的-15终态不改。认证控制台证实原UID0f0e8eee/d5aab172均失败、Worker已删除后，07:05写入两个platform_terminal gate。随后创建三份已准备任务。cluster02中null-a与q96case0-a均出现实际FailedScheduling/insufficient nvidia.com/gpu、没有Host/Pod IP；其科学operator和run_manifest不存在。

null-a（52f5db98…）于07:28确认已停止/Worker已删除；07:42再次确认无原生启动后才创建null-b。q96case0-a（50c8d57b…）第一次停止核查因仍为“停止中”超时；未再点停止，07:53回读已停止，07:54确认Worker已删除，07:55原生确认未启动后才创建case0-b。两个handoff admission保留原件和SHA；原冻结launch、固定case/query/模型/标签/指标及所有旧时钟不变。五次云创建对应三个逻辑阶段，两个未运行排队实例已关闭，没有同时运行同一阶段或未知创建重试。

q96各最多4h且含180s退出/归档，并始终受表中原截止约束；null最多3.5h且含退出、hard11UTC不变。累计准入预留143.8054115725/144GPUh，不新增预留、不退还未知退出、不重置旧clock。CCI没有新增科学准入，不因GPU空闲绕过旧bootstrap时钟。AFS本次截面4,859,973,632B，4GiB准入/512MiB运行保护保持，不清理当前3000/原specialist/CAS或唯一轨迹。

## 当前运行并不等于新增机制结论

q96只执行先前封存的原策略提案及Full动作，GPU用于原四token DINO效果读取与原生物理回放，不生成新的策略样本。case1截面已封口anchor-a和anchor-aa，正在原顺序rank-5父状态重建；case0正在首次资格前缀。两组仍无完整32候选/四Full/CPU审计，不能发布整池预测、修复收益或总体成功率。

null父心跳正常、case0子程序仍在原模型加载阶段；尚无可发布的Full/null/Full行为结果。依次执行两个原固定q8，不按结果换case；原始动作、真实history、RNG及两侧Full控制必须通过精确复现。原namespace q48两条中断不重跑；shared matched 82角色/41个0–0对及13历史unknown也不混入本实验。

## CPU训练语义检查仅作为工程诊断

ITER50_EFFECT_HEAD_STATIC_AUDIT_V2.json纠正V1误选最大.mot字典的检查方法。原checkpoint将九个WARM模块另存于warm_retrospection_modules；10000/12000两固定step均完整存在。只读CPU/mmap检查了adapter与gate的56个小tensor，54个不同，effect_mixer四个参数均不同；原配置loss_effect=0.05，训练target为储存事件的事实观测效应。V1的零匹配不表示头缺失或未训练。此检查没有模型forward/物理动作、没有选择新checkpoint、没有改模型或主指标，也不证明跨状态后果校准、修复或自然撤回收益。

## 写作基线和下一步

v35的49页PDF/完整源码ZIP/三位回复/20项绑定/17评论块保持封存；37整段有利条件论证继续待证。新的CPU静态检查及运行记录不填入行为指标空位。收到真实q96/null终态后，先保全完整轨迹、invalid和unknown，独立核对原前缀、H32/计数标签/四Full与原DINO，再创建下一版材料、完整工程与干净ZIP编译和视觉验收。原specialist、shared3000与固定12000范围保持区分。作者新真机材料尚未回传；不重复索取旧录像，不发表或正式上传。

原始收据位于本机tmp/rebuttal/execution/20261113_iter50及AFS实验根的ITER50_EXECUTION_RECEIPTS_V1；全部科学结果仍在原指定目录。current_iteration和ITER50_EXECUTION_ADDENDUM_ZH提供机器可恢复指引，不修改任何活动冻结源码。
