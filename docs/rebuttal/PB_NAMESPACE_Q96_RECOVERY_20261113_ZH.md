# 原namespace任务中断后的固定q96资源恢复

两原run_manifest分别于Nov13 04:38:15/03:49:00UTC记录failed、exit=-15；q48在rank22/rank11重建父状态时无封口，不能当正常完成。旧父operator的running字段是残留，不覆盖原件来伪造终态。实际调度/Pod原因尚未认证，保留SIGTERM事实，不称已确认被抢占。原q8/factual完整证据保持。

q48完整原件先双端封存，两个36分支序列分别25封口+1未知+10未开始、14封口+1未知+21未开始。未知分支不重跑、不补0，完整pool主指标仍缺失。新资源只拥有原登记且未启动的q96，不从q48结果挑选source或更换query。之后若安排q48剩余工作，须另有原序列/未知保护/跨资源Full资格协议，本实现没有授权或执行该续跑。

每case一个独立idle1GPU/8CPU/128GiB/restart0任务，最多4h含180s退出归档，截止不得晚于原10:34:23/10:35:04UTC。前段实际首准入至SIGTERM时间加本段上界不得超过原10GPUh；总预留143.8054115725/144不增加、不退还未知credit，不把计划当计费。两个q96可并行，另有独立q8 null任务，仍低于6GPU并发上限。

原物理collector仍使用dc6b8ab冻结快照，原specialist12000/source4a1/weighta765/sim a97/DINO四token/H32/prefix4/严格TOPP与PB rubric全部不改；新wrapper只管理资源、收据、归档和CPU复核。q96 plan只相对原q48 plan改变query、其原始预执行快照选中的rank及原分配pool-q0096目录。沿用各case原private runtime/lease；q48不并行，worker自身排他lease防止原owner仍活跃。新operator/退出/归档放在独立恢复根，旧父operator、原q48原件和原clock不重写。

实际派发前必须在登录恢复后认证旧Pod/任务终态，保留源收据并写platform_terminal.json；缺少此gate时程序拒绝推理。未知创建和活动任务不得重复。预检不需要该gate，也不构造policy/仿真或执行物理动作。已提交或已运行只能根据实际创建UID、Pod、原生心跳和退出收据更新。

运行顺序：原collector CPU输入封存→固定32候选/四Full完整物理序列→TAR/manifest原件→禁CUDA CPU独立审核。重复Full不精确、invalid、未知、资源终止均保留，终态失败不自动重启。负的预测/repair结果不会改变预定指标/任务、阈值或原主文成绩；仍与原稿DEV选定step未核实的范围区分。
