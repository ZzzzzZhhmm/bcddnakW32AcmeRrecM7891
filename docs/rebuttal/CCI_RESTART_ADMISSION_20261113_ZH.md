# Nov13 21:42UTC：RB45加载期中断及CCI窗口准入修正

本页优先于本轮21:29的“RB45正在加载”快照。CCI workload bootstrap实际于21:33:45.223041UTC重新启动；旧monitor最后心跳21:33:42.017620，launcher2587139、child2587173/2587178全部不存在，跨节点两个lease可获取、GPU无进程。原operator/runner文件仍写running且没有退出收据；它们是过时原件，不代表活动进程。

RB45第二次Pooled尝试只留下loading/null、0 query，未形成typed-policy加载完成收据、生成输出或物理命令；45 Full从未启动。29个原件逐字节保全为`ITER43_CCI_RESET_UNKNOWN_RAW_V1.tar.gz`（b3f6c12dba8341af6f1e7229c03c7496126fe850622d06d0bb8d9b9d54910f19），manifest f0b9674b821177583a914ee6e3086a225884e2f372909c78883e4b59d6906715，本机独立验收。原running/null/缺失exit全部不改，不伪造exit0或exit-15。

新增独立CCI重启加载unknown1后，matched历史unknown尝试为**13**（原cohort10、旧加载1、guard1、本次重启加载1）。完整科学证据仍**82回合/41固定0–0配对**。原161命令中断与本次0命令中断各自保留；不把任何一项补0。禁止自动第三次RB45尝试、恢复旧ACP或跳过条件启动45 Full。

本次转交检查确认了空闲GPU及上次runner终态，却未核实workload剩余重启窗口，准入检查不充分。新增只读`cci_window`：从唯一已观察到的CCI bootstrap取得PID/start ticks/年龄，只返回命令hash，不返回命令内容；PID1可能是存活数月的pause，kernel boot ID也不能作为workload年龄。要求新鲜观测、完整任务时长与退出/收据余量都在保守4小时窗口内，近重启、旧观测、身份不明或预算非法均拒绝。

17项Windows和17项原生CPU检查通过；原生实际观测确认新bootstrap年龄约543秒。新鲜窗口的7200+180秒检查仅为CPU准入验证，**没有新GPU派发或重试授权**。第一次临时测试包因namespace解析失败在collection阶段停止，补齐隔离包入口后17项通过；此工程检查不计为策略尝试或科学证据。

全部49个既有冻结入口保持；guard只用于之后独立新登记的CCI任务，不能改活动快照。119.8054115725/120 GPUh预留保持，三项尾窗credit已经一次生效；没有因为未知退出自行退回本RB45预留，所有旧时钟保留。

现在唯一实际GPU流水线是原PB ACP85e374b421a24794ae0e9f76d0257378的q96（21:42已进入rank20父状态重建），原23:15:38UTC结束；标准0GPU审计1ae90ef25fac4730a0ac38ba1405df73等完整终态至23:25:38UTC。先读取这两项真实心跳、完整32候选/Full控制/退出及原始归档，再更新机制段；不将中途计数当整池统计。37整段pending仍不放行，v30/C.6固定75回合成本分母保持。
