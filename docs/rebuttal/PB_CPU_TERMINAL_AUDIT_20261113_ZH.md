# PB固定候选池的0GPU终态审计

`scripts/audit_rebuttal_PB_terminal_pools.py`只等待既有PB GPU监督器的query退出和完整原件归档，不派发、不重试、不修改任何物理/模型实验。ACP配置0GPU、4vCPU、16GiB、标准CPU规格、单副本、重启0；三个固定query必须全部记账。审计输出目录与科学原件分离。CPU绝对截止事前固定，不能重置GPU原15:15:38→23:15:38UTC时钟。

每个query的已归档终态到达后，可独立CPU子进程逐项核验TAR/manifest与原目录、冻结输入/动作/预测、accepted物理和计数连续链、只读盲判标签、重复Full控制，以及原768维H32均匀DINO delta。只有exit0、非资源截断、完整operator且验证成功的query输出数值表；失败、未开始、无归档和未知单列，不补0、不自动重跑。三个query的结果仍来自同一factual episode，不是三个独立成功率试验。

各参照严格使用相同的完整H32候选支持。invalid和提前成功的H32 effect留空；所有32个原rank仍显示其状态。未记录required query effect时constant absolute MSE留空，不能暗用零向量。输出JSON、CSV、LaTeX行和静态指标图只供审核；不替代原稿benchmark、不作群体置信区间，也不自动放行有利PENDING结论。q8开发验证详见台账23.266与Git外iter39证明；相同原件的重复审计不是新科学样本。

运行器只使用冻结源码和输入SHA，不访问Git远端，不加载Torch/模拟器，不因canonical工作树脏而失败。进入新的正式0GPU ACP任务前，须先完成CCI实际原件审计、源码commit/push/离线同步、固定计划/预检，再回读唯一云UID和真实进程。未知创建不重复提交；运行期GPU与CPU材料均保存在Git外。
