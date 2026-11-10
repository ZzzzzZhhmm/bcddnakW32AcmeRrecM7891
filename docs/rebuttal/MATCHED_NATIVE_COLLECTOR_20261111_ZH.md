# Full/Pooled双checkpoint原生评测入口

`scripts/run_rebuttal_matched_native.py`运行一个固定case的一个显式角色；`scripts/run_rebuttal_matched_campaign.py`在持久phase/shard窗口内顺序调度角色。纯准入、配对、实际采样见证和未知结果规则在`src/fastwam/research/matched_native.py`。旧Full/null入口、原模型/部署文件不改，原训练/候选实验继续使用其冻结快照。

角色绑定包含独立bundle args、checkpoint SHA/更新步/训练attestation和明确config/class/schema：Full使用sim_rmbench / WarmRetrospectionFastWAM；Pooled使用sim_rmbench_pooled_memory / PooledMemoryPriorFastWAM。共同模型和部署仍从已冻结source87199cc6…导入，新增collector单独冻结并验SHA。不同训练模型不会被当成同一checkpoint消融。

两个阶段必须严格区分：

- recorded_dev只允许原最早PB DEV episode435、frame0至36每4帧十个前缀、Full200和Pooled100，用相同新namespace检查真实高斯、当前图像/状态、bank rows和因果历史。生成前不传后续观察；生成后才登记原记录的四条已执行动作。不加载仿真、不执行生成命令、不估计任务成功率。
- formal只允许事前设计3484ad14…、固定final3000/seed3407、RB700或BT1000、各50对里的固定10-case分片。需要真实原生并发/可写路径隔离收据、事前50-reset roster及两final完整训练状态/checkpoint等准入证据。尚未完成这些门槛时不得启动正式评测。

采集器不启用Full专用probe，不改机制模式；实际infer/source包装返回原对象并在结束恢复方法。H32原输出在消费者验证与动作执行前封存，原生支持的[32,14]和[1,32,14]都保留。所有重规划保存动作、种子、输入/来源/hash，预定ordinal0/8/48额外保存有限context和当前图像；每个原生命令记录调用前后实际关节与TOPP异常。TOPP观察器只记录官方异常并原样返回/抛出，不改变官方继续执行语义、不提前中止原生回合。

只有官方成功/原生上限，以及有已封存原始返回动作证明的明确输出形状/非有限错误，才有确定任务标签。基础设施、装载、未知退出或未完成轨迹保留unknown，不补0。配对核验共同初始RGB/状态/指令与实际初始Gaussian；轨迹分叉后不强行宣称后续输入相同。DEV共同记录轨迹则检查全部十个query的输入和Gaussian。

campaign lease由子进程继承，未知退出阻止重复worker。每个attempt保留exit、日志、原输出与档案；明确未知终态需检查后显式resume，不能在多个已知结果间挑选。持久first-start和固定绝对截止同时生效，max-new-roles只限制本次调用，不重置窗口。程序不探测Git远端，不因dirty工作树失败。

19:00 UTC CCI实际实现检查：35项通过（包含新入口、旧池化配置、lease/档案生命周期和有界调用），四文件语法通过；0新GPU推理/0仿真/0正式TEST。后续仍需新入口实际Full/Pooled DEV匹配资格、最终权重严格装载、50-roster及真实并发隔离验证。接口实现和单元测试不等于正式评测已就绪或已有收益。
