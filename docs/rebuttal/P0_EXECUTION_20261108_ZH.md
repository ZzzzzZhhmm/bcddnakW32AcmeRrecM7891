# 直接证据优先的P0执行

目标是通过正面解决核心科学问题争取讨论后332/333。当前模型固定为作者指定v8-step3000；生产算法和原论文数值保持各自身份，新实验单独记录。主配置为`configs/nonreal/rebuttal_p0_evidence_plan.json`。

## 三个主实验包

1. **生成修复与闭环贡献（BB-W3/Q2、HH-W5）**：两个任务的同状态raw/refined/candidate-null，主局部分支H32，修复/损害/净增益同时报告；独立Full/null完整回合给出配对SR及不确定性。Full与null保留相同模型、观测和原生回合记忆；null同时撤回检索source及conditioning。正式任务为Rearrange Blocks与Battery Try，分别覆盖既有M1负差和Mn反馈收益问题。
2. **自然缺技能退让（RK-Q2、HH-W4、BB-W4）**：非空、规模受控的技能家族holdout库，对比clean与holdout的自然gate、实际候选覆盖及Full/null配对SR。先标注交互家族、再冻结缺失定义；不能把删去同任务名称等同于全库没有相关原语。按预定查询执行完整候选池验证实际覆盖。
3. **同信息、同预算的简单训练对照（BB-W2、HH-W3）**：保留当前观测、事实回合记忆、相同候选池、共同骨干与数据，单独训练不依赖显式后果比较的检索先验控制。与作者模型3000次更新/有效batch128对齐，先测单H100吞吐/显存和更新完整性，再将训练拆成可恢复分段。训练控制不是推理开关。

候选区分性的附加采集复用实验包1：每任务预定20个来源episode、ordinal8/48处各32候选，报告中心化误差、排序、覆盖与选择率；未到达的查询单独计数，不按得分另挑状态。实际WARM记忆与每个阶段的恢复资格是分支采集的必要前置，独立fresh-reset闭环不依赖快照恢复，可先启动。

## 首先启动的可执行入口

`scripts/run_rebuttal_rmbench_pairs.py`：独立Full/null模型和retriever能力令牌，按照相同专家准入seed与官方unseen指令重置；核对初始物体/目标、关节状态、相机和指令。按固定case ID交替执行顺序，保留全部seed准入、逐动作qpos、policy telemetry、预定query的源/候选数组与图像及每个完整回合的原始结果。运行错误、截断和正常失败分别记录。

正式规模固定每任务/条件100对，candidate seed从340800000顺序开始，最多检查2000个；pilot用3407/3408且不混入正式cohort。续跑必须沿原seed序列与全局episode ID，不根据成功率决定何时停止。运行从immutable源码快照启动，不在任务中访问Git远端或拒绝dirty checkout。

`src/fastwam/research/policy_state.py`覆盖真实working-memory snapshot、执行历史、retriever发放/消费集合与continuation、模型thread和待执行队列；分支期间任何变化均可检出。7项纯软件测试已经通过，真实集成由CCI资格作业另验。

后续真实历史入口为`scripts/collect_rebuttal_rmbench_recovery.py`：核心来自已绑定合同的immutable源码，采集扩展另记source hash。先在固定Full factual轨迹上到达有真实执行历史的query，封存同上下文的raw/refined/null，再做H32分支和重复refined。独立标签`physical-next-interaction.v1`只读实际物理/任务谓词，不能替代历史适用性标注；几何边界及无效分支保留unknown。该入口当前先做排除pilot，完成真实恢复与policy不变性验收后才晋级科学采集。

技能库干预在`src/fastwam/research/repertoire.py`：预定家族holdout、等规模clean和stable全局row tie-break，共用原bank及EventID。新增搜索域是显式实验干预，不把受限检索伪装成原全库部署。query候选实际执行覆盖仍是零相关技能声明的必要证据。

## 理论证据同时推进

`P0_THEORETICAL_EVIDENCE.tex`给出共享目标候选项的精确均值/方差分解、事实监督的条件Bayes风险差、同候选同历史下有限Euler步的撤回/稳定性推导。共同目标确实不能独立识别不同反事实；后两类理论结论都列出假设。自然缺技能识别和闭环成功必须由实验包1/2验证，不由条件零门恒等式代替。

## 本次基础设施状态

北京时间约19:24起CCI SSH超时；已在超过正常重启窗口后异步请求作者查看控制台状态。本地继续完成采集器、协议、理论与提交。任何未成功启动的作业都不标为GPU在跑；恢复后先核对容器启动时间、已有进程和持久化输出，再从已冻结入口启动。

作者随后切换到`slai_debug_workspace_1`、180.184.249.158:21276，路径不变。北京时间19:51已实际连通，新H10080GB空闲、driver580.95.05，AFS源码HEAD为d578a78且无本轮外的服务器修改。当前PID1显示为8月启动，不能用它推算此次CCI四小时租期；本批继续采用有界任务并保留退出/归档余量。

## 验收与材料流向

逐query/逐trial原件与失败日志保存在Git外并双端归档，哈希、source/checkpoint/config和精确支持主张写入`docs/nonreal72h/EXPERIMENT_RECORD_ZH.md`。合格结果才进入逐审稿人回复与原LaTeX修订；优先交付能直接回答问题的表、配对区间、可追踪轨迹和明确实验设置。当前新代码尚未取得任何新的效能数字。
