# 三个固定初始状态的非空经验库自然门控诊断

这是对既有missing-skill计划的补充诊断，不替代其独立coverage标签、闭环Full/null配对或原200-pair主分析。它回答：在当前事实输入完全相同、检索可用集合发生变化时，现用模型自然输出什么门值？

## 冻结输入与条件

- 所有三个既有固定RB source：episode0/1/2，原seed340800000/1/2，仅query0、frame0。不按模型成败、门值或对照差异另选状态。
- 直接读取已封存的初始RGB/官方关节API输入与原指令，验证原始数组哈希；使用控制器原生空初始历史，不人为补零或构造过去动作。
- 原作者指定shared-stage v8-step3000，原科学core、相机/DINO/normalizer、H32、prefix4、NFE10、K32保持。先独立核验数值runtime与四份原事实DINO精确数组，再加载policy。
- 只比较已冻结的两个非空等规模搜索域：clean_size_matched和target_skill_family_holdout，均19,540个原bank事件，seed3407。原89,680-event bank、事件ID和task context标签不改；排除续接事件泄漏，保留原bank provenance。
- 两条件各在独立进程中读取同三个初始状态，串行占用同一GPU。每条件只生成三个H32，不执行任何命令，不更新任何模拟器；每个状态拥有正常的新episode/query能力。

## 记录与资格

保存全部32候选的原row/event ID、原始probe、生成动作、实际alpha、g、source gate、deterministic eligibility、固定配置阈值，以及每个query的Gaussian/input/DINO哈希。两条件必须复用同一source的当前输入；后续聚合核验实际Gaussian一致性。额外的task-family类别计数只描述事件来源，不作为独立的行为相关性标签。

任何未来/teacher字段、非空旧history、原始输入改变、检索row越过已冻结搜索域、强制null或去除comparison均停止，不将其记作自然拒绝。代码不调阈值、不训练、不重标候选。

## 论述边界

`g>0`仅称positive effective gate。没有实际候选任务谓词标签时，不能称false activation，也不能断言heldout库没有可用的通用grasp/transport。初始状态诊断不证明长时序历史可靠性、技能缺失时整任务不劣或任务成功率。

生产policy清理episode时的success=false字段是无outcome诊断的清理接口值，不计作任务失败；本诊断执行命令数固定为0、task_success字段为null。两条件的完整输出与原生检查、失败及source/plan身份都保存在Git外。
