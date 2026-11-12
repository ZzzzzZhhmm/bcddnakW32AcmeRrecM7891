# 原稿Press Button specialist的固定状态直接机制补证

这是原稿训练运行的补充机制取证，使用`press_button-s3407-v2`的step12000、source47277e5、root3407，不能与shared final3000混称。作者确认该运行属于原稿；原DEV选定step仍未恢复，因此本试验不称原稿准确率复现。

模型SHA为`a765d987b2df0d228782980592c39aaed0e837040c7f6f48c905e4641f79c285`，原模型source SHA为`4a1d05cb548f16bf6a9b239082b02d4697aa8e8c1f43c2f5d8ffbaec196462fd`。已有严格加载和静态runtime/encoder投影证明保持；新步骤推进到实际命令和完整轨迹，不重复把加载成功当成机制证据。

## 固定输入与顺序

- 原seed协议200项的首值0，seed文件SHA`ee29037b74c4ee65358ccbab91549e1aa0e5e05e75226b646cf07b4aca0be221`。固定首值，不按模型表现或可修复程度更换。先按原评测规则做expert setup，随后重新reset为learned episode，expert动作/未来结果不进入模型历史。若原固定setup不准入，保留失败，不顺延到另一个seed。
- 第一阶段只做1次Full replan和4条实际原生命令；保存初始RGB/qpos、模型输出、event-adapter输出、指令、随机源、实际编码器runtime与四命令前后物理状态。
- 第二阶段重新构建相同episode；初始观测、首个模型动作输出和前四条真实物理轨迹必须与第一阶段精确相同，才执行到原生成功或1500命令上限。失败/超时保留为未知，不换seed。
- 固定探针q8/q48/q96，保存执行前infer kwargs、Gaussian/RNG、模型输出与候选证据；没有到达的query明确标`not_reached_no_replacement`。不因看到成功/倒下/候选分数才换query。
- 取证不训练、不修改threshold/候选数/采样/控制频率/原生成功定义；H32、每次执行4步、K32、recent16/summary8与原投影保持。

## 原生成功检查的状态变化

官方`press_button.check_success()`本身会更新按压标志、计数并重置按钮。额外调用会改变被测状态。因此新审计只读取`card_id_1/2`、三项`press_cnt`和三项`press_flag`，不用该方法观测状态，也不把这些隐藏任务字段输入policy。正常原生`take_action`及原expert准入的调用顺序保留。

资格和factual trace同时记录每步计数/标志，除了articulation qpos/qvel，也精确核对四步历史状态。这是防止取证改变被测任务的工程措施，不构成新的模型收益结果。

## 资源与退出

独立新PB阶段接在CCI原46Pooled实际终态之后。先核对旧monitor退出、worker实际不在、GPU无进程和两旧lease空闲，并完整封存45Full/46Full/46Pooled所有已存在原件；旧失败、未知和原10:44:53UTC截止不改。

PB阶段此前只有静态准备，没有实际新episode时钟。新阶段在真正准入时一次性写`admission.json`，最多7200秒；资格最多2400秒，完整trace继承同一绝对截止。最迟11月12日11:00UTC准入，最迟13:00UTC结束，拒绝通过重启重置窗口。单一supervisor lock及original-specialist lease防重复，输出必须为新路径。任何阶段失败即保全并停止，不自动换样本或隐去负结果。

CCI开发验证使用独立源码副本；源码英文commit/push后，从CCI已同步commit冻结三个脚本再运行。新资格/collector/监督器不访问Git远端、不因canonical工作树脏而失败。运行状态以Git外`original_PB_mechanism_stage_20261112_v1/operator.json`、child退出收据、archive及实际轨迹为准。

## 与20项意见和文稿的关系

此研究优先服务BB-W3、BB-Q2、HH-W5的P2直接证据。先有合格parent，后按预定三query执行候选物理分支，再按原先声明的后果指标/参照给出结果；资格或完整factual episode本身不关闭P2。后续分支尚未执行时，所有候选预测优越性、修复比例与最终成功收益整段仍为PENDING。

P3自然missing-skill不能由一次手动null或按钮计数推断；BB-W4、BB-Q3、RK-Q2等的可靠性空缺继续保留。原100对Full/Pooled、原q8局部稳定性和q96不利结果分别保存，不能用新PB替换它们。v27回复/论文保持封存；新数据通过完整身份/轨迹/分母审计后才同步填数并重审论断。当前没有新PB成功率或candidate-effect结果可写进论文。

源码检查：Windows及原生CCI各49项测试通过，覆盖BRT/BT旧绑定、PB模型/seed/1500步绑定、只读计数审计、资源时钟/探针不可漂移和缺失输出不可冒充归档。Windows初次因全局fcntl导入失败的记录保留；现将fcntl限于Linux运行入口，跨平台计划校验通过。测试只是工程验证。
