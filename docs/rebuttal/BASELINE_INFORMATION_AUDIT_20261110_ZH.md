# 简化对照的信息使用审计与后续门槛

本审计针对新匹配训练线的 `ConditionedPriorFastWAM`，不修改或否定原稿中已报告的Gaussian、no-comparison或learned/constant-utility实验。现有Full/control200均保留原身份和完整state；200不是最终选择点，尚无匹配TEST结论。

## 已核实的实际输入路径

- 数据适配器向两组提供共同的当前观测、历史和检索候选记录；“记录存在于context”不等于策略消费了记录。
- `DirectActionEvent.forward`只显式读取`warped_actions`和`candidate_valid_mask`，通过`**unused`忽略其余事件输入。它将完整H32×14动作确定性打包为4个token；未使用检索事件的视觉状态、历史效应或时间特征来生成这些token。
- `ConditionedPriorFastWAM._source_only_conditioning`组合8个当前回合history gist token和4个所选检索动作token。`CoarseRank`按原候选顺序选择；`ConstantGate`使用预先指定0.25强度，保留共同的有效性/线程规则。
- 父类构造`required_gist`时显式将retrieved event pre/delta置零并屏蔽event mask；source-only分支随后不走Full的`predictive_gist`。因此这些记录也没有通过另一条gist路径进入当前简化conditioning。
- 因此，当前对照可以回答“共同骨干、回合记忆与直接检索动作先验是否足够”，但还不能单独回答“在相同检索视觉/效应信息下，简单策略是否足够”。它不是ReCAP或WarmPrior的官方实现复现。

## 应当如何使用已有训练结果

现有control200作为直接动作先验消融保全，不更名为同信息最终对照；不因审计而删除或覆盖它。Full200仍可沿原数学配方独立续训向固定选择点3000。Full200→500的资源安排不构成已完成的公平性能比较。

更强同信息对照必须在首次训练前明确冻结输入接口：当前视觉/状态/文本、相同因果回合历史、相同K个检索事件的动作与视觉/效应记录、有效mask、相同bank与query约束。简化机制可以直接编码或池化这些记录，但不能仍忽略其中一组信息。若token数或额外参数发生变化，须明示并报告实际计算成本，不能声称接口/预算天然完全相同。

不同结构须从共同初始化开始独立训练，不能把control200迁移后的权重当成该新结构完成了200步匹配训练。配方、数据顺序、种子、最终选择点3000、推理H32/prefix4/NFE10及统计cohort保持预先约定。新结构和固定协议尚待实现/验证，本审计不将候选设计或工程通过写成实证结果。

## 直接对应的审稿问题

BB-W2、HH-W1、HH-W3继续保持未关闭；原稿组件对照和原始benchmark均值可以支持各自范围内的主张，但不代替这项同信息比较。论文与三位审稿人的公开回复不得把control200的训练完成当作公平对照已完成，也不得据此解释M(1)差异。

## 22:46 UTC接续

已实现独立的`PooledMemoryPriorFastWAM`，原生45项及真实训练/在线K32记录的BF16模块前向/反向通过，独立0更新配置已冻结。细节和证据边界见`POOLED_MEMORY_CONTROL_PROTOCOL_20261110_ZH.md`。旧对照的信息缺口没有被重命名掩盖；新对照的最终训练和匹配评测仍待完成。
