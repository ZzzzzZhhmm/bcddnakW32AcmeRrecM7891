# 原BT q8事实输入恢复

原`infer_kwargs.pt`对NumPy数组只保存类型标记；不能据此直接构造完整source context。本轮使用原f77不可变代码恢复实际历史，原记录不改，也不填零或伪造历史online capability。

## 已通过的真实记录核验

`recover_rebuttal_original_history.py`仅使用原controller、原记忆更新、已保存的factual world/VAE输出及实际执行命令，恢复固定case0/q8之前的历史。9个query身份与history evidence、8次factual update和32条原环境/模型空间动作hash全部精确一致。q8 snapshot为`c0282bd0eabee3d30e40c9cb68beb07f64e55ee83cbc11a1eb5c99a24547392c`，含16个历史tokens、8条action summaries；没有运行模型、retriever、GPU或仿真。

原件在Git外`ITER30_ORIGINAL_Q8_HISTORY_CPU_V1`，恢复history NPZ SHA256为`1346e51bd60e573c8f6eaf8d7ba5e858671a829ca41368b877cdc1baf09ecfe0`。同一输入不是新独立回合，不增加SR或机制收益。

`recover_rebuttal_original_query.py`的CPU preflight也通过：使用原部署的typed stats loader、原processor与M1 recipe validators，恢复proprio并与原hash精确匹配，0模型加载/推理/物理命令。首次preflight因把原始JSON列表传给processor而失败，已改用原`_stable_stats`，保留原失败与源码hash；没有修改normalizer或降低数值检查。

## 接续：受控的只读检索恢复

下一步仅对原同一相机/qpos快照签发一个新的、合法归属的retriever step，使用原DINO、bank、合同和GPU数值环境。要求step digest、CLS context key、candidate payload、图像、proprio、prompt、seed及候选行全部与原记录精确匹配。通过原所有权validator读取候选事实；它只消费当前新签发的step，不重建或复用历史capability。

原`_context_from_online_facts`是无模型参数计算的转换函数，可将已核验的当前事实与恢复历史转换为原BF16 source context，供后续显式研究kernel使用。当前完整context与真实checkpoint Full GPU回放尚未通过；CPU资格不能替代它们。

只读GPU阶段必须从新冻结控制器运行，单次retrieval，0策略推理/0物理命令，明确时间/存储/显存上限，保护正在运行的固定Full/Pooled任务。失败、未知或输入不匹配均保留并阻止后续null物理执行，不按结果另选query。
