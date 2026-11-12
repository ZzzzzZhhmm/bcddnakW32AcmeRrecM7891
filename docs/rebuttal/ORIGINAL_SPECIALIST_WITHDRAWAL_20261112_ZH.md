# 原代际 specialist 的候选撤回资格与下一步

本轮实现位于 `src/fastwam/research/legacy_gate_intervention.py`，不改原模型、checkpoint、训练或部署默认行为。仅在显式研究干预的单次推理作用域内，将原学习门的 probability 置零；保留其自然输出作为见证，不把强制干预称为模型自发撤回。

## 已完成的工程资格

对原 f77c633 与 47277e5 两个不可变源码工程分别运行 `scripts/qualify_rebuttal_legacy_gate_intervention.py`。两份原 v3/timing4 模型实现文件 SHA256 相同：`d89796cc5da67d431d911e5f9191c5d60a14599bd3763cc64198e1384dc0b67d`。每份使用原工程的小型 CPU fixture 完成 23 项检查；未装载真实 checkpoint，未运行 GPU 动作生成或物理分支。

检查覆盖 Gaussian 初始化精确相等、候选条件同时归零、回合历史仍影响条件、候选内容扰动不改变撤回后的条件、保留原检索元数据、干预前后 Full 恢复精确，以及梯度/训练/重复调用拒绝和异常清理。原模型没有统一的 `force_null` 研究开关，因此不能把新版 shared 模型的入口直接套到旧模型。置零选择概率也不等于检索库为空；原选择元数据仍保留。

首轮 qualifier 因重复调用的拒绝消息不符合预期而失败：第二次调用已被 gist 计数守卫拒绝，但在较晚位置才拒绝。修复为 source 方法入口即拒绝第二次调用后重验通过；旧失败另存 `ITER29_LEGACY_CPU_V1_FAILURE_AND_CORRECTION.json`，不掩盖失败。真实作用域仍要求参数和 buffer 的版本计数保持不变。

原始收据存放在 Git 外、本机 `tmp/rebuttal/execution/20261112_iter29` 与 AFS `rebuttal_20261109_acp`：

- `ITER29_ORIGINAL_V3_CPU_QUALIFICATION_V2.json`：f77 原工程的 23 项检查。
- `ITER29_ORIGINAL_472_CPU_QUALIFICATION_V1.json`：472 原工程的 23 项检查。
- `ITER29_ORIGINAL_PB_BINDING_INVENTORY_V1.json`：PB 旧严格加载证明的恢复核对。

以上仅关闭原代际入口的 CPU 工程检查，不填回复 P2/P3 槽，不证明自然 missing-skill 检测、修复收益、非劣效或原稿成功率复现。

## 真正 GPU/物理证据的准入次序

1. 在不可变的新控制器中绑定原 checkpoint、训练源码、runtime、完整事实输入、已有固定 query 与共同 Gaussian。不得改用按结果筛出的状态，不重开此前已到期的 shared-q8 资源窗口。
2. 用原 checkpoint 精确重现已有 Full H32，逐项核对图像、proprio、历史/候选、RNG、源与条件及归一化和原生动作。任一不一致即保留为工程未知，阻止物理 null 分支；不修改数值 validator 以通过检查。
3. 在同一事实 context 上生成候选撤回 H32，并通过完整生成器的候选扰动不变性和再一次 Full 重现检查。CPU source/conditioning 相等不能代替整个 GPU 动作生成器的检查。
4. 在原固定状态重建成功、Full 重现成功且动作在物理分支前封存后，按前瞻协议运行 Full/null/Full 同状态 H32；单独记录下一交互、掉落与最终成功，保留失败与未知。

当前两路 GPU 任务来自原冻结实验，不能为这一步重复创建或争抢模拟器 lease。现有 q8/q96 不利样本、全零标签和每个原资源窗口全部保留。实际 GPU 准入须另有明确身份、预算、退出收据；本文件本身不表示已启动。

## Press Button 原模型的独立用途

PB step12000 已于 Nov10 完成真实 v3 严格加载（权重 `a765d987b2df0d228782980592c39aaed0e837040c7f6f48c905e4641f79c285`），本轮恢复原证明，无需重复读入 12GB 权重。训练原代码为 47277e5，配置 recent-event capacity 为 16、action-summary capacity 为 8，原生任务上限为 1500，不能复用 BT 的 20/1000。

PB 在线合同应由其原代码构建，并显式绑定 root seed 3407；旧 matrix 默认 17，原 builder 不支持 `--seed` 参数，应通过输入 matrix 设定。真实 DEV 选中 step 仍未知；不得声称 step12000 已恢复原稿选点，也不得根据 TEST 成绩挑选权重。后续有界 DEV 能力与分支资格测试属于独立补证，不替代已冻结的 shared 100 对对照或 BT 原状态。
