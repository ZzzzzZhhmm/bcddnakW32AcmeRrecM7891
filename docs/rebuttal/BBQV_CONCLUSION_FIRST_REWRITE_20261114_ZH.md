# Bbqv 结论先行回复重写（2026-11-14）

本轮仅处理作者指定的 Bbqv 英文回复优化；所有实验与定时任务继续按作者要求暂停。不修改冻结的首轮材料、论文 PDF 或其他审稿人回复，不发布 OpenReview。

## 工作材料

工作目录：`tmp/rebuttal/responses/20261114_Bbqv_conclusion_first/`。

- `Reviewer_Bbqv.md`：完整英文正文，2,539 词、十张表；六组覆盖该审稿人的六条 weakness 和四条 question。每组先给明确结论，随后给实验控制、数据及解释；感谢和总结保持作者第一人称。
- `AUTHOR_CHECKS_ZH.md`：证据来源、统计口径、作者核对及真实尚未关闭的科学缺口。
- `CHECKPOINT_DIAGNOSTICS_AUTHOR_ONLY.md`：按 checkpoint/协议统一保留新增效能诊断，含正、负和未知结果，不按结果筛选。
- `archive_inputs/`：冻结首轮 Bbqv 正文与原 v40 机制附录的字节副本及 SHA 记录；它们不是新 GPU 轨迹。
- `CONTENT_VERIFICATION_V1.json`：问题映射、算术核对、原件 SHA、正文检查；正文 SHA256 为 `4a761461b980ee5d127fe6981c79f82287d0bd36deebe7e64112fd9e56f4652c`。

恢复入口：`tmp/rebuttal/planning/writing_current_Bbqv.json`。此入口不覆盖旧 `current_iteration.json` 的历史实验截面；实验暂停记录继续具有优先级。

## 本轮实质改进

1. 以 M(n) 任务增益、同骨干干预、候选池内判别和三训练种子自适应门控对照组织论证，避免仅依赖 0.1/0.9 点总体领先。
2. 保留审稿人明确追问的 M(1) 数值及真实分解；不把描述性差距定位写成已验证的失败原因。
3. 纳入原有 constant-utility 表及三种子差值，不再将已有结果隐藏在计划段落后。
4. 区分主文 19.0% 诊断、扩展 selector 27.3%/65.9%、候选覆盖、raw-proposal 适用与最终任务 SR；不把不同分母的差值当 repair rate。
5. 给出完整成本表、模块时间边界，以及真机目的地记忆/操作顺序的具体要求与完整基线计数。
6. 对外正文无 pending、提交提醒、作者内部部署说明；没有模拟填数或将未完成比较写成已完成。外部方法完整复现、人口级 repair frequency、>50% 曲线和更广真机泛化仍按真实状态在作者文件中说明。

## 核对与保护

- M(n) 的四任务成功次数合计 349/400 vs 326/400；失败减少 23/74，正文使用 approximately 31%，避免用四舍五入后的组均值推导另一精度。
- +7.8 与 +7.3 属于不同消融，不相加；11.0 的 paired-seed t 区间与 episode-clustered 区间保持独立。
- 新增效能诊断隔离依据为模型/协议归属尚需核对，同样不在新版正文中挑选有利 BT q8 个例。所有历史原件保留。
- 当前作者关于另一台机器的完整模型的说明不能自动认定旧测量无效，也不能生成新实测数值。
- 本轮没有新的训练、评测、ACP 创建、CCI 执行、删除或原始数据更新。原有 36 个 tracked deletions 未操作。
