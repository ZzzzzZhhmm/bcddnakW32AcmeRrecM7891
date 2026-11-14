# Bbqv 结论先行回复重写（2026-11-14）

最新工作稿为下文记录的第二版 `20261114_Bbqv_question_aligned`；第一版保留作对照。当前入口始终以 `tmp/rebuttal/planning/writing_current_Bbqv.json` 为准。

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

## 问题标题与分类回应第二版（最新）

作者进一步规定条目标题只概括审稿人原问题、首句加粗给出答复，并按证据回应/温和澄清/承认呈现疏漏与实际修订组合回应。该规范已落实至 `AUTHOR_DIALOGUE_WRITING_STANDARD_ZH.md` 和 `AUTHOR_DIALOGUE_STANDARD_ZH.md`，以 Bbqv 为其他审稿人的结构示范。

- 工作正文：`tmp/rebuttal/responses/20261114_Bbqv_question_aligned/Reviewer_Bbqv.md`，2,712 词、六个问题标题、十张表。SHA256 `75a92649ec68dba32dd6b8573217a5e23c207fd3d667717da8fd9a3468066f43`。
- 标题保留 W1/Q1、W2、W3/Q2、W4/Q3、W5/Q4、W6；全部为问题摘要，无我方结论；首句分别用一个完整粗体句给出答复。
- 十张表与首版逐字相同；未新增实测或模拟数值。真机正文 +35/+30 pp 为原表百分比的直接差值，query-only 的 0.500 ordering 明确为 tie convention。
- “已修改”说明逐项绑定当前 v40 LaTeX 的十五个实际位置。仅核对现有内容，不修改 PDF，不伪称本轮重新编译或已上传。
- 同目录 `QUESTION_STRATEGY_AND_REVISION_MAP.json` 绑定十项原文、标题/首句、内部回应策略、修订位置和仍未关闭的实证请求；`AUTHOR_CHECKS_ZH.md` 保持作者信息独立。原稿关于真机规模/更广变化的疑问没有被改写成“任务简单”。
- 既有第一版、冻结首轮材料与诊断原件保留；其他两位回复未改动，实验/自动任务继续暂停。文字覆盖不表示所有科学缺口已关闭。
