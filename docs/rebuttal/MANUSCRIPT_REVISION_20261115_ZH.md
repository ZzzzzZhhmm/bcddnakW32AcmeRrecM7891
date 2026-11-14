# 2026-11-15 修订论文交付说明（作者材料）

本轮以已核验 v40 完整 LaTeX 工程为底稿，对照 33 页最终投稿 PDF 及 Bbqv、Hh4E、rkWY 三份最新优化回复独立修订。三份回复的正文与表格 SHA 未改；原稿、冻结首轮材料、数据和 checkpoint 均未改动。CCI、ACP 与定时任务保持暂停，本轮没有运行新实验或对外发布。

## 交付

- PDF：`C:\Users\admin\Downloads\slai_download\WARM\tmp\rebuttal\revision\20261115_rebuttal_aligned\WARM_ICLR2027_revised_20261115.pdf`，完整 39 页，主文 10 页，References 从第 11 页开始。原投稿主文 9 页，增加一页；原 ICLR 样式、纸张、页边距、行距和字号设置保持。
- 完整工程：`C:\Users\admin\Downloads\slai_download\WARM\tmp\rebuttal\revision\20261115_rebuttal_aligned\paper`；干净源码 ZIP：`C:\Users\admin\Downloads\slai_download\WARM\tmp\rebuttal\revision\20261115_rebuttal_aligned\WARM_ICLR2027_revised_20261115_source.zip`，28 个文件，包含所有 6 个图件，包内无作者说明、原始运行输出和过时素材。
- 逐条变更映射：`REVIEW_RESPONSE_PAPER_MAP_ZH.md` / JSON，20 项意见对应 16 组回复及具体 LaTeX label、页码、源文件行号。
- PDF SHA256：`28d522eb8c8a41579b2c953ddd887e88ef0fa61b1edab6c57b83c438985132e0`；ZIP SHA256：`9c4f20932caeee87019f5b2d99a40af843f860137451601c2dd91c73a9e38f00`。

## 内容和排版

1. 以“当前交互证据—候选后果判断—生成影响调节”组织摘要、引言、方法、实验与结论。精简重复动机及无关的动作模型分类介绍，保留原核心方法、公式与性能结论。
2. 主文八张表按 benchmark、memory/source/comparison controls、候选诊断、配对对照、常量门控、门控响应、部署开销展开。新增三张主文表使用已有附录证据；26 张保留的既有定量表数值序列完全一致。原汇总表的 utility 一行去重，并在直接门控对照及其段落保留 11.0 [7.7,14.3]。
3. Battery Try 图保留原始三幅图像裁剪，蓝/橙对照 Full (no drop) / candidate-null (drop)，以模拟器最低高度 0.759883 / 0.311422 m 和 0.70 m 阈值明确差异。它是同状态、32 命令的局部稳定性示例；附录 D 保留相应 strict completion 为 0 的数据，未改写成完整任务修复率。
4. 重新排制鲁棒性图；总览与结构图替换旧字体并清除重复标签。六个图件的所有可见文字均为嵌入的 Times New Roman；照片、原图像像素和原有数值未生成或修改。源码包图件的作者元数据与旧元数据对象已清理。
5. 保留并强化 R2P/SOR 的目的地、顺序、平衡重置及 episode 记忆定义；20 次/任务 SR、20 训练 episodes、1,356 event windows、5 DEV episodes、七条件定性场景保持各自口径。
6. 对齐 specialist 监督、shared-query regularizer、独立执行目标、宏/条件选择指标和各类统计单位；将 conditional episode equivalence 片段整合入附录 A，保持其匹配时序/停止、状态更新及随机性条件。
7. 正文采用作者第一人称和正面证据论证；局限与未来工作在附录。主文段尾、caption、公式引导语已逐项处理；算法扫描的唯一主文“短行”是公式 (4) 的编号，已人工排除。摘要段尾填充率约 91.6%。附录的公式、表格、参考文献及证明符号保持其自然排版，不用改字号或页边距追求行宽。
8. 与三份回复无关的开发诊断与原生运行计时章节移到独立作者档案，原字节内容分别保存在 `SUPPLEMENTARY_DEVELOPMENT_MEASUREMENTS_UNABRIDGED.tex`、`SUPPLEMENTARY_NATIVE_LOGGED_COST_UNABRIDGED.tex`。保留所有有利、不利和未知结果，不以其他模型替换原报告；主文成本采用受控 H100 recorded-input profile。

## 验收

完整工程经过 pdfLaTeX、BibTeX、后续三遍 pdfLaTeX；干净 ZIP 独立解压后再次完整编译。两份 PDF 的全部 39 页文本及 108 dpi 像素逐页完全相同。最终 0 undefined references、0 overfull、0 font fallback；全页渲染检查、主文逐页核查、匿名性、AI use statement 和无作者提醒检查通过。指标和数据修改仅为呈现、去重与位置调整，未加入模拟数值或把计划写为完成实验。

完整数值、编译、图像来源和交付校验证明在 `DELIVERY_QA.json`、`TABLE_NUMBER_COMPARISON.json`、`PUBLIC_SOURCE_MANIFEST.json` 与其余来源文件中。本轮完成写作同步，不把 20 项意见的编辑覆盖误标为所有实证问题均已关闭；补证事宜按作者要求留待下一轮讨论。

Git 版本仅保存 `docs/rebuttal/manuscript_20261115/public_text_snapshot` 下 22 份公开文本源文件；图件与完整工程位于上述源码 ZIP，原始证据和生成 PDF 均在 Git 外。
