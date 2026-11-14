# 2026-11-15 实验章节为核心的限范围修订（作者交付说明）

本轮执行作者最新范围要求：保留最终投稿 PDF 的非实验章节框架、段落及核心行文，将主要增量集中在实验章节。此前较大幅度精简的版本单独冻结保留；本版本不覆盖原稿、首轮包或此前修订件。CCI、ACP、实验和定时任务继续保持暂停，本轮仅做本地写作、编译与验收，未对外发布。

## 当前交付入口

- PDF：`C:\Users\admin\Downloads\slai_download\WARM\tmp\rebuttal\revision\20261115_experiment_focused\WARM_ICLR2027_revised_20261115_experiment_focused.pdf`。
- 完整 LaTeX 工程：`C:\Users\admin\Downloads\slai_download\WARM\tmp\rebuttal\revision\20261115_experiment_focused\paper`。
- 干净源码包：`C:\Users\admin\Downloads\slai_download\WARM\tmp\rebuttal\revision\20261115_experiment_focused\WARM_ICLR2027_revised_20261115_experiment_focused_source.zip`。
- 原意见—回复—修订位置映射：同一目录下 `author\REVIEW_RESPONSE_PAPER_MAP_ZH.md` 与 JSON；覆盖三位审稿人 20 项原始意见、16 组当前回复，包含实际 LaTeX label、源文件行号和编译页码。

PDF 为 **40 页，主文完整 10 页**。Conclusion 的整个段落及最后一句均在第 10 页，末行下边界为 728.181 pt，距离正文下边界约 4.1 pt。AI/ethics/reproducibility statements 与 References 从第 11 页开始；没有缩小字号、边距或行距来换取篇幅。

| 交付物 | SHA256 | 大小（字节） |
|---|---|---:|
| PDF | `ef224b19c51c6eaa7e77aa0ceeb62abe7abec0bdcd6275b0e99d55226ef46820` | 43,610,171 |
| LaTeX ZIP | `66b70cc23fd071ed7e94f6ed5a99458c40e7bbdb78721fdf11e7093271273057` | 35,306,088 |

## 范围控制与原稿对应

基准为作者指定的 33 页最终投稿 PDF：`C:\Users\admin\Downloads\11572_World_Action_Retrospecti.pdf`，SHA256 `aa04f9519881983b9f6f408dcfb5b912c9290b883dbe31567f671136fa864594`，与既有 baseline 归档同字节。对照其原有行文，非实验章节采用 `tmp\first_submission\WARM_ICLR2027_new` 中对应源文件恢复。

| 章节 | 本轮处理 |
|---|---|
| Abstract | 保留原段落；仅明确 RMBench 的 task-specialist 设置，并作一处与现有证据一致的局部措辞调整。 |
| Introduction | 恢复原稿全部叙述，包括 Analects 引入及原贡献段落；不保留上一版本的整体精简改写。 |
| Related Work | 恢复原稿三个段落及原有分类、引用顺序。 |
| Method | 保留全部章节和段落标题、顺序、公式及核心叙述，仅在四处已有段落内澄清 specialist/shared-policy 接口、raw utility 与 eligibility、指定零门控撤回、原 specialist 监督和 shared-query regularizer 的对应。 |
| Conclusion | 恢复原稿完整五句，不为排版删句或重写。 |

忽略注释、标签与图环境后的叙述 token 对齐检查中，Introduction、Related Work、Conclusion 与原源文件完全一致；Abstract 对齐相似度为 98.64%，Method 为 97.46%。这些是局部文本差异的辅助检查，不作为科学相似度指标。所有上述章节的标题序列和显示公式逐项相同。完整 diff 和五文件内容锁保存在 `author`，避免后续排版再次扩大修改范围。

## 实验章节的核心增量

1. 主文保留原有四张核心定量表，结合 benchmark 和记忆复杂度分组解释整体性能及任务差异。
2. 新增三张来自已有证据的主文汇总表：同候选池的配对预测/选择比较、三训练种子的 adaptive/constant utility 控制、H100 recorded-input 部署成本。按独立 cohort、模型设置和统计单位分别阐述，避免将局部机制诊断写成完整任务成功率。
3. 保留后果预测、干扰经验与两类记忆互补的证据链，删除重复的主文门控中位数表；关键 raw/effective medians 以一句话留在实验正文，完整人数、条件中位数和全查询口径留在附录 C.2.3。原数据未删除或改值。
4. 保留真机任务、20 次/任务原 SR、20 训练 episodes、1,356 个 event windows、5 DEV episodes 和七条件定性场景，明确目的地、顺序记忆与评测重置规则。
5. 保留 Battery Try 的同状态 Full (no drop) / candidate-null (drop) 图，蓝/橙并列及实际最低高度明确区分。它支持 32 命令局部稳定性；附录 D 同时保留严格完成指标，不改变观察的实验含义。
6. 主文现为 7 表、5 图；工程共 6 个图件。六图与此前已验收的 TNR 图件逐字节相同，所有可见文字均为 Times New Roman，原始照片/模拟器图像像素未重新生成或修改。

当前 29 张带 label 的定量表，其表内数值序列与此前修订中相同 label 的表逐项一致；唯一移除的是主文重复门控汇总表，完整对应附录表仍保留。六个附录内容源文件与此前修订逐字节相同。三份最新 question-aligned 回复及其数值表保持原 SHA；本轮更新的是论文的呈现和位置映射。

## 验收与保护

完整工程执行 pdfLaTeX、BibTeX、后三遍 pdfLaTeX；28 文件干净 ZIP 独立解压后再次全流程编译。两份 PDF 的全部 40 页文本与 108 dpi 像素逐页相同；28 个成员 SHA 与工作工程、ZIP、独立解压工程三方一致。编译最终为 0 undefined references、0 overfull、0 font fallback，原 ICLR 样式、数学宏定义及 bibliography 保持。全 PDF 已渲染，主文逐页直接视检，附录通过整册 contact sheet 核查；匿名性、AI use statement、无作者提醒检查通过。

非实验章节按本轮“保留原稿”优先级保留其原有自然换行，不为了段尾填充率再次改写句子；因此不把“所有主文段尾均达到 60%”写为本轮验收结论。实验段落与图表已在完整 10 页范围内去重整理，Conclusion 末句落页检查按完整文字执行，避免仅检查标题而漏掉末句跨页。

作者说明、原稿差异、范围锁、数值核对与编译证据全部放在工程外的 `author`；公开 LaTeX 工程和 ZIP 中不加入这些作者材料。没有模拟填数、新测量或将写作同步标成实证问题全部关闭。冻结首轮、之前的修订版和全部原证据保持；本轮未访问 CCI、提交 ACP、恢复定时任务、发布 OpenReview 评论或上传正式稿。

Git 仅新增此说明及 `docs/rebuttal/manuscript_20261115_experiment_focused/public_text_snapshot` 下 22 份公开文本源，并更新实验台账的写作记录；PDF、图件、ZIP 和 QA 均保存在 Git 外。完整验收见交付目录 `author\DELIVERY_QA.json`、`ORIGINAL_NONEXPERIMENT_FIDELITY.json`、`TABLE_SCOPE_NUMERIC_CHECK.json`、`INDEPENDENT_RECOMPILE_COMPARISON.json`。
