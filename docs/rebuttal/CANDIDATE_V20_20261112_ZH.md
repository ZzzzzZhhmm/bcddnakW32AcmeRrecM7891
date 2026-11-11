# v20：作者对话、证据联动与精准表达

本轮依据作者明确提出的四项要求完成：采用自然的we/our作者视角；每位开场感谢并复述其实际认可；逐项回应后总结已完成修订；对实验补证，对论述硬伤作精准替换。公开回复始终面向各自审稿人，未出现其他决策者、索分或录用宣传。

## 可审阅材料

- 三位回复：`tmp/rebuttal/responses/20261112_v20_author_precision/{Bbqv,Hh4E,rkWY}.md`，20项意见、17个可拆分网页评论块。
- 论文完整原工程：`tmp/rebuttal/revision/WARM_ICLR2027_v20_author_precision/`。
- 编译PDF：`tmp/rebuttal/revision/WARM_ICLR2027_revision_v20_author_precision.pdf`，SHA256 `cc162884906c2ec4e9dc88fa78cef1cf7781533f05447ae93324d462eb3a53dc`。
- 原工程ZIP：`tmp/rebuttal/revision/WARM_ICLR2027_revision_v20_author_precision_source.zip`，SHA256 `8e4fa62e714359f997f7bafa7a5db5e8a6d6b4b2833278a3789dc6f666a57b30`。
- `CHANGES_FROM_V19_ZH.md`、`SELF_REVIEW_V20_ZH.md`及20项问题—主张—证据—回复—修订映射位于回复目录。

## 具体改进及其依据

三份开场分别对照原始Critical Strengths。Bbqv认可实用性、后果驱动扩展、两类经验的互补及消融/干扰证据；Hh4E认可问题重要性与整体信息整合思路；rkWY认可动机与可适配、可调节的经验复用。未把认可问题重要性改写成认可有效性。

主体由作者直接解释设计、证据、实验和修订；结尾列明已经澄清的歧义以及方法、实验和成本附录的位置。v20进一步修正并列动词语法、待填差值的百分点单位，以及成本段可能提前暗示行为收益的表达。

论文保持原贡献和原结果，以三项控制决策组织方法；解释event-specific监督与固定replan下的撤回条件。真机段明确R2P/SOR的目的地和时序记忆要求，并与受控回忆消融联动，避免把真机总体成功率单独归因为某模块的因果效应。主文任务差异直接依托已报告数据，不借尚未完成的匹配比较为其背书。

## 验收及证据边界

完整原工程和独立解压ZIP均完成含BibTeX的四遍编译，44页文字相同；主文含结论10页。原九图、已有附录、所有公式/表格及LaTeX数值token保持；0未定义引用、0overfull，45条继承字体warning。相对v19仅第3/7/8页像素变化，已直接验看；其余41页像素一致。匿名与每页内部待证标记通过。

72成员123,816,401B内部交付包已本机和AFS逐成员核验：TAR SHA256 `3fe839215e47cddb67d7f0418b6f3d8ad249f0352f5f0818e2d11cb8cb42d187`，manifest `15c50f6021c7a4bc1ba4e89a10038b40b796319f5aaad33bf5b832b78dc9709f`。AFS存于`WARM_evaluations/rebuttal_20261109_acp/writing_candidates/20261112_v20/`，验证收据`ITER22_V20_AFS_VERIFICATION_V1.json`。

这是完成论证与排版自审的内部候选，不是证据齐全的公开版。P1–P5整段定性方向与37个空数值仍待验证；20项覆盖不等于20项实证关闭。最新执行事实见独立`EVIDENCE_PROGRESS_ZH.md`和实验台账。未发布评论、上传正式投稿或联系他人。
