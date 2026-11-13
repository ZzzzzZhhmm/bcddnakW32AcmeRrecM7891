# WARM v40：训练协议统一与自包含回复

本机入口 `tmp/rebuttal/responses/20261114_v40_verified_clean`，三位回复各 1,811 / 2,440 / 818 词，10 / 15 / 4 张表，17 个评论块。感谢与审稿人真实 strengths、作者 we/our 口吻、逐项回答及修订归纳保持。读回复即可看到关键表格，不要求回论文定位证据。20 条原 review 的原文哈希与序号保持，映射改为 v40 对应评论块及论文标签。

本版按四份原 specialist 训练收据纠正 B.1 的学习率说明并补齐 batch/update 预算，Hh4E W1 同步加入四行训练配置表。实测原始收据均是 peak LR **0.0001**，有效 batch **128 = 8×4×4 H100**；BT/PB/Cover 12k，BRT14k。原成绩、候选池、负结果和旧 shared3000 队列不变；没有用训练元数据冒充原 DEV 选点恢复或新评测。

完整 LaTeX 实证工程：`tmp/rebuttal/revision/WARM_ICLR2027_v40_verified_clean`。

- 实证 PDF：`WARM_ICLR2027_revision_v40_verified_clean.pdf`，50 页/主文10页，SHA `a3a63311869fbe965df5b5afaa179dbf2912c8e8e36c96857ced627ca477c2d7`。
- 完整源码 ZIP：`WARM_ICLR2027_revision_v40_verified_clean_source.zip`，SHA `55220d496f0b331e4cc4bb04714ba3c259c142365ca75b14507cbe8ad49e9217`。
- 独立演练：`tmp/rebuttal/rehearsal/20261114_v40_hypothetical`；PDF 51 页/主文10页，SHA `717f8a6b8570c4a7f3f75e2bf66824fcb3442947fd794d9c6e7bc19d67f6e8de`，源码 ZIP SHA `1b1b9086e253b8c69174d4f9e31442738e952e02f1072abc813d41ebfbe81da9`。

两套原工程均 BibTeX+四遍 pdflatex，完整32文件源码 ZIP 分别独立解包重编译；全部50/51页文本与像素精确，只有第24页较 v39 改变且已直接验看，主文像素不变。10图、47张原表及另加的实现表全部保留；0 undefined-reference/overfull，45条既有字体回退警告。匿名与 AI 声明保持。

正文没有 pending、do not submit 或作者执行提醒，采用正常黑色排版。理想数值仅在独立演练版本，段/表前源码注释标记；实证候选未插入未测值。37 实证字段仍 null/未验证/未放行，P1–P5 没有因材料排版完成而关闭。41对 shared 完整结果仍全部0–0，13历史 unknown 保持；新真机素材0。

`tmp/rebuttal/execution/20261114_iter57` 保存 V40_AUTHORING/COMPILE/QA_FINAL/REPLY_QA、原训练协议逐收据 SHA 及封包核验。AFS 完整封存以实际 `ITER57_V40_FULL_AFS_VERIFICATION_V1.json` 为准，不能在上传验真前声称双端完整。

新原 specialist 兼容 control 的生产采样和 CPU 续检见 `LEGACY_CONTROL_POLICY_QUALIFICATION_20261114_ZH.md`；它不改变本版任何成功率。代码后续有界 GPU step 资格与正式匹配训练另行登记。本版尚未发布 OpenReview 或上传正式投稿，v38/v39 四大文件保持。
