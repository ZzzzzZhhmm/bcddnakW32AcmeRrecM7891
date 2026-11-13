# WARM v40：训练协议统一与自包含回复

本机入口 `tmp/rebuttal/responses/20261114_v40_verified_clean`，三位回复各 1,811 / 2,440 / 818 词，10 / 15 / 4 张表，17 个评论块。感谢与审稿人真实 strengths、作者 we/our 口吻、逐项回答及修订归纳保持。读回复即可看到关键表格，不要求回论文定位证据。20 条原 review 的原文哈希与序号保持，映射改为 v40 对应评论块及论文标签。

本版按四份原 specialist 训练收据纠正 B.1 的学习率说明并补齐 batch/update 预算，Hh4E W1 同步加入四行训练配置表。实测原始收据均是 peak LR **0.0001**，有效 batch **128 = 8×4×4 H100**；BT/PB/Cover 12k，BRT14k。原成绩、候选池、负结果和旧 shared3000 队列不变；没有用训练元数据冒充原 DEV 选点恢复或新评测。

完整 LaTeX 实证工程：`tmp/rebuttal/revision/WARM_ICLR2027_v40_verified_clean`。

- 实证 PDF：`WARM_ICLR2027_revision_v40_verified_clean.pdf`，50 页/主文10页，SHA `a3a63311869fbe965df5b5afaa179dbf2912c8e8e36c96857ced627ca477c2d7`。
- 完整源码 ZIP：`WARM_ICLR2027_revision_v40_verified_clean_source.zip`，SHA `55220d496f0b331e4cc4bb04714ba3c259c142365ca75b14507cbe8ad49e9217`。
- 独立演练：`tmp/rebuttal/rehearsal/20261114_v40_hypothetical`；PDF 51 页/主文10页，SHA `717f8a6b8570c4a7f3f75e2bf66824fcb3442947fd794d9c6e7bc19d67f6e8de`，源码 ZIP SHA `1b1b9086e253b8c69174d4f9e31442738e952e02f1072abc813d41ebfbe81da9`。

两套原工程均 BibTeX+四遍 pdflatex，完整32文件源码 ZIP 分别独立解包重编译；全部50/51页文本与像素精确，只有第24页较 v39 改变且已直接验看，主文像素不变。10图、此前修订版47个表环境及另加的实现表全部保留；这里的47不是重新清点的33页投稿表数。0 undefined-reference/overfull，45条既有字体回退警告。匿名与 AI 声明保持。

正文没有 pending、do not submit 或作者执行提醒，采用正常黑色排版。理想数值仅在独立演练版本，段/表前源码注释标记；实证候选未插入未测值。37 实证字段仍 null/未验证/未放行，P1–P5 没有因材料排版完成而关闭。41对 shared 完整结果仍全部0–0，13历史 unknown 保持；新真机素材0。

`tmp/rebuttal/execution/20261114_iter57` 保存 V40_AUTHORING/COMPILE/QA_FINAL/REPLY_QA、原训练协议逐收据 SHA 及封包核验。AFS 完整封存以实际 `ITER57_V40_FULL_AFS_VERIFICATION_V1.json` 为准，不能在上传验真前声称双端完整。

新原 specialist 兼容 control 的生产采样和 CPU 续检见 `LEGACY_CONTROL_POLICY_QUALIFICATION_20261114_ZH.md`；它不改变本版任何成功率。代码后续有界 GPU step 资格与正式匹配训练另行登记。本版尚未发布 OpenReview 或上传正式投稿，v38/v39 四大文件保持。

## 纯文本阅读与粘贴入口

实证合并阅读稿 `tmp/rebuttal/responses/WARM_v40_reviewer_replies_verified.md` SHA825278342853fde0e565a9c36e1ce0ab0b4e194a160e46e769f89c3043c59f6f；独立构造值演练阅读稿 `tmp/rebuttal/rehearsal/WARM_v40_reviewer_replies_hypothetical.md` SHA66f629ac860e7a323b88e40a758c7d211293ebbd817f5923f4260a0561e2300b。两者分别29与42张表，已双端验真；原稿和17评论块保持。

实证另有 `tmp/rebuttal/responses/WARM_v40_reviewer_replies_paste_clean.md`，SHAecf0ca1c439fc0125e5b353ab0090fc90022b725e814a7b80bbd682536172559；对应目录 `20261114_v40_paste_clean` 含三位单独回复、17块及manifest。只去除HTML源码注释，不修改可见段落、数字或表格，不依赖OpenReview如何显示注释。22成员另包本机与AFS逐项全验，TAR922c74ad4a00f465e55f60440c1022b97aa004fc3e3e3d2b4898c58471534153、manifest a9b5d7b39c949e43cf689dd549ec7a545e4847786fb000e9f22af92c597d05ee。该导出没有把37个待证项转换成实证；正式发布仍未授权。

原完整71成员写作包含两PDF/两完整源码ZIP及回复、QA、原四训练收据，TARb97c1dfc8b035f56c4e1aea5ebfa41fd805612db38735549a9c60659d48c29b9，本机/AFS全部精确。后续执行文档另行迭代，不重写该冻结包。正式新specialist对照读 `LEGACY_SPECIALIST_CONTROL_CAMPAIGN_20261114_ZH.md` 与最新执行状态。
