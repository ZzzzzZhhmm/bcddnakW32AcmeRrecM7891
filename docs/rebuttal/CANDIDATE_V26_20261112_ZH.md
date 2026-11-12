# v26：作者交流、原模型同父对照与论文联动

当前可审阅候选，未发布OpenReview或上传正式文稿。稳定候选目标仍为北京时间11月13日23:00。v25及所有原件保留。

## 四项写作要求的落实

三位回复以we/our逐项说明作者的判断、做法和证据。感谢开场来自各自原review真实strength：Bbqv认可后果驱动选择和两类记忆互补，rkWY认可动机与适配/调节复用的灵活性，Hh4E认可问题重要性与综合pipeline；没有扩张最后一位对效果的认可。

17评论块对应20项原weakness/question，字段、序号和hash均核验。先给直接答案和最有力的相关证据，再说明测量范围；结尾只总结已完成的澄清及论文修改。公开草稿没有面向第四方、索分或评价审稿人动机的措辞。

新增证据用于巩固受控复用的实际作用。原方法、九图、公式、旧表数据和所有任务结果保留。逻辑表达精准调整：原specialist事件事实监督与shared-stage诊断分开；原训练run固定12,000-update机制fixture与DEV选点准确率区分；历史shared加载失败用明确模型与过去时表述；未完成非空bank试验明确为protocol，避免暗示有利结果已经完成。没有删除原核心观点或用新诊断替换原基准数值。

## 新的直接证据

原BT固定q8先精确复现Full，再在同checkpoint/history/当前输入/common Gaussian下撤回候选source与conditioning，保留query gist；候选扰动不变性及最终Full恢复通过，参数/buffer版本和对象拓扑不变。

三条独立重建父状态的H32物理分支均完整，96命令。Full/Null/Full-repeat的最低电池高度为0.759883/0.311422/0.759883m，既有掉落谓词为z<0.70m；Full与重复Full未掉落，Null掉落。原Full和最后Full的端点及效应精确；全部strict下一交互完成标签仍为0。这是一个预先固定状态的局部稳定性因果对照，不冒充总体修复率、完整任务SR或自然missing-skill可靠性。

55项14,163,126B物理及原失败档案已双端逐项/动作/测量链/原父状态typed值与数组bytes核验，效应独立重算误差0。TAR `a3da55a1787785d6d3ba936c72ea263277d17f1f812bea891bc55fd214e74faa`，manifest `471718c70926d69ed5bfdf963377548aea81ba6f3edcd244b8d088fb086719ae`。pickle往返的task/robot/rng序列化差异在原件/台账保留；原运行时严格hash检查保持，未声称重获相同序列化metadatahash。新图像取自原始RGB，无生成或重建。

这些证据联动Bbqv W3/Q2、rkWY Q2、Hh4E W5和附录D.1：图6、表33，PDF第42页。三位结尾同步总结真实新增内容。q96完整32候选/0unknown及不利指标、shared零结果、全部失败和未知均保留。

## 交付与验收

- 回复入口：`tmp/rebuttal/responses/20261112_v26_verified_withdrawal/`，含三位Markdown、17评论块、20项映射、`SELF_REVIEW_V26_ZH.md`与改动清单。
- 原完整LaTeX工程：`tmp/rebuttal/revision/WARM_ICLR2027_v26_verified_withdrawal/`。
- PDF：`tmp/rebuttal/revision/WARM_ICLR2027_revision_v26_verified_withdrawal.pdf`，SHA256 `f5766b90fa0ecc0b37e76ae05f52fcedd8299417d3f609f8e76ffc3894999957`。
- 同名`_source.zip`：SHA256 `2e2e269caca0197a042c33e66d222006071b185fe76ce3e2c603e73ade50b9bb`。

完整工程及独立干净ZIP均含BibTeX四遍重编译46页，主文至结论10页；所有页文本相同。改变的42–46页已直接验看，最终历史范围精炼仅43页改变并再验看，其余45页与已审版本像素一致；与v25的前41页像素一致。0undefined/overfull，45继承字体warning；匿名、内部待证横幅与第11页原AI use statement保持。q96表留在D.1，不漂入待证D.2。

92成员125,784,549B写作包已本机及04:58:45UTC AFS逐项/解包核验，`writing_candidates/20261112_v26`。TAR `7e3ca714f591c11e8f4aab2f6c253f335e73c1007c501586c84bae53d395f81d`，manifest `b584b204f428e1b5bb2a3a7eb91036fe691627c10cbbc8b3bb7beeeda1aadbbe`。

## 当前执行与剩余门槛

本版写作封存快照为同信息研究35完整角色/17固定0/0对/6unknown。后续执行增量2已独立核验43Full配对和44Full，最新37完整角色/18固定0/0对/6unknown，44Full尚未配对，原100对仍未完整。新唯一两role outer已06:03准入44Pooled及条件45Pooled，原bt40 10:44:53UTC截止不变。不得把排队、加载或当前少量状态当成最终SR。最新20项执行映射读取`responses/20261112_v26_execution_addendum2`，本版92项写作封存包及PDF保持。

P1–P5整段有利定性论证与37空槽保留，均待各自直接证据审核。后果预测仍须同支持候选误差，整体修复须完整分母，自然missing-skill须独立absence标签、自然门与配对行为，不能由本次显式撤回替代。作者独立真机采集尚未收到新trial。ACP仍待现有重新登录请求，不重复创建未知或活动任务。后续接续以`current_iteration.json`、实际退出及实验台账为准。
