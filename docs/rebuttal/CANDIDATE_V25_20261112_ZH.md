# v25：作者交流与完整q96证据联动

当前内部审阅候选，未发布OpenReview或上传正式文稿；实证发布门槛尚未关闭。v24及其执行补充封存保留。稳定候选目标为北京时间2026年11月13日23:00。

## 作者视角与逐项回应

三位回复以we/our说明我们做了什么、证据回答了什么、修订落在何处。感谢开场经原review strength字段核验：Bbqv认可后果驱动选择与两类记忆的互补作用；rkWY认可动机及适配/调节复用的灵活性；Hh4E认可问题的重要性和综合pipeline。最后一位的认可不扩张为性能或机制有效性认可。

17评论块覆盖20项weakness/question，保持原review字段、序号与hash绑定。正文先给直接答案和最相关的已验证依据，再解释对照定义与范围；结尾总结真实完成的澄清和论文修改。没有显式第四方措辞、索分或夸大认可。原核心贡献、全部任务结果及较低分任务保留。

主文4.7以作者口吻区分提案适用性、同状态生成修复与完整任务成功，并将自然撤回连接到独立候选适用性及自然门值测量。弱实验用证据补强；监督定义与逻辑表述经源码核对后精准重写。没有因新诊断尚未提供有利结论而删除原方法或原实验。

## 新增证据与可审阅材料

原specialist BT case0/q96已完整32候选、0未知、四次精确Full。附录D.1表35给出原uniform-mean descriptor的共同支持centered MSE：predicted/history/constant为0.170002/0.164821/0.144898；absolute predicted/history为0.211722/0.205980。query requirement未捕获，constant absolute明确不可得。干预前已掉落、全部严格下一交互标签0、单query依赖性与无同父null范围保留。Bbqv W3/Q2、Hh4E W5与修订总结同步。其余主文、原九图、方法、公式及既有表格数据逐字节保持，仅替换过时interim附录段、增加一张完整原件支持的表。

- 回复：`tmp/rebuttal/responses/20261112_v25_completed_q96/{Bbqv,rkWY,Hh4E}.md`；同目录含17评论块、20项映射、溯源、自审与改动清单。
- 原完整LaTeX工程：`tmp/rebuttal/revision/WARM_ICLR2027_v25_completed_q96/`。
- PDF：`tmp/rebuttal/revision/WARM_ICLR2027_revision_v25_completed_q96.pdf`，SHA256 `8df62ab4811ad831f72fb28e9f4c7db66104f5c91b2531714c7d50d2132f69f3`。
- 同名`_source.zip`：SHA256 `2cab23bf6b023959dfe6e0ce37085a7b7985fe018c1dc7df8ccab9671e9980f4`。

完整工程及独立干净ZIP均含BibTeX四遍编译45页、主文至结论10页，45页文本一致。42–45页直接验看、其余41页像素一致。0undefined/overfull、45继承字体warning；匿名、内部待证横幅及第11页原AI use statement保持。

94成员123,962,579B写作包已在本机及04:00:12UTC AFS逐成员/解包核验，`writing_candidates/20261112_v25`。TAR SHA256 `3757c283c225fd09a6be8b0efea5f21dd714daa1d19bf07103e0ec521c0662e0`，manifest `809e03373c5f5c82cb8d3b8c625e791b94a131ef9981350c7784ba928d9d2578`。

## 待关闭的证据门槛

快照包括34完整角色/17固定对全0/0/6unknown，原100对未完成，不提供最终SR差值、CI或非劣效。P1–P5全部条件段及37槽保持未释放，有利定性论证和数值须一起对实际结果审核。

原q8事实context恢复已验真，属于工程资格；原checkpoint同context Full/显式Null/最终Full检查另行运行，先精确复现Full再允许Null封存。策略加载不等于推理完成或修复证据。作者独立真机采集尚未收到新trial。晚到证据以实验台账、真实退出收据及`current_iteration.json`为准，不覆盖本版封存内容。
