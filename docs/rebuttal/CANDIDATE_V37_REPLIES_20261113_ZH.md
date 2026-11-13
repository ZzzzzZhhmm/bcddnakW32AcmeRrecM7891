# v37回复与v36论文：最终审阅指针

截至2026-11-13 12:54UTC（北京时间20:54），本次仅优化回复和可恢复元数据，不生成新PDF，不新增试验。

## 审阅入口

- 三位回复：`tmp/rebuttal/responses/20261113_v37_verified_leads/REVIEW_GUIDE_ZH.md`。20项原review字段/序号/hash、17评论块、实际strength感谢、we/our和完成修订总结保持。
- 配套论文仍为`tmp/rebuttal/revision/WARM_ICLR2027_revision_v36_namespace_withdrawal.pdf`及同名`_source.zip`：50页/主文10页。完整原工程及干净ZIP已编译验真，PDF SHA `96384b358046799ce5020766fecbba4f072b7d5140c18cc20269fb9b171fea4e`、源码ZIP SHA `fb78b7e802af1bc8bb936828cbce70f53f988e1d5197ae4d78553d89cc94b97a`，本次逐字节未改。
- 当前QA：`tmp/rebuttal/execution/20261113_iter51/V37_RESPONSE_QA_V1.json`和`V37_PAPER_BINDING_V1.json`；原50页编译/视觉QA保持在iter50/V36_QA_V3.json。

八处回复引导句原先用“we use/evaluate the test below”等现在时带出尚未执行的位移、扩库或自然缺技能协议，容易与其后的PENDING发生歧义。本版明确写成已经澄清的方法、已经观测的配对，或所规定的独立协议。所有数字、原有有效证据、感谢/结尾和37整段条件论证精确相同，不收缩有证据支持的核心结论。

DATA_DEPENDENCIES曾继承17对的历史截面；现绑定最新41对/13unknown、显式撤回完成而总体收益未建立的审计。回复—证据—论文位置映射同步，避免把过时控制文件当作当前状态。73成员写作包本机/AFS逐项验真：TAR SHA `242187cffd0ce694112d848f04969cb0f20cdf58d0f988ae386482e5337275b3`，manifest SHA `9a0b0350ba6aa66ea0138d0e8210777a3b21aa64c75f9a93eedc14086de040aa`；AFS路径`WARM_evaluations/rebuttal_20261109_acp/writing_candidates/20261113_v37_responses/verified`。旧v36的110项与PDF/ZIP继续封存。

## 原模型谱系追查的结果

只读扫描四个指定训练/历史评估根共1156文件，355个稳定小型元数据原件完成独立双端SHA/TAR验证；原始总字节33,174,949，压缩包3,349,825字节，SHA `491c9e86428c46f91570a403f0efc7cee6e421615e3f8bd710ab80b421353361`。本机位置iter51/raw_verified；AFS为`ITER51_MODEL_LINEAGE_V1`。

四个作者确认的specialist运行均恢复了末步训练attestation与配置；BT/PB/Cover为12000，BRT为14000，`source_policy=fixed_context_top1`，各配置`eval_every=0`。这是训练身份，不等于DEV选择记录；此次未重新加载或哈希巨型权重。指定范围未找到DEV选step的严格标记或specialist eval元数据。找到的BRT formal100-v3 stdout以缺少sapien在导入阶段失败，不能当作完成评估或投稿成绩来源。没有据此修改任何原稿成绩，也不据有限搜索声称其他位置不存在最终评估。

下一步继续找正式SR表—最终权重/配置/评估manifest的直接绑定；不要将版本审计或这次未找到误写为论文错误，亦不要盲目换checkpoint追求有利结果。

## 继续执行的约束

当前科学GPU程序0；两最后runner complete/0，旧0GPU审计仅两个q8报告。两个原窗口和独立余额阶段均关闭，143.8054115725/144预留及一次性credit保持，不重置时钟、不重试未知、不复活旧UID。本次仅CPU只读工作，无新模型/物理动作。12:53原生核查AFS可用4,769,820,672B，当前12000/3000、CAS、唯一轨迹和并发真机工作保持。

ACP登录请求仍为既有待办，未再次索取登录或标准GPU配额；未收到新的认证确认时不尝试创建。关键P1–P5和37整段有利结论仍开放，新增真机收到0次。23CST前以回复v37＋论文v36作为当前内部稳定写作候选继续逐条审核，不将“材料已封存”等同于“所有科学证据已闭合”。未经授权不发布、正式上传或向他人发消息。

## 13:57UTC晚间交叉验收

iter52/NIGHTLY_CONSISTENCY_V1核验20原review绑定、17评论块重构、所有数值范围与PDF/ZIP哈希。37回复槽与31论文宏（34处引用）全部对应，P1差值/CI及P2最终SR/差值CI为复合字段；四个回复专用支持字段为discordant、unknowns、coverage、gate distribution。没有遗漏或自动填值，全部保持整段PENDING。统一填数要求在docs/rebuttal/DATA_SLOT_FILL_CONTRACT_ZH.md，包含每个字段的单位、原始分母及哪些结果不能互换。

本轮无新试验或PDF。浏览器作者console读取在派发代码之前超时；原任务闲置过期后确认用户tab保留，清理finished/forcedCleanup=false/closed0。公共OpenReview页面要求浏览器验证、API工具不可读，因此最新讨论是否有变化仍未核实，不冒称没有新评论。原生已完成轨迹与封存数据不受此影响，既有登录请求不重复发送。
