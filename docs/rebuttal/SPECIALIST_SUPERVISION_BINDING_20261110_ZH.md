# 原稿 specialist 绑定与 W5 监督语义：新增核对

作者已确认四个目录属于原稿训练运行。2026-11-10 07:08 UTC 的原件保全包含四份配置、25份 checkpoint training attestation、其他小型元数据及清单，共34项。BT v1 的6份与 BRT v1 的7份记录均绑定 f77c633；PB v2 与 Cover v1 各6份均绑定47277e5。各run内部的 resolved config hash 一致。没有据此选择原稿DEV step，没有重新计算25个大权重的全文件hash。

四个末次可用权重的CPU FakeTensor/mmap读取分别为12000/12000/12000/14000，均为retrospection schema v3、timing_dim4。它们不能装入现用v8/timing_dim8，也不能自动称为原稿最终DEV选点。两份冻结原源码的retrospection_model.py字节完全相同，SHA `d89796cc5da67d431d911e5f9191c5d60a14599bd3763cc64198e1384dc0b67d`。其中1219–1230行的event-effect loss使用各检索事件自身的observed_effect；当前示范target_effect用于requirement、排序和gate等目标。这与v8的当前示范factual项加common-query candidate regularizer并不是同一训练语义。

这是针对已确认run的实现事实，不证明原稿所有表格或LIBERO模型都属于同一代际，也不推断它导致了v8诊断的负结果。原稿表12–18仍按作者确认保留为真实报告结果；现在需要逐表恢复模型代际与协议绑定，而不是删去不利结果、把旧权重迁入新结构或按新测试结果选点。

## 对回复与论文的直接影响

Hh4E W5必须正面解释“公共query目标能否学习候选差异”。公共目标本身不能识别各候选的反事实后果，这个数学判断不因版本改变。另一方面，也不能继续把v8的两项loss解释成这四个原稿specialist的实际训练方式。原始event-factual监督的目标来源、当前requirement监督和query下候选的独立执行验证应分别写清。

v7完整PDF、LaTeX和回复保持不可变。新增`responses/20261110_v7_engineering_addendum2`维护20项最新映射，单独准备W5替换候选和LaTeX澄清片段。它们是待合入的可审阅材料，不是已经完成修订或可以发布的稳定回复；W5实施描述和各机制cohort模型绑定仍是release gate。不要把“原稿有该公式”当成“已确认四run执行该loss”的证明。

## 接续顺序与验收

1. 原始四run/config/attestation身份已确认，不再重复索取归属。原DEV选点未知保持未知。下一次原run兼容资格从固定顺序BT、PB、Cover、BRT开始，不参考新测试表现改顺序。
2. 新补证若使用末次可用权重，须在任何推理结果前单独冻结为12000/12000/12000/14000的补充协议，明确“原run上的固定点”而非“恢复原稿选点”。实际准入前重算对应权重完整hash，并绑定原bank/normalizer/DINO/camera、训练合同和原冻结源码。
3. 先通过原代际factory/config/strict state loading及单次原生推理资格，再运行候选后果与同状态生成对照。操作层只允许用不可变snapshot身份代替Git工作树依赖；不改模型数学、不迁移结构、不让eval访问Git远端或因dirty失败。资格成功不计任务成功率。
4. 机制测试需要独立执行候选、Full与candidate-null，保存实际父状态/RNG/编码身份和完整轨迹。后续语义阶段须由真实任务状态验证；不能仅按query序号称为跨阶段。不可到达或不适用的固定状态保留为未知，不换成成功状态。
5. 监督描述与对应cohort绑定后，逐一更新W5、Bbqv W3/Q2、相关公式/附录，再完整编译revision并验收。既有v8全部六query/192候选/182完整/10未知、六组零局部标签和闭环负结果继续单独公开保留。

原件根目录：`WARM_evaluations/rebuttal_20261109_acp`。34项档案SHA `0d56b4020f1da242ca99e172f63c3a813e544c75b6e76ce85c8a0150a6698960`、manifest `fd06676aa86c1f88d95fc1e1f05413a192cb33db06924d50d979b2183d761ab5`；四权重元数据/源码语义报告`AUTHOR_CONFIRMED_SPECIALIST_ARCHITECTURE_AND_EFFECT_SOURCE_V2.json` SHA `0f4806d2d5d8a89cc168163145d48be3686940b1762a71fc2a0e23925b47d822`。C盘/AFS均保全，metadata审计不等于原准确率复现。


08:03 UTC接续：BT12000与PB12000已经分别用原f77/472 factory与原config完成strict loading，实际完整checkpoint SHA在加载前后重算并匹配原attestation，source前后不变。两份9项工程档案1f9fddd6…/90071fee…双端核验。这使“两份已核实可加载的原代际权重”从metadata候选变为实际加载事实；仍不确定原DEV点，仍无在线推理或任务收益。Cover原代际资格已独立有界准入，BRT按固定顺序随后处理。


10:06 UTC更新：四个原任务run的固定末次权重现在均已实际native strict load，完整checkpoint hash在加载前后匹配原attestation，不再仅是元数据候选。Cover/BRT新增9项档案分别5b1b8cf4…/7fbc3e8d…；原DEV点仍未知，不据新结果选点。BRT原完整在线RMBenchWarmPolicy亦实际加载成功，原runtime projection和DINO数值合同匹配；随后操作程序instruction导入错误退出，推理0，失败保全。原窗口v3修复仍须查实际终态。

监督澄清已合入v8完整主文与Appendix A.4，以及Hh4E W5；不再只是一份LaTeX proposal。37页/10主文完整工程及干净ZIP编译、视觉检查通过，原表数值与9图保持。新段指出event目标也存在复制历史输入的退化，不能因不同候选target不同就声称反事实已识别。全部原机制cohort的代际/DEV绑定及原模型独立执行仍是实证gate。v7/旧addendum2保持不可变，当前20项映射转v8。
