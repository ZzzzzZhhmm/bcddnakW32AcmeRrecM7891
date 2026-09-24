# 服务器实查后的论文回填与最后24小时实验安排

核验窗口：2026-09-24 15:45 UTC起（北京时间9月24日23:45起，跨9月25日）。对应作者提供的28页 `WARM_ICLR2027_new (1).pdf` 及活动LaTeX源包。资源按作者最新确认的 **4×H100 80GB、RMBench暂不能渲染**；本轮不修renderer、不启动GPU训练或评测、不移动/覆盖原产物。

## 1. 先给结论

1. 已找到可追溯的历史LIBERO全套结果，但为 **1936/2000=96.8%**，不是稿中98.7%。不能把旧模型结果悄悄换成当前Full WARM结果。
2. 在已检查路径中，尚未定位到支持 **RMBench 83.9%、六行组件消融、五轮干扰** 的完整原始运行链。这不是宣称它们绝对不存在，而是目前不能验收。
3. 真机目录确实有数据、memory和训练配置，但找到的 `outcome.json: success` 属于**遥操作示范**，不是WARM、Fast-WAM或pi0.5的自主trial。不能据此填图3的12个成功计数。
4. 最新真机 `pilot_20hz_joint` 有 **25 episodes、20 train/5 dev、12,333帧、1,356个train-only事件、K32**。可回填数据与预处理说明。
5. 真机动作/状态不能写错：**动作是6维绝对关节目标+夹爪宽度，proprio仍为TCP位置+rotvec+夹爪宽度**。两者都是7维，但语义不同。`delta_action_mask`在此版本仅处理padding，不把绝对关节目标变成相对TCP动作。
6. R2不是已验收的15k模型：最后日志step3680，最近验过权重字节的发布点3000；16:00 UTC再次读到的bundle仍是旧`running`记录，没有完成收据。ACP平台终态仍需控制台确认。

## 2. 找到了哪些目录

服务器基准路径 `B=/mnt/afs/task3_2/L202500276_lwz`。作者所说的 `B/dataset` 不存在，实际检索了 `B/datasets`。

### 2.1 历史LIBERO：已有真实结果，但不是98.7%的证据

汇总根：

```text
B/projects/WARM_evaluations/batches/step_019100/
  spatial-formal-s3407-v2/summary.json
  object-formal-s3407-v2/summary.json
  goal-formal-s3407-v2/summary.json
  libero10-formal-s3407-v2/summary.json
```

逐任务运行根：`B/projects/WARM_evaluations/results/step_019100/`。
原训练根：`B/projects/WARM/runs/libero_warm_2cam224_1e-4/warm-full-4xh100-zero1-numerics-20260718-110409/`。

| Suite | 成功/总数 | 成功率 | 与当前稿的关系 |
|---|---:|---:|---|
| Spatial | 488/500 | 97.6% | 稿中写98.4% |
| Object | 496/500 | 99.2% | 稿中写100.0% |
| Goal | 483/500 | 96.6% | 稿中写98.6% |
| Long / LIBERO-10 | 469/500 | 93.8% | 稿中写97.8% |
| 合计 | 1936/2000 | 96.8% | 稿中98.7%对应1974/2000 |

这批结果的root seed为3407，每任务50次；另外的seed17 pilot不能并入或替换它。现有验证和原始归档详见[实验记录](EXPERIMENT_RECORD_ZH.md)第1–3节。只有核实该模型与当前稿的方法身份后，才决定是否用于主表；目前可作为单独标明的历史结果。

### 2.2 真机：最新可核实的数据版本

```text
B/projects/WARM/real/piper/processed/pilot_20hz_joint/
  prepared/catalog.json
  prepared/audit.json
  prepared/config.json
  prepared/data_config.yaml
  prepared/dataset_stats.json
  prepared/source_manifest.json
  prepared/dataset/meta/conversion.json
  features/COMPLETE.json
  features/contracts/{camera,encoder,normalizer}.json
  memory/event_bank/{manifest.json,events.npz}
  warm_from_stage_a/run_20260924_100346/
  warm_from_stage_a/run_20260924_122118/
  warm_from_stage_a/run_20260924_150003/
```

原始25条示范在 `B/projects/WARM/tmp/WARM_real/pilot_20hz/`，`real/piper/raw/pilot_20hz` 是指向它的symlink，不应重复计数。更老的pilot_v1/v2/v2_16p67、不同表示的pilot_20hz，以及临时pilot_v3不应全部相加当独立样本。

| 任务 | train | dev | 总示范 |
|---|---:|---:|---:|
| Banana-to-Box | 4 | 1 | 5 |
| Duck-to-Box | 4 | 1 | 5 |
| Return-to-Plate | 8 | 2 | 10 |
| Same-Order Retrieval | 4 | 1 | 5 |
| 合计 | 20 | 5 | 25 |

- 处理后帧数：train 9,976、dev 2,357，共12,333。它们是command-bearing rows，不是自主试验次数。
- 本轮实际读取bank的dataset/episode索引数组，确认1,356条事件来自20个不同episode，全部split=train；没有把dev episode加入bank。未重算全部大数组payload哈希。
- bank manifest文件SHA-256：`1b4fc6087b98863056a055711139c0a2ee1a123921c9468cb40a4a93f6e4a4e0`；manifest记录的bank content hash：`a5cf005dd30627efdb2edd7bdc3110708944e4f48e64a1e688e721e8bb3c4de3`，两者不是同一种哈希。
- 配置：external/wrist，输入shape声明640×480，每视角224×224，水平拼为224×448；nominal20Hz，H32，K32，context width=768+4。
- 动作转换来源：服务器 `scripts/real/build_piper_joint_action_cache.py` 及 `prepared/dataset/meta/conversion.json`。该转换保留TCP proprio，重新发布关节目标动作并重算训练归一化和事件bank；不是切换到7维关节proprio。
- `piper_table_20260919`只是calibration identifier，不能代替实际标定矩阵、误差报告或精确硬件型号。
- `_holdout`是episode拆分后的命名。旧侧录notes明确包含把最后episode标成dev并另命名session的过程；不能据名称写成独立采集session。没有找到足以验证未见布局/姿态范围的记录。

本轮真机训练配置核对：BF16，per-device batch1，accumulation1，LR1e-4，cosine，weight decay0.01，clip1.0，**training seed42**；corruption seed3407是另一字段。配置8 epochs不是8 epochs已完成。归档时run100346记录到step13340/已保存state13200；run122118记录到step6260/state6000；run150003只有配置未归档到metrics。它们是不同目录，不是可相加的完整训练预算；最终部署checkpoint未确认。`eval_num_samples=0`及`eval_num_inference_steps=10`不能证明做过实机eval或部署用了NFE10。

## 3. 可以直接粘贴的LaTeX，以及准确放置位置

作者源包不在Git，本轮未直接改写它。以下四个片段在 `docs/nonreal72h/latex/`，均不改主表数字。`\res`宏已由源包定义。四个片段已在最小LaTeX夹具中以`pdflatex -draftmode`通过编译检查；表格行按下述方式直接粘贴到table body，不单独在tabularx里嵌套`\input`。这只是语法检查，完整论文编译仍应在合入后检查交叉引用、篇幅和图表。

### A. 真机数据、表示与memory：可立即新增

位置：`sec/appendices/c_additional_results.tex`，C.4.1 `Platform, Data, and Shared Interface`，label `app:realworld_hardware`，放在表11之前。

粘贴[real_data_verified_20260925.tex](latex/real_data_verified_20260925.tex)。其中包含25/20/5、四任务划分、12,333帧、动作与TCP proprio语义、1,356事件和K32。

注意：该段只报告当前数据版本。原段落“所有方法均使用相同训练split/控制接口并已评测”仍须trial证据支持，不能由这个新段落自动变真。若保留尚未执行的协议，使用planned/prospective语态，并删除结果性推断。

### B. 表11：只替换已知字段，保留真正未知项

位置：同文件 `tab:realworld_hardware`。

- 用[real_settings_rows_20260925.tex](latex/real_settings_rows_20260925.tex)替换六行body。
- 第二列表头改为 `Recorded setting / remaining verification`。
- caption改为：

```latex
\caption{\textbf{Recorded real-robot data and adaptation configuration.}
The verified data fields are distinguished from physical deployment
details that remain to be linked to autonomous trial records.}
```

这是一份诚实的**工作稿部分回填**，不是可以带红色待确认直接投稿的最终表。尚缺camera型号/测得同步精度/实机command rate/prefix/部署decoder/最终checkpoint/NFE/GPU/真实trial规则。将来有trial后应按实际运行替换这些剩余项，而不是提前填默认值。

### C. 已测运行成本：可立即新增，但必须保留范围说明

位置：`sec/appendices/c_additional_results.tex`，C.2.2 `app:gate_probes` 中现有forced-null检查之后、C.3 `app:reuse_controls`之前。

粘贴[online_cost_verified_20260925.tex](latex/online_cost_verified_20260925.tex)。数值：200 replans +20 warm-up，median500.0ms、p95553.8ms、peak allocated23.62GiB、bank89,680events。

不可删掉该片段中的smoke300、natural gate全部0、记录回放且动作不执行、排除模拟器/加载等限制。它不是成熟模型完整评测FPS、不是baseline加速比，也不是SR。现有50/50 forced-null段落已经正确填过，不再重复运行。

### D. R2优化动态：可选附录，不填到主表配置里

可在C.2另起一个`Additional shared-model optimization diagnostic`段落，粘贴[r2_training_verified_20260925.tex](latex/r2_training_verified_20260925.tex)。该片段同时报告action/effect下降和gate loss/Brier上升，防止选择性只报有利趋势。

**不要**用这些值替换表4的候选端点MSE、自然gate TA/FA、或表8尚未定位的主结果训练参数。R2的4卡/b8a4/GBS128/seed3407仅属于这个额外shared run。

### E. 历史LIBERO一行：仅当明确展示历史模型时使用

```latex
Historical WARM checkpoint (step 19,100)
& 97.6 & 99.2 & 96.6 & 93.8 & 96.8 \\
```

该行顺序是Spatial/Object/Goal/Long/Average。不能直接把它贴到当前Full WARM行而不先确认模型和协议；主表应决定报告哪个真实模型，并同步摘要、引言、图1及结论。

## 4. 目前不能回填的单元格/结论

| 位置 | 缺少的证据 | 现在怎么处理 |
|---|---|---|
| 表1及摘要98.7% | 对应1974/2000的真实episode和模型身份 | 不能标记核验完成；历史96.8%单独处理 |
| 表2及摘要83.9% | 对应755/900的九任务逐trial及checkpoint | 不能从训练loss或旧失败eval补成此值 |
| 表3六行消融 | 每行T/I身份、模型、预算及episode结果 | 保留内部待核验，不当已完成独立训练 |
| 图4/表10五轮干扰 | 每run/condition outcomes和donor/replacement/eligible日志 | 不能从均值SD倒推原始run |
| 表4十个红色数值 | H32真实候选分支端点、独立适用标签 | 缺新实验，不可用train effect loss代填 |
| 图3十二个红色k/20 | 三方法×四任务自主trial记录 | 25条遥操作success不能填；未找到不是0%SR |
| 表8四组红色配置 | 与主表/消融/干扰各自绑定的实际运行配置 | 不能使用R2或real的配置统一填满 |
| C.2自然gate TA/FA | 独立正负适用标签及natural decisions | gate梯度/训练alpha/forced-null均不能替代 |
| C.3三个因果/匹配对照 | 已声明的训练或干预与实际测量 | 一天内不能默认补齐，必要时删/改为未完成协议 |

图3左侧仍缺真实retrieved-event对应图像；不能拿任意遥操作图片当模型在线检索实例。真机“held-out layouts”“跨任务复用”“任务关键历史线索已移出近期窗口”等叙述，也需实际reset、event和timing证据。

## 5. 还需要运行哪些实验：按4卡、无RMBench渲染重新排序

### 5.1 首要决策：不要把24小时全押在新训练上

若主表原始记录仍找不到，需要的是**协议匹配的成熟checkpoint的完整suite评测**，而不只是再把shared模型多训若干步。若没有对应模型或合格renderer，这一主证据缺口不能靠更多离线diagnostic解决。

按已记录吞吐，从step3000续至15000约13.4小时纯4卡训练，尚未含恢复、保存、排队。按已有0.5秒/replan作粗略容量示例，九任务各100次全部达到稿中上限的策略推理约36.6 GPU小时，理想4卡约9.2小时，尚未含模拟器；实际成功提前终止可能更短，模型/机器差异可能更慢。两项串行已约22.6小时，不能同时承诺“完整训练+全部主评测+机制实验+4小时论文收尾”。

### 5.2 必需队列与条件

| 优先级 | 实验/工作 | 资源与前置 | 对应论文 | 本日决策 |
|---|---|---|---|---|
| P0 | 原记录恢复、逐episode重算、模型映射 | CPU；先复用现存结果 | 表1/2/3、图4/表10 | 先完成，不能用新cohort冒充旧run |
| P0 | 若缺原记录，固定合格模型完成全部目标suite | 可将4卡按task/reset分片；相同模型、协议、预声明seeds | 主表 | 当前RMBench被renderer阻塞；不可只挑好任务 |
| P0 | 实机自主trial与失败/接管记账 | 机器人组；先有已验证安全部署 | 图3、C.4 | 现存示范不够；20次/格是计划，不是事实 |
| P1 | R2确认中断后，从完整3000state续至15000 | 原4卡runtime，全state恢复，新输出/plan；约13.4h纯训练 | 新研究模型，不自动属于主表 | 仅在没有合适当前模型且明确选择训练路线时执行；不能重复旧R2命令 |
| P1 | 新固定checkpoint的DEV50 natural gate/source诊断 | 单卡，无renderer；现有bundle工具可复用，但须绑定新checkpoint | C.2/C.3的限定离线证据 | 只在模型验收后做一次；自然g全零就报告零覆盖，不强行开门 |
| P1 | H32候选后果与适用性标签，共用gate统计 | 合格renderer +真实六类state restore +独立标签；当前后端未验收 | 表4、C.2 | 不是仅“能显示画面”即可运行；在渲染最后解决的安排下暂列blocked |
| P2 | 专用history-read critical/noncritical干预 | 需新接口和语义标注，当前未验收 | C.3.3 | 不挤占主评测；本日不默认承诺 |
| 延后 | 新retrieval-residual、Gaussian/scale-only匹配训练；重做7000次干扰 | 多模型训练/大量rollouts | C.3、表10 | 除非已有模型/记录，否则删除必达承诺 |

**不需要再跑：**smoke300 forced-null、同一smoke300零激活source对照、已完成的200次成本、已有完整历史LIBERO那一批。只有更换研究问题/模型并明确单独身份才有理由新增，不为填表重复工程smoke。

### 5.3 具体时间门槛

- **T0–2h：**结束主结果路径恢复；确认ACP控制台终态；固定可用checkpoint及真实主张范围。实机组交出实际trial ledger（如果仍未执行，明确尚无k/n）。
- **T+2h：**在“使用已有成熟模型评测”和“4卡13.4h续训”之间作明确资源选择。不能把同4卡重复安排给两条线。新诊断可不依赖renderer，但不冒充主评测。
- **T+4h：**若原证据/renderer/合格branch backend仍缺，正式下调稿件claim与表格，不能一直假定稍后全部补回。
- **T+16h：**不再开长训练、匹配对照或新的大规模干扰协议；只收尾已能完成的运行。
- **T+20–24h：**保留4小时归档、回填、同一数字全稿同步、编译与逐页核对。

按作者安排，本轮把renderer问题留到后面，没有尝试修复。但必须直说：**把renderer留到最后，也就把RMBench闭环主结果和表4采集的开始时间一起推迟；离线CUDA任务不能解除这条依赖。** 若最终没有合格渲染机/足够时间，应诚实缩减主张，不能以训练loss或占位数代替。

## 6. 证据保存、搜索边界与复核

定向目录清单包含3,361个匹配的小型元数据/日志文件、9个symlink，扫描错误0。覆盖WARM、WARM_artifacts、WARM_evaluations、WARM_training、5090同步/备份根及RMBench数据目录；读取源/配置时只处理与本任务相关内容。跳过大权重、媒体、依赖缓存和重复不可变code快照；记录了跳过目录，未声称遍历所有压缩包内文件。

扩展内容搜索于2026-09-24 16:09:29 UTC完成：遍历projects/datasets下76,118个目录，读取112,926个文本文件，2,409个超过2MB的文件跳过，访问错误0，记录168个归档路径。最大目录深度15；排除了依赖、媒体、大权重和部分重复源码目录。2,230份关键词命中中包含大量其他项目的配置/计划/日志，不能计作WARM实验。WARM相关命中主要是实验计划、协议、source manifests与配置，没有建立98.7/83.9、六行消融、五run干扰的原始结果链。完整带路径和行号的检索结果保存在仓库外`claim_search.json`。

原始小文件已单独归档，**没有下载大权重**：

- 服务器：`B/projects/WARM_evaluations/evidence_archive_20260924/paper_fill_20260924T1545Z.tar.gz`。
- 本地：`F:/WARM/result/paper_fill_20260924/paper_fill_20260924T1545Z.tar.gz`，解包同名目录。
- 622个文件，2,468,470 bytes，SHA-256 `ddba6c7a277714635ebe9cf44539cad621afc27e5d96f763428f372c26229f59`。
- 双端archive SHA一致，本地逐文件字节/哈希核验622/622通过。`files.json`将每份小文件绑定到原服务器路径。
- 本地同级 `inventory.json`、`real_analysis.json`、`real_bank_check.json` 保存目录扫描、各版本统计、实际bank索引验证与最新R2共享盘状态；`supplement/`保留关节转换源码、conversion及stats原件。真机训练文件是时点快照，不作为训练终态收据。
- 无权重或optimizer大文件进入Git；提交的是文档与可粘贴LaTeX，不是原始数据。

对“没有找到”的表述限定于本轮已检索范围：其他机器、个人存储、未挂载路径及未解包归档可能另有记录。匹配到论文/手写说明里的98.7或83.9，也不算找到该组原始运行。
