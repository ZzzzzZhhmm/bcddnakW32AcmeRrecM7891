# Nov12：原PB剩余分支与匹配评测的独立资源阶段

本页是恢复入口，精确实时状态以 Git 外 `current_iteration.json` 及其所指 LIVE/退出/原件核验收据为准。旧窗口不重置，运行程序不访问 Git 远端。作者已明确 GPU 仅可使用闲时；纯 CPU 沿用已验证的标准 0GPU/4CPU/16GiB。

## 原PB阶段已结束，不能把云端“成功”当作整池完成

旧物理阶段 UID `50ea739accc14fc3a1030a64583d3eed` 在原 13:15UTC 截止后退出，监督器实际归档完成。q8、q48 各有四个完整 H32 干预，分别是进展 0–0 和父状态事先超额导致的 NA。q96 的两个原 Full 尝试各接受17条命令后遇原生 TOPP 拒绝，属于有完整收据的 invalid branch，不能称为完整 H32；所选 raw rank0 完整32条命令，但父计数5/1已经超过目标2/7，进展标签 NA。旧最终 Full 控制在前缀重建时中断，保持 unknown，禁止重跑或用新控制取代。

原40项 q96 档案 SHA `fc8e1d2924afe05da2e137aef6fabc6a5eebc3d1c973052c82f08391e6dc146c`。三query始终来自一个 factual episode。原 H32/768维 mean-DINO 指标、无效/删失处理、原 BT q96 不利结果及 q8 forced-null 局部稳定性结果均保留。

## CPU封存实际完成，剩余GPU任务已唯一提交

`warm-pb-remain-prep-20261112-p36-cpu`，UID `a2d7118029704b04b656a882ee7734fd`，标准0GPU/4CPU/16GiB，实际13:42:24–13:43:33UTC完成。三个CPU子进程并行验证原件并冻结剩余计划，0新策略推理、0新物理候选。旧stage17项及q8/q48/q96的51/51/40项原件均已本机/AFS独立核验。

新物理任务 `warm-pb-remaining-20261112-p36-idle`，UID `85e374b421a24794ae0e9f76d0257378`，cluster02，闲时1GPU/8CPU/128GiB，单副本/重启0，沿用已确认镜像和AFS。其程序位于 Git 外 `original_PB_remaining_resource_stage_20261112_v1/launch.sh`；源冻结为 `original_PB_remaining_controller_e23e634`。三个query各31个从未执行原候选和原pool-final保持原顺序。旧已知完整/invalid收据逐字节沿用，中断分支只记未知；新pre/post Full控制须精确复现原Full，且明确不替代旧AABA控制。

新阶段准入后最多8小时，最迟Nov13 10:00UTC结束；原13:15窗口字节保持。plan SHA `ad42486a56a1cfb813ed8f73dc2894744cee985ffdc3a64aedb19be5fff59939`。旧源码13个冻结入口与新控制器分开保存。

13:56UTC 新Worker仍等待，实际事件是 CPU/内存 `nodeSpec quota` 不足，尚无 AFS runner。全局空闲GPU数不能替代实际调度证据。已检查两页规格表，最小1GPU规格仍须8CPU/128GiB，无4CPU的GPU规格；未重复创建另一同stage任务。若之后另一集群出现可用卡，应先核实该UID仍未准入/无runner，再按已知终态、显式核销后迁移；禁止两个job同时执行同一个stage。

## CCI只推进原名单中的从未开始角色

已结束的BT47 Pooled单独资源段完整1000命令/250生成、success0，781原件已双端验真。与原bt40逐字段比较仅改变资源起止时间及登记引用，模型、roster、源、bank、推理和指标全部相同，可在保留allocation身份的前提下计入同一科学名单。

新 `matched_never_started_CCI_next_roles_stage_iter36_v1` 独立13:49:40–16:19:40UTC，仅原next_request的47Full及条件48Full。monitor2570134、初始launcher2570165已实际启动；须读取实际进程/心跳/轨迹和退出，不能以GPU低利用率认定完成。单角色最多3600秒，监督器最多两角色；未知退出不自动重试。原bt40的10:44:53UTC窗口和master18:00UTC上限不改。

原native plan SHA `44c23a4b88e6424b802a6fab25a8792181e63a08a4bdb18e2fe684caa10ab179`，新plan SHA `be7e3f93863ea0150ae0fc444e1592b9fc367824e3e477bc51993e55bb75477b`。保守资源预留上界105.619 GPUh低于原120GPUh，上界并发6不变；这是包含队列与间隔的预留上界，不冒称实测GPU消耗。CCI后端重启时刻未验证，任何中断原件必须保留。

## 证据与写作恢复

截至iter36独立核验，原allocation48完整角色/23对，加上述资源等价的BT47 Pooled为49角色/23对。23对均0–0；BT47/RB31/RB41 Pooled尚未配对。原7个cohort基础设施unknown和独立CCI加载unknown1保持。新原件在 `ITER36_NEW_TERMINAL_ARCHIVES_V1`；`NEW_TERMINAL_LOCAL_VERIFICATION_V2.json`、`PB_TERMINAL_AND_PREP_LOCAL_VERIFICATION_V1.json`、`RESOURCE_ONLY_EQUIVALENCE_LOCAL_VERIFICATION_V1.json`给出完整计数/链/源/哈希。后续完成角色须先独立核验再加计。

审阅面为封存v27及执行增量5；增量5合并增量3/4的必要事实，旧包均保留。20项/17回复块/37整段pending，最终匹配SR、预测优越性、总体修复与自然missing-skill可靠性、新真机trial尚未关闭。含第一人称的已核验协议澄清、完整待证段和LaTeX插入片段已更新；未修改封存PDF，未声称重新编译，未发布。
