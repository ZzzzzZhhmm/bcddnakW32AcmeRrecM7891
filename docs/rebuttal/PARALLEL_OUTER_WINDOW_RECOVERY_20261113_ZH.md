# 两路实际ACP恢复与76角色证据

本轮恢复入口为`tmp/rebuttal/planning/current_iteration.json`的iter42及台账23.283–287。保留封存v30/47页主文10页和所有旧候选；新证据仍为补充片段，37个有利结果槽未放行。

两路闲时ACP已经实际获1GPU/8CPU/128GiB、单副本/重启0并启动真实runner，均在cluster02：

| 任务 | UID | 固定角色次序 | UTC截止 |
|---|---|---|---|
| warm-mf3k-bt30-tail-20261113-p42-idle | 38a8fb6c994d42a483f92f027e417d78 | BT32 Full显式第二次→32 Pooled | 21:39:26.094270 |
| warm-mf3k-rb30-tail-20261113-p42-idle | 613e7ebe9c1b40669c6c3c4c406c8da0 | RB35 Pooled显式第二次→35 Full | 21:42:51.386072 |

原signal2外层中断、unknown和全部原件已核销封存。此处只恢复同一固定case/role的第二次，不能自动第三次、替换已知失败、跳样本或再建条件后继。各monitor持有自己的两角色。原18UTC master及所有旧窗口保留；新BT窗口也沿用失败CPU预检之前的登记。两路新预留各1.8GPUh，保守累计119.8928711239/120、最多6GPU槽；无新额度时不能继续无界申请，先核对真实剩余和已用资源。

新控制器`acca89ba…`只含资源准入与同绝对时钟guard变化，1106源码成员、551其余文件及科学AST独立验真；61本地/91原生CPU检查通过。两次CPU准备错误已保留，最终准备禁用bytecode写入，不改活动快照。32原冻结入口保持。原件、checkpoint、CAS、Git外结果未删除。

BT3 Full于19:37UTC完整1000命令/250生成、success0，与旧Pooled3构成第38个固定0–0对。当前累计76完整角色（66原allocation+10资源等价），12历史unknown尝试保留。1554双方原件及输入/噪声/候选匹配已独立核验。这不是100对完成或优势/非劣结论，不能填正向pending。

CCI同monitor继续BT4 Full→条件4 Pooled至21:30UTC。原PB ACP85e仍q96整池，23:15:38UTC截止；原0GPU审计1ae继续至23:25:38UTC。不要重建它们，不能用旧result loading头代替实际progress，也不能把ACP容器PID当CCI PID。

Bbqv/Hh4E已核验补充位于`tmp/rebuttal/responses/supplements/20261113_iter42_verified_followups`；rkWY v30保持。C.6计时是冻结75回合/15450replans，后来的76角色分母不可混写。q48仍为独立已验真Markdown/LaTeX片段，尚未合入封存PDF；q96终态到手后统一形成下一完整候选并重编译。作者新真机材料仍未收到，不写已完成；未发布OpenReview/上传正式稿或联系他人。
