# 匹配原生评测：固定 reset 名单与工作进程隔离

原始科学设计仍为 `3484ad14…`：RB/BT 各首批 50 个官方专家准入状态、种子从 340800000 顺序尝试、候选预算 2000、native 上限 700/1000。只用专家是否成功决定准入，不加载学习策略、不据 Full/Pooled 表现改样本。旧 RB10 必须是新 RB50 的精确前缀。

新增 `scripts/qualify_rebuttal_matched_resets.py`，保留旧十例采集入口。每个候选先落盘 started/expert_result；每个准入候选保存完整 expert_info，再做两次重新播种、全新 setup 的初始 reset，核对三相机 RGB、API qpos、任务逻辑/物体状态、实际关节状态以及官方 unseen 指令。两次 reset 不一致必须停止并保留该准入种子，不能跳到下个候选。保留所有拒绝、不稳定种子和运行错误；未知运行错误不能冒充专家失败。

正式 worker 与该采集器使用完全相同的 task_values 读取函数（AST 检查），相同的 setup、指令生成及采样顺序。两次 reset 原件为压缩 NPZ 和 JSON；不保存专家长视频或完整训练 tensor。最终 roster 只在 50 项全部通过且旧 RB10 前缀相同时生成。该文件不是策略成功率或机制优势证据。

每个固定分片使用独立 runtime 目录，拷贝 envs/task_config/description，保留原代码与配置 bytes；只读共享原资产。HOME、TMP/TEMP/TMPDIR、XDG、Torch extensions/kernel、CUDA、Triton、Numba、Matplotlib 和图形 shader 缓存位于各自 private 目录。不同 worker 的 runtime/private 不得相同、嵌套或通过链接指向同一可写目录。每个 private 目录带独立 fcntl lease；必须在 import Torch 前选择这些路径。旧共用 runtime 的全局 lease 仍保护旧原生实验，实际隔离资格前不可并发争用它。

已检查原生 robot.set_planner：双臂通过本进程创建的匿名 multiprocessing.Pipe 通信，不使用固定 TCP 端口。仍需实际同时启动多个独立 runtime，记录两次 reset、四条原生 hold 命令和真实物理轨迹相同，检查实际 planner 子进程/匿名描述符及时间重叠，再生成并发资格收据。静态代码/目录创建/55 项检查不等于该资格已通过。Python 写审计记录并限制 Python 层写入；它不覆盖 C++ 系统调用，不将其称作完整 syscall 审计。

每项资格使用事前声明的单一、至多两小时窗口，退出与全部原件持久保留。发生失败后不重置窗口，不自动重试未知进程。真实 eval 仍须两份 final3000 的完整 state/CAS、typed bundle、实际 DEV 共同输入/噪声资格及显式 scope 声明；此处不降低这些门槛。

2026-11-10 20:27 UTC：CCI 55 项相关检查通过（pytest 11.10 秒，完整过程 14.84 秒），原模型/训练/部署数学不改，0 新策略推理/0 正式 TEST。20:21 实际训练 Full2160、Pooled1462 连续有限，最终权重未完成；当前回复和 v12 PDF 内容不因新增工程入口而改变。

实际NVRTC检查补充：过长AFS TMPDIR在首次setup触发编译error6，0专家结果，全部原件559a6bc1…保留。缓存/临时目录因此改为每worker独立的短容器本地目录，可在新节点重建；runtime、输出和跨节点lease继续存AFS。目录归属/长度和persistent lease门槛经57项检查通过。原v1路径声明及失败不能覆盖，新路径必须另冻并实际资格；原两小时窗口保持。

官方完整观测包含额外front_camera。策略仍只消费原三相机，但reset审计按实际static_camera_name与双腕精确校验并保全所有RGB，不将额外相机丢弃。短缓存后首专家已accepted；旧三相机schema错误终态及原件e80a5e53…保留，59项反例/回归通过。两个任务的名单采集可作为并发隔离工程资格在不同私有runtime同时运行；这不是正式策略TEST，也不提前宣告并发资格通过。

官方专家seed过滤核对：eval_policy.py SHA edff5aa9…在254–262行捕获专家阶段异常后递增seed。首次BT32号的精确grasp_actor→Action构造target_pose=None属于该源确定的几何失败；旧unknown原件不改，以独立恢复证明追加窄分类，其它基础设施/未知错误仍停止。完整RB18/BT14两reset前缀与全部退出档案9200d955…双端核验；显式恢复只复用已哈希原件，接续未完成reset，不重采已完成样本、不重置22:31UTC截止。

实际并发资格程序按固定两个批次6+4运行十个分片runtime。每个probe两次初始化，在四条hold前加入ready barrier，验证所有scene在任一关闭前已同时存在，并核对不同planner endpoint inode。正式角色在首query前还必须比较roster的真实关节qpos/qvel；相同API drive target不足以替代该检查。77项工程检查通过，实际十worker资格仍须等完整50-rosters后执行，原22:31UTC期限不重置。

2026-11-10 22:23UTC补记：两个任务各首50专家准入初态及200次独立reset已完整；892原件deae79ec…/本机全部数组核验。十worker真实资格6+4两批、20 reset/80 hold/初始化场景重叠/不同planner endpoint全部通过，143原件3f99eb0d…/资格2ad5c09e…双端核验。既有22:31UTC窗口未重置，不需要重采名单或重跑并发。source861004fa…保持。后续正式100对仍等两final3000完整state/CAS/typed bundles，且须显式绑定原严格DEV失败及单独external-input因果反馈资格，不把此工程结果写成策略效能。
