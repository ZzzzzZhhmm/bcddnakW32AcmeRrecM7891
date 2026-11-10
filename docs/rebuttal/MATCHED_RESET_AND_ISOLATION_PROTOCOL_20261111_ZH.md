# 匹配原生评测：固定 reset 名单与工作进程隔离

原始科学设计仍为 `3484ad14…`：RB/BT 各首批 50 个官方专家准入状态、种子从 340800000 顺序尝试、候选预算 2000、native 上限 700/1000。只用专家是否成功决定准入，不加载学习策略、不据 Full/Pooled 表现改样本。旧 RB10 必须是新 RB50 的精确前缀。

新增 `scripts/qualify_rebuttal_matched_resets.py`，保留旧十例采集入口。每个候选先落盘 started/expert_result；每个准入候选保存完整 expert_info，再做两次重新播种、全新 setup 的初始 reset，核对三相机 RGB、API qpos、任务逻辑/物体状态、实际关节状态以及官方 unseen 指令。两次 reset 不一致必须停止并保留该准入种子，不能跳到下个候选。保留所有拒绝、不稳定种子和运行错误；未知运行错误不能冒充专家失败。

正式 worker 与该采集器使用完全相同的 task_values 读取函数（AST 检查），相同的 setup、指令生成及采样顺序。两次 reset 原件为压缩 NPZ 和 JSON；不保存专家长视频或完整训练 tensor。最终 roster 只在 50 项全部通过且旧 RB10 前缀相同时生成。该文件不是策略成功率或机制优势证据。

每个固定分片使用独立 runtime 目录，拷贝 envs/task_config/description，保留原代码与配置 bytes；只读共享原资产。HOME、TMP/TEMP/TMPDIR、XDG、Torch extensions/kernel、CUDA、Triton、Numba、Matplotlib 和图形 shader 缓存位于各自 private 目录。不同 worker 的 runtime/private 不得相同、嵌套或通过链接指向同一可写目录。每个 private 目录带独立 fcntl lease；必须在 import Torch 前选择这些路径。旧共用 runtime 的全局 lease 仍保护旧原生实验，实际隔离资格前不可并发争用它。

已检查原生 robot.set_planner：双臂通过本进程创建的匿名 multiprocessing.Pipe 通信，不使用固定 TCP 端口。仍需实际同时启动多个独立 runtime，记录两次 reset、四条原生 hold 命令和真实物理轨迹相同，检查实际 planner 子进程/匿名描述符及时间重叠，再生成并发资格收据。静态代码/目录创建/55 项检查不等于该资格已通过。Python 写审计记录并限制 Python 层写入；它不覆盖 C++ 系统调用，不将其称作完整 syscall 审计。

每项资格使用事前声明的单一、至多两小时窗口，退出与全部原件持久保留。发生失败后不重置窗口，不自动重试未知进程。真实 eval 仍须两份 final3000 的完整 state/CAS、typed bundle、实际 DEV 共同输入/噪声资格及显式 scope 声明；此处不降低这些门槛。

2026-11-10 20:27 UTC：CCI 55 项相关检查通过（pytest 11.10 秒，完整过程 14.84 秒），原模型/训练/部署数学不改，0 新策略推理/0 正式 TEST。20:21 实际训练 Full2160、Pooled1462 连续有限，最终权重未完成；当前回复和 v12 PDF 内容不因新增工程入口而改变。
