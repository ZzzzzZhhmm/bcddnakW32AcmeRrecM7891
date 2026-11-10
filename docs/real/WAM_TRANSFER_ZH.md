# 从 WARM 迁移到新 WAM 项目的 Piper 真机训练交接

版本：2026-11-10。对象：使用 FastWAM codebase、已有 LIBERO 预训练 checkpoint、采用同一套 Piper 遥操作设备的新项目实验负责人。

**本次已确认：新项目不保留 WARM 的 retrospection / memory。** 推荐复用真机记录格式、数据转换、图像与动作处理、训练器和离线推理接口；模型结构与研究模块保留新项目自己的实现。本文给出一条不需要 DINO、event bank、recollection 或 candidate cache 的完整训练路线。它并不要求在机器上先跑一遍 WARM 训练。

## 1. 先读这一页

推荐顺序：确认命令语义与数据划分 → 转换新任务数据 → 用新任务 TRAIN 计算统计量 → 适配自己的 LIBERO checkpoint → 缓存新指令的 T5 embedding → 微调 → DEV 离线推理 → 交给现场团队做低速闭环。

| 项目 | 本次继承的接口 | 新项目要做的事 |
|---|---|---|
| 机械臂 | 单个活动 Piper，6 轴＋夹爪 | 复核现场 SDK、关节顺序、零位、工具与限位；双臂设备不表示模型同时输出双臂动作 |
| 采集 | 两路 RGB：`external`、`wrist`；对应同一活动臂 | 保持标定、安装、颜色、时钟和命令日志一致 |
| 原始状态 | TCP xyz 米、xyzw 四元数、夹爪宽度米；另存六关节反馈 rad | 不把六关节反馈直接塞进下面的 TCP proprio |
| **模型 proprio** | **`[tcp_x,tcp_y,tcp_z,rotvec_x,rotvec_y,rotvec_z,measured_gripper_width]`，7D** | 保持 TCP/控制坐标系一致；旋转向量不是 Euler 角 |
| **模型 action** | **`[joint_target_rad[0:6], commanded_gripper_width_m]`，7D 绝对目标** | 使用实际下发的关节目标；不是反馈差分、速度或 TCP 增量 |
| 时间与窗口 | 参考 profile 为 20 Hz，H=32，action/video ratio=4 | 从原始时间戳验证频率，不只相信 metadata；20 Hz 不是推理调用频率 |
| 图像 | 每路 resize 到 224×224，external 在左、wrist 在右，拼成 224×448 | 训练/推理使用相同变换；不能换相机顺序或只在推理时改变裁剪 |
| 归一化 | 新任务 TRAIN 上的 min/max，动作与状态分别计算；夹爪范围取二者 TRAIN 的并集 | 不使用 LIBERO 或旧任务的统计量 |
| 权重 | 新项目自己的 LIBERO 预训练权重 | 检查模型键名/形状；只把权重当初始化，重新建立真机优化器与数据配置 |
| 记忆 | 本次不启用 | 不运行 `features`、`memory` 或 `train_piper_warm.py` |

**交付边界：**代码包包含完整的参考 Python 包、配置、相关脚本、测试和本文；不包含模型权重、Wan/T5/VAE 资产、真实数据、SSH key 或现场控制器。参考源树中的 WARM 代码是历史实现及部分模块依赖，不要求把它们接入新项目。不要将整个 `src/fastwam` 覆盖到已修改的目标项目；先在独立环境跑通本参考流程，再按第 9 节接入目标模型。

## 2. 我们之前实际上做了什么

以代码、历史训练配置和已保存的数据审计为依据，旧流程分为三个阶段：

1. `pilot_v1` / `pilot_v2` 先使用 TCP 增量动作微调 FastWAM。LIBERO proprio 为 8D，Piper proprio 为 7D，使用专门的投影迁移脚本。随后才训练 WARM 记忆模块。
2. 实测采集曾混有约 16.67 Hz 和 20 Hz 数据，因此后来建立独立的 `pilot_20hz`。不能将两者仅改名后混为 20 Hz。
3. 后期 `pilot_20hz_joint` 将动作标签改为实际 `joint_target_rad + gripper_width_m`，保留 TCP proprio 与两相机图像。旧转换脚本复用已有图像特征、重算关节动作统计量，再重建 WARM memory。

本地保存的 2026-09-24 审计记录对应 25 条示范、20 TRAIN / 5 DEV，12,333 个 command-bearing rows，其中 TRAIN 9,976、DEV 2,357。这些是**旧任务的数据记录**，不是新项目要求的样本数、训练步数或自主任务成功率。旧 holdout 的命名也不证明与 TRAIN 是独立物理采集 session。

历史位置（仅用于团队找回资产，非本文命令默认值）：

```text
CCI 项目根：/mnt/afs/task3_2/L202500276_lwz/projects/WARM
原始数据：tmp/WARM_real/pilot_20hz
处理数据：real/piper/processed/pilot_20hz_joint
历史配置：configs/real/piper_warm_20hz_joint.local.yaml
历史 joint 初始化引用：pilot_20hz_joint/init/step_012400_mot.pt
本地审计：F:/WARM/result/paper_fill_20260924/
```

本次核对的是本地代码与归档配置，没有重新连接服务器验证这些大文件仍在或确认现场最终部署 checkpoint。新项目不使用这些历史路径作为自动回退。硬件型号后缀、固件、相机型号、当前部署频率和实际执行 prefix 仍需现场确认。

## 3. 代码地图：该看哪些文件

路径均以代码包中的 `code/` 为根。

| 路径 | 作用与是否复用 |
|---|---|
| `src/fastwam/real/episodes.py` | 原始 episode 的 writer、加载与结构/时序校验，复用 |
| `src/fastwam/real/preprocessing/piper.py` | 原始记录 → TCP state 和动作；本次支持直接读取绝对关节命令，复用 |
| `src/fastwam/real/preprocessing/contract.py` | TCP 增量 / 绝对关节两种动作语义的明确名称 |
| `src/fastwam/preprocessing/{config,prepare,lerobot_writer}.py` | 配置、LeRobot 风格发布、TRAIN-only 统计与审计；入口仅执行 prepare |
| `configs/real/transfer/preprocess_joint.template.json` | 新任务的关节目标预处理模板 |
| `scripts/real/{validate_episode,index_teleop_episodes,preprocess_teleop}.py` | 数据质检、固定 split 清单、转换入口 |
| `src/fastwam/real/libero_to_piper.py`、`scripts/real/migrate_libero_fastwam_to_piper.py` | 兼容原版 FastWAM payload 的 proprio 8→7 投影迁移 |
| `scripts/real/configure_transfer_training.py` | 从新任务 prepared 数据生成完整训练 YAML；无需 memory |
| `scripts/precompute_text_embeds.py` | 为新任务指令生成 T5 cache |
| `scripts/train.py`、`src/fastwam/{runtime,trainer,training_config}.py` | 原训练主入口、模型构建、优化、验证、checkpoint 保存和恢复 |
| `configs/model/fastwam.yaml` | 原 FastWAM 模型参考配置；新项目可替换为自己的 model factory/config |
| `src/fastwam/datasets/lerobot/robot_video_dataset.py` | 实际训练 batch 的视频、动作、proprio、文本与 padding |
| `src/fastwam/datasets/lerobot/processors/fastwam_processor.py` | 图像和状态动作处理；务必与部署一致 |
| `src/fastwam/datasets/lerobot/utils/normalizer.py` | min/max 映射、常量通道处理和反归一化 |
| `scripts/real/probe_transfer_model.py` | 严格加载权重，在录制 DEV 当前帧上输出动作；无电机执行 |
| `scripts/real/package_training_transfer.py` | 打包源码与逐文件 SHA256 清单；训练时不调用它 |

历史文件保留，但新项目不直接启动：

- `scripts/real/build_piper_joint_action_cache.py`：旧的缓存转换脚本，含 `pilot_20hz`、20 TRAIN 等历史假设。新路线直接从原始关节目标 prepare，不需要这一步。
- `scripts/real/pack_piper_20hz.py`：特定历史数据的筛选、复制、时间戳修复和 holdout 改名脚本；不是通用采集纠错器。不要给新数据自动补时间戳、掩盖重复帧或用重命名证明 session 独立。
- `scripts/real/acp_piper_*.sh`：含旧服务器路径、旧数据量、旧 checkpoint 和 WARM 两阶段启动逻辑。本交接使用显式新路径的命令。
- `scripts/real/smoke_train_piper.py`：虽名为 smoke，也曾承担完整 FastWAM 微调，但旧入口依赖 `memory/warm_data_config.yaml`。本交接用新配置生成器加原 `scripts/train.py`，消除此依赖。
- `scripts/real/train_piper_warm.py`、`eval_piper_warm.py`：仅供核对旧 WARM；新项目不运行。旧 `eval_piper_warm.py` 的 DEV loss 不是闭环成功率。

## 4. 原始数据交付格式

优先接收 WARM sidecar 格式，不要仅交视频。只有视频无法恢复已执行的专家控制目标。若现场仍同时输出 LeRobot 数据，`opst_cam/wrist_cam` 与其 7D state 不能靠名字或维度自动视为本格式；请核实该 state 是关节还是 TCP，再单独适配。

```text
real/piper/raw/new_tasks_v1/
  session_train_A/episode_001/
    episode.json
    observations.jsonl
    commands.jsonl
    outcome.json
    command_provenance.json       # 建议保留现场语义说明
    external/000000.jpg ...
    wrist/000000.jpg ...
  session_dev_B/episode_001/ ...
  session_test_C/episode_001/ ...
```

`episode.json` 的必要字段见 `docs/real/README.md`。核心为：

```json
{
  "schema": "warm.real.episode.v1",
  "episode_id": "session_train_A__episode_001",
  "session_id": "session_train_A",
  "task_id": "new_task_identifier",
  "instruction": "A precise instruction for the new task.",
  "split": "train",
  "synthetic": false,
  "camera_order": ["external", "wrist"],
  "action_semantics": "commanded_absolute_tcp_pose_and_gripper_width",
  "calibration_id": "VERIFIED_SITE_CALIBRATION_ID",
  "control_frame": "VERIFIED_CONTROL_FRAME",
  "tcp_frame": "VERIFIED_TCP_FRAME",
  "clock_id": "site_monotonic_clock",
  "clock_domain": "client_monotonic_ns",
  "nominal_action_hz": 20,
  "max_observation_gap_ns": 100000000,
  "max_sensor_skew_ns": 50000000
}
```

上面的时间阈值仅示例，必须依照采集接口核验。新关节 profile 仍读取原 sidecar schema：记录中保留 TCP 命令和状态，**额外必须有 `joint_target_rad`**。不通过改 `action_semantics` 字符串绕过原始格式验证；训练输出的关节语义写入 processed profile 与 conversion provenance。

观察 i 的结构：

```text
seq: i
timestamp_ns: 当前事实观察的统一单调时钟边界
cameras.external / cameras.wrist:
  path: 相对 episode 的 RGB 图片路径
  timestamp_ns, width, height, color_space: RGB
robot:
  timestamp_ns
  joint_position_rad: [6 个实测关节角]
  tcp_position_m: [3]
  tcp_quaternion_xyzw: [4，单位四元数]
  gripper_width_m: 实测夹爪总宽度，米
```

命令 i 的结构：

```text
seq: i
observation_seq: i
timestamp_ns: 实际下发时间，位于观察 i 与 i+1 之间
accepted: true
joint_target_rad: [6 个实际下发的关节目标，rad]
tcp_position_m: [3，对应已下发关节目标经已验证 FK/TCP 得到的目标]
tcp_quaternion_xyzw: [4，同上]
gripper_width_m: 实际下发的夹爪目标宽度，米
```

历史 `accepted` 表示 JointCtrl 调用返回，不代表 SDK 对 Cartesian 目标的确认，更不代表目标已经到达。请保留 `command_provenance.json` 中真实的 `accept_semantics`、FK 和控制模式来源。不要用下一帧反馈替代 `joint_target_rad`；即使所有字段都是 7D，物理含义也不同。

N 个 commands 对应 N+1 个 observations。记录末帧用于验证最后一条命令的后继事实。发布训练数据时保留 N 个 command-bearing rows，不伪造 terminal action。最低长度应超过 H=32；正式示范应覆盖完整任务，不拼接中断段。失败原始记录保留，索引器默认只选 outcome=success；如研究需要失败示范，显式修改选取策略，不静默删除失败段。

split 按事先定义的独立采集 session 划分；DEV 用于模型选择，TEST 不参与训练统计、超参数选择或反复调试。每个新任务应有足够多独立示范及 DEV 覆盖，数量由任务难度和数据多样性决定，不照抄旧项目每任务 4/1。

## 5. 转换后数据和模型张量

```text
real/piper/processed/new_tasks_joint_v1/prepared/
  dataset/
    data/chunk-000/episode_000000.parquet
    videos/chunk-000/observation.images.external/episode_000000.mp4
    videos/chunk-000/observation.images.wrist/episode_000000.mp4
    meta/{info.json,tasks.jsonl,episodes.jsonl,episodes_stats.jsonl,conversion.json,...}
  catalog.json                 # 固定 episode/task/split 与路径
  dataset_stats.json           # 只用 TRAIN 计算
  train_stats_manifest.json    # 统计来源
  data_config.yaml             # 可直接送入普通 FastWAM 训练器
  audit.json
  source_manifest.json
  config.json
  COMPLETE.json                # 阶段输出完整性；不是训练或真机成功标志
```

Parquet 中每行含 `action`、`observation.state`、`timestamp`、`frame_index`、`episode_index`、`task_index` 等。`timestamp=i/fps` 是通过真实时轴校验后发布的相对时间，原始时间戳保留在转换信息。prepare 会把 RGB 序列编码为 MP4，原始 JPEG/PNG 继续保留；它不是论文素材的无损排版流程。

| 层级 | 张量 / 字段 | 形状与含义 |
|---|---|---|
| 原始训练行 | `observation.state` | `[7]`，TCP xyz+principal rotvec+measured width |
| 原始训练行 | `action` | `[7]`，六绝对关节目标 rad+commanded width m |
| dataset sample | `video` | `[3,9,224,448]`，RGB，浮点 [-1,1] |
| dataset sample | `action` | `[32,7]`，经新 TRAIN 统计归一化 |
| dataset sample | `proprio` | `[32,7]`，与 action 对齐；模型当前条件取第 0 步 |
| dataset sample | `context` / `context_mask` | 原 T5 接口为 `[128,4096]` / `[128]` |
| batch | 上述张量 | 前面增加 batch 维 B |
| 推理输入 | `input_image`、`proprio` | 当前 `[3,224,448]` 和 `[7]`，或单 batch 形式；不输入未来帧 |
| 推理输出 | `infer_action(...)["action"]` | `[32,7]`，仍在模型归一化空间，必须反归一化 |

H32 表示 32 个动作步；数据加载先组织 33 个观察位置，再按 0,4,…,32 采 9 个视频帧。20 Hz 时预测跨度约 1.6 秒，视频采样相当于 5 Hz。采集频率、模型动作时间基准、实际下发频率、重规划频率和低层伺服频率需分别记录，不能混为一个“20 Hz”。

**归一化注意事项：**典型非恒定通道为 `y=2*(x-min)/(max-min)-1`；实际代码对近常量范围有特殊处理，forward 另 clamp 至 [-5,5]。反变换应调用同一 normalizer，不能手写简化公式用于正式推理。统计 min/max 不是硬件限位。`delta_action_dim_mask=[true×6,false]` 在这里保留历史 padding 行为；它不把关节目标变成 delta，也不允许减去 TCP proprio。

## 6. 环境与外部资产

建议复用已验证的 Linux GPU 环境。以下是当前仓库固定依赖的安装方式，不自动升级现场 SDK；GPU 型号/显存/driver 与目标模型的新模块仍须在训练机核验。

```bash
conda create -n wam_real_transfer python=3.10 -y
conda activate wam_real_transfer
cd /absolute/path/to/unpacked/code
python -m pip install -U pip
python -m pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
python -m pip install -e . --extra-index-url https://download.pytorch.org/whl/cu128
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"
python -m pip check
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
nvidia-smi
```

依赖以包内 `pyproject.toml` 为准。此前工程使用 BF16、混合 attention gradient checkpointing 和 ZeRO-1；新模型的显存需求需实测，不能由已有 7D 动作接口推导。训练资源由本方负责，现场 GPU 只需通过该模型实际推理的显存和延迟验收。没有在本次 Windows CPU 检查中运行 5B GPU 训练。

需单独提供的资产：

- 新项目的 LIBERO checkpoint，以及与它一致的模型源码/config。
- 原参考配置需要 `Wan-AI/Wan2.2-TI2V-5B` 的模型/VAE/text encoder 文件，以及 `Wan-AI/Wan2.1-T2V-1.3B` 的 tokenizer 资产；实际解析见 `models/wan22/helpers/loader.py`。不能只拷贝一个训练 `.pt` 就认为依赖齐全。
- 原参考配置的 `checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`；如目标架构采用不同 ActionDiT，以目标配置为准。
- 新任务原始 sidecar 数据；现场标定与 SDK 版本记录。

`scripts/preprocess_action_dit_backbone.py` 可按原 README 生成缺失的 ActionDiT 初始化资产。外部资产已完整缓存后可设置 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MODELSCOPE_OFFLINE=1`；缺文件时应显式补齐，不能用其他 checkpoint 自动回退。本次不需要 DINO 权重或 WARM bank。

## 7. 按顺序执行：新任务的普通 WAM / FastWAM 微调

以下都从解压包的 `code/` 根目录执行。用独立项目目录，输出版本不可复用。先按代码原型跑通，再换入新模型 factory。

### 7.1 校验并固定新数据

```bash
python scripts/real/validate_episode.py \
  real/piper/raw/new_tasks_v1/session_train_A/episode_001
python scripts/real/index_teleop_episodes.py \
  --root real/piper/raw/new_tasks_v1 \
  --output configs/real/transfer/episodes.local.json
cp configs/real/transfer/preprocess_joint.template.json \
   configs/real/transfer/preprocess_joint.local.json
```

编辑新 `.local.json`：填真实 `calibration_id/control_frame/tcp_frame`，确认 source.root、manifest、output、fps 和时间容差；保留绝对关节 `control_mode`。模板相对路径以**配置文件位置**解析，不是以 shell 工作目录解析。新项目普通 prepare 不使用 encoders 区域中的 DINO/VAE 配置，不需要为这些字段下载资产。

```bash
python scripts/real/preprocess_teleop.py \
  --config configs/real/transfer/preprocess_joint.local.json --stage plan
python scripts/real/preprocess_teleop.py \
  --config configs/real/transfer/preprocess_joint.local.json --stage prepare
```

只执行这两个 stage；不要用 `--stage all`。plan 仅展示配置，不检查真实文件。prepare 会解码图像、核对时钟/单位字段、验证 split、转换关节命令、计算 TRAIN-only 统计并发布不可覆盖的数据目录。失败时保留 staging/FAILED.json，修复根因并使用新版本输出，不能直接删除校验或增大容差掩盖缺帧。

检查：`dataset_stats.json` 的 TRAIN 数与预期一致，`audit.json` 没有跨 split 重复，随机检查训练/DEV 的首、中、末行 action 与 raw command 的 `joint_target_rad` 一致；state 与 raw TCP 对齐。转换后的任务文本必须是新任务，而不是旧 banana/duck 指令。

### 7.2 适配自己的 LIBERO checkpoint

对于与原版 FastWAM 相同的 payload（顶层 `mot` 和 `proprio_encoder`）：

```bash
python scripts/real/migrate_libero_fastwam_to_piper.py \
  --source /absolute/path/to/YOUR_LIBERO_FASTWAM.pt \
  --output real/piper/releases/new_tasks_v1/libero_init_piper7d.pt
```

该工具只做 `proprio_encoder.weight[:,0:6]` 复制，以及 LIBERO 最后两维 gripper 投影权重取均值成为单宽度通道；bias 复制。它不把 Piper state 填充到 8D，不迁移 WARM heads，并保存迁移 SHA256 报告。

**维度迁移不是动作语义迁移。**LIBERO 与 Piper action 都可能是 7D，但前者为 Cartesian delta/夹爪开关，本文为绝对关节目标/米制夹爪宽度。ActionDiT 权重只能作为预训练初始化，必须在新真实标签上学习；不可直接上机。若目标模型已是 7D proprio，不重复执行 8→7；若键名、payload 或模型结构不同，由目标项目实现等价迁移，并逐项说明复制/新建的参数。

### 7.3 生成可运行的训练 YAML

```bash
python scripts/real/configure_transfer_training.py \
  --processed-root real/piper/processed/new_tasks_joint_v1 \
  --text-cache real/piper/processed/new_tasks_joint_v1/text_embeds \
  --run-dir real/piper/runs/new_tasks_v1_run01 \
  --init-checkpoint real/piper/releases/new_tasks_v1/libero_init_piper7d.pt \
  --model-config configs/model/fastwam.yaml \
  --epochs 20 --batch-size 1 --accumulation 1 \
  --learning-rate 1e-4 --save-every 400 --eval-every 1000 \
  --output configs/real/transfer/train_new_tasks.local.yaml
```

20 epochs、1e-4 等是历史设置导出的**起始方案**，不是对新任务的最优承诺。根据 DEV 与数据量调整；保存和验证间隔单位是 optimizer step。有效 batch 约为 per-GPU batch × GPU 数 × gradient accumulation。不要拷贝旧脚本的 2465、9336、9976 作为新数据每 epoch 步数，以新 dataset/sampler 的实测长度和训练日志为准。

生成器只做配置和 prepared 完整性检查，不加载权重，因此成功不表示 checkpoint 已兼容。新模型使用独立、已完成继承合并的 `--model-config`，其中 `_target_` 指向目标项目 factory。所有 dataset/stat 路径会写入本机绝对路径；迁移到另一台机器时重跑配置生成，不手改 sealed prepared 文件后继续宣称校验有效。

### 7.4 生成新指令的文本缓存

```bash
python scripts/precompute_text_embeds.py \
  --config-path "$PWD/configs/real/transfer" \
  --config-name train_new_tasks.local overwrite=false
```

这一步只读新 dataset 的任务文本。prompt 模板为：
`A video recorded from a robot's point of view executing the following instruction: {task}`。
缓存文件名为 prompt 的 SHA256 加 `.t5_len128.wan22ti2v5b.pt`。更改指令或 T5/tokenizer/长度必须重新生成相应 cache，不能把旧任务 cache 改名冒用。部署时使用训练一致的模板和文本 encoder。

### 7.5 GPU 冒烟、正式训练和恢复

正式消耗 GPU 前先检查配置解析：

```bash
python scripts/train.py --config-path "$PWD/configs/real/transfer" \
  --config-name train_new_tasks.local --cfg job --resolve
```

先用独立目录验证一两个真实 batch 的 forward/backward、有限 loss、梯度更新、checkpoint 保存及严格重载；该结果只算工程验收。冒烟时设 `run_steps=2 save_every=1 eval_every=1 eval_num_samples=1`，保持总 num_epochs/schedule 配置，不用冒烟输出冒充完整训练。

正式启动参考：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes 4 --num_machines 1 --machine_rank 0 \
  --main_process_ip 127.0.0.1 --main_process_port 29531 \
  scripts/train.py --config-path "$PWD/configs/real/transfer" \
  --config-name train_new_tasks.local
```

GPU 数和端口按实际分配修改。脚本不拉取 Git、不探测 origin，也不因工作区未提交而拒绝训练；实验使用固定源码副本。训练期间记录配置、数据版本、checkpoint 初始化来源、Python/torch/CUDA/driver、吞吐、显存、loss 和 DEV 结果。

`--init-checkpoint` 在本训练器中对应 weights-only `resume=<文件>`：模型参数加载，优化器/步数重新开始。它不等于断点续训。中断后应使用 `checkpoints/state/step_XXXXXX/` 完整状态目录，保留 optimizer、scheduler、sampler、step，并保持同一数据、model 和总 schedule：

```bash
accelerate launch --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes 4 --main_process_port 29531 scripts/train.py \
  --config-path "$PWD/configs/real/transfer" --config-name train_new_tasks.local \
  resume=/absolute/path/to/new_tasks_v1_run01/checkpoints/state/step_XXXXXX
```

上面的 `step_XXXXXX` 必须替换为确实存在的本次 run 保存点；不自动寻找全项目“最新”权重。也可以通过配置生成器的 `--resume-state` 生成另一份配置，但必须显式保持原超参数和总 schedule。选择新任务权重时依据预定 DEV 指标和可重复 rollout，不默认最大 step 最好。

## 8. 离线推理与交给现场的内容

参考 FastWAM 接口可直接运行：

```bash
python scripts/real/probe_transfer_model.py \
  --config configs/real/transfer/train_new_tasks.local.yaml \
  --checkpoint /absolute/path/to/selected/checkpoints/weights/step_XXXXXX.pt \
  --samples 8 --nfe 10 \
  --output real/piper/releases/new_tasks_v1/dev_probe.json
```

这会严格核对 `mot`/proprio 加载，只向 `infer_action` 传当前帧、当前 proprio 与文本条件；保存 `[32,7]` 归一化动作、反归一化的关节目标/夹爪宽度和 open-loop MAE。录制后继画面不受预测动作影响，不能据此计算闭环成功率。第一条计时标记为冷启动，计时范围不含相机、磁盘解码、网络与机器人执行。

原 `FastWAM.load_checkpoint` 的普通默认路径允许部分非严格加载；新架构接入前必须审阅 missing/unexpected keys 和尺寸，并运行 probe 的 strict 路径或目标项目的等价严格检查。目标模型更换了 loader 或 `infer_action` 签名时，在该项目写显式适配器；不要捕获 TypeError 后静默改成宽松加载。

动作反归一化应使用本 processor 的 action normalizer：

```python
# pred: CPU float Tensor [32,7]，从模型返回。
physical = processor.normalizer.normalizers['action']['default'].backward(pred)
joint_targets_rad = physical[:, :6]
gripper_width_m = physical[:, 6]
```

不要直接套用旧 `processor.postprocess`：该通用历史方法还包含按 `num_obs_steps-1` 裁动作前缀的逻辑，对已输出的完整 32 步可能得到空数组。这里单 action/default、无额外 action_state_transforms，直接使用上述同一个 normalizer 的 backward。

现场部署包至少包含：选定权重、对应完整 model/training config、`dataset_stats.json`、相机与状态/动作契约、文本 cache/encoder 配置、固定代码与依赖版本、模型资产清单、离线 probe 结果，以及现场已验证的单位/限位/执行 prefix 配置。train/dev 原始数据可另交，不必发送到推理端。没有 memory bank 或 WARM source contract 的要求。

控制客户端收到动作后必须确认是绝对 joint target；不加当前关节角、不加 TCP、不做笛卡尔 IK 来解释这七维。SDK 整数单位转换由同一已验证现场桥负责。模型动作块通常只执行选定前缀后重新观察；H32 不表示必须开环执行完整 1.6 秒。prefix、插值、平滑、请求有效期和异步策略按真实端到端延迟设计并记录，不能从本次离线代码推断。

现场继续遵守：离线 → 无电机 dry-run → 单步 → 低速小工作区 → 闭环任务。检查关节/速度/加速度限制、全连杆碰撞、工作区、夹爪范围、异常动作拒绝、超时、watchdog 和独立急停。所有采集与真机推理在合作方现场完成。仓库中的 shadow harness 不包含已验收的 CAN/相机驱动或生产 inference server/client；请从原现场团队取得他们实际运行的桥接代码和版本，不能从“硬件相同”推断该软件已经随包交付。

## 9. 如何接入另一个 WAM 项目

推荐保留目标项目的模型与算法，移植以下边界：

1. **数据入口：**复用 raw schema、校验和 Piper adapter，输出统一 Parquet/video/meta 与 train stats。若目标项目另用 LeRobot 版本，写 reader 映射，核对 action/state 语义与时轴，不改标签来源。
2. **batch 入口：**按第 5 节对齐 video/action/proprio/context/mask。未来视频是训练监督，不是在线可读的当前输入。不要把 TEST 的未来、文件名答案或 outcome 塞进输入。
3. **模型入口：**保留目标模型 factory、训练 loss 和 checkpoint loader。若仍兼容 `run_training` / `training_loss` / `infer_action`，用独立 model YAML 替换 `_target_`。否则沿用目标训练器，只接本数据与 processor，并做一 batch 的等价检查。
4. **checkpoint：**先确认 payload、proprio dim、action dim、ActionDiT hidden dim、T5 dim、模型版本；8→7 脚本仅适用于本文明确的 FastWAM 格式。不同算法不要以 `strict=False` 忽略关键分支。
5. **部署输出：**提供固定 `[32,7]` 模型空间动作到关节目标的适配，反归一化和相机处理完全复用训练资产。

不移植到目标模型主路径：`models/warm/` 的 retrospection、recollection 更新、repertoire 构建、candidate cache、source-run contracts 或 WARM-specific loss。参考源码保留这些文件是为了完整可导入和追溯，不意味着新项目必须使用它们。旧 `context_dim=770/772` 是 WARM 检索上下文，**不是 T5 的 4096，也不是新模型应固定继承的维度**。

## 10. 常见失败与处置

| 现象 | 首先检查 |
|---|---|
| 找不到 `memory/warm_data_config.yaml` | 误用了旧 smoke 入口；本路线使用 prepared/data_config＋新配置生成器 |
| `joint_target_rad` 缺失/维度错误 | 采集未记录已下发目标，或给了另一种 LeRobot 格式；回到原始记录，不用反馈补造 |
| state 8D/7D 不匹配 | LIBERO proprio 投影未迁移，或模型仍创建 8D head |
| 动作形状正确但上机方向错误 | 关节顺序、绝对/增量、rad/SDK units、工具定义或 statistics 错误，立即退回离线 |
| `Nonuniform ... clock` | 实测频率、缺帧、暂停、命令与观察对齐；不能只修改 nominal fps |
| 同一个 session 跨 split | 采集划分不独立；重新定义实际划分并保留来源，不只改目录名 |
| `UNCONFIRMED_*` 报错 | 模板仍未填写现场事实；不替换成随意字符串通过检查 |
| 缺 T5 cache | 新指令未编码、模板/长度/目录不一致 |
| 缺 Wan/ActionDiT 文件 | 单个 `.pt` 不是完整运行资产；核对 model factory 的加载路径 |
| OOM | 先确认实际 GPU/残留进程；降低 per-GPU batch，保留有效 batch 需调整累积；按模型验证 checkpointing/分布式方案 |
| loss 下降但闭环差 | 检查 domain/action/state/latency 对齐及数据覆盖，不把 open-loop loss 当成功率 |
| 恢复后学习率/step 不对 | 将 weights-only 初始化误当 full-state resume，或换了总 schedule/data |

## 11. 验收与尚待确认

交接方已完成：源代码定位，后期 joint/TCP 语义核对；新任务 joint 直接 prepare；不依赖 memory 的训练配置生成；严格离线推理入口；源码 ZIP 的逐文件校验。CPU 测试使用合成小样本，不代表真机或新模型训练完成。

接收方验收顺序：

1. 解压后 `python verify_bundle.py` 通过；阅读 source commit 和依赖文件。
2. 核实自己的 model config、LIBERO checkpoint 键与形状，以及全部外部资产。
3. 提供新任务独立 session 的 TRAIN/DEV 样本；逐行核对七维语义、图像方向和实测时序。
4. prepare 通过；统计只来自 TRAIN；批量 shape、mask、任务文本正确。
5. GPU 实际 forward/backward/保存/严格重载通过；记录环境、有效 batch 与显存。
6. 正式训练后选定明确权重；DEV probe 给出有限 `[32,7]` 及合理物理量，无 silent partial load。
7. 现场依照上一节低速验收后才进行完整任务，保留每次失败和人工介入日志。

待接收方确认：新任务列表与示范量、是否改变新模型结构/loader、实际初始化权重、目标 GPU/显存、现场 SDK/固件/相机和标定版本、最终指令模板、真实控制桥、执行 prefix 与延迟预算。本交接不替他们虚构这些信息，也不启动任何 GPU 训练或机械臂动作。

## 12. 可复现检查

```bash
python -m pytest tests/test_real_training_transfer.py \
  tests/test_unified_preprocessing.py tests/test_preprocessing_torch.py \
  tests/test_libero_to_piper_migration.py tests/test_real_handoff.py -q
```

这些检查覆盖原始 TCP 路径未改变、直接关节命令转换、TCP proprio 保留、TRAIN-only 统计、可选旧 memory 接口兼容、异常目标拒绝、独立训练 YAML 和打包排除规则。验证环境与实际执行结果记录在本次交付包旁的验收文件中；Linux GPU 环境需接收方执行第 11 节验收。
