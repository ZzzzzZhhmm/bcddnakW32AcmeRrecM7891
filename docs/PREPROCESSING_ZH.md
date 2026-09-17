# WARM 统一数据预处理

本模块把不同数据源接到仓库现有的 LeRobot、事实特征、事件库和候选检索缓存接口。
它处理离线数据，不连接 CAN、不使能电机、不执行 policy。旧 LIBERO/RMBench 脚本仍保留，
正式仿真实验无需迁移；新数据版本使用新入口，不能与旧版特征、统计量、bank 混搭。

## 先区分需要预先准备的内容

| 产物 | 数据来源及用途 | 泄漏边界 |
| --- | --- | --- |
| LeRobot episode、catalog | 原始图片、状态、动作、指令、明确划分 | 以完整 episode 划分；真机以 session 隔离 |
| dataset_stats | 仅 train 的动作、状态 | dev/test 不参与统计 |
| 事实特征 | train/dev 的 DINO CLS、2×2 spatial tokens、可选逐帧 VAE | 所有特征来自实际图像；VAE 每次只编码单帧 |
| repertoire / event bank | 仅 train 的 context–action–consequence 事件 | 每个事件必须有真实后果；不补造末端帧 |
| candidate caches | train/dev 查询同一 train bank | 排除查询 episode 身份、原始内容和特征内容 |
| recollection 训练输入 | 同一 episode 已发生的事实前缀 | 由现有 RetrospectiveFeatureStore 按查询时刻组装；未来特征只能用作监督目标 |
| recollection 真机运行态 | 当前试验已收到的观测和已执行动作 | 每个 episode 从空历史开始；不加载 dev/test 完整轨迹作为初始记忆 |

LIBERO 预训练权重可以作为微调起点，但它的 Panda 状态、动作尺度和经验库不能直接成为
Piper 的模型接口或经验库。真机版本需要自己的 train stats、encoder/camera/action contract
及真实 train bank；更新数据划分、相机/标定、动作定义、encoder 或归一化后，重新生成完整版本。

## 数据源与明确支持范围

| adapter | 输入 | 输出接口 |
| --- | --- | --- |
| `libero_lerobot` | 仓库现有 LeRobot v2.1 roots + 已划分的 catalog | 7D delta EE action，8D Panda state，image/wrist_image；不重复转换视频 |
| `rmbench` | 官方 RMBench demo_clean 发布包 | 调用原 `RMBenchConversionConfig`/converter，保留 9 任务、50 episode、45/5 协议及检查 |
| `robotwin2` + `legacy` | `/joint_action/vector`、三个 `/observation/*/rgb`、seen 指令 JSON | 原始 N 个观测变 N−1 行：state[t]=qpos[t]，action[t]=qpos[t+1] |
| `robotwin2` + `xpolicylab_v1` | 原生 `data_format_version=v1.0`、state/action、vision、instructions、frequency | 读取已经错位的 state/action，绝不再次 shift |
| `piper_teleop` | `warm.real.episode.v1` 单活动臂、外部/腕部 RGB、已接受 TCP 目标命令 | 7D base-frame TCP delta + 绝对夹爪宽度；7D 真实状态 |

RoboTwin 当前实现限定双臂各 6 关节 + 1 夹爪的 14D qpos 数据；不把其他 embodiment、EE pose
或 16D/更高维向量裁剪成 14D。只接受明确选择的格式，不按维数自动猜语义。
RoboTwin 的 action 来自“下一实测 qpos 参考轨迹”，不是独立记录的控制命令。
其控制合同沿用 `robotwin_bimanual_qpos_plus_grippers`，原始标签来源另记在 conversion manifest。

新版 RoboTwin/XPolicyLab 使用 `vision/cam_head/colors`、`cam_left_wrist/colors`、
`cam_right_wrist/colors`，`instructions` 是根级 JSON 字符串，关节字段名为复数 `*_joint_states`。
这与在线 observation 的单数键不同。

JPEG 必须指定 `legacy_opencv_rgb_jpeg` 或 `xpolicylab_jpeg`；后者识别 `XPL-RGB1` COM 标记，
兼容上游历史无标记编码。已有 RGB uint8 HWC 数组可明确选择 `rgb_array`。不自动修正未知 BGR/RGB。
代码参考已核对的上游版本：

- [RoboTwin 原生导出](https://github.com/RoboTwin-Platform/RoboTwin/blob/6dde57155eafa3e4ebf6ad1f93a7cf7d5d41a755/envs/utils/pkl2hdf5.py)
- [XPolicyLab 状态与图片契约](https://github.com/XPolicyLab/XPolicyLab/blob/bb9a0b5f5136a74503b679af830bfd0a3a837d5c/utils/process_data.py)

上游可能继续改变，收到新数据先核对结构、帧率及版本。模板 task/目录只是填写示例，不声明当地已有这些数据。

## 安装与统一命令

CPU 转换使用独立 Python 3.10+ 环境，避免把依赖升级写进现有 GPU 训练环境：

```bash
python -m venv .venv-preprocess
# Linux: source .venv-preprocess/bin/activate
# Windows PowerShell: .venv-preprocess/Scripts/Activate.ps1
python -m pip install -r requirements-preprocessing.txt
```

在仓库根目录复制相应模板，例如 RoboTwin legacy：

```bash
cp configs/preprocessing/robotwin2_legacy.template.json configs/preprocessing/robotwin2.local.json
cp configs/preprocessing/robotwin2_episodes.template.json configs/preprocessing/robotwin2_episodes.local.json
# 编辑 local config 的 source、manifest、output、实际 fps、encoder 本地资产及 pinned revision。
python scripts/preprocess_warm_data.py --config configs/preprocessing/robotwin2.local.json --stage plan
python scripts/preprocess_warm_data.py --config configs/preprocessing/robotwin2.local.json --stage prepare
```

Windows 用 `Copy-Item` 替代 `cp` 也可。配置中的路径相对于**配置文件所在目录**，不是调用目录。
`plan` 只核对配置并展示计划，不证明原始文件、GPU 或 checkpoint 已可用。
填写所有 `UNCONFIRMED_*` 项；不把模板默认 15/20 Hz 当成实测频率。

在我们自己的现有 GPU 训练环境执行事实特征编码，继承仓库已经验证的 CUDA/PyTorch/torchvision、
Hydra、Transformers、Wan VAE 依赖；不要求第三方为离线训练预处理安装 CUDA：

```bash
python scripts/preprocess_warm_data.py --config configs/preprocessing/robotwin2.local.json --stage features
python scripts/preprocess_warm_data.py --config configs/preprocessing/robotwin2.local.json --stage memory
# 首次创建新版本也可以 --stage all；已完成的 stage 不会被自动复用或覆盖。
```

`features` 使用本地冻结 DINOv2-base（768 维）和配置指定的 VAE，不自动下载模型或降级成假特征。
`include_vae=false` 仅可生成 DINO-only 特征供检查；现有完整 recollection 必须有 VAE，
因此 `memory` 会拒绝这种不完整特征，不会发布一个训练时才报错的完整 WARM 配置。
这里只测过 CPU 转换与合成特征编排；真正的 DINO/VAE 权重编码需要训练机执行验收。

## 划分与原始数据清单

新 raw adapter 要求 `warm.source-episodes.v1` manifest，例如：

```json
{
  "schema": "warm.source-episodes.v1",
  "episodes": [
    {"id": "task_000", "task": "put_back", "split": "train", "path": "task/demo_clean/data/episode0.hdf5", "instructions": "task/demo_clean/instructions/episode0.json"},
    {"id": "task_001", "task": "put_back", "split": "train", "path": "task/demo_clean/data/episode1.hdf5", "instructions": "task/demo_clean/instructions/episode1.json"},
    {"id": "task_002", "task": "put_back", "split": "dev", "path": "task/demo_clean/data/episode2.hdf5", "instructions": "task/demo_clean/instructions/episode2.json"}
  ]
}
```

native XPolicyLab 不需要 `instructions` 文件；Piper 的 task/instruction 从原始 episode metadata 读取。
manifest 明确列出当前版本纳入的 episode，不扫描目录后悄悄随机分割。真实数据可用
`scripts/real/index_teleop_episodes.py` 从已有 metadata 生成清单，保留原始 split，并列出被 outcome 过滤的记录。
LIBERO 使用既有 `build_warm_episode_catalog.py` 产出的完整固定划分；不要为导入临时重抽 train/dev。
RMBench 沿用原 converter 的固定协议，模板 split_seed=3407；旧实验使用别的已固定 seed 时保持原值。

统计量对全部 train command-bearing rows 求 min/max 或总体 mean/std（ddof=0），输出 FastWAM
`action.default.global_*` / `state.default.global_*` 格式。特征阶段调用原 FastWAMProcessor 及
normalizer，包含原有范围退化处理与 [-5,5] clamp，不另写一套近似归一化。
Piper 的第 7 维 action/state 都是夹爪总宽，因此专门用 train 中指令宽度与实测宽度的并集范围
归一化这两个通道；避免控制跟踪误差造成不同尺度，使现有夹爪时序模块可直接比较起始状态与指令。
其余通道分别统计。该规则写入 train_stats_manifest，不使用网页标称范围代替训练统计。

## 时间轴：不要补造 terminal action

现有 WARM feature contract 规定：LeRobot 有 L 行，则缓存 L 个事实状态及 L−1 个动作。
本模块保持该约定，未修改训练主流程：

- RoboTwin legacy：原始 N 个观测 → L=N−1 个 command-bearing rows → L−1=N−2 个可建库 transition。
- XPolicyLab native：已有 L 行对齐 state/action → L 行 → L−1 个可建库 transition。
- Piper：原始 N 条已接受命令 + N+1 个观测 → L=N 行 → N−1 个可建库 transition。

最后一条 command 仍在训练 parquet；最后的额外 terminal 观测/图片保留在原始数据中，
conversion manifest 记录 terminal 和原始文件 hash。当前 bank 不消费最后一条 command，
因为统一 LeRobot 行中没有它的后果帧。没有补零 action、重复末帧或把下一 episode 的帧接过来。
若将来要完全利用 terminal，须显式扩展 catalog/feature 时间轴契约并同步测试，不能只改一个切片。

train episode 至少要有 H+1 个发布行（默认 H=32）。短片段原样保留，不拼接穿越 reset；
当前版本要单独整理成符合长度的采集段再进入 manifest。失败、不规则控制、aborted 记录应归档，
不要用 padding 或重采样假装它们是正常演示。

## 输出和训练接入

```text
output/version/
  prepared/
    dataset/                        # 新 raw adapter 的 LeRobot；LIBERO 引用已有 roots
      meta/conversion.json          # episode 原始文件/内容 hash、来源、terminal、outcome
    catalog.json / audit.json
    dataset_stats.json / train_stats_manifest.json
    data_config.yaml                # 已固定 episode_split，不随机拆数据
    config.json / COMPLETE.json
  features/
    contracts/{normalizer,encoder,camera}.json
    dataset_000/episode_000000.npz + .manifest.json
    train_features.list / dev_features.list
    COMPLETE.json
  memory/
    event_bank/
    train_candidates/ / dev_candidates/
    training_bindings.json          # 路径、维度、gripper、recollection action_mode
    warm_data_config.yaml           # data 配置 + warm_candidates 的完整绑定
    COMPLETE.json
```

`warm_data_config.yaml` 可作为训练配置的 `data` 部分，或由我们按既有 Hydra 实验配置合并；
所有 bank/candidate/catalog/stats/feature 路径已经配套生成。仍需由我们提供实际 checkpoint、
文本 embedding cache、模型参数、优化器与训练发布配置。本模块没有发起训练或自动迁移权重。
Piper 的 proprio 从 LIBERO 的 8D 变成 7D，需要显式处理输入投影的迁移并做模型前向验收，
不能用伪造双指状态凑成 8D。

训练包装器只新增了 Piper `7 action / 7 state / 2 camera / global min-max` 的严格分支；
既有 LIBERO 与 RoboTwin 的维度、归一化、mask 和图片校验保持有效。
数据接口通过不代表 checkpoint 已兼容，也不代表真机运行态已实现。

每个 stage 独立不可覆盖，以 `COMPLETE.json` 发布成功；出错保留 `.stage.staging-*/FAILED.json`，
不会把半成品命名为最终产物。发现数据错误后保留失败目录，修正配置并选择新的 output 版本。
source/target 不能互相嵌套。原始文件与视频 hash 在转换/读取阶段核对，统计量文件或配置变更后
拒绝继续构建下游。更换机器时建议在训练机用 raw 重新 prepare，避免手改已绑定的绝对路径。

## 验收

1. 先用 2 train + 1 dev 的短样本跑 prepare，检查图片红蓝通道、任务指令和动作方向/单位。
2. 核对原始时长、fps、命令数、发布行数、memory transition 数；确认没有跨 session/reset 拼接。
3. 查看 audit 的 cross-split duplicates 为 0，train stats 的来源只有 train。
4. GPU 上用实际 DINO/VAE 执行 features，检查 semantic shape、显存与耗时；禁止拿测试 encoder 做训练。
5. memory COMPLETE 中 event 数应 >0；train 至少 2 个不同 episode，确认 empty candidate 数符合预期。
6. 用配套配置运行一个训练样本与前向，核对 causal recollection、future target mask 与候选同 episode 排除。
7. 微调权重交给第三方后，仍须按离线回放 → 无电机 → 单步 → 低速闭环完成硬件验收。

CPU 回归入口：

```bash
python -m pip install pytest
python -m pytest tests/test_unified_preprocessing.py tests/test_real_handoff.py -q
```

扩展新 benchmark 时：实现 `EpisodeAdapter.read(entry) -> RawEpisode`，明确 action/state/图片/时间语义，
在配置与 prepare 的 adapter 选择处注册，补充自己的 raw fixture 与往返测试。保持训练 stats、
事实特征、bank、candidate cache 这四个公共阶段不变；新的控制语义还需要明确的训练/部署契约校验。
