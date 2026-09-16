# 真机合作方技术接入

先阅读 [对外协作说明](PARTNER_HANDOFF_ZH.md) 或同名 Word 文件，再填写
[设备回填模板](../../configs/real/partner_intake.template.json)。当前只确认对方有两台
带夹爪的 Piper。所有采集、现场推理及真机执行由合作方负责；训练和模型交付由 WARM 团队负责。

## 可立即运行

Python >=3.10，标准库即可；无需 `pip install -e .`、CUDA、Piper SDK 或 ROS。
以下命令从仓库根目录运行。输出目录与文件不要复用。

```bash
mkdir -p outputs/partner_round01
python scripts/real/collect_inventory.py --output outputs/partner_round01/inventory.json
python scripts/real/make_mock_episode.py outputs/partner_round01/mock_episode
python scripts/real/validate_episode.py outputs/partner_round01/mock_episode --allow-synthetic
python scripts/real/replay_inference.py outputs/partner_round01/mock_episode --mock --output outputs/partner_round01/mock_shadow.json
python -m unittest discover -s tests -p test_real_handoff.py -v
```

inventory 可以分别在采集机和现场推理机运行。缺少诊断程序会被记录为不可用，不会安装或激活
设备。mock 为 4×4 PPM 合成图、36 个静止目标，仅证明接口；不能去掉 synthetic 标记冒充真机数据。
零动作也不是安全运动指令。未提供电机执行入口。

## 接入真实采集器

首个原始数据交换格式 `warm.real.episode.v1` 只支持单臂两相机、实际下发的绝对 TCP 位姿和
夹爪开度。它不是 LeRobot 训练数据、不是归一化 action，也不是 LIBERO 的 8 维 proprio。
合作方现有 SDK/ROS/相机采集器继续负责设备操作；将已经通过核验的记录传给 `EpisodeWriter`。

```text
episode_directory/
  episode.json          # 固定 metadata
  observations.jsonl    # N+1 个边界观察
  commands.jsonl        # N 个实际接受的命令
  outcome.json          # 独立结果，不输入模型
  external/             # 逐帧原始 RGB，PNG/JPEG 等
  wrist/
```

原始视频、rosbag 和厂商日志可另存。第一版交换格式要求可直接定位的图像文件，不接受仅视频
路径且无 frame index 的模糊引用；后续可单独约定视频容器扩展。不要改变相机方向或裁剪后丢弃原图。

完整可运行的结构示例由 `make_mock_episode.py` 生成；参考 `src/fastwam/real/synthetic.py`。
真实接入时自行构造 metadata，尤其是任务、标定、坐标与时钟，不复制 synthetic 身份。

| 字段 | 定义 |
|---|---|
| metadata schema | `warm.real.episode.v1` |
| episode_id session_id task_id instruction | 非空字符串；episode 不重复；instruction 不包含本轮隐藏答案 |
| split synthetic | train/dev/test；真实采集为 false。划分由双方预定，不能按帧随机切分 |
| calibration_id control_frame tcp_frame | 标定版本、位姿参考系、被控制工具端点；状态和命令必须相同 |
| clock_domain clock_id | client_monotonic_ns；所有时间映射到一个采集时钟，设备原时钟另存 |
| camera_order | 固定 `["external", "wrist"]` |
| action_semantics | `commanded_absolute_tcp_pose_and_gripper_width` |
| nominal_action_hz | 约定的动作边界采样频率，首轮建议研究 20 Hz；不是推理频率或 CAN 帧率 |
| max_observation_gap_ns max_sensor_skew_ns | 双方实测后确定的正整数阈值；不是安全阈值，不应放宽来掩盖丢帧 |

`observations.jsonl` 每条包含：

```json
{
  "seq": 0,
  "timestamp_ns": 1000000000,
  "cameras": {
    "external": {"path": "external/000000.png", "timestamp_ns": 995000000, "width": 640, "height": 480, "color_space": "RGB"},
    "wrist": {"path": "wrist/000000.png", "timestamp_ns": 996000000, "width": 640, "height": 480, "color_space": "RGB"}
  },
  "robot": {
    "timestamp_ns": 998000000,
    "joint_position_rad": [0, 0, 0, 0, 0, 0],
    "tcp_position_m": [0.2, 0, 0.2],
    "tcp_quaternion_xyzw": [0, 0, 0, 1],
    "gripper_width_m": 0.04
  }
}
```

数值仅为结构示例，不是推荐的机器人姿态。记录观察时，图像与状态必须已产生且新鲜；全部相机
时间及状态时间须向前推进。单机 receipt time 不等于真实 exposure time，若只能记录到达时间，
应额外提供采集延迟估计及原始时钟信息，让双方评估是否满足要求。

命令记录字段为 `seq`、`observation_seq`、`timestamp_ns`、`accepted`、`tcp_position_m`、
`tcp_quaternion_xyzw`、`gripper_width_m`。seq 从 0 连续；第 i 条命令关联观察 i，发送时间落在
观察 i 和 i+1 之间，后继状态时间晚于发送时间。accepted 只说明控制接口接受，不说明目标已到达。
若 SDK 没有接受反馈，应先约定实际可观察的发送状态并扩展格式，不伪造 accepted。

高频伺服子步另保留原始日志；不得直接当作策略时间步。先按约定动作边界记录目标与反馈。
暂停或拒绝导致不连续时保留原始诊断并结束该连续段；跨段 history 的处理由训练转换确定，
不能把丢失的提示段静默删除。当前 writer 对拒绝命令报错，调用方应保留其原始日志并安全结束记录。

接入方式示意，变量均来自合作方已经验证的采集器：

```python
# PYTHONPATH=src；本类不连接设备，不负责采集或发送动作。
from fastwam.real.episodes import EpisodeWriter

writer = EpisodeWriter(new_episode_directory, agreed_metadata)
# 先保存 observation 引用的图像到 writer.root，再提交记录。
writer.start(first_observation)
for actual_command, measured_successor in validated_recorder:
    writer.append_transition(actual_command, measured_successor)
writer.finish(status="success", interventions=0, notes="operator-reviewed")
```

首次质检：

```bash
python scripts/real/validate_episode.py /absolute/path/to/episode --report outputs/partner_round01/episode_audit.json
```

真实数据不加 `--allow-synthetic`。返回码 0 只代表结构通过，2 代表数据失败；报告文件已存在会拒绝
覆盖。检查顺序、有限数值、单位四元数、相对文件路径、时间新鲜度、N/N+1 关系及末帧。
工具目前不解码图片，不证明声明的分辨率/颜色正确，不检测实际画面是否停帧，不判断物理可达性。
前五条仍需双方逐帧检查，并核对真实命令单位、轴向、FK/TCP、夹爪和标定。

原生关节示教或仅被动拖动轨迹应先提供原始格式，不能为通过校验伪造 Cartesian commands。
由 WARM 团队新增对应 profile/转换并回放核验后，再批准批量采集。

## 录制数据上的真实模型联调

`replay_inference.py` 提供后端接口，当前没有随附可用的 WARM real backend 或 checkpoint。
该工具用录制动作更新历史并在录制画面上推理，属于 teacher-forced 离线诊断；预测动作没有改变
后续画面，不能计算闭环成功率，也不能用其结果证明恢复能力。

真实后端由 WARM 团队随交付包提供，实现工厂 `create(release_path)` 和四个方法：

```python
class Backend:
    def reset(self, context): ...
    def observe_executed(self, recorded_command, measured_successor): ...
    def predict(self, observation): ...  # JSON-compatible list [32, 7]
    def synchronize(self): ...          # CUDA synchronize, not a no-op for GPU
```

reset 的 context 包含任务、指令、相机顺序、坐标系及 image_root。插件只能从 image_root 解析
当前/过去观察指定的图片，不读取 outcome、未来图片、文件名中的隐藏答案或任务评价器。
插件是可信 Python 代码，导入会执行代码；harness 不是安全沙箱。离线阶段在没有 CAN 权限的
容器或物理断开运动通道的推理机运行，不能只依赖脚本注释。

后端交付时负责严格核验 checkpoint、真实统计、经验库、编码器及相机约定；重置旧 episode；
将原始观测转换为训练时的图像/state，将已接受命令转换为真实模型空间的执行账本；正确使用
`OnlineEpisodeController` 与 `BoundOnlineStep`，不能把普通 FastWAM 函数作为完整 WARM。
本 harness 不补做这些语义适配。

shadow release JSON 最少字段：

```json
{
  "schema": "warm.real.shadow-release.v1",
  "release_id": "assigned-by-WARM-team",
  "action_shape": [32, 7],
  "action_space": "real_profile_normalized",
  "execution_prefix": 4
}
```

工厂还应从该描述读取部署包资产配置并验证。前缀 4 是拟议值，须与真实微调后的历史摘要契约一致；
不能把 LIBERO 的执行前缀 10 仅在现场单独改成 4。收到完整交付包后才运行：

```bash
# backend、release、prefix 按团队交付值替换；此处不是已有可用模块名称。
python scripts/real/replay_inference.py /absolute/path/to/real_episode \
  --backend delivered_backend:create --release /absolute/path/to/shadow_release.json \
  --prefix 4 --output outputs/partner_round01/real_shadow.json
```

输出标明 mock、teacher_forced、冷启动及包含冷启动的 p50/p95/p99。正式 profiling 应由后端
增加分阶段 GPU 计时、峰值显存，先预热再测量足够次数；此处小样本时间不是实时达标结论。
工具检查形状和有限数值，不做动作限幅、碰撞或电机执行。

## 后续实时接口验收

尚待现场信息完成的模块：相机/SDK 采集桥、raw→training 转换、真实 profile、WARM 后端、
本地安全控制进程及网络 server/client。请在排期中列入这些工作，不将它们当作本次已实现。

实时请求至少携带 session、episode、obs_seq、client 采集时间、相机/状态及已实际执行的动作回执。
响应绑定同一身份、plan_id、动作空间、前缀长度、有效期。client 用自己的单调时钟判断年龄；
不能直接相减不同主机的 monotonic 时间。安全检查后的实际命令及实测响应分别记录，未执行的
预测动作不进入事实记忆。模型推理期不在线训练或更新正式经验库。

验证顺序为离线→隔离运动输出的 dry-run→监督单步→低速小空间→完整闭环。独立 watchdog、
急停、全连杆碰撞、限位、限速与加速度、夹爪力、IK/反馈异常、过期/重复/乱序响应都由控制侧
验证。推理崩溃或断网时控制侧必须自行停止接受新动作，清空旧队列，恢复时显式重新初始化。

## 交付记录

每轮数据和模型使用新的版本目录。保存设备清单、标定版本、模型版本、运行配置、环境、完整
试验表和原始日志。记录 train/dev/test 的整个 episode/session 分组，不按帧随机划分。
测试日志和录像全部交付，包括失败与接管；测试集不能加入训练统计或检索 bank。
