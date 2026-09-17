# Piper 遥操作数据预处理与第三方协作

推荐分工：第三方负责采集、原始数据备份、格式校验及全部真机推理；我们负责 raw 转换、
离线 DINO/VAE 特征、记忆事件库、微调、模型选择。这里的离线特征编码属于训练准备，
不是要求我们远程运行对方机器人。
这份说明补充此前的第三方交接文档；其中曾列为待开发的标准格式数据转换现已实现，
现场采集桥、推理后端和物理验收仍需按实际设备接入。

## 第三方要先确认并交付什么

| 项目 | 我们的最低输入要求 | 对方回填 |
| --- | --- | --- |
| 机械臂 | 首版一台活动 Piper 6 轴 + 夹爪；另臂固定停放 | 型号/固件/SDK commit、主从关系、活动臂 |
| 命令记录 | 实际已接受的绝对 TCP 目标，而非预测未发出的动作或反馈差分 | 遥操作下发模式、原始字段、控制帧与 TCP 定义 |
| 状态 | 同步的 6 关节 rad、TCP xyz 米、xyzw 四元数、夹爪总宽米 | SDK feedback 字段、频率、单位 |
| 图片 | 外部 RGB + 活动臂腕部 RGB，每帧独立时间戳、实际尺寸、固定相机顺序 | 相机型号/内外参/安装照片、实际 fps/延迟 |
| 时间 | 同一 client monotonic 时钟；命令夹在当帧与下一事实观测之间 | 记录时钟、相机时间映射、控制周期实测 |
| 标定 | calibration_id、control_frame、tcp_frame 一致且有对应资产 | 标定文件、方法、日期与验证结果 |
| 划分 | 按采集 session 预先分 train/dev/test；测试不能回流训练 | session/episode/task/instruction/outcome |

用户提供的 [Piper 产品页](https://www.agilex.ai/page/690abe7b5e78cfa260412c92)
不能确定现场型号、固件、夹爪零位或姿态约定。[官方 SDK V2 接口](https://github.com/agilexrobotics/piper_sdk/blob/master/asserts/V2/INTERFACE_V2.MD)
中 EndPoseCtrl 平移与夹爪开度使用 0.001 mm 整数，姿态/关节使用 0.001 degree 整数；
若现场确认使用该接口，前者乘 1e-6 得米，后者乘 π/180000 得弧度。
SDK 姿态 RX/RY/RZ 到四元数的旋转顺序、主动/被动约定仍需现场验证，本模块不猜测。
关节限位、速度、夹爪范围也不得从网页数值直接作为现场可执行安全配置。

若遥操作只有 joint target，没有 TCP target，或只有反馈 qpos，先提交一个原始样本：
需确认 FK、工具变换和命令语义后新增导入桥，当前脚本会拒绝缺少目标命令的数据。
不能用 `state[t+1]-state[t]` 冒充专家下发动作。双臂同时执行、深度图与不同相机数不在首版 profile 中。

## 文件位置与记录格式

把原始数据置于 `real/piper/raw/<session>/<episode>/`；该目录已在 Git 中忽略。
每个 episode 包含 `episode.json`、`observations.jsonl`、`commands.jsonl`、`outcome.json`
及两路图片，详细字段见 [原始格式与记录器](README.md)。我们将处理结果放到
`real/piper/processed/<version>/`，不与 LIBERO/RMBench/RoboTwin 的数据相混。

采集初期先交 2 个 train episode + 不同 session 的 1 个 dev episode；正式批量之前由我们检查。
默认演示模板是 20 Hz、horizon 32，均为拟议设置，必须与实际采集和微调配置一致。
至少记录 33 条连续命令及 34 帧事实观测才满足默认 bank 最短长度，实际任务通常更长。
避免暂停、丢帧后继续当成连续 episode。默认允许的全局时轴误差为 5 ms，可由双方根据实测
修改，但必须小于半个 tick；不对不合格时间轴静默插值。

```bash
# 第三方：已有格式后做 CPU 校验；这条命令不接机器人。
python scripts/real/validate_episode.py real/piper/raw/session_A/episode_001

# 从全部已记录的 metadata 生成清单，保留原来的 train/dev/test。
python scripts/real/index_teleop_episodes.py \
  --root real/piper/raw --output configs/real/piper_episodes.local.json
```

索引器默认纳入 success，其他 outcome 在输出 `excluded` 列表中注明；原始失败数据继续备份。
恢复示范中的早期失败步骤可以留在最终 success episode 中。若研究明确需要完整 failure episode，
同时在索引器增加 `--include-outcome failure`，并在配置 `include_outcomes` 纳入 failure；
失败后的中止片段不得与下一次试验拼接。

## 我们这边的转换与记忆初始化

```bash
python -m pip install -r requirements-preprocessing.txt
cp configs/real/preprocess_piper.template.json configs/real/preprocess_piper.local.json
# 编辑 source.manifest 为 piper_episodes.local.json，填写 source 的标定/坐标帧、actual fps、output 和 encoder 资产。
python scripts/real/preprocess_teleop.py --config configs/real/preprocess_piper.local.json --stage plan
python scripts/real/preprocess_teleop.py --config configs/real/preprocess_piper.local.json --stage prepare
# 以下两步在我们的 GPU 训练环境中执行：
python scripts/real/preprocess_teleop.py --config configs/real/preprocess_piper.local.json --stage features
python scripts/real/preprocess_teleop.py --config configs/real/preprocess_piper.local.json --stage memory
```

`preprocess_teleop.py` 是统一入口的 Piper 专用包装，切换仿真用
`scripts/preprocess_warm_data.py`。stage 分开是为了先检查便宜的转换结果，再消耗 GPU；
也可对全新 output 使用 `--stage all`。所有 `UNCONFIRMED_*` 必须先填写。

每条动作由**当时观测**和**实际目标命令**计算：

```text
state[t] = [TCP_xyz_m, Log(R_observed), measured_gripper_width_m]       # 7D
action[t] = [p_command - p_observed,
             Log(R_command @ R_observed.T),
             commanded_gripper_width_m]                             # 7D
```

旋转增量以已确认的 base/control frame 表达。未来部署应从**获取该 action 的观测锚点**
恢复绝对目标 `R_command = Exp(delta) @ R_observed`、`p_command = p_observed + delta_p`；
action chunk 中每一步的 delta 都对应采集时该步的观测，不可把整块都绑定首帧或一边执行一边
随意更换参考。具体 receding-horizon 执行、插值、超时和安全控制需要现场后端独立验收。
反归一化后的夹爪通道是米制绝对宽度，不是 LIBERO 的 ±1 开关。

Piper raw 有 N+1 个观测，发布 N 行训练样本；按当前 WARM 缓存契约只建 N−1 个 transition。
末端观测仍在 raw，并且写入 conversion manifest；没有补造末尾动作。详情见
[统一时间轴说明](../PREPROCESSING_ZH.md#时间轴不要补造-terminal-action)。

## 第三方最终会收到什么

我们在真实 train 数据上构建统计量与 repertoire，完成微调并验收模型输入投影后，交付固定版本的
checkpoint、processor/normalizer、相机/动作契约、train bank、运行配置及对照版本。
第三方运行的是这一整套配套包，不能只换 checkpoint 或把 LIBERO bank 拷贝过去。
每轮真机 episode 都清空 recollection；只用当前轮已发生的信息更新，不提前加载测试录制轨迹。

本次实现完成了离线预处理和严格训练接口，不包含现场 SDK/相机采集桥、可自主运动的模型后端、
关节/碰撞/急停执行器。尚未拿到真实遥操作文件、DINO/VAE 资产或做物理测试。
部署仍按既有 [第三方工作说明](PARTNER_HANDOFF_ZH.md) 的离线 → dry-run → 单步 → 低速小工作区
流程推进，不因为 memory COMPLETE 或合成测试通过而跳过安全验收。
