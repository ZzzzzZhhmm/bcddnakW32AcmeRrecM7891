# 动作与时间索引核对

本核对用于排查新固定查询中后果预测未优于query-constant的原因，不用于更改主指标、truth尺度、固定样本、H32或原投稿结果。

## 已核实

- 原训练source的`num_frames=33`给出32个动作目标和33个原始观察索引。`action_video_freq_ratio=4`仅把视频取为索引0、4、8、…、32，共9帧；不是128步动作视野。
- 转换器和实际Parquet元数据均定义`observation.state[t]=qpos[t]`、`action[t]=qpos[t+1]`。实际第0个episode的前64个action向量与随后64个state向量逐元素精确相同，frame索引连续。不能据此采用一行偏移“修正”。
- 原训练feature store的后果target为`Phi(o[frame+32])-Phi(o[frame])`，与32个下一帧动作目标的索引范围相符。
- 实际数据metadata的fps为15，对应名义编码时间32/15秒。该值是索引/视频metadata，尚不证明原始示教的物理elapsed time。
- 正式native执行器每次`take_action(..., action_type='qpos')`接收一个绝对关节/夹爪目标，并对左右臂调用TOPP（1/250秒步长），随后按规划轨迹长度执行多次`scene.step()`。H32指32次控制器请求，不是32个物理步，也不保证所有候选经历相同物理时间。

## 尚不能推断

目前没有证明离线和native存在四倍视野错误或一帧错位，也没有证明控制器elapsed-time差异导致了新负结果。物理持续时间、速度/接触差异及训练分布对应关系仍须直接测量。即使新增诊断，也保留原冻结H32/cohort/指标，不能事后缩放truth来制造优势。

`OFFLINE_NATIVE_TIME_SEMANTICS_V1.json`绑定原训练converter/dataset/target源码、原native executor、实际info.json和第0个Parquet的SHA；它仅涉及上述范围，没有运行新policy或优化器。原件已纳入35文件双端档案`same_information_control_and_time_audit_v1.tar.gz`，SHA `c7dddd91d7239cc84fdf5a1e5b699570d1dbdad87a556420f73cda37cba94596`。
