# Piper 真机数据工作区

本目录用于第三方采集数据的本地副本，与所有仿真 benchmark 隔离。
`raw/`、`processed/`、`releases/` 已加入 Git 忽略规则；目录中的数据不要 commit/push。

```text
real/piper/
  raw/session_A/episode_001/       # 原始记录、RGB 文件；不覆盖或原地改写
  raw/session_B/episode_001/       # 不同采集 session 可预先分配 dev/test
  processed/session_v1/           # prepare/features/memory 的独立版本
  releases/                      # 后续双方验收的部署包；不是自动生成的可执行驱动
```

第三方采集输出遵循 `warm.real.episode.v1`；桥接代码位于
`src/fastwam/real`，预处理适配器位于 `src/fastwam/real/preprocessing`，
命令位于 `scripts/real`。原始 vendor HDF5/ROS bag 可以作为旁路档案保留，
目前不能不经字段、单位、时间与动作语义确认就直接导入。

操作说明：[真机数据预处理与协作说明](../../docs/real/PREPROCESSING_ZH.md)。
