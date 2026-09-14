# RMBench Benchmark 配置对照与 LIBERO 结果

---

## 1. RMBench：本仓库 vs MemoryWAM

对照 [MemoryWAM (arXiv:2606.20562)](https://arxiv.org/abs/2606.20562) 论文中 RMBench 设定。


| 配置项 | MemoryWAM | 本仓库（WARM） |
|---|---|---|
| Benchmark | RMBench（9 任务，M(1)/M(n)） | 同左 |
| 训练数据（official 设定） | 每任务 **50** 条 expert demo | 同左 |
| 评测 | 每任务 **100** rollout，报 SR | 同左（Gate D 协议） |
| 动作空间 | 14D qpos（双臂） | 同左 |
| 相机 | 3 路（head + 双腕）拼成 **384×320** 单图 mosaic | 3 路独立 **240×320**（`cam_high`, `cam_left_wrist`, `cam_right_wrist`） |
| 动作 chunk / horizon | **H=16**（frame stride 4 × VAE stride 4） | **H=32** |
| 在线 replan | 按 16-step chunk 执行 | **4** step replan（`replan_steps=4`） |


---

## 2. LIBERO 仿真结果

4×H100。  
评测：四套件 × 10 任务 × **50** rollout，`root_seed=3407`（与 MemoryWAM / 常见 LIBERO 协议一致）。

### 2.1 按套件

| 套件 | 成功 / 总数 | 成功率 |
|---|---:|---:|
| LIBERO-Spatial | 492 / 500 | 98.4% |
| LIBERO-Object | 500 / 500 | 100.0% |
| LIBERO-Goal | 493 / 500 | 98.6% |
| LIBERO-10 | 489 / 500 | 97.8% |
| **合计** | **1974 / 2000** | **98.7%** |
