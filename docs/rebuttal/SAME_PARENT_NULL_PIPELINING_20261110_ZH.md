# 固定后期query的同父null补测与依赖管理

目的：补BB-W3/Q2要求的refined与candidate-null直接局部比较，并为后续raw→refined修复/损害汇总取得同状态证据。保留原v8模型、完整科学core、固定source0/q48、原H32、同Full事实历史及同Gaussian。此为原local-recovery计划中的query，不增加或替换按结果选择的样本。

## 不变的执行要求

- source0/q48对应第192条事实命令后的父状态；q48是replan ordinal，不是只回放48条命令。每个新分支都独立fresh reset并精确回放原192条命令。
- 原Full事实700步及六个query的Full/null采样动作已经封存；新任务不重新生成policy动作、不消费未来teacher、不修改当前事实history。
- 两份已有Full初始anchor必须完整H32，且原始父状态、端点和语义效应精确相同。新分支再逐项匹配完整native/RNG父状态、保存RGB/官方policy输入及原DINO数组；任何差异均停止并保留原件，不放宽容差。
- 执行封存null H32，然后独立reset/replay执行原Full H32。后者必须精确复现原Full端点和效应。新的两个分支不会写入事实模型记忆。

## 允许的工程流水安排

已有K32 collector按固定rank顺序工作。原q48选中rank29；其动作与选择身份在任何新分支真值前已封存。为避免等待整个K32结束而浪费并行资源，新null执行可在两个初始Full anchor就绪后开始。这个安排不读取或改变rank29的结果，也不替换它。

新显式选项`--allow-pending-pool-records`仅允许选中raw及pool-final-anchor两个记录暂缺。缺失raw的标签始终为null，完整query状态保持pending。已有任何记录均要完整验证；重复terminal attempt、哈希损坏、原query不符、父状态或anchor端点不符均停止，不能在流水选项下跳过。默认q8入口仍要求所有原依赖就绪。

新null/post-null-Full结束时再次读取原pool依赖。仍缺依赖则状态为`complete_fixed_null_branches_pending_pool_dependencies`；只有原raw与final anchor也验证后，才能形成完整三角色query。原K32 collector的完整query及其预测分析继续使用自己的固定协议，不能因null已结束提前填全。

## 可报告与不可报告

逐query记录原raw、refined、null的独立任务谓词标签、完整H32/invalid/early-success状态、所有未知、原件及source/model/plan身份。未完整执行或不可判定的标签不填0。单query比较不等于整任务SR或总体修复率；重复Full是执行一致性证据，不增加独立query/episode数。

作者指定v8与原稿逐任务specialist的对应关系仍待核实。所有新局部结果保留当前共享模型范围，不自动替代原论文数据。新增真机取证仍延期。
