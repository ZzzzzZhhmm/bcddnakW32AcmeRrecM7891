# 待填数据的统一映射与放行条件

核验时间：2026-11-13T13:57:43.549782+00:00。回复v37配套论文v36；本表不填入任何假设数值。

37个回复槽位对应31个论文宏；P1差值/区间与P2最终SR/差值区间在论文各合成一处，另4个回复专用支持字段保留。不存在漏配的论文宏。所有槽位均位于整段PENDING之内，当前全部未放行。

|回复字段|论文宏|单位和证据对象|
|---|---|---|
|P1_CI|P1-delta-CI|paired difference95CI in percentage points|
|P1_DELTA|P1-delta-CI|paired SR difference in percentage points|
|P1_DISCORDANT|回复专用支持数据|paired discordance counts, task-stratified|
|P1_FULL|P1-Full|success count /100 fixed reset pairs, Full role|
|P1_POOLED|P1-Pooled|success count /100 fixed reset pairs, Pooled role|
|P1_TASK_RESULTS|P1-task-results|per-task success/assigned/missing and paired estimates|
|P2_BAD_DENOM|P2-bad-denominator|complete identifiable proposal-failure denominator|
|P2_EFFECTS|P2-effects|centered effect MSE for each prespecified reference on identical support|
|P2_EFFECT_CI|P2-effect-CI|paired primary metric contrast and prespecified interval|
|P2_EPISODES|P2-episodes|independent source episode count|
|P2_FINAL_DELTA_CI|P2-final-SR-delta-CI|paired complete-task difference and interval|
|P2_FINAL_SR|P2-final-SR-delta-CI|separate complete-task continuation successes/assigned for each role|
|P2_GOOD_DENOM|P2-good-denominator|complete identifiable proposal-success denominator|
|P2_HARMS|P2-harms|Full failures among proposal-success states|
|P2_NULL|P2-null|same-state candidate-null outcomes and missingness|
|P2_REPAIRS|P2-repairs|Full completed-interaction count among proposal-failure states|
|P2_STATES|P2-states|predeclared factual state count, not candidate count|
|P2_UNKNOWNS|回复专用支持数据|all invalid/unknown/unstarted records by fixed support|
|P3_ABSENT_N|P3-absent-N|independently certified truly uncovered decision count|
|P3_COVERAGE|回复专用支持数据|applicable-candidate coverage with reached/assigned denominators|
|P3_FALSE_ACTIVE|P3-false-active|false-activation numerator/identified-absence denominator plus interval|
|P3_GATE_DISTRIBUTION|回复专用支持数据|natural effective-gate distribution by applicability stratum|
|P3_NI_MARGIN|P3-NI-margin|prospective noninferiority margin, never fit from observed outcomes|
|P3_RATES|P3-rates|prospectively frozen event-replacement proportions|
|P3_SR_DELTA_CI|P3-SR-delta-CI|Full/null complete-task paired difference and interval|
|P3_TASKS|P3-tasks|prospectively frozen task list|
|P4_CI|P4-CI|task-specific paired complete-task difference and interval|
|P4_MEMORY_ACC|P4-memory-accuracy|correct/reached memory decisions by task and condition|
|P4_R2P_SR|P4-R2P|new paired Full/history-masked successes/20 each on R2P, not original15/20|
|P4_REACHED|P4-reached|reached memory-decision denominator;all assigned trials retained for SR|
|P4_SOR_SR|P4-SOR|new paired Full/history-masked successes/20 each on SOR, not original15/20|
|P5_BANK_EVENTS|P5-bank-events|training-only repertoire event count with source provenance|
|P5_BANK_SR_DELTA_CI|P5-bank-SR-delta-CI|original/expanded-bank matched-reset outcomes and difference interval|
|P5_COVERAGE_DIVERSITY|P5-coverage-diversity|coverage and independent source-episode diversity on fixed queries|
|P5_DISPLACEMENT_RESULTS|P5-displacement-results|frozen displacement levels in physical units with per-level counts/SR/unknowns|
|P5_NEW_DEMOS|P5-new-demos|independent newly teleoperated episode count|
|P5_SPLIT|P5-split|fixed train/development episode allocation|

## 统一填数前的硬条件

- 必须绑定真实raw/manifest、固定样本和query支持、模型/config/source、完整失败与未知；不拿原稿数值、工程检查、shared或固定12000别的范围填入本组。
- P1的100对是设计分母，当前41对全部0–0和13次历史unknown不能静默改成完整100对。
- P2以独立来源episode/state为统计单位，同一状态32候选不是32独立episode；不能把局部无掉落改写为交互完成或最终任务成功。
- P3独立确认非空bank确实缺技能；自然gate与显式g=0干预分开。NI margin须事先冻结，不能事后拟合。
- P4仅接作者新采集的配对结果；原15/20或无历史遮蔽资格的录像不填入新对照槽。记忆决策以reached为分母，完整SR保留所有分配试验。
- P5保持独立demo的训练/开发隔离、固定reset与位移等级；扩库和移动位置是不同实验轴。
- 先审核实际方向和区间，再一起放行整段结论、回复、论文宏和20项证据映射。填数字不会自动放行有利结论。

本次不新增试验或PDF，不更改已封存回复、论文、原结果及37个槽位。
