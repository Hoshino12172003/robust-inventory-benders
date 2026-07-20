# Cut-Strengthened Joint V3 Development Decision

## Decision status

本文件冻结 Cut-Strengthened Joint Adaptive Inexact Benders V3 的 development 阶段决策。它记录既有 development 结果和预注册选择规则的应用，不修改原始协议、不继续调参，也不代表 validation、final test 或论文最终算法结论。

选定进入后续 validation 协议设计的唯一候选为：

`joint_v1_core_point_strengthened`

其机制保持：

- `cut_strengthening_policy: core_point`；
- `precision_policy: joint_error_budget`；
- `master_error_budget_ratio: 0.25`；
- `subproblem_error_budget_ratio: 0.50`；
- `max_cuts_per_iteration: 1`。

本轮不使用 development 阶段允许的一次参数修订。Validation 尚未开始，因此不能声称 core-only 已通过 validation、final test，或已经成为论文最终算法。

## Auditable evidence

全部正式 development 运行使用代码提交：

`8eabc10f9248878f4f5e409bbcd75ead288e168b`

独立复算的归档哈希为：

| Scale | ZIP | SHA256 |
| --- | --- | --- |
| medium-large | `cut_strengthened_joint_v3_development_medium_large_results.zip` | `D778D9B988BB360BBFF898A7A80887D8E68103E525512617EBC924C9DC76C492` |
| large | `cut_strengthened_joint_v3_development_large_results.zip` | `8516D345BDB752F946CB5643B10A560C14B6B7DBF8109F2BCE5F2A78521826E4` |

两个 manifest 均记录 20/20 completed、0 failed、0 remaining，以及上述代码提交。40 行结果覆盖 seeds 75–79 和四个预注册消融方法。原始结果目录与 ZIP 不进入本提交。

机器可读冻结文件为 `experiments/configs/selected_cut_strengthened_joint_v3_candidate.yaml`，SHA256 为 `7E8AAF39DE8C100B4CE9B46256A074FBD324B07DDC347D256494ED070D4E0EB6`。

## Correctness and completeness

只读分析确认：

- 40 次运行无 failed，所有状态和 final gap 均有记录；
- 所有运行具有有效 LB 和 UB；
- LB 非递减、UB 非递增，MP/SP requested gap 非递增；
- 原目标鲁棒 SP objective bound 是唯一 UB 来源；
- 核心点辅助 LP 与第二受限 robust MILP 均未更新 UB；
- 接受的核心点割对偶可行，并满足当前点保持及不弱于原割的容差条件；
- 强化失败时保留原 V1 主割；
- 第二割仅由不同扰动模式、有效 incumbent、非重复且被违反的割产生；
- 终止核验期间 V3 辅助机制关闭；
- 每轮割数、cooldown、时间预算和有限模式记忆符合预注册协议。

## Medium-large summary

| Method | Solved rate | Mean PAR-2 | Mean iterations | PAR-2 vs V1 | Iterations vs V1 | Development gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `proposed_joint_rho025_050` | 100% | 96.16216770003084 | 937.4 | baseline | baseline | reference |
| `joint_v1_core_point_strengthened` | 100% | 21.390878800023348 | 258.6 | −77.7554% | −72.4131% | pass |
| `joint_v1_stall_secondary_cut` | 100% | 105.78482709999662 | 899.8 | +10.0067% | −4.0111% | fail |
| `proposed_cut_strengthened_joint_v3` | 100% | 23.42112904000096 | 259.8 | −75.6441% | −72.2850% | pass |

Core-only 与 full V3 均有 5/5 个 seed 的 PAR-2 不高于 V1。Secondary-only 平均 PAR-2 高于 V1，且 3/5 个配对实例比 V1 慢超过 3%，触发预注册的系统性尾部退化判据，因此未通过 medium-large 门槛。

## Large summary

| Method | Solved rate | Mean PAR-2 | Mean iterations | PAR-2 vs V1 | Iterations vs V1 | Development gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `proposed_joint_rho025_050` | 80% | 1470.499560820032 | 2857.2 | baseline | baseline | reference; seed 79 time limit |
| `joint_v1_core_point_strengthened` | 100% | 125.79281519998331 | 624.8 | −91.4456% | −78.1324% | pass |
| `joint_v1_stall_secondary_cut` | 80% | 1524.8088922199793 | 2745.4 | +3.6933% | −3.9129% | fail; seed 79 time limit |
| `proposed_cut_strengthened_joint_v3` | 100% | 132.6898214599816 | 629.8 | −90.9765% | −77.9574% | pass |

Core-only 与 full V3 均有 5/5 个 seed 的 PAR-2 不高于 V1，并通过全部 large 门槛。Secondary-only 的 solved rate 未改善、平均 PAR-2 上升且迭代数降幅不足，未通过 large 门槛。

## Pre-registered candidate selection

分析工具得到：

- eligible：`joint_v1_core_point_strengthened`、`proposed_cut_strengthened_joint_v3`；
- rejected：`joint_v1_stall_secondary_cut`；
- selected：`joint_v1_core_point_strengthened`；
- decision：`freeze_one_candidate`；
- `configuration_or_parameter_changes_performed: false`。

选择 core-only 严格遵循预注册排序：

1. large mean PAR-2 为 125.7928，低于 full V3 的 132.6898；
2. large mean iterations 为 624.8，低于 full V3 的 629.8；
3. medium-large mean PAR-2 为 21.3909，低于 full V3 的 23.4211；
4. core-only 只有核心点强化组件，结构比 full V3 更简单。

因此 full V3 虽通过两个规模的门槛，但不被选为 validation 候选。Secondary-only 在两个规模均未通过性能门槛，明确淘汰。其代码、测试和 development 配置仍保留为消融与审计材料。

## Interpretation boundary

Development 阶段仅用于机制筛选和候选冻结，不进行正式统计推断。该结果不允许被解释为 validation 或 final-test 通过，也不支持提前写成论文最终算法结论。

核心点割继续称为 Magnanti-Wong-type core-point strengthened cut，不称为严格 Pareto-optimal cut。Validation seeds 80–89 和 final-test seeds 90–109 仍仅为预留，本阶段没有创建相应运行配置或结果。管理敏感性与任何公平模型实验均未启动。
