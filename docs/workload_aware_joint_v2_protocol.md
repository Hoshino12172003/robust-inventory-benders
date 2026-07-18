# Workload-Aware Joint Adaptive Inexact Benders V2 Protocol

## Scope and status

Workload-Aware Joint Adaptive Inexact Benders V2（工作负载感知的双层自适应不精确 Benders）是与冻结 Joint V1 并行存在的实验性精度策略。V1 名称仍为 `proposed_joint_rho025_050`，策略标识仍为 `precision_policy: joint_error_budget`；V2 名称为 `proposed_workload_aware_joint_v2`，且只有 `precision_policy: workload_aware_joint` 才启用 V2。

本阶段仅冻结算法机制、开发/验证配置、选择规则、审计和测试，不运行开发、验证、最终测试或管理敏感性正式实验，不形成经验结论。V2 可以被拒绝而无需回滚代码、修改 V1 参数或删除 V1 结果。

## Mechanism

第 k 轮完成后的主问题和子问题时间记为 $t_k^M$ 和 $t_k^S$。只使用已经完成的历史迭代，通过指数移动平均更新：

$$
\bar t_k^M=\beta\bar t_{k-1}^M+(1-\beta)t_k^M,
\qquad
\bar t_k^S=\beta\bar t_{k-1}^S+(1-\beta)t_k^S,
$$

其中冻结值 $\beta=0.80$。首次有效观测直接初始化对应 EMA。原始 MP 工作负载占比与权重为：

$$
s_k^M=\frac{\bar t_k^M}{\max(\epsilon_{time},\bar t_k^M+\bar t_k^S)},
\qquad
w_k^M=\operatorname{clip}(s_k^M,1/3,2/3),
\qquad
w_k^S=1-w_k^M.
$$

冻结总误差比例为 $\rho_{total}=0.75$，因此：

$$
\rho_{M,k}=0.75w_k^M,
\qquad
\rho_{S,k}=0.75w_k^S,
\qquad
\rho_{M,k}+\rho_{S,k}=0.75.
$$

在上下界截断前，MP 负担上升会把更多允许误差分配给 MP，同时减少 SP 的分配。候选 gap 为：

$$
g^M_k=\operatorname{clip}(\rho_{M,k}g_k,10^{-4},0.02),
\qquad
g^S_k=\operatorname{clip}(\rho_{S,k}g_k,10^{-4},0.05).
$$

实际选择继续单调收紧：

$$
\hat g^M_k=\min(\hat g^M_{k-1},g^M_k),
\qquad
\hat g^S_k=\min(\hat g^S_{k-1},g^S_k).
$$

V2 仅改变正常迭代的 MP/SP requested MIP gap。鲁棒对偶 MILP、LB/UB 更新、有效割、全局 gap、`theta >= 0`、重复割、Gamma、终止核验触发和最终终止条件均不变。

## Initialization, fallback, and timing order

冻结配置为：

- `workload_ema_decay: 0.80`
- `workload_total_error_budget_ratio: 0.75`
- `workload_master_weight_min: 0.3333333333333333`
- `workload_master_weight_max: 0.6666666666666666`
- `workload_time_epsilon: 1.0e-9`
- `workload_initial_master_weight: 0.3333333333333333`
- `workload_initial_subproblem_weight: 0.6666666666666666`

第一轮、时间缺失、非有限、负值、平滑时间之和过小或全局 gap 无效时，V2 精确回退到 V1 权重，即 $\rho_M=0.25$、$\rho_S=0.50$。全局 gap 无效时仍沿用既有 `precision gap fallback = 1`，不定义新的全局 gap。

每轮开始时只用截至上一轮的 EMA 选择 gap；随后依次求解 MP 和 SP，最后用这两个求解器的本轮时间更新 EMA。日志、文件写入、管理评价和其他开销不计入 workload。终止核验强制两个 gap 为 0；核验轮不作为正常策略输出，也不更新 workload EMA。退出核验后继续使用进入核验前的 V2 精度与 EMA 状态。

## Logged evidence

V2 迭代日志记录策略是否活跃、EMA 衰减、MP/SP 时间 EMA、原始 MP share、两个选定权重、两个动态比例、总误差比例、是否回退及回退原因。最终 metadata 记录启用状态、最终 EMA、最终权重、平均权重和回退次数。字段均可序列化为 CSV/JSON；V1 对应字段为 `false`、空值或零，不改变 V1 已有字段含义。

## Frozen experiment stages

阶段顺序固定为：

1. 静态测试和 mock 测试；
2. 中大规模开发集；
3. 大规模开发集；
4. 中大规模验证集；
5. 大规模验证集；
6. 按预先规则选择 V1 或 V2；
7. 只有 V2 通过验证后才建立独立最终测试配置；
8. 最终算法确定后才运行完整管理敏感性实验。

四个已冻结配置均比较 `mp_adaptive_rho050`、`proposed_joint_rho025_050` 和 `proposed_workload_aware_joint_v2`：

| 阶段 | 规模 | 种子 | 时间限制 | 运行数 |
| --- | --- | --- | ---: | ---: |
| development | medium_large | 40–44 | 600 秒 | 15 |
| development | large | 40–44 | 1800 秒 | 15 |
| validation | medium_large | 45–54 | 600 秒 | 30 |
| validation | large | 45–54 | 1800 秒 | 30 |

开发集只用于调试、机制分析、稳定性检查和是否进入验证阶段的决定，不进入正式统计推断。验证开始前 V2 参数必须冻结，验证集不得用于继续修改参数或选择阈值。

## Pre-registered V1/V2 selection rule

首先检查正确性门槛。V2 必须无 failed 运行；保持有效 LB；所有可更新运行保持有效 UB；LB 非递减；UB 非递增；requested MP/SP gaps 非递增；不因缺失 incumbent 产生伪割；无算法逻辑异常；终止和超时状态解释正确。任一正确性条件失败，立即选择 V1。

正确性门槛通过后，V2 还必须同时满足以下必要条件。

中大规模非退化条件：

- V2 solved rate 不低于 V1；
- V2 平均 PAR-2 不高于 V1 的 1.03 倍；
- V2 的平均性能退化不得超过 V1 3 个百分点。

大规模改善条件：

- V2 solved rate 不低于 V1；
- V2 solved rate 不低于 MP-only；
- V2 平均 PAR-2 至少比 V1 降低 5%；
- V2 平均 PAR-2 至少比 MP-only 降低 3%；
- 在 10 个 large 验证实例中，至少 6 个实例的 V2 PAR-2 不高于 V1；
- V2 平均名次优于 V1 和 MP-only。

这些阈值是验证阶段的采用规则，不是最终统计显著性标准。任何必要条件不满足即选择 V1，并停止 V2 正式化；不得在查看验证结果后修改阈值。

## Reserved final-test seeds

中大规模最终测试预留种子为 `[55, 56, 57, 58, 59, 60, 61, 62, 63, 64]`，大规模最终测试预留种子为 `[65, 66, 67, 68, 69, 70, 71, 72, 73, 74]`。这些种子不得用于实现调试、参数选择、开发、验证或 smoke test。本 PR 不创建最终测试配置。

若 V2 通过验证，后续最终测试预计比较六种冻结方法；若 V2 未通过，则不运行 V2 最终测试，继续采用 V1。

## Commands

先运行中大规模开发：

```powershell
python -m src.experiment_suite `
  --config experiments/configs/workload_aware_joint_v2_development_medium_large.yaml `
  --resume
```

只有机制与正确性检查通过后，再运行大规模开发：

```powershell
python -m src.experiment_suite `
  --config experiments/configs/workload_aware_joint_v2_development_large.yaml `
  --resume
```

验证配置采用同一 CLI，并仅在开发阶段通过后执行。所有四个配置也支持 `--overwrite` 与不调用求解器的 `--dry-run`。运行管线不自动并行启动多个 Gurobi 实例。

## Managerial sensitivity postponement

现有 190-run 管理敏感性配置保持为 V1，不在本阶段改指 V2，也不运行。若最终选择 V1，则运行原配置；若选择 V2，则另行新增保持相同敏感性水平的 V2 配置。并列最坏场景和其他管理指标解释继续遵循既有管理敏感性协议。
