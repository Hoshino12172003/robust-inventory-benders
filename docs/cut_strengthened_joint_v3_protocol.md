# Cut-Strengthened Joint Adaptive Inexact Benders V3 Protocol

## Scope

Cut-Strengthened Joint Adaptive Inexact Benders V3（割强化的双层自适应不精确 Benders V3）是与冻结 Joint V1 并行存在的实验扩展。完整机制称为“核心点与停滞触发割强化的双层自适应不精确 Benders”。本实现只建立算法、development 配置、日志、审计、测试和只读分析工具，不运行 development、validation、final-test 或管理敏感性正式实验。

V1 继续使用 `proposed_joint_rho025_050`、`precision_policy: joint_error_budget` 和 `cut_strengthening_policy: none`。V3 的所有消融方法同样使用 `joint_error_budget`，不包含或依赖 Workload-Aware V2。将 `cut_strengthening_policy` 设为 `none` 即恢复 V1。

## Terminology and validity

本文使用 core-point strengthened cut、Magnanti-Wong-type core-point strengthening 和“核心点强化割”。不声称该割严格 Pareto-optimal，因为动态核心点尚未证明始终位于第一阶段可行域凸包的相对内部。固定任一可行扰动模式及任一对偶可行解得到的仿射函数均不超过完整鲁棒 recourse 函数，因此所生成的割仍是有效 Benders 割。

论文面向用户的终止阶段称为“终止核验”；既有内部 `final_certification_*` 字段保持兼容。

## Frozen V1 foundation

V3 不改变鲁棒对偶 MILP、需求不确定集、MP objective bound 对 LB 的更新、目标鲁棒 SP objective bound 对保守 UB 的更新、SP incumbent 生成有效割、全局相对 gap、`theta >= 0`、Gamma、重复割、终止核验、终止条件以及 `best_x_values`/`best_y_values` 的语义。

V1 精度参数保持 `rho_M=0.25`、`rho_S=0.50`，MP gap 范围 `[0.0001, 0.02]`，SP gap 范围 `[0.0001, 0.05]`，并保持单调收紧。

## Core-point strengthened primary cut

设当前库存为 $x^k$，原目标鲁棒 MILP incumbent 的扰动模式为 $z^k$，对应需求为

$$d^k_{rj}=\bar d_{rj}+\hat d_{rj}z^k_{rj}.$$

固定模式对偶仿射函数为

$$
L(z^k,\lambda,\mu,\nu;x)
=\sum_{r,j}d^k_{rj}\lambda_{rj}
-\sum_{i,j}x_{ij}\mu_{ij}
-\sum_j(1-\alpha_j)\left(\sum_r d^k_{rj}\right)\nu_j.
$$

第一阶段求解纯连续 LP：

$$q_{z^k}=\max L(z^k,\lambda,\mu,\nu;x^k).$$

仅当第一阶段为 optimal 时，第二阶段才以迭代开始前保存的核心点 $x_k^c$ 为目标，并加入

$$L(z^k,\lambda,\mu,\nu;x^k)\ge q_{z^k}-\delta_k,$$

其中

$$\delta_k=10^{-7}+10^{-8}\max(1,|q_{z^k}|).$$

两个 LP 均不包含二元 `z`、`w`、`g` 或 McCormick 约束。第一阶段或第二阶段非 optimal、对偶不可行、当前点保持失败、核心点改善不足、割重复、原主割不违反或处于终止核验时，均回退到原 V1 主割；强化失败绝不导致跳过原割。

归一化核心点改善为

$$
\frac{\max(0,L_{strong}(x^c)-L_{original}(x^c))}
{\max(1,|L_{strong}(x^c)|,|L_{original}(x^c)|)}.
$$

最低阈值冻结为 `1.0e-7`。辅助 LP 的 objective、incumbent 和 bound 仅用于割强化，`core_point_auxiliary_bound_used_for_UB` 必须始终为 false。

核心点初始为空。第一轮不强化，正常迭代完成后保存 $x^1$。其后仅使用截至上一轮的状态，并在正常迭代末更新

$$x^c_{new}=0.5x^c_{old}+0.5x^k.$$

代码使用欧氏距离检查 `core_point_min_distance=1.0e-9`。终止核验期间不强化、不更新核心点。

## Stall-triggered differentiated secondary cut

第二割仅在 `stall_secondary` 或 `core_point_stall_secondary` 下启用。它使用截至上一轮的 LB 历史。窗口为 5，最近相对改善为

$$
\frac{\max(0,LB_{last}-LB_{first})}
{\max(1,|LB_{first}|,|LB_{last}|)}.
$$

只有该值不超过 `1.0e-4`，且不是第一轮、Gamma 已到 target、原 SP 有 incumbent、全局 gap 大于 `1.0e-3`、不在 10 轮 cooldown、剩余时间至少 30 秒、累计第二次受限鲁棒求解时间占已用 wall-clock 时间的比例不超过 0.10、且不处于终止核验时才触发。

第二求解时间上限为

$$\min(10,0.05\times remaining\_global\_time),$$

若不足 1 秒则跳过。第二 robust MILP 通过 no-good 约束排除当前主模式，并排除最近最多 10 个已使用第二模式。模式记忆采用固定长度 FIFO 去重更新，不会无界增长。

第二求解有 incumbent 即可生成割，不要求 optimal。只有模式不同、未在记忆中、割非重复且当前点绝对违反量超过 `cut_violation_tol` 时才加入。V3 不启用相对割筛选。受限第二 MILP 的 objective 和 objective bound 均不得更新 UB，`v3_secondary_bound_used_for_UB` 必须始终为 false。

## Per-iteration order

完整 V3 正常迭代顺序冻结为：V1 精度选择、MP、原目标 robust SP、保存原 SP bound 作为唯一 UB 来源、尝试核心点强化主割、必要时回退原主割、基于历史 LB 决定第二求解、检查违反量与重复性、最多加入一条主割和一条第二割、更新终止核验、正常迭代末更新核心点、检查原终止条件。

终止核验时 MP/SP 请求 gap 仍强制为 0，不执行两种 V3 辅助机制，也不更新核心点。`useful_primary_cut_added` 只表示主割加入，第二割不参与该标志。

## Frozen parameters

核心点参数：更新权重 0.50；距离阈值 `1.0e-9`；两阶段各 2 秒；最少剩余时间 10 秒；最小 gap `5.0e-4`；当前点绝对/相对容差 `1.0e-7`/`1.0e-8`；最小归一化改善 `1.0e-7`。每个 LP 的实际限制同时受全局剩余时间约束。

第二割参数：LB 窗口 5；停滞阈值 `1.0e-4`；cooldown 10；最小 gap `1.0e-3`；最少剩余时间 30 秒；单次最多 10 秒；剩余时间比例 0.05；额外时间占比上限 0.10；模式记忆 10。

## Development ablation

两个 development 配置均使用种子 `[75, 76, 77, 78, 79]`，比较：

- `proposed_joint_rho025_050`：V1；
- `joint_v1_core_point_strengthened`：core-only，最多 1 割；
- `joint_v1_stall_secondary_cut`：secondary-only，最多 2 割；
- `proposed_cut_strengthened_joint_v3`：完整 V3，最多 2 割。

medium-large 为 20 runs、600 秒、10000 轮；large 为 20 runs、1800 秒、20000 轮。所有方法使用 `robust_dual_milp`、Gamma `[2]`、关闭 continuation、旧 cut selection、旧 adaptive secondary generation 和 adaptive gap。

开发顺序固定为：静态审计与测试；medium-large development；正确性和非退化验收；large development；消融分析；最多选择一个候选；必要时只允许一次基于 development seeds 的参数修订；冻结新提交；另建 validation protocol PR；validation 开始后禁止调参。

正式命令仅供后续使用，本 PR 不执行：

```powershell
python -m src.experiment_suite `
  --config experiments/configs/cut_strengthened_joint_v3_development_medium_large.yaml `
  --resume
```

中大规模验收后：

```powershell
python -m src.experiment_suite `
  --config experiments/configs/cut_strengthened_joint_v3_development_large.yaml `
  --resume
```

## Pre-registered development selection

正确性门槛包括：无 failed；LB/UB 有效且分别非递减/非递增；MP/SP 请求 gap 非递增；强化割对偶可行且满足当前点保持；辅助 LP 和第二受限 MILP 不更新 UB；第二模式不同；无伪割；终止核验正确。任一组件失败则不得进入候选。

中大规模门槛：solved rate 不低于 V1；平均 PAR-2 不高于 V1 的 103%；不得出现明显系统性尾部退化。分析工具将“至少 3/5 配对实例的 PAR-2 比 V1 高超过 3%”标为明显系统性尾部退化。

大规模门槛：solved rate 不低于 V1；平均 PAR-2 至少降低 5%；平均迭代数至少降低 10%；5 个实例中至少 3 个 PAR-2 不高于 V1；额外割时间完整记录。

通过门槛的方法按 large 平均 PAR-2、large 平均迭代数、large 超时数、中大规模 PAR-2 排序；若性能差距小于 1%，优先组件更少的方法。full V3 不因名称而优先。若无新方法通过，停止 V3，继续采用 V1。

## Reserved validation and final tests

Validation 只在文档预留种子 `[80, 81, 82, 83, 84, 85, 86, 87, 88, 89]`，不创建配置。预计比较 MP-only、V1 和至多一个选定 V3。

Validation 门槛：中大规模 solved rate 不低于 V1、平均 PAR-2 不高于 V1 的 103%；大规模 solved rate 不低于 V1 和 MP-only、平均 PAR-2 至少比 V1 降低 7.5%、平均迭代数至少降低 15%、至少 6/10 实例 PAR-2 不高于 V1、至少 6/10 实例迭代数低于 V1、平均名次优于 V1 和 MP-only。任一必要条件失败则选择 V1。

最终测试仅预留 medium-large `[90, 91, 92, 93, 94, 95, 96, 97, 98, 99]` 和 large `[100, 101, 102, 103, 104, 105, 106, 107, 108, 109]`。这些种子不得用于调试、单元测试、development、参数调整、validation 或 smoke test。

## Managerial sensitivity

管理敏感性继续暂停，直到论文最终算法确定。V3 失败时使用原 V1 管理敏感性配置；V3 完成开发、验证和最终测试后，才可另建保持相同敏感性轴、水平、种子和指标的 V3 配置。
