# Model Implementation Audit

Date: 2026-07-08

## 结论摘要

当前代码与论文给出的两阶段预算型鲁棒库存模型在核心变量、第一阶段约束、第二阶段约束、预算型二进制需求扰动集合方面基本一致。只要 `src/scenarios.py` 能完整枚举 `sum z <= Gamma` 的全部场景，`src/monolithic.py` 和 `src/benders.py` 求解的是论文模型的有限场景等价形式。

主要不一致或需澄清之处有三类：

- 大规模时 `max_scenarios` 触发候选场景 fallback，代码不再严格求解完整预算鲁棒模型。
- `adaptive_gap_gamma_benders` 是算法加速策略，不改变最终目标 Gamma 的上界评估，但论文中需要清楚说明它是求解策略而非新的数学模型。
- 测试目前能验证核心一致性，但规模较小，投稿前建议补充更多数值回归和边界测试。

整体判断：不构成论文投稿的根本障碍；但如果论文声称“大规模实验均为精确预算鲁棒最优解”，则必须避免使用候选场景 fallback，或在论文中明确写成启发式/近似鲁棒评估。

## 1. 已经和论文一致的部分

### `src/instance.py`

- `InventoryInstance.fixed_cost` 对应论文中的 `F_i`。
- `inventory_cost` 对应 `c_ij`。
- `capacity` 对应 `K_i`。
- `volume` 对应 `v_j`。
- `budget` 对应第一阶段库存资金预算 `B`。
- `transport_cost` 对应 `a_irj`。
- `shortage_penalty` 对应 `p_rj`。
- `service_penalty` 对应 `phi_j`。
- `service_level` 对应 `alpha_j`。
- `base_demand` 对应 `dbar_rj`。
- `demand_deviation` 对应 `dhat_rj`。
- `inventory_ub` 对应 `M_ij`。
- 随机生成器保证 `phi_j` 大于各产品对应的最大缺货惩罚量级，符合论文中 `phi_j > p_rj` 的设定意图。

### `src/scenarios.py`

- `_scenario_from_units` 实现了：

```text
d_rj = dbar_rj + dhat_rj z_rj
```

- `active_units` 是所有 `z_rj = 1` 的需求单元集合。
- `enumerate_budget_scenarios` 在场景数不超过 `max_scenarios` 时，枚举所有 `|active_units| <= Gamma` 的组合，对应：

```text
z_rj in {0,1}
sum_rj z_rj <= Gamma
```

### `src/subproblem.py`

- 第二阶段变量完全对应论文：
  - `q[i,r,j] >= 0`
  - `u[r,j] >= 0`
  - `e[j] >= 0`
- 需求满足约束一致：

```text
sum_i q_irj + u_rj >= d_rj
```

- 库存供应约束一致：

```text
sum_r q_irj <= x_ij
```

- 服务水平约束一致，代码写成等价形式：

```text
sum_r u_rj - e_j <= (1-alpha_j) sum_r d_rj
```

- 第二阶段目标使用运输成本、缺货惩罚和服务水平违反惩罚，和论文模型叙述一致。
- Benders cut 使用 LP 对偶值构造。对供应约束的对偶系数通常为非正，符合库存增加会降低二阶段成本的经济含义。

### `src/monolithic.py`

- 第一阶段变量一致：
  - `y[i]` 是二元变量。
  - `x[i,j] >= 0`。
  - `theta >= 0` 表示鲁棒二阶段成本上界。
- 第一阶段容量约束一致：

```text
sum_j v_j x_ij <= K_i y_i
```

- 库存配置逻辑约束一致：

```text
x_ij <= M_ij y_i
```

- 第一阶段预算约束一致：

```text
sum_i F_i y_i + sum_ij c_ij x_ij <= B
```

- 对每个枚举需求场景建立一套第二阶段变量和约束，并使用：

```text
theta >= scenario_cost
```

等价表达 `theta >= max_{d in U(Gamma)} Q(x,d)`。

### `src/benders.py`

- `_build_master` 中的主问题变量和第一阶段约束与论文一致。
- `_solve_worst_recourse` 在给定 `x` 后求所有当前场景下二阶段成本的最大者，对应鲁棒子问题：

```text
max_{d in U(Gamma)} Q(x,d)
```

- `_add_cut` 将固定场景的二阶段对偶割加入主问题；当场景完整枚举且最终 Gamma 等于目标 Gamma 时，标准 Benders 与 monolithic 模型一致。
- `adaptive_gap_gamma_benders` 每轮都用目标 Gamma 场景计算可行解上界，因此最终报告的上界具有目标 Gamma 下的鲁棒解释。

### `src/policies.py`

- 该文件不定义论文数学模型，只控制主问题 `MIPGap`。
- `RLInspiredGapPolicy` 借鉴 RL-iGBD 的离散动作思想，是算法层面的求解策略，不改变论文模型的可行域或目标函数。

### `tests/test_core.py`

- `test_subproblem_cut_matches_current_point` 验证 Benders 割在当前点等于子问题目标值。
- `test_standard_benders_matches_monolithic_on_tiny_instance` 验证小规模下标准 Benders 与 monolithic 目标值一致。
- `test_adaptive_benders_converges` 验证自适应版本能返回收敛解。

## 2. 不一致的部分

### A1. 大规模场景 fallback 不是完整预算鲁棒集合

- 文件：`src/scenarios.py`
- 函数：`enumerate_budget_scenarios`, `candidate_budget_scenarios`
- 风险等级：高

当 `count_budget_scenarios(instance, gamma) > max_scenarios` 时，代码不再枚举全部 `sum z <= Gamma` 场景，而是调用 `candidate_budget_scenarios`，按 `dhat * shortage_penalty` 排序生成少量候选场景。

这与论文中的完整预算型不确定性集合不完全一致。此时求解结果只对候选场景鲁棒，不保证对完整 `U(Gamma)` 鲁棒。

修改建议：

- 对投稿主实验，如果要声明精确鲁棒最优，设置足够大的 `max_scenarios`，或限制问题规模，使所有场景均可枚举。
- 在 `SolveResult.metadata` 中增加 `scenario_mode = "full"` 或 `"candidate"`，并在实验 CSV 中输出。
- 若使用 candidate fallback，应在论文中写成启发式加速或近似鲁棒评估，不应写成精确求解完整预算鲁棒模型。

是否影响投稿：

- 若论文实验只使用完整枚举：不影响。
- 若论文大规模实验使用 fallback 但仍声称精确鲁棒：会显著影响投稿可信度。

### A2. `theta` 被设为非负，论文未显式给出下界

- 文件：`src/monolithic.py`, `src/benders.py`
- 函数：`solve_monolithic`, `_build_master`
- 风险等级：低

代码中 `theta = model.addVar(lb=0.0, name="theta")`。论文模型只写 `theta`，未显式说明 `theta >= 0`。

由于当前第二阶段成本均为非负成本项，`Q(x,d) >= 0`，所以 `theta >= 0` 与模型经济含义一致，不会改变最优解。

修改建议：

- 在论文模型中补充 `theta >= 0`，或在代码注释中说明由于二阶段成本非负，故设定 `theta` 非负。

是否影响投稿：

- 基本不影响。属于表述补充问题。

### A3. 自适应 Gamma 是算法策略，不是原始鲁棒模型本身

- 文件：`src/benders.py`
- 函数：`_gamma_for_iteration`, `solve_benders`
- 风险等级：中

论文模型定义的是固定目标 `Gamma` 下的问题。代码的 `adaptive_gap_gamma_benders` 在迭代早期使用较小 `active_gamma` 生成割，再逐步推进到目标 `gamma_target`。

这并不直接改变最终目标模型，因为代码每轮都用 `target_scenarios` 计算目标 Gamma 下的可行上界，并且只有在 `active_gamma == gamma_target` 时才允许终止。但这属于算法加速策略，需要在论文算法章节明确说明。

修改建议：

- 在论文中写明：动态 Gamma 仅用于早期割生成与加速，最终收敛判据和上界评估均对应 `Gamma^tar`。
- 在实验输出中保留 `gamma_schedule` 和每轮 `gamma`，当前代码已在 iteration log 中记录。

是否影响投稿：

- 不影响模型一致性，但如果论文没有解释清楚，审稿人可能认为模型目标在迭代中被改变。

### A4. inexact Benders 的理论下界处理需要论文说明

- 文件：`src/benders.py`
- 函数：`solve_benders`
- 风险等级：中

代码使用 Gurobi 的 `model.ObjBound` 作为主问题有效下界，即使主问题用非零 `MIPGap` 求解。Gurobi 的 `ObjBound` 通常是有效全局下界，因此工程上可接受。

但若论文参考 RL-iGBD 的理论写法，可能需要说明为何使用 `ObjBound`，而不是采用论文中类似 `v(l)(1-epsilon_MP)` 的 true lower bound 形式。

修改建议：

- 在论文算法描述中写明：实现中直接使用 MIP 求解器返回的有效 best bound 作为下界。
- 在结果日志中保留 `realized_master_gap`，当前代码已记录。

是否影响投稿：

- 不影响代码结果，但影响理论叙述严谨性。建议在算法章节补充一句。

### A5. 测试覆盖偏小，尚不足以支撑全部论文实验声明

- 文件：`tests/test_core.py`
- 函数：全部测试
- 风险等级：中

当前测试验证了核心机制，但只覆盖很小的实例。没有覆盖：

- `Gamma = 0`
- `Gamma` 大于需求单元数
- 多产品服务水平约束边界
- `max_scenarios` fallback 行为
- 不同 `MIPGap` 策略的回归结果
- monolithic 和 Benders 在更多随机种子上的一致性

修改建议：

- 增加参数化测试：多 seed、多 `Gamma`、多产品、多区域。
- 增加一个测试确认 fallback 时返回的 `scenario_mode` 被标记为 candidate。
- 增加固定实例 JSON 的数值回归测试，避免后续改代码导致结果漂移。

是否影响投稿：

- 不直接影响投稿，但会影响复现实验可信度。建议投稿前补强。

## 3. 逐文件风险表

| 文件 | 函数/区域 | 一致性结论 | 风险等级 | 修改建议 |
|---|---|---:|---:|---|
| `src/instance.py` | `InventoryInstance`, `generate_instance` | 参数映射一致 | 低 | 论文或 README 中说明随机算例生成规则 |
| `src/scenarios.py` | `enumerate_budget_scenarios` | 完整枚举时一致 | 高 | 标记 full/candidate 模式，投稿实验避免 fallback 或明确其近似性质 |
| `src/scenarios.py` | `candidate_budget_scenarios` | 与完整鲁棒集合不一致 | 高 | 仅作为启发式候选场景生成，不作为精确鲁棒模型声明 |
| `src/subproblem.py` | `solve_recourse_subproblem` | 第二阶段模型一致 | 低 | 增加对偶符号和割有效性的注释/测试 |
| `src/monolithic.py` | `solve_monolithic` | 完整场景枚举时一致 | 低 | 在论文中说明 monolithic 是场景枚举等价模型 |
| `src/benders.py` | `_build_master` | 主问题变量和约束一致 | 低 | 补充 `theta >= 0` 的说明 |
| `src/benders.py` | `solve_benders` | 标准 Benders 一致 | 中 | 论文说明 inexact 下界使用 Gurobi best bound |
| `src/benders.py` | `_gamma_for_iteration` | 算法策略，不是原模型 | 中 | 论文强调最终收敛对应 `Gamma^tar` |
| `src/policies.py` | `RLInspiredGapPolicy` | 不改变模型 | 低 | 论文中作为求解策略描述，不写入数学模型 |
| `tests/test_core.py` | 全部 | 核心验证通过但覆盖小 | 中 | 增加多 seed、多 Gamma、fallback、回归测试 |

## 4. 修改建议优先级

### P0：投稿前必须处理

- 若实验使用完整鲁棒模型，确保 `max_scenarios` 不触发 fallback。
- 若使用 fallback，论文必须明确写成近似/启发式候选场景方法。

### P1：强烈建议处理

- 增加 `scenario_mode` 元数据，区分完整枚举和候选场景。
- 在论文算法章节说明 inexact Benders 使用 Gurobi best bound 更新下界。
- 在论文算法章节说明 adaptive Gamma 只用于割生成加速，最终收敛对应 `Gamma^tar`。

### P2：建议处理

- 增加更多测试实例和数值回归测试。
- 在代码注释中说明 Benders cut 的对偶符号来源。
- 在论文模型中显式写 `theta >= 0`。

## 5. 是否影响论文投稿

当前实现可以作为论文原型代码和小中规模实验基础。核心模型一致性没有发现致命问题。

对投稿影响最大的点是 `candidate_budget_scenarios`：它是工程上合理的扩展接口，但不是论文给定预算型不确定性集合的完整精确求解。如果投稿实验需要严格支撑“预算鲁棒最优解”，应只报告完整枚举或开发精确的鲁棒分离子问题。

若论文表述改为：

- 小中规模：完整场景枚举，精确验证；
- 大规模：候选场景或自适应策略作为启发式加速；

则当前代码方向是可以支撑投稿准备的。
