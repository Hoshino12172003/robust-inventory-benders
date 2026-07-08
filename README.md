# Paper Code

预算鲁棒库存分配与自适应不精确 Benders 分解原型。

## Quick Start

```powershell
python -m src.cli generate --config configs/default.yaml
python -m src.cli solve --method adaptive_gap_gamma_benders --instance data/processed/instance.json
python -m src.cli experiment --config configs/experiment.yaml
```

## Methods

- `monolithic`: 枚举预算型需求场景的单体鲁棒模型，用于小规模校验。
- `standard_benders`: 固定目标 Gamma、精确主问题的 Benders 分解。
- `inexact_benders`: 固定目标 Gamma、固定主问题 MIPGap 的不精确 Benders。
- `adaptive_gap_gamma_benders`: 参考 RL-iGBD 的离散动作思想，按 Benders 进展自适应选择主问题 MIPGap，并将 Gamma 从小预算推进到目标预算。

## Scenario enumeration modes

1. `exact_scenarios: true`

This mode fully enumerates the budgeted uncertainty set `U(Gamma)`. It is the default setting for paper experiments and exact benchmarks. If the full scenario count exceeds `max_scenarios`, the program raises an error instead of silently switching to an approximation.

```yaml
robust:
  exact_scenarios: true
```

2. `exact_scenarios: false`

This mode allows `candidate_budget_scenarios` when full enumeration exceeds `max_scenarios`. It is a heuristic / approximate mode for larger exploratory runs. Candidate mode must not be reported as an exact robust optimum in paper experiments.

```yaml
robust:
  exact_scenarios: false
```

3. `monolithic`

The monolithic method is intended for small-scale exact validation. It uses full scenario enumeration and should be treated as an exact benchmark, not as a heuristic candidate-scenario solver.

```yaml
robust:
  gamma_target: 2
  gamma_schedule: [0, 1, 2]
  max_scenarios: 5000
  exact_scenarios: true
```

Solve result metadata records the target scenario mode, exact-scenario flag, number of scenarios used, estimated full scenario count, `max_scenarios`, and whether any heuristic candidate scenarios were used.

## Benders subproblem modes

The default paper-experiment setting is:

```yaml
algorithm:
  subproblem_mode: robust_dual_milp
```

- `robust_dual_milp`: solves the robust dual MILP subproblem derived from the paper model. This directly evaluates `Q^R(x; Gamma)` and is the recommended default for main experiments.
- `scenario_enumeration`: evaluates recourse over full or candidate demand scenarios. This mode remains available for small-scale exact benchmarks, validation against `robust_dual_milp`, and heuristic large-scale experiments.

## RL-iGBD Reference

`E:/浏览器文件/RL-iGBD-main/` 中的源码确认是论文 *Learning to control inexact Benders decomposition via reinforcement learning* 的实现。当前项目借鉴了其中的策略接口思想：

- 状态记录迭代数、Benders gap、log gap、gap 改善、上下界。
- 动作采用 11 个离散等级，并映射到当前 Benders gap 以下的主问题 `MIPGap`。
- 第一版不训练 PPO，但 `src/policies.py` 已保留策略接口，后续可接入 RL 策略。

## Structure

- `src/instance.py`: 合成库存算例生成与 JSON 读写。
- `src/scenarios.py`: 预算型需求扰动场景枚举与候选场景生成。
- `src/subproblem.py`: 给定库存和需求场景的二阶段配送 LP 与 Benders 割。
- `src/monolithic.py`: 单体鲁棒模型。
- `src/benders.py`: 标准、不精确、自适应 Benders 主循环。
- `src/policies.py`: MIPGap 策略接口，预留 RL 扩展。
- `src/experiment.py`: 多方法、多随机种子的实验对比与 CSV 输出。
