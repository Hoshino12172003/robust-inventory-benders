# Cut-Strengthened Joint V3 Validation Protocol

## Scope and isolation

本 validation 的唯一目的，是检验 development 阶段冻结的 `joint_v1_core_point_strengthened` 能否在独立实例上泛化。它只比较冻结的 Joint V1 `proposed_joint_rho025_050` 与冻结的 core-only 候选，不恢复 secondary-only、full V3、MP-only 或任何新候选。

Development seeds 75–79 仅用于候选筛选。Validation seeds 固定为 80–89，二者结果不得合并进行正式统计推断。Final seeds 90–109 保持封存，不用于 validation、调试、参数修改、smoke test 或单元测试。本阶段不形成论文最终效应量，也不决定管理敏感性或公平模型结论。

候选来源固定为 `experiments/configs/selected_cut_strengthened_joint_v3_candidate.yaml`，其 SHA256 必须为 `7E8AAF39DE8C100B4CE9B46256A074FBD324B07DDC347D256494ED070D4E0EB6`。Validation 不允许修改任何模型、实例生成逻辑、算法开关、容差或时间限制。

## Configurations and execution isolation

两个配置各有 10 seeds × 2 methods = 20 runs：

- `cut_strengthened_joint_v3_validation_medium_large.yaml`: medium-large，600 秒，最多 10000 轮；
- `cut_strengthened_joint_v3_validation_large.yaml`: large，1800 秒，最多 20000 轮。

除 experiment name、seed、输出目录、候选集合及 validation 阶段锁定字段外，每个 validation 配置必须与相应 development 配置一致。候选集合缩减为 V1 与 core-only 是 development 决策直接要求的唯一方法变化。

运行必须使用与 development 相同的软件、硬件和 Gurobi 线程策略。代码不自动并行执行多个 Gurobi 实例，也不新增线程覆盖。若实际硬件或求解器环境不同，必须记录并停止跨阶段运行时间解释。

Validation 输出分别写入 `experiments/results_cut_v3/validation_medium_large` 与 `experiments/results_cut_v3/validation_large`。Experiment name、output directory 和稳定 run key 均与 development 分离，因此 `--resume` 只能读取相应 validation 目录，不能读取 development 或 final 结果。

## Frozen algorithm settings

两种方法均使用 `robust_dual_milp`、Gamma target 2、`gamma_schedule: [2]`、关闭 Gamma continuation、旧 cut selection、旧 secondary generation、adaptive gap 和自动并行。

V1 保持 `precision_policy: joint_error_budget`、`rho_M=0.25`、`rho_S=0.50`、MP gap `[0.0001, 0.02]`、SP gap `[0.0001, 0.05]`、单调精度收紧、`cut_strengthening_policy: none` 和每轮最多一条割。

候选与 V1 使用相同精度控制，唯一算法差异是 `cut_strengthening_policy: core_point`。核心点参数完全继承冻结候选；stall secondary、workload-aware V2 和所有旧 secondary 机制保持关闭，每轮最多一条割。

## Outcomes and paired comparisons

同一规模、同一 seed 的 V1 与候选构成配对实例。所有聚合均以 10 个预注册 seed 为分母，不删除超时、失败或不利实例，也不以成功运行子集替代主分析。

主要指标是 large 上相对 V1 的 mean PAR-2 改善。PAR-2 定义保持现有管线语义：达到容差时等于实际算法 runtime；否则等于 `2 × time_limit`。`solved_to_tolerance` 仅在存在 objective、final gap 有限且不高于配置 `tol=1e-4` 时为真。

次要指标包括：medium-large mean PAR-2 非退化、两个规模的 solved rate、large mean iterations、配对 seed 的 PAR-2 胜负数、配对 seed 的迭代数胜负数、成功运行 runtime，以及所有未求解运行的 final optimality gap。运行时间不包含文件写入或管理事后评价时间。

Timeout 保留实际 LB、UB、final gap、iterations 和 runtime，并以 PAR-2 计入。失败或缺失运行视为未求解，以 `2 × time_limit` 计入 PAR-2，同时使正确性完整性门槛失败。Timeout 的实际有限 final gap 必须报告；失败或缺失 final gap 保持缺失，不得填零。迭代数使用每个完整运行记录的实际值；缺失迭代记录不得插补，并触发完整性失败。

配对 PAR-2 比较直接比较同一 seed 的两种 PAR-2；相等计为候选“不高于 V1”。配对迭代比较要求两边都有有效实际迭代数，严格较低才计为候选胜出。

## Correctness gate

性能判断前必须全部满足：20/20 运行记录完整、无 failed、状态解释正确、所有运行保持有效 LB、所有可更新运行保持有效 UB、LB 非递减、UB 非递增、MP/SP requested gaps 非递增、候选核心点割保持对偶可行和当前点保持、辅助 LP 不更新 UB、无伪割、终止核验语义不变。任一项失败即 validation fail，并选择 V1；不得在 seeds 80–89 上修复参数后重新验证。

## Pre-registered decision rule

Validation **pass** 必须同时满足：

1. 正确性门槛全部通过；
2. medium-large 候选 solved rate 不低于 V1，mean PAR-2 不高于 V1 的 103%；
3. large 候选 solved rate 不低于 V1；
4. large mean PAR-2 至少比 V1 降低 7.5%；
5. large mean iterations 至少比 V1 降低 15%；
6. large 10 个配对 seed 中至少 6 个候选 PAR-2 不高于 V1；
7. large 10 个配对 seed 中至少 6 个候选 iterations 严格低于 V1。

Validation **fail** 是以下任一情况：正确性门槛失败；medium-large solved rate 低于 V1；medium-large mean PAR-2 高于 V1 的 103%；large solved rate 低于 V1；或 large mean PAR-2 未达到 7.5% 改善。失败后选择 V1，禁止在同一 validation 数据上修改候选并重新验证。

Validation **inconclusive** 指主要安全与 PAR-2 条件通过，但一个或多个 large 次要迭代/配对门槛未通过，或完整运行环境不可比。结果不确定时不采用 V3 作为论文算法，保持 V1；任何新假设必须使用全新、另行预注册且未访问的数据，不能复用 seeds 80–89。

该两方法协议不计算三方法平均名次，也不声称满足原预留文本中涉及未纳入方法的比较条件。此限制在观察 validation 结果前冻结，不能事后改变。

## Commands reserved for later execution

本 PR 不执行以下命令。协议验收后，后续正式运行顺序为先 medium-large，再 large：

```powershell
python -m src.experiment_suite `
  --config experiments/configs/cut_strengthened_joint_v3_validation_medium_large.yaml `
  --resume
```

仅在 medium-large 正确性验收完成后：

```powershell
python -m src.experiment_suite `
  --config experiments/configs/cut_strengthened_joint_v3_validation_large.yaml `
  --resume
```

本协议 PR 只允许 `--dry-run`、静态审计和单元测试。不得运行 validation、final、管理敏感性或公平实验。
