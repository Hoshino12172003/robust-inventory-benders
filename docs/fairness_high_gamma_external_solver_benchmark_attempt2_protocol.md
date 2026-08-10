# Certified fairness high-Gamma external benchmark Attempt 2 protocol

## Incident basis and scientific scope

Attempt 1 is frozen under archive SHA256 `17ABAC73952D7A6C62EFAC313EC5A3771D904750BEE917BC51FBA3F1C76FDD47` as `baseline_gamma_identity_defect`. Its Gamma 2 subset remains scientifically usable, but its Gamma 3 and 4 cells are not scientifically usable because the baseline solver received Gamma 2. No Attempt 1 instance, baseline, anchor, checkpoint, frontier, post-evaluation, result, or summary may be read or reused by Attempt 2.

Attempt 2 repeats the complete preregistered matrix from scratch. It is not a selective rerun and is not a tuning stage.

## Frozen matrix

- stage: `HIGH_GAMMA_EXTERNAL_BENCHMARK`
- scale: `small`, with 4 warehouses, 4 products, and 5 regions
- seeds: 185, 186, 187, 188, 189
- Gamma: 2, 3, 4
- rho: 0.025
- execution attempt: 2
- baseline: 15 independent Gamma-specific runs
- Hybrid frontier: 15 runs using `certified_hybrid_scenario_benders_fairness`
- direct frontier: 15 runs using `gurobi_direct_extensive_form`
- total: 45 runs

For 20 demand components, the complete scenario counts are 211, 1,351, and 6,196 for Gamma 2, 3, and 4. Each seed-Gamma cell has one fresh instance archive, one fresh Gamma-specific baseline, and one fresh certified anchor shared only by the two paired frontiers in that cell.

## Baseline Gamma identity and T=1 lifting gate

After applying all frozen selected and candidate parameters, the runner overwrites the final baseline solver configuration with the requested Gamma in every authoritative location: `gamma`, `robust.gamma_target`, `robust.gamma_schedule=[Gamma]`, disabled Gamma continuation, and `robust.max_scenarios=|U_Gamma|`. The baseline result must report the same `gamma_target`, final `active_gamma`, singleton `gamma_schedule`, and scenario count. The run, checkpoint, anchor, and manifest preserve requested Gamma, target Gamma, active Gamma, active policy, schedule, scenario count, and canonical instance SHA.

Before either frontier model is created, the baseline first-stage solution is lifted with `T=1` and evaluated over every scenario in the requested uncertainty set using the same anchor, rho, cost budget, Threads=1, Seed=0, FeasibilityTol=1e-7, and 30-second per-scenario limit. Every scenario must have a feasible policy, all acceptance evidence must pass, and the actual robust cost must not exceed `(1+rho)C_anchor` within 1e-7. The checkpointed lifting evaluation is independent of frontier post-evaluation and must resume without repeated scenarios. Any mismatch or failed lifting blocks both frontiers fail closed.

## Frozen methods and solver settings

The Hybrid mathematical model, fairness definition, budgeted uncertainty set, complete scenario recourse blocks, certified Farkas separation, proposal pool, final exact objective-bound certification, and scientific branching are unchanged. Committed scenario blocks and certified Farkas cuts are append-only; proposal eviction cannot remove master constraints; robust optimal status requires final exact certification and valid complete post-evaluation.

The direct method remains the complete deterministic equivalent with shared first-stage variables and full scenario-specific recourse. It does not use Hybrid separation, cache, candidate pools, Farkas cuts, or Gurobi built-in Benders.

All baseline and frontier models use Threads=1, solver Seed=0, FeasibilityTol=1e-7, and an 1,800-second algorithm limit. Post-evaluation uses 30 seconds per scenario and chunk size 25. PAR-2 is based only on algorithm runtime, with 3,600 seconds for uncertified tasks. Rho remains 0.025.

## Identity, recovery, and authorization

Attempt 2 writes only to `experiments/results_fh_ext/hg2`, requires strict `--resume`, and does not support overwrite. The first formal output root must not exist. Instance, baseline, T=1 lifting, anchor, algorithm, post-evaluation, manifest, result, and summary identities fail closed on drift or corruption. Attempt 1 paths are prohibited inputs.

Formal execution requires a separately reviewed Attempt 2 authorization file. Authorization must be checked before output creation, instance generation, solver configuration, or `gurobipy` import. Approval of implementation alone does not authorize the run.
