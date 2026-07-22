# PR #31 mathematical pre-merge acceptance

## Scope and disposition

PR #31 was reviewed against base `bfff1a9e7bb9cc3cae277a686beac8659a214b62`
and original head `8dbd40add34a1b319f08458cea4bc9c331bc99b4`. The review found three
merge-blocking implementation defects: the baseline cost anchor was read from
the ambiguous `objective` field, the Farkas normalization allowed the zero
ray, and separation certification did not whitelist trustworthy termination
statuses. These defects were corrected without changing the fairness metric,
rho grid, seed allocation, V3 parameters, recourse model, robust dual, or
uncertainty set.

Subject to the recorded checks below, the disposition is `approve_for_merge`.
The PR remains Draft for the requested human Ready/merge action. No formal
development run was executed.

## Model semantics

The primary problem minimizes `T` over a common first-stage `(y,x)` and one
adaptive recourse `(q(z),u(z),e(z))` per scenario. The same recourse variables
simultaneously satisfy original demand, inventory and service constraints,
the robust total-cost cap, and every applicable regional shortage-rate cap.
`T` is bounded in `[0,1]`. Regions with zero demand are excluded. WGap and WWD
are post-solve reporting metrics, not objectives, so the formulation does not
seek equality by lowering well-served regions.

The cost expression is exactly fixed opening plus inventory plus transport,
shortage and service-violation costs. No term is omitted or counted twice.

## Certified cost anchor and rho=0

For each seed, `C_anchor` is the full-precision frozen-V3
`SolveResult.upper_bound`, accepted only with `status=optimal`, `valid_UB=true`,
and final gap at most `tol`. Its decimal and IEEE-754 hexadecimal values,
baseline run key, Git commit, config SHA256, candidate SHA256 and anchor SHA256
are persisted. Lower bounds, midpoints, master objectives, rounded summaries,
and single-scenario costs are prohibited. Every rho point for the seed reuses
that one anchor and computes `B_rho=(1+rho)*C_anchor`.

Diagnostic fair-best fixes the diagnostic first-stage decision and removes
recourse degeneracy. Integrated rho=0 may instead choose another first stage
inside the same cost boundary. Therefore it may achieve a cost-neutral fair
reconfiguration and is not required to equal fixed-x diagnostic fair-best.
When x is explicitly fixed, the same shared-cap logic must agree with the
fixed-x extensive form. Both cases have hand-built regression coverage.

## Independent Farkas derivation

Writing recourse feasibility as `A v <= b`, `v>=0`, with nonnegative
multipliers `(a,b,c,k,ell)` for demand, supply, service, cost and regional
fairness rows gives:

- shipment column: `-a[r,j] + b[i,j] + k*cq[i,r,j] >= 0`;
- shortage column: `-a[r,j] + c[j] + k*cu[r,j] + ell[r] >= 0`;
- service-violation column: `-c[j] + k*ce[j] >= 0`.

The ray cut is

`-d*a + x*b + service_rhs*c + (B-first_stage)*k + T*D*ell >= 0`.

Thus the master coefficients are `-k*fixed_cost` for y,
`b-k*inventory_cost` for x, positive `D*ell` for T, and the remaining demand,
service and budget terms form the constant. The inequality direction in the
master is `>=0`. The equality normalization `sum(a,b,c,k,ell)=1` excludes the
zero ray without deleting a nonzero ray because the cone is homogeneous. It
also proves every multiplier lies in `[0,1]`, making the binary-continuous
McCormick bounds exact rather than empirical Big-M values.

## Separation and Benders certification

An incumbent with positive violation may generate a valid cut. Robust
feasibility is certified only when an `optimal` or normal `time_limit` solve
has an objective bound no greater than the frozen feasibility tolerance.
Interrupted, numeric, suboptimal, infeasible, unbounded and unknown exits never
certify. The absence of a found violation is not a certificate.

The master objective bound remains the lower bound. A candidate T becomes an
upper bound only after both cost and fairness feasibility are certified over
the uncertainty set. Final termination requires the relative optimization gap,
an optimal final master and zero-gap separation certification. Time-limit and
uncertified records are not marked successful. Post-evaluation independently
checks that realized worst normalized shortage does not exceed T.

## Extensive-form independence

The tiny-instance oracle enumerates uncertainty-set extreme points, constructs
an explicit recourse block for every scenario, and directly adds the shared
cost and regional fairness constraints. It does not call the Farkas cut
generator. Tests cover one region, symmetric regions, a material gap, zero
demand, rho=0, a large allowance, an infeasible budget, distinct cost/fairness
worst scenarios, Gamma 0/2, recourse degeneracy, fixed-x shared caps, and
cost-neutral first-stage reconfiguration.

## Development decision and recovery

Each scale contains seeds 120--129, ten frozen-V3 baselines and five rho points
per seed, for 60 tasks. The primary endpoint is paired robust-minimum-fill-rate
improvement over rho=0. Correctness is mandatory, each scale needs at least
80% solved frontier tasks, and material improvement requires at least 4/10
instances at one positive rho to improve by at least 0.05. The smallest
eligible positive rho is selected. Failure yields
`stop_no_material_improvement`; insufficient completion without correctness
failure yields `development_inconclusive`.

Only experiment name, phase, reserved validation seeds 130--139 and output
directory may change in a later validation protocol. Model, uncertainty set,
generator, rho grid, tolerances, time limits, V3 parameters, success/PAR-2
definitions, certification and thresholds are frozen.

The runner uses stable baseline/frontier keys, a single-writer lock, atomic run
records, generic and fairness-specific manifests, deterministic result/summary
rebuilds, and identity checks for config, code, candidate, baseline, anchor and
rho. Only certified successes are skipped by resume. Summary fields distinguish
optimal, time-limit, uncertified, invalid post-evaluation and infeasible tasks.

## Validation evidence

- fairness development audit: 72/72 passed;
- regional fairness diagnostic audit: 63/63 passed;
- V3 Final protocol audit: 41/41 passed;
- V3 Validation protocol audit: 36/36 passed;
- original V3 audit: 49/49 passed;
- targeted fairness/model/protocol tests: 45 passed;
- complete collected set: 377 unique nodeids;
- partition: group 1 = 203, group 2 = 173, isolated CLI = 1;
- group 1: 203/203 passed;
- group 2: 173/173 passed (executed in short sub-batches because of the desktop
  host's external command window);
- isolated CLI dry-run: 1/1 passed.

The exact partition is stored in `docs/audits/pr31_pytest_nodeids.txt`; the
three groups are disjoint and their union equals all 377 collected nodeids.

Dry-runs report 10 baselines plus 50 frontier tasks for each scale, 1,831 and
4,657 scenarios respectively, no audit errors, no instance generation and no
solver call. Formal output directories were absent before and after dry-run.

## Frozen artifact hash transition

| Artifact | Original PR hash | Accepted hash |
|---|---|---|
| medium-large config | `71656984A67AFA15CFEF677AA8E70A72CC6D65163486E9527BC81554C317C62D` | `E52FAA5EB93E43C6C6B611E47239DD98378DBF039BD9ECA3C49FF7B2D5AE59E1` |
| large config | `E984CA8B6ACE03CD46BB824C163799BFEC229A288952B58678297C5CD8BC067E` | `358629EBB7BC15371B8D2295C6B7E468E9E872FA1AF2E9BA801982057096925B` |
| development protocol | `BF3891775B19BC210E0E03A390C7CE2D169C8E0F2C658C1FAB570F237AD1B4D8` | `BBA9973BB8A4D660202FA3D99DBDC957DB00A8A336632667A09B9F47974E22B5` |
| mathematical model | `166265DED7A20AE6CEFB8548A0D27343BBA6FBE0646293956E7AEF07C8FB1FBA` | `0EAC981F815709F37A7ACD5B5B70A604888C522DD86F58B4E55F18F40FCF4413` |

Frozen V3, Final decision, diagnostic decision, base recourse, robust-dual
subproblem, uncertainty set and `src/benders.py` remain byte-identical to the
base. No seeds 120--159 were generated or solved.
