# Hybrid final holdout reconciliation decision

```yaml
decision: approve_final_holdout_after_reporting_reconciliation
scientific_solution_valid: true
optimization_rerun_required: false
```

The formal ZIP passed 120/120 run checks, 100/100 exact-certification checks, and 13,050/13,050 post-evaluation chunk SHA checks. The archive SHA remained `BD253A4DE967143B8CD88E7DEADCB821D6062208CB71664B900F9BD8EBDF3839` before and after the read-only audit.

The sole crossing is `large`, seed 172, rho 0.10 (`r_45f40e77afd919415d895390`). The implementation stores `max(historical_lower_bound, current_master_ObjBound)` while the persistent master receives additional scenario blocks. Gurobi's numerically solved objective decreased by 2.0694404762322538e-05, within the frozen 1e-4 tolerance. The original gap function clipped the resulting negative numerator to zero without separately recording the crossing.

Scientific validity does not rely on that historical maximum. In the final iteration, the current persistent scenario master was optimal and its solver best bound equaled both its incumbent and the reported UB (0.2003226594124341). Complete exact separation was optimal with objective bound -0 and certified that same incumbent robust feasible. Thus the current master bound supplies the relaxation lower bound and exact separation supplies full robust feasibility; together they prove final optimality under the frozen protocol tolerance.

The derived reports therefore preserve `historical_recorded_lower_bound` as a trajectory field and use `final_master_solver_best_bound` as `reported_certification_lower_bound`. No run, checkpoint, instance, post-evaluation artifact, or source ZIP was changed, and no optimization or Gurobi call was made.
