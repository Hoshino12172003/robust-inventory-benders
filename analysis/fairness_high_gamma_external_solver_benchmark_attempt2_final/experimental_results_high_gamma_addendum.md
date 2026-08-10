---
source_zip_sha256: 4F8CEC3F9EAF69B053AE3DBAE6C29D5AAEF7DAAD87C49A798039DBCF9FADD783
run_git_commit: 797caafd12c006e85bc3394b01905bbfb137b0a9
config_sha256: A377D5B040FED160B323B58D42D9FFD1DE57E52F6C64D2050D6667E47DCA9334
protocol_sha256: 4C76B5C7A02E245174BE02B6FCEBBCD744EB6B684A1F0CA71D05964EB1F1A32F
generation_schema: fairness_high_gamma_attempt2_final_reporting_v1
---

# High-Gamma stress test and external solver benchmark

This independent subsection is intended for insertion after the frozen Final Holdout and Gamma-sensitivity sections. It uses only the High-Gamma Attempt 2 sample and does not alter or pool the Final Holdout observations.



We conducted a controlled stress test on instances with 4 warehouses, 4 products, and 5 regions. This experiment is separate from the Final Holdout and from the cross-scale Gamma=0/1/2 sensitivity analysis. With 20 uncertain demand components, the complete scenario set grows from 211 at Gamma=2 to 1,351 at Gamma=3 and 6,196 at Gamma=4. The external comparator is described as a general-purpose solver benchmark based on the direct deterministic equivalent formulation.

## Certification and computational evidence

Hybrid certified all 15 cells and completed final exact separation and exhaustive post-evaluation in every cell. The direct formulation certified 5/5 cells at Gamma=2 and 5/5 at Gamma=3. At Gamma=4, all five direct runs reached the 1,800 s time limit without an incumbent. These cells therefore enter the analysis through their time-limit classification and 3,600 s PAR-2 penalty; no direct objective value or optimality gap is reported. Among jointly certified cells, the maximum absolute differences in T were 5.079e-07 at Gamma=2 and 1.511e-06 at Gamma=3.

## Rolling proposals versus committed master blocks

The rolling pool limits only proposal memory. It does not delete scenario blocks already committed to the master:

$$
S_{k+1} = S_k \cup \{\hat s_k\}.
$$

At Gamma=4, the mean counts of proposal evictions, rediscoveries, and duplicate proposals were 24.8, 11.8, and 0.0, respectively. All committed scenario and certified-cut ledgers were append-only, and every Hybrid cell ended with exact certification. Thus, in this small-scale stress test, we observed no cycling or certification failure caused by proposal-pool eviction. This empirical result is not a proof of computational scalability for arbitrary Gamma, nor a claim that Hybrid dominates all state-of-the-art methods.

## Anchor quality and interpretation

For each seed-Gamma cell, `C_anchor` is the certified upper bound from the matching robust-cost baseline. The valid statement is $C^* \in [LB,C_{anchor}]$, not $C_{anchor}=C^*$. The maximum relative certified anchor gap was 0.00009349, below the frozen $10^{-4}$ tolerance and small relative to $\rho=0.025$.

## Boundary of the conclusion

The evidence is limited to five pre-registered seeds and the stated small-scale Gamma=2/3/4 design. A Gamma=4 direct time limit without an incumbent is not a certified solution, and Gurobi is not used as a proxy for every industrial algorithm.
