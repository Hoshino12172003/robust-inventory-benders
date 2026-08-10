---
source_zip_sha256: 4F8CEC3F9EAF69B053AE3DBAE6C29D5AAEF7DAAD87C49A798039DBCF9FADD783
run_git_commit: 797caafd12c006e85bc3394b01905bbfb137b0a9
config_sha256: A377D5B040FED160B323B58D42D9FFD1DE57E52F6C64D2050D6667E47DCA9334
protocol_sha256: 4C76B5C7A02E245174BE02B6FCEBBCD744EB6B684A1F0CA71D05964EB1F1A32F
generation_schema: fairness_high_gamma_attempt2_final_reporting_v1
---

# Response to reviewer comments: high Gamma and external benchmark

## A. Dependence on very small Gamma and possible cycling of the rolling window

We added a pre-registered small-scale stress test with Gamma=2, 3, and 4, corresponding to 211, 1,351, and 6,196 complete scenarios. Hybrid certified all 15 cells. We also clarified the mechanism: the rolling pool governs only candidate-proposal memory, whereas committed master blocks satisfy $S_{k+1}=S_k \cup \{\hat s_k\}$. The audit verifies append-only scenario and cut SHA ledgers, reports evictions, rediscoveries, and duplicates, and confirms final exact separation in every cell. Our revised claim is deliberately limited: in the 4-warehouse by 4-product by 5-region stress test, we observed no cycling or certification failure caused by pool eviction. We do not claim validity for arbitrary Gamma or equate finite theoretical convergence with practical scalability.

## B. Missing external general-purpose solver benchmark

We added a general-purpose solver benchmark based on the direct deterministic equivalent formulation, with shared first-stage variables and explicit recourse blocks for every scenario. Gurobi's built-in Benders strategy was disabled. The direct formulation certified all Gamma=2 and Gamma=3 cells. At Gamma=4, all five runs reached the 1,800 s limit without an incumbent as model size grew sharply; these cells are retained in the pre-registered comparison with a 3,600 s PAR-2 penalty. We do not report fabricated objectives or gaps for those runs and do not claim that Gurobi represents all industrial algorithms.

## C. Conservatism of C_anchor and the economic meaning of rho

We now state the anchor relationship as $C^* \in [LB,C_{anchor}]$. The anchor is a certified upper bound, not an exact optimum. Across all 15 baseline cells, the maximum certified relative gap was 0.00009349, below $10^{-4}$. This residual conservatism is small compared with $\rho=0.025$; it is reported explicitly in the anchor-quality table and is not used to overstate economic precision.
