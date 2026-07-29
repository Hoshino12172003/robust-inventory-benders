# Hybrid fairness D1 read-only decision

The immutable source is `fairness_hybrid_ccg_benders_d1_large_seed160_rho0_results.zip`, SHA-256 `7E89115E3BE325C9A37C31D28D32EA80EEA95F09528DBC8AEDA833EF0129A4A9`. A direct ZIP audit passed CRC, JSON/CSV parsing, all run/checkpoint/chunk hashes, all 4,657 post-evaluation scenarios, and the complete identity chain. The source archive is not committed or modified.

The scenario master began with 13 complete recourse blocks. Iterations 1--8 each appended exactly one independently certified scenario and one certified Farkas cut; no scenario or cut was removed or replaced. The recorded master solver bound increased monotonically from 0.005034684692266179 to 0.0750160357508425. Iteration 9 appended nothing, ran complete exact separation, obtained a valid separation objective bound of negative zero, and only then set robust certification. Post-evaluation followed the algorithm certificate and cannot replace it.

The frontier finished with LB=UB=T=0.0750160357508425, gap zero, 21 scenario blocks, 8 Farkas cuts, 9 iterations, and robust minimum fill rate 0.9249839642491575. Its algorithm runtime was 109.36750799999572 seconds, post-evaluation solver/wall times were 8.734002351760864/61.325600400567055 seconds, total wall time was 207.18595419987105 seconds, and PAR-2 was 109.36750799999572 seconds. The baseline runtime was 722.7178315999918 seconds. All 4,657 post-evaluation scenarios were valid; maximum accepted residual was 7.275957614183426e-12.

## Legacy post-evaluation field

PR #46's generic post-evaluation implementation hard-coded `execution_attempt: 4`. Code inspection proves that this value labels the fourth-generation post-evaluation pipeline; it is not consulted to locate run artifacts. The formal D1 attempt remains 1 in the manifest and canonical run key. The post-evaluation identity itself stores that canonical attempt-1 run key plus the exact Git commit, resolved config SHA, anchor SHA, solution SHA, scenario-order SHA, chunk size, and time limit; the canonical identity hash locks every chunk and its index. Thus the mislabeled field cannot import or resume another run. D2 replaces it with separate `run_execution_attempt` and `post_evaluation_pipeline_generation` fields. The D1 archive remains untouched.

`D1_review.zip` is a seven-entry convenience copy of the manifest/results/run/status material used for human review. It is not referenced by the manifest, run key, checkpoints, or scientific status; it is excluded from both formal evidence enumeration and recursive identity. Its SHA-256 is `6BD61236C041027A52B4498274FC0FA1C83222C65674A664B5BDC9BE27A911B1`.

Decision: `approve_for_d2_controlled_large_expansion`. D1 is a single controlled development observation, not statistical evidence.
