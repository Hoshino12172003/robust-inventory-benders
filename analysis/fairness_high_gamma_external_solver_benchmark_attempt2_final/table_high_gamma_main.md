---
source_zip_sha256: 4F8CEC3F9EAF69B053AE3DBAE6C29D5AAEF7DAAD87C49A798039DBCF9FADD783
run_git_commit: 797caafd12c006e85bc3394b01905bbfb137b0a9
config_sha256: A377D5B040FED160B323B58D42D9FFD1DE57E52F6C64D2050D6667E47DCA9334
protocol_sha256: 4C76B5C7A02E245174BE02B6FCEBBCD744EB6B684A1F0CA71D05964EB1F1A32F
generation_schema: fairness_high_gamma_attempt2_final_reporting_v1
---

# High-Gamma stress test and external solver benchmark

| Gamma | Scenarios | Hybrid certified | Hybrid mean runtime (s) | Direct certified | Direct mean runtime (s) | Direct mean PAR-2 (s) | Ratio of means | Median paired ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 211 | 5/5 | 1.251 | 5/5 | 5.545 | 5.545 | 4.43 | 3.08 |
| 3 | 1351 | 5/5 | 2.685 | 5/5 | 230.578 | 230.578 | 85.87 | 86.13 |
| 4 | 6196 | 5/5 | 7.607 | 0/5 | 1800.090 | 3600.000 | 236.64 | 234.02 |
