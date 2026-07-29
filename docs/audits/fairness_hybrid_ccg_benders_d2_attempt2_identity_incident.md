# Hybrid D2 Attempt 2 identity-interface incident

## Disposition

D2 Attempt 2 is frozen as `execution_incomplete`. It is not a scientific D2 result and must never be resumed, overwritten, or imported. The original evidence remains read-only at `E:\rfhd2\experiments\results_fairness_hybrid_ccg_benders\controlled_d2_large_seeds160_162_rhos0_001_010`.

## Read-only identity and observed state

- Git commit: `EE4F91986F7DAC1CA7BB42CC9EDA8803791C07E6`
- Execution attempt: 2
- Config file SHA256: `AE449C3E1551532AA772E1E51F4348860FC16A7CFE9970D6E2A9477F2E2DBFF1`
- Resolved config file SHA256: `40D670CAA6E3871BC4A33BFFD8536E6273282D0FB65A1BCCD279F6EC65804E19`
- Protocol SHA256: `D1DD2DC7417204CDB0B9A70986B975EC621D400CF8384E6EC026ED1A90D9367B`
- Candidate SHA256: `8AF2687A4340D03BE44C5A73FFD3BE1F1E015F5447D2B56FD9A8919049D46BA0`
- Accessed seeds: exactly 160
- Baseline: complete and `certified_robust_optimal`
- First frontier: complete `implementation_error`, reason `incomplete_expected_run_identity`
- No frontier algorithm checkpoint or post-evaluation exists; the failure occurred during T=1 initial-UB identity validation before Hybrid solver construction.

The D2 runner supplied a full run identity with four additional fields to the frozen D1 T=1 upper-bound interface, whose contract requires exactly ten fields. The hotfix projects the full D2 identity onto that exact contract. The full identity remains locked in the D2 manifest and checkpoint identities.

## Formal output file SHA256

| Relative path | SHA256 |
| --- | --- |
| `instances/160.json` | `352159DD65C8D0240C124839CAB15FDEF8DFE9D984435F45AEFFA92CDBABB410` |
| `manifest.json` | `1A13ED4E15B4555EA960CE44A426D9B1DE7D526570B913CD294E27D55FBEE7F2` |
| `resolved_config.yaml` | `40D670CAA6E3871BC4A33BFFD8536E6273282D0FB65A1BCCD279F6EC65804E19` |
| `results.csv` | `7C58A344C9400102D03B492A20A47CBC5B39D5B18679E408C5E39D877938C23A` |
| `runs/r_53bea95ab3aafd83bf25c8d8/baseline_checkpoint.json` | `25B5382FB1741333DAD17A76F7D7809F8E725A6278CA4EDF4FFBA47A3D2BCA06` |
| `runs/r_53bea95ab3aafd83bf25c8d8/run.json` | `298E9A65AD83AFAAEED13949A0F050ABD4F62064A9C35932B5D59131E3E27DAC` |
| `runs/r_53bea95ab3aafd83bf25c8d8/status.json` | `686654360F2061C3C514D142D87EAE0EE5AA5C70714273F2408A71941800CECF` |
| `runs/r_6b87dd4d243020ca52435be5/run.json` | `3A780C83F8C9563C40923B67E0E31E375A0AB2763DD6867552F0E691CCA69A12` |
| `runs/r_6b87dd4d243020ca52435be5/status.json` | `8F5CE31039274A21F346C38A363D4DD0C82D2062C4F8E452A94DDF6CE244EACF` |
| `summary.csv` | `24462D84D78C539478EA6E43467AB3F82FD5819B4160476BCF834C459B7E16B8` |

## Attempt 3 isolation

Attempt 3 uses new canonical run keys and the new output directory `experiments/results_fairness_hybrid_ccg_benders/controlled_d2_a3_large_s160_162_r0_001_010`. `previous_attempt_results_reused=false`; no Attempt 2 instance, baseline, anchor, run, checkpoint, manifest, result, or aggregate may be read or reused.
