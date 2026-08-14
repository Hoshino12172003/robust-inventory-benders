# Paired cross-scale Gurobi Direct delivery freeze

This document freezes the delivery identity of the medium-large and large paired Direct benchmark. It is documentation-only and does not change the runner, configuration, formal results, or paired Hybrid source evidence.

## Git identity

- Execution commit recorded in all ten formal `run.json` files: `0a8d3e71af4472be03763ab7720ce90fbf111f68`.
- Audited delivery HEAD pushed before Draft PR creation: `e520af1acc1e7bf4c5b8c3f1cf6be73e2be3bd19`.
- `e520af1` contains `0a8d3e7` plus the solver-free final-audit module and its test. It is the audited delivery HEAD, not the Git commit recorded during the Gurobi executions.
- Branch: `agent/gurobi-direct-cross-scale-paired`.
- Draft PR: <https://github.com/Hoshino12172003/robust-inventory-benders/pull/71>.

No rebase or run-core change is authorized after this freeze. Derived documentation or reporting tables may be added in later commits only if they preserve the execution and archive identities below.

## Archive identity

| Role | Archive | SHA256 |
|---|---|---|
| Direct medium-large result evidence | `gurobi_direct_cross_scale_paired_medium_large_results.zip` | `BF2D219D3EEDF232BE59E3483870059C2F5650B55A90ABE296FAB39C5343CF31` |
| Direct large result evidence | `gurobi_direct_cross_scale_paired_large_results.zip` | `4FA092A5FD91262AE577321CD574B55DED52AADF34B0C23410F6AA28F3F15674` |
| Frozen Hybrid source evidence | `fairness_hybrid_gamma_sensitivity_attempt3_results.zip` | `EE45A00AA341EE5EB2894DE43EE2F47022C27F1D29146FCFEC803236EF59DB6F` |

The two Direct result archives are external evidence and are not committed to Git. Each contains exactly five `run.json` files for seeds 180--184 of its named scale, the matching checkpoints and status records, plus the common manifest, resolved configuration, paired table, summary, and final audit. Archive inspection confirmed that all ten run records identify execution commit `0a8d3e71af4472be03763ab7720ce90fbf111f68`.

## Scientific outcome

- Direct: 10/10 `time_limit_uncertified` after the 1800-second limit.
- Direct incumbent count: 0/10.
- Paired frozen Hybrid source: 10/10 `certified_robust_optimal`.
- Hybrid reruns: 0.
- Baseline reruns: 0.
- Direct results without an incumbent have no reportable objective difference or certified raw-runtime ratio. Their reporting value is certification rate, PAR-2, and deterministic-equivalent model size.

## Per-seed identity chain

Every row is bound by the frozen Hybrid source archive SHA above. The complete Hybrid and baseline run keys remain embedded in each Direct `run.json`; the table records their compact manifest linkage through the Hybrid directory, instance identity, and anchor identity.

| Scale | Seed | Direct directory | Instance canonical SHA256 | Anchor SHA256 | Source Hybrid directory |
|---|---:|---|---|---|---|
| medium_large | 180 | `r_7ebe7b5538a2afade91e1917` | `72D9415391E1C9DCB865144A6CB0D3DAEEA7DAB6A9143660C30970A7AC69C15D` | `63737C9FFABB50CE6CA8BB7C812FFCFECD456E18209D9A1406C7581D06FA1B5F` | `r_1bebc8e77e5883bb97defc24` |
| medium_large | 181 | `r_2045953b25f20672a7b3173d` | `D6F53ED22EC87A8EBA9E5CAE3E3A01AFBA20FAB92DDCB0F7192E57142133FF9E` | `653A7846FD750BC16CD19B378B79AE76A33510DBBF5D7AE0DC60CFC5D705CA2B` | `r_a925dc4701348df230c3e45c` |
| medium_large | 182 | `r_55eb71479a6449f468fe368b` | `EC76EF57060C94F826F1BC4F34927C27A35996375A73E4552B12B95F7FE00CC2` | `C4E4C2294657A2BD43D7F0F4469F94244306F71914DDFD9C8D1C645922045E21` | `r_555bd0175b754b4303f94e63` |
| medium_large | 183 | `r_8456118030ec9474ae3623f4` | `F1D5134111FD59B5B038B201F06CFD4B1BBAC73AB6A67C704F74FD1D03B52CFA` | `BF5CB1AF7A5F30E455B1B35CC57BB21D748AB1D9006C43B68DD40574861E02C8` | `r_d907fe1cb2a19ec9ee50eb94` |
| medium_large | 184 | `r_58f741bdd7c146b5853780b1` | `017A4812FE72FDED4A7BBCE2B89207F536B2E0855BFDCE08DD504C844B861EA5` | `CCD3F9C10AC96F0658020A4FEAC8C82894BDAC0C72CA8C8739A4658CDB726BD1` | `r_7aebdadeeb3f5e1f7c5f133a` |
| large | 180 | `r_78b570bc0810690f4e5dbe00` | `C4C2015393262C8F13AC820C7B8A714F9C25A22B66655216EA33E0EA64FE986C` | `525C20FD3A96D5ECED53C6FDF7C4EEBCB0D50BA037FCA038E028540B06A1D417` | `r_6a8ba350b0c8f8425e4092f4` |
| large | 181 | `r_22da964693db429a6f34f248` | `AB950915301505F5C073F716DFE4D9A984E46F336A2FAB18B3D976B5145103CD` | `E96F81C79AF606FBF718DC32F9F317F3874CA0F10572557237BDC78040505E84` | `r_3bb5628274b5a4dc1833635f` |
| large | 182 | `r_9cbfdafd6282b1e97488378a` | `EBE76A1C1A7BFBA7305C126DB319C3CB3EB767DE00313CE0C2B02D6996445E11` | `F2C9D9F052DBB51870358C35C43FE0801C7E7E50954EA2A02B1EDDF9C495D2DE` | `r_6bf1fe31d7fa1c5fd9b9eddb` |
| large | 183 | `r_67e3844893209af4fd6565c3` | `499AF36AA674D2C92A09C3C3BE7998F09180409879532A45348417146384CE69` | `A8D4A9D50B5E8D9A2FB3A4376CCB1309BA88BA87006BB7E31288F3D5B456F536` | `r_cefc6871ca65262ec611411a` |
| large | 184 | `r_69c005dbfa751556ae31a55b` | `D3321BA690C5CAC58E1AA9E3ED11EDBDBDE398DB8549B347A9F1197C1AAA4594` | `792854936D35FB381951ED851E83A7F48221AD41685678F5AAED7B6A4BFC38BB` | `r_416eb309da09c1c9a811087c` |

## Independent read-only merge gate

Before merging, an independent reviewer must verify without rerunning optimization:

1. the ancestry relationship `0a8d3e7 -> e520af1` and the absence of changes to the run core after execution;
2. both Direct archive SHA256 values and the Hybrid source archive SHA256;
3. five unique seeds per scale, ten unique Direct run directories, and ten matching instance/anchor/Hybrid identities;
4. 10/10 Direct time limits without incumbents and 10/10 paired Hybrid certifications;
5. the reporting boundary: this benchmark compares Hybrid with Gurobi's direct deterministic-equivalent formulation and does not establish universal superiority over Gurobi.

