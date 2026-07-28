# Large remediation L0 Attempt 4 precision adapter incident

## Disposition

Attempt 4 is frozen as `execution_incomplete`. It is not a scientific result
and must never be resumed or reused. The original evidence remains read-only at
`E:\rfl0a4\experiments\results_fairness_large_final_remediation\attempt4_l0_large_pilot_baseline_schema_hotfix`.

The baseline completed with `scientific_status=certified_robust_optimal`. The
frontier failed before master construction because the production adapter did
not pass the frozen precision payload to
`solve_certified_adaptive_multicut_fair_benders()`; the solver therefore
received its fail-closed legacy default and raised
`precision_policy must remain joint_error_budget`.

S0 tests passed `FROZEN_PRECISION` directly to the solver, while fake pipeline
tests replaced the production frontier adapter. Neither exercised the missing
production argument.

## Read-only identity

- Git commit: `9e77e6d097d1825c3f4bf3a02400ea16496d6f4c`
- execution attempt: `4`
- L0 config file SHA256: `C18AE2CA1BEA5D222197268462D6BE342553FA88CB06CB060A2D7CED28F24B2E`
- resolved config file SHA256: `5975424A79A35EC722A689A62485FE4D04BB36C78D1012CB017E7CC57A12D589`
- resolved config canonical SHA256: `0497679275C8336AF888534C0BB300103F08BC7E9BAA7E64A57170567841AB0E`
- protocol SHA256: `79A3F87EBE7BFB00951E255D61E9FCA109D1F1CCA09DF7ED6A6042547BCB3742`
- candidate SHA256: `DAC7A01941215624DBC5D8831814B71FDDDCC2CFEA54D1FE15FA5EAEA7C6F305`
- accessed seed: exactly `160`
- manifest completed run count: `0`
- manifest certified solved count: `0`

## Formal output file SHA256

| Relative path | SHA256 |
| --- | --- |
| `instances/160.json` | `352159DD65C8D0240C124839CAB15FDEF8DFE9D984435F45AEFFA92CDBABB410` |
| `manifest.json` | `9F36CECFBF2626FA6CFB14887227BBA8242495F39BB020C3734E6EBD918D4CE0` |
| `resolved_config.yaml` | `5975424A79A35EC722A689A62485FE4D04BB36C78D1012CB017E7CC57A12D589` |
| `results.csv` | `30BEA293D6762CE585FC633163780C550BF82D0828868E0A3B4999EA8DBDE782` |
| `runs/r_18bea9a750d7e9f06c1b4684/run.json` | `730C32F6F93747E9DDCBCB5245106B8FC1D658BC005C171D4FCFB1956EFAF192` |
| `runs/r_18bea9a750d7e9f06c1b4684/status.json` | `8F5CE31039274A21F346C38A363D4DD0C82D2062C4F8E452A94DDF6CE244EACF` |
| `runs/r_f18f5ddc9534b5202c05e668/baseline_checkpoint.json` | `24FAF616DD5C7DC3943256805EDA5991B75F414ADEB8CF452D1A4B4718EFF907` |
| `runs/r_f18f5ddc9534b5202c05e668/run.json` | `BF22B432C5FB5CB3AD592B9E794678A12FFC964DA80C3A605FAEAC8E9D737C09` |
| `runs/r_f18f5ddc9534b5202c05e668/status.json` | `8785BDF70128C89B2AAA18B7C88A5D7952EE0CCE1235525A62C4F43605A7F7EA` |
| `summary.csv` | `42EF67C4E908EA40BD737827537E382F039E7D4FA7F2B50F87C3E17DB8312703` |

## Attempt 5 isolation

The corrected L0 plan uses execution attempt 5 and the new output directory
`experiments/results_fairness_large_final_remediation/attempt5_l0_large_pilot_precision_adapter_hotfix`.
It does not import or reuse Attempt 4 instances, baseline, anchor, runs,
checkpoints, manifest, or aggregates. L1 and M1 remain unauthorized.
