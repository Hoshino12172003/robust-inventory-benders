# Large remediation L0 Attempt 3 baseline schema incident

## Disposition

Attempt 3 is frozen as `execution_incomplete`. It is not a scientific result and must never be resumed or reused. The original evidence remains read-only at `E:\rfl0\experiments\results_fairness_large_final_remediation\attempt3_l0_large_pilot`.

## Read-only identity

- Git commit: `3E00ED7E7C250E3649931C1CC807072E5AA098B1`
- L0 config file SHA256: `3716F496481C65B3442A61252EE880A7922074505E65C212A77530DE457A7F52`
- Resolved config file SHA256: `B8351B882E952CA757E95C8A57900B7CCD0DDEF85AC3FC1A03B0BF1C6EDCCE1A`
- Resolved config canonical SHA256: `B108D343AF588DA9AD97C944444DCCC4A07FF58251005121FF4316DC94BEB262`
- Protocol SHA256: `79A3F87EBE7BFB00951E255D61E9FCA109D1F1CCA09DF7ED6A6042547BCB3742`
- Candidate SHA256: `DAC7A01941215624DBC5D8831814B71FDDDCC2CFEA54D1FE15FA5EAEA7C6F305`
- Instance scientific SHA256: `02A83AF3929A7CABF2BB35E545787611868591F331E3E707DA93E9754445B5B0`
- Accessed seeds: exactly `160`

The instance has eight warehouses and eight products. Its canonical indices are `I = range(8)` and `J = range(8)`.

## Observed state

- One instance, one manifest, one baseline checkpoint, two `run.json` files, and two `status.json` files exist.
- The baseline is scientifically certified: `status=optimal`, `valid_UB=true`, gap `8.245244391009812e-05`, and `scientific_status=certified_robust_optimal`.
- The baseline payload uses `best_y_values` as a JSON list of length 8 and `best_x_values` as a JSON list of 8 rows, each of length 8.
- The payload does not contain `y_values` or `x_values`.
- The frontier is `implementation_error` with reason `invalid_baseline_y_shape` and is not solved.
- No algorithm checkpoint exists. The failure occurs in initial-UB construction before `gp.setParam`, master construction, or any frontier solver call.

The S0 and fake fixtures had serialized `y_values` and `x_values`, so they did not exercise the production `SolveResult.summary_dict()` schema (`best_y_values` and `best_x_values`).

## Formal output file SHA256

| Relative path | SHA256 |
| --- | --- |
| `instances/160.json` | `352159DD65C8D0240C124839CAB15FDEF8DFE9D984435F45AEFFA92CDBABB410` |
| `manifest.json` | `0E95265EAE6F7A9E6AE5E3FB22958AD88055331686AABD4B4982F3D746DDCA2B` |
| `resolved_config.yaml` | `B8351B882E952CA757E95C8A57900B7CCD0DDEF85AC3FC1A03B0BF1C6EDCCE1A` |
| `results.csv` | `5FDA20F25C0FBA51234C1E42BF7A1D69FE34F8A01A54CABBC1CCA7EA2D39B63A` |
| `runs/r_9f01d62df9e1de0407ebfbc0/run.json` | `5163E9FF1EF76D6F8C23C43E612EEB2706647EBFCFF369C281527AA61831140A` |
| `runs/r_9f01d62df9e1de0407ebfbc0/status.json` | `8F5CE31039274A21F346C38A363D4DD0C82D2062C4F8E452A94DDF6CE244EACF` |
| `runs/r_e1cc500cf08a56f2faae0e13/baseline_checkpoint.json` | `4EAE25F8D8E9F75D27A3192BDAED368C172F21B61656A6959026620E407EC927` |
| `runs/r_e1cc500cf08a56f2faae0e13/run.json` | `4E3CB32D74974AE7B8B54393AA559EC1E030FD994DDCD1A95312EFEEC447DA95` |
| `runs/r_e1cc500cf08a56f2faae0e13/status.json` | `8785BDF70128C89B2AAA18B7C88A5D7952EE0CCE1235525A62C4F43605A7F7EA` |
| `summary.csv` | `93161AC9CF18D08D2C7AD96C7026646BE236059B1E336027F5BF154FC17277DF` |

## Hotfix isolation

The hotfix L0 plan uses execution attempt 4 and the new output directory `experiments/results_fairness_large_final_remediation/attempt4_l0_large_pilot_baseline_schema_hotfix`. Attempt 3 instances, baseline, anchor, runs, checkpoints, manifest, and aggregates are explicitly not reused. L0, L1, and M1 remain formally unauthorized in the hotfix PR.
