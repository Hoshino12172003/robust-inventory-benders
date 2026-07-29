# Hybrid C&CG–Benders D1 delivery and operations

This delivery freezes the Large Attempt 5 failure as immutable development evidence and introduces only `certified_hybrid_scenario_benders_fairness`. D1 is authorized by its checked-in configuration only after this Draft PR is independently reviewed and merged. Before merge, `formal_run_authorized` remains false at the workflow level even though the future-run configuration contains its audited D1-only switch.

## Attempt 5 recovery point

The source archive is `fairness_large_final_remediation_l0_attempt5_results.zip`, SHA256 `09B41862A5BFED724EDBEC1E64996B54AA878119F5C0DEDFE5B10126B2525A98`. The derived stop decision, provenance, and per-entry SHA inventory are under `analysis/fairness_hybrid_ccg_benders_d1_freeze`. They are evidence only and cannot be consumed as D1 instances, baselines, anchors, runs, cuts, checkpoints, manifests, results, or summaries.

Read-only recovery of this code node after merge uses its recorded merge commit:

```powershell
git fetch origin
git worktree add --detach E:\rfhd1-freeze <MERGED_D1_COMMIT>
```

## Sole formal command after merge and review

Run from a new short worktree whose output directory does not exist:

```powershell
git fetch origin
if (Test-Path 'E:\rfhd1') { throw 'E:\rfhd1 already exists; do not overwrite or reuse it' }
git worktree add --detach E:\rfhd1 origin/main
Set-Location E:\rfhd1
$python = 'E:\论文代码\robust-inventory-benders\.venv\Scripts\python.exe'
& $python -m src.fairness_hybrid_ccg_benders_runner --config experiments\configs\fairness_hybrid_ccg_benders_d1.yaml --stage D1 --resume
```

No other seed, rho, stage, or configuration may borrow this authorization. `--overwrite` does not exist.

Progress can be inspected without modifying output:

```powershell
Get-ChildItem 'E:\rfhd1\experiments\results_fairness_hybrid_ccg_benders\development_d1_large_seed160_rho0\runs' -Recurse -Filter status.json | ForEach-Object { $_.FullName; Get-Content -LiteralPath $_.FullName }
```

Use `Ctrl+C` once in the foreground terminal for a safe interrupt. Resume with the exact same Python command above; identity drift, damaged state, and partial persistent state fail closed.

The baseline plus algorithm envelope is at most about one hour. The conservative post-evaluation envelope is `4,657 × 30` seconds, so the combined solver envelope is about 39 hours 49 minutes plus checkpoint and aggregation overhead. This is an upper envelope, not a runtime prediction.

After a complete run, package without modifying the output directory:

```powershell
$source = 'E:\rfhd1\experiments\results_fairness_hybrid_ccg_benders\development_d1_large_seed160_rho0'
$archive = 'E:\论文代码\fairness_hybrid_ccg_benders_d1_results.zip'
if (Test-Path $archive) { throw "$archive already exists; choose a new immutable archive name" }
Compress-Archive -LiteralPath $source -DestinationPath $archive -CompressionLevel Optimal
Get-FileHash -LiteralPath $archive -Algorithm SHA256
```

## Statistical boundary

D1 contains one development seed, so it supports no inferential comparison. For any later frozen evaluation, seed is the independent unit: resampling is by seed cluster, every rho and both methods stay together within a sampled seed, each rho has one pair per seed, five rho-specific tests use Holm correction, and an overall comparison aggregates rho within seed or uses the seed-cluster bootstrap. Seed-rho tasks are not independent replicates.
