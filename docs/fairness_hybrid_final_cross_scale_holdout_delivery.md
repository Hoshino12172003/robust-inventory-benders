# Final cross-scale holdout execution

This delivery authorizes only the frozen `FINAL_HOLDOUT` matrix after this change is independently reviewed and merged. Never run it from a PR branch or a non-detached worktree.

```powershell
git fetch origin
if (Test-Path 'E:\rfhfinal') { throw 'E:\rfhfinal already exists; do not overwrite or reuse it' }
git worktree add --detach 'E:\rfhfinal' origin/main
Set-Location 'E:\rfhfinal'
$python = 'E:\论文代码\robust-inventory-benders\.venv\Scripts\python.exe'
$config = 'experiments\configs\fairness_hybrid_final_cross_scale_holdout.yaml'
& $python -m src.fairness_hybrid_final_holdout_runner --config $config --stage FINAL_HOLDOUT --dry-run
```

The dry-run must report 20 baselines, 100 frontiers, 120 unique tasks, no duplicates, no reserved-seed access evidence, `instances_generated=false`, `solver_called=false`, absent output directories, and a portable path. Then the only formal command is:

```powershell
& $python -m src.fairness_hybrid_final_holdout_runner --config $config --stage FINAL_HOLDOUT --resume
```

Use `Ctrl+C` for a safe interruption. Resume with the identical formal command; do not edit the config, use another checkout, or delete partial output. A compact progress view is:

```powershell
Get-ChildItem 'experiments\results_fairness_hybrid_final_holdout' -Recurse -Filter status.json |
  ForEach-Object { Get-Content -Raw $_.FullName | ConvertFrom-Json } |
  Group-Object state,scientific_status,algorithm_status |
  Select-Object Count,Name
```

The serial solver-limit envelope is 10 hours for 20 baselines plus 50 hours for 100 frontier algorithms. The post-evaluation solver-limit envelope is about 2,703.3 hours across all registered scenarios; these are limits, not wall-time forecasts. Checkpoint/resume exists because actual completion time depends on instance difficulty and post-evaluation solve times.
