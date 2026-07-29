# Controlled Large D2 delivery

This delivery authorizes only the 12-task D2 matrix after this Draft PR is independently reviewed and merged. It does not authorize any other seed, rho, candidate, holdout, or full grid. Attempt 2 is frozen as execution-incomplete and must not be resumed; the command below starts isolated Attempt 3.

## Frozen identities

- D1 archive: `7E89115E3BE325C9A37C31D28D32EA80EEA95F09528DBC8AEDA833EF0129A4A9`
- D1 decision: `1F7101CB722C4A4E6974C2D8597F4ED37BF89C2072FA039502EC69E834C7F17E`
- D2 protocol: `A1D1655F4D66B79ADB9AF28E69F8E04D50F0EAEFB8577F645080D5713D1426BC`
- D2 config: `ED8F145A9ACAA1AC799DBBDE2BAEBF1A35F2F614FE41B0F73EF8F278690EF63A`
- candidate: `8AF2687A4340D03BE44C5A73FFD3BE1F1E015F5447D2B56FD9A8919049D46BA0`

## Merge-time run procedure

Use a new path that has never existed. The production gate requires a clean detached worktree at the exact local `origin/main`; it runs before output creation, instance generation, or solver configuration.

```powershell
$repo = 'E:\论文代码\robust-inventory-benders'
$worktree = 'E:\rfhd2a3'
$python = Join-Path $repo '.venv\Scripts\python.exe'
git -C $repo fetch origin
if (Test-Path -LiteralPath $worktree) { throw "$worktree already exists; choose another never-used short path" }
git -C $repo worktree add --detach $worktree origin/main
Set-Location $worktree
& $python -m src.fairness_hybrid_ccg_benders_d2_runner --config experiments\configs\fairness_hybrid_ccg_benders_d2.yaml --stage D2 --resume
```

View completed/running task records without modifying them:

```powershell
Get-ChildItem 'E:\rfhd2a3\experiments\results_fairness_hybrid_ccg_benders\controlled_d2_a3_large_s160_162_r0_001_010\runs' -Recurse -Filter status.json | ForEach-Object { $path=$_.FullName; $status=Get-Content -Raw $path | ConvertFrom-Json; [pscustomobject]@{path=$path; state=$status.state; scientific_status=$status.scientific_status; algorithm_status=$status.algorithm_status} }
```

Use `Ctrl+C` once for a safe interruption. Resume from the same detached worktree with the same command:

```powershell
& $python -m src.fairness_hybrid_ccg_benders_d2_runner --config experiments\configs\fairness_hybrid_ccg_benders_d2.yaml --stage D2 --resume
```

After all 12 tasks complete and the audit passes, package a copy outside the output tree:

```powershell
Compress-Archive -Path 'E:\rfhd2a3\experiments\results_fairness_hybrid_ccg_benders\controlled_d2_a3_large_s160_162_r0_001_010\*' -DestinationPath 'E:\论文代码\fairness_hybrid_ccg_benders_d2_attempt3_large_results.zip' -CompressionLevel Optimal
Get-FileHash 'E:\论文代码\fairness_hybrid_ccg_benders_d2_attempt3_large_results.zip' -Algorithm SHA256
```

The solver-limit envelope is six hours for baselines plus frontier algorithms and 349.275 hours for all post-evaluation scenario limits. It is not a wall-time prediction. D1 observed 722.718 seconds for its baseline, 109.368 seconds for the frontier algorithm, 61.326 seconds for frontier post-evaluation wall time, and 207.186 seconds total frontier wall time; D2 must not shorten certification based on that observation.
