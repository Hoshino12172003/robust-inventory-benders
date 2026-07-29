# Controlled Large D2 delivery

This delivery authorizes only the 12-task D2 matrix after this Draft PR is independently reviewed and merged. It does not authorize any other seed, rho, candidate, holdout, or full grid.

## Frozen identities

- D1 archive: `7E89115E3BE325C9A37C31D28D32EA80EEA95F09528DBC8AEDA833EF0129A4A9`
- D1 decision: `1F7101CB722C4A4E6974C2D8597F4ED37BF89C2072FA039502EC69E834C7F17E`
- D2 protocol: `D1DD2DC7417204CDB0B9A70986B975EC621D400CF8384E6EC026ED1A90D9367B`
- D2 config: `AE449C3E1551532AA772E1E51F4348860FC16A7CFE9970D6E2A9477F2E2DBFF1`
- candidate: `8AF2687A4340D03BE44C5A73FFD3BE1F1E015F5447D2B56FD9A8919049D46BA0`

## Merge-time run procedure

Use a new path that has never existed. The production gate requires a clean detached worktree at the exact local `origin/main`; it runs before output creation, instance generation, or solver configuration.

```powershell
git -C 'E:\论文代码\robust-inventory-benders' fetch origin
if (Test-Path -LiteralPath 'E:\rfhd2') { throw 'E:\rfhd2 already exists; choose another never-used short path' }
git -C 'E:\论文代码\robust-inventory-benders' worktree add --detach 'E:\rfhd2' origin/main
Set-Location 'E:\rfhd2'
& 'E:\论文代码\robust-inventory-benders\.venv\Scripts\python.exe' -m src.fairness_hybrid_ccg_benders_d2_runner --config experiments\configs\fairness_hybrid_ccg_benders_d2.yaml --stage D2 --resume
```

View completed/running task records without modifying them:

```powershell
Get-ChildItem 'E:\rfhd2\experiments\results_fairness_hybrid_ccg_benders\controlled_d2_large_seeds160_162_rhos0_001_010\runs' -Recurse -Filter status.json | ForEach-Object { Get-Content -Raw $_.FullName | ConvertFrom-Json | Select-Object @{n='path';e={$_.PSPath}},state,scientific_status,algorithm_status }
```

Use `Ctrl+C` once for a safe interruption. Resume from the same detached worktree with the same command:

```powershell
& 'E:\论文代码\robust-inventory-benders\.venv\Scripts\python.exe' -m src.fairness_hybrid_ccg_benders_d2_runner --config experiments\configs\fairness_hybrid_ccg_benders_d2.yaml --stage D2 --resume
```

After all 12 tasks complete and the audit passes, package a copy outside the output tree:

```powershell
Compress-Archive -Path 'E:\rfhd2\experiments\results_fairness_hybrid_ccg_benders\controlled_d2_large_seeds160_162_rhos0_001_010\*' -DestinationPath 'E:\论文代码\fairness_hybrid_ccg_benders_d2_large_results.zip' -CompressionLevel Optimal
Get-FileHash 'E:\论文代码\fairness_hybrid_ccg_benders_d2_large_results.zip' -Algorithm SHA256
```

The solver-limit envelope is six hours for baselines plus frontier algorithms and 349.275 hours for all post-evaluation scenario limits. It is not a wall-time prediction. D1 observed 722.718 seconds for its baseline, 109.368 seconds for the frontier algorithm, 61.326 seconds for frontier post-evaluation wall time, and 207.186 seconds total frontier wall time; D2 must not shorten certification based on that observation.
