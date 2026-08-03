# Gamma sensitivity protocol delivery

This delivery freezes the `GAMMA_SENSITIVITY` protocol without running it. The only permitted commands in this pull request are solver-free:

```powershell
python -m src.fairness_hybrid_gamma_sensitivity_runner --config experiments/configs/fairness_hybrid_gamma_sensitivity.yaml --stage GAMMA_SENSITIVITY --dry-run
python -m src.fairness_hybrid_gamma_sensitivity_audit --root .
python -m pytest tests/test_fairness_hybrid_gamma_sensitivity.py -q
```

The production pipeline is complete but remains fail-closed because `formal_run_authorized=false` and no reviewed authorization file exists. It has no `--overwrite` option. A later formal-run authorization is a separate reviewed, Git-tracked JSON file after a fresh seed-access audit; it does not require another runner change and must not change this protocol, candidate, matrix, solver identity, or certification rules.

After that separate authorization-only change is merged, the formal command is:

```powershell
python -m src.fairness_hybrid_gamma_sensitivity_runner --config experiments/configs/fairness_hybrid_gamma_sensitivity.yaml --stage GAMMA_SENSITIVITY --resume --authorization-file experiments/configs/fairness_hybrid_gamma_sensitivity_authorization.json
```

The command is accepted only from the clean detached current `origin/main` worktree at `E:\rfgs`. Authorization, Git, seed-access, existing-output, and complete path gates execute before solver code is imported or an output directory is created.

Expected dry-run totals are 30 baseline, 30 frontier, and 60 unique runs. Scenario counts are Medium-large `1/61/1831` and Large `1/97/4657` for Gamma `0/1/2`. The algorithm solver-limit envelope is 108,000 seconds (30 hours); the complete post-evaluation envelope is 997,200 seconds (about 277 hours). Neither is a wall-time prediction.

## Frozen delivery evidence

- Base commit: `827b1373702972ae780231899afe17cf6eff0d53` (PR #52 merge commit).
- Protocol SHA256: `D9552F83E292CB1EDB262DF2B113A8044AECD6607207FDBC857770083C35EC1A`.
- Config SHA256: `82834FB7BC91C3CE2BB4759A2C4571E3E812E9A9C030A2CD9F56F8CD60B61A59`.
- Frozen Holdout ZIP SHA256 before and after audit: `BD253A4DE967143B8CD88E7DEADCB821D6062208CB71664B900F9BD8EBDF3839`.
- Repository seed-access audit: only the config preregistration declaration was found; generated-instance, solved-run, and formal-result evidence counts are all zero.
- ZIP seed-access audit: 13,640 entries, CRC error count zero, target-seed record count zero.
- Dry-run creates no formal output. Under the frozen formal worktree `E:\rfgs`, exhaustive planning covers root files, 60 run/status pairs, 30 baseline checkpoints, 30 algorithm checkpoints, post-evaluation final/index/chunks, and all atomic temporary names. The longest path is `post_chunk_tmp`, 123 characters, leaving 97 characters below the 220-character limit.
- Solver-free Gamma tests: 37 passed, including a complete synthetic 60-run execution through the real detached Git gate, byte-stable exact CSV plan projection, second-pass resume with zero repeated instance/baseline/frontier/post calls, baseline-checkpoint interruption and corruption recovery, rejection of unrelated untracked and tracked changes, Gamma-specific baseline solver settings, pre-solver corrupt-output rejection, and opaque filename/ZIP-member seed detection. Static audit: 15/15 passed. PR #51/#52 regression results are rerun in the final delivery check.
- `git diff --check` passed. Protected candidate, `src/benders.py`, `src/scenarios.py`, corrected-results, and paper-metrics hashes passed.

No formal sensitivity result exists. No seed 180--184 instance was generated or loaded, no Gurobi model was created or optimized, and no Final Holdout, D1, or D2 artifact was modified or reused. During the repair review, one mistakenly selected historical D2 regression reached `gp.setParam`; Gurobi environment initialization immediately failed on the sandbox/license username mismatch before model construction. No further solver-touching historical tests were run. The Gamma dry-run and all 35 Gamma tests remained solver-free, and the final seed audit found zero access evidence.
