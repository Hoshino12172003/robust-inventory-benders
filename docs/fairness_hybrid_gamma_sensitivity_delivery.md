# Gamma sensitivity protocol delivery

This delivery freezes the `GAMMA_SENSITIVITY` protocol without running it. The only permitted commands in this pull request are solver-free:

```powershell
python -m src.fairness_hybrid_gamma_sensitivity_runner --config experiments/configs/fairness_hybrid_gamma_sensitivity.yaml --stage GAMMA_SENSITIVITY --dry-run
python -m src.fairness_hybrid_gamma_sensitivity_audit --root .
python -m pytest tests/test_fairness_hybrid_gamma_sensitivity.py -q
```

The runner intentionally rejects a non-dry invocation even when `--resume` is supplied, because `formal_run_authorized=false`. It has no `--overwrite` option. A later formal-run authorization must be a separate reviewed change after a fresh seed-access audit; it must not silently change this protocol, candidate, matrix, solver identity, or certification rules.

Expected dry-run totals are 30 baseline, 30 frontier, and 60 unique runs. Scenario counts are Medium-large `1/61/1831` and Large `1/97/4657` for Gamma `0/1/2`. The algorithm solver-limit envelope is 108,000 seconds (30 hours); the complete post-evaluation envelope is 997,200 seconds (about 277 hours). Neither is a wall-time prediction.

## Frozen delivery evidence

- Base commit: `827b1373702972ae780231899afe17cf6eff0d53` (PR #52 merge commit).
- Protocol SHA256: `D8CEA1249E92E9594D8308F5617D8767F23FFF7472644012DD4FD031CC7EF245`.
- Config SHA256: `F09074EA83E91514A0D1DFBF49F31EA24D2AABD9C94530B5F6A7E5504DA4585C`.
- Frozen Holdout ZIP SHA256 before and after audit: `BD253A4DE967143B8CD88E7DEADCB821D6062208CB71664B900F9BD8EBDF3839`.
- Repository seed-access audit: only the config preregistration declaration was found; generated-instance, solved-run, and formal-result evidence counts are all zero.
- ZIP seed-access audit: 13,640 entries, CRC error count zero, target-seed record count zero.
- Dry-run was generated twice with byte-identical output; no formal output directory was created. The longest planned Windows absolute path is the `post_chunk_tmp` path at 216 characters in this review worktree.
- Solver-free Gamma tests: 23 passed. PR #52 deterministic paper-artifact test: 1 passed. PR #51 full read-only archive reconciliation tests: 4 passed, including 120 runs, 100 exact certificates, 13,050 chunks, and deterministic report regeneration.
- `git diff --check` passed. Protected candidate, `src/benders.py`, `src/scenarios.py`, corrected-results, and paper-metrics hashes passed.

No formal sensitivity result exists. No seed 180--184 instance was generated or loaded, no Gurobi environment/model was configured or optimized, and no Final Holdout, D1, or D2 artifact was modified or reused.
