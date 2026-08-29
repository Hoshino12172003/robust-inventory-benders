# Hybrid v8 reproducibility guide

## Scope

This package reproduces the frozen Hybrid v8 comparisons against standard single-scenario CCG and Batch-4 CCG. The candidate algorithm, confirmation protocol, prepared inputs, run order, raw per-case outputs, and summaries are versioned together. The five confirmation weeks were selected and hashed before their optimization runs.

## Requirements

- Python 3.12;
- `gurobipy` with a working Gurobi license;
- `pandas`;
- the repository dependencies in `requirements.txt`.

The environment used for the reported run is recorded in `environment_snapshot.json`. Solver settings are frozen in `confirmation_protocol.json`: one thread, solver seed zero, zero final MIP gap, and fixed feasibility tolerances.

## 1. Verify the frozen package

From the repository root, run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python real_data_studies/real_data_olist_v1/algorithm_v8_adaptive_hybrid/verify_reproducibility.py
```

This checks the candidate source SHA-256, all ten sealed confirmation input files, and the archived result artifacts. Do not proceed if any hash differs.

## 2. Reproduce the five-week sealed confirmation

The prepared and frozen inputs are already included. Do not rerun the `prepare` stage when reproducing the reported confirmation, because that stage exists only to document how the inputs were originally derived.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python real_data_studies/real_data_olist_v1/algorithm_v8_adaptive_hybrid/run_sealed_confirmation.py --stage run
python real_data_studies/real_data_olist_v1/algorithm_v8_adaptive_hybrid/run_sealed_confirmation.py --stage summarize
```

The run stage solves the common cost anchor and complete Direct model, then executes Hybrid v8, pure single-scenario CCG, and Batch-4 CCG in the precommitted rotating order. It writes one JSON file per week under `sealed_confirmation/results/`. The summarize stage regenerates `sealed_confirmation/summary.json` and `sealed_confirmation/summary.csv`.

Runtime values need not be bit-for-bit identical across machines. Certification status, iteration structure, cut triggers, and objectives should agree within the recorded numerical tolerances. Compare paired methods only within the same fresh run.

## 3. Reproduce the twelve-week development comparison

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python real_data_studies/real_data_olist_v1/algorithm_v8_adaptive_hybrid/run_v8_vs_pure.py
python real_data_studies/real_data_olist_v1/algorithm_v8_adaptive_hybrid/summarize_v8.py
```

The twelve weeks were observed during algorithm development. They explain algorithm behavior but are not an independent confirmation set.

## 4. Reproduce the algorithm-development screens

The following commands reproduce the static block/cut screen, cut-normalization screen, and repeated development pairing:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python real_data_studies/real_data_olist_v1/algorithm_v8_adaptive_hybrid/screen_candidates.py
python real_data_studies/real_data_olist_v1/algorithm_v8_adaptive_hybrid/screen_cut_selection.py
python real_data_studies/real_data_olist_v1/algorithm_v8_adaptive_hybrid/run_dev_pairing.py
```

These steps are development evidence and must not be presented as sealed validation.

## 5. Interpretation boundary

The sealed experiment supports that the complete Hybrid v8 framework was faster than standard single-scenario CCG on all five confirmation weeks while preserving certification and Direct-objective agreement. It does not support a claim that the Farkas-cut component alone, or Hybrid v8 as a whole, was faster than Batch-4 CCG. The scenario-block mechanism explains most of the measured speedup.
