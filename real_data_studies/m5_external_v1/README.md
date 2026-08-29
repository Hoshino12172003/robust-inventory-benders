# M5 external real-data validation

This directory is reserved for a preregistered external validation of Hybrid
v8 using the official M5 Forecasting - Accuracy data.

The merged protocol is in
`docs/hybrid_v8_m5_external_holdout_protocol.md`.  Raw sales files remain
outside the repository.  The committed `raw_manifest.json` records the exact
official-file identities without exposing any demand values.

Before processing, verify an authorized official download with:

```text
python real_data_studies/m5_external_v1/verify_official_download.py RAW_DIRECTORY
```

The deterministic input processor is frozen separately from its outputs.  It
must not be run on the official files until the processor pull request has
merged.  After that merge, create a new repository-external output directory
with:

```text
python -m real_data_studies.m5_external_v1.prepare_m5_inputs RAW_DIRECTORY OUTPUT_DIRECTORY
```

The processor first checks the raw hashes, then writes factor membership,
calibration diagnostics, 12 weekly cases, 22 scenarios per case, and an input
freeze manifest.  Optimization is a later gated step and must consume those
frozen inputs without rewriting them.

After the processed-input PR and the formal-runner PR have both merged, verify
the sealed execution plan without solving:

```text
python -m real_data_studies.m5_external_v1.run_formal_holdout --stage verify
```

The `run` stage is the only command authorized to create formal optimization
results.  It verifies every frozen input hash before solving and stores one
raw file per method and case so an interrupted run can resume without rerunning
completed methods.
