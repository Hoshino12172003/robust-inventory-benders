# M5 external real-data validation

This directory is reserved for a preregistered external validation of Hybrid
v8 using the official M5 Forecasting - Accuracy data.

The protocol is in
`docs/hybrid_v8_m5_external_holdout_protocol.md`.  Until that protocol is
merged, this directory must contain no raw sales files, processed demand data,
case inputs, or optimization outputs.

Raw Kaggle files are intentionally not committed.  After authorized download,
their hashes and structural metadata will be recorded before deterministic
processing.  The repository will clearly distinguish observed M5 demand from
calibrated optimization parameters.
