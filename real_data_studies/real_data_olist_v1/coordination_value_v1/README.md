# Coordination value v1

This additive Olist study compares flexible multi-warehouse fulfillment with an
optimized single-source policy. Read `PROTOCOL.md` before running anything.

The implementation is intentionally gated. Before review and merge, only the
following commands are valid:

```powershell
python real_data_studies/real_data_olist_v1/coordination_value_v1/run_experiment.py --stage prepare
python real_data_studies/real_data_olist_v1/coordination_value_v1/run_experiment.py --stage dry-run
python real_data_studies/real_data_olist_v1/coordination_value_v1/test_coordination_model.py
```

`prepare` records paths and hashes of the 17 existing weekly instances and their
scenario files. It invokes no solver. `dry-run` verifies configuration, input
identity, case count, and the formal authorization gate.

Formal execution and reporting will be enabled only by a separate reviewed
authorization after this protocol and implementation are merged.

After authorization, the formal commands will be:

```powershell
python real_data_studies/real_data_olist_v1/coordination_value_v1/run_experiment.py --stage run
python real_data_studies/real_data_olist_v1/coordination_value_v1/run_experiment.py --stage summarize
```

The formal runner retains completed case files when resumed and never overwrites
an existing result or analysis artifact.
