# Olist real-data validation (`real_data_olist_v1`)

This directory is an isolated, additive study. It does not replace or rerun any synthetic development, holdout, sensitivity, paired-benchmark, or stress-test experiment.

## Freeze rule

Before this study was created, SHA-256 hashes were recorded for every existing repository file outside this directory. The baseline is stored in `provenance/preexisting_files_sha256.csv`. At completion, `scripts/freeze_existing_experiments.ps1 -Mode verify` must report zero modified, missing, or unexpected files outside this directory. A failed verification invalidates the study delivery.

## Data lineage

- `raw/source_archive/`: immutable downloaded archive.
- `raw/extracted/`: immutable source CSV files extracted from that archive.
- `processed/`: cleaned and aggregated tables derived from the raw data.
- `instances/`: model-ready instances created from the training and calibration periods only.
- `configs/`: frozen real-data experimental configurations.
- `results/`: outputs for Hybrid, pure CCG, and Gurobi Direct.
- `analysis/`: statistical summaries, figures, and the experiment report.
- `logs/`: execution logs.
- `provenance/`: source metadata, hashes, data dictionaries, and freeze verification.

The original source is the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/olistbr/brazilian-ecommerce/home), version 2, licensed under CC BY-NC-SA 4.0.

## Interpretation boundary

Olist records marketplace transactions rather than a retailer's internal warehouse planning system. Demand, customer regions, seller geography, product categories, freight values, and timestamps are observed. Candidate warehouses, capacities, opening costs, holding costs, shortage penalties, and policy service targets are calibrated modeling inputs and must not be described as observed facts.
