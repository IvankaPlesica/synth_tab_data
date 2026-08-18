# Synthetic Data Pipeline

This project generates synthetic datasets for **CKD** and **hepatitis**.

It includes:

- missingness pattern mining
- S1 synthesis using shadow columns
- S2 synthesis using missingness patterns
- a real data baseline
- subgroup and outlier analysis
- optional disease staging
- cross-validated R/S1/S2 evaluation with fairness and privacy checks

## Setup

Install the project dependencies.

```bash
pip install -r requirements.txt
```

## Generate synthetic data

Run:

```bash
python export_synthetic.py --dataset ckd
```

or:

```bash
python export_synthetic.py --dataset hepatitis
```

Use a different seed with:

```bash
python export_synthetic.py --dataset ckd --seed 123
```

To also generate disease-staging columns:

```bash
python export_synthetic.py --dataset ckd --with-staging
```

## Output

Files are written to `outputs/`.

For each dataset:

- `<dataset>_synthetic_s1.csv` — S1 synthetic data
- `<dataset>_synthetic_s2.csv` — S2 synthetic data
- `<dataset>_synthetic_b.csv` — real data baseline
- `<dataset>_synthetic_s1_staged.csv` — S1 with staging, if requested
- `<dataset>_synthetic_s2_staged.csv` — S2 with staging, if requested
- `<dataset>_synthetic_b_staged.csv` — baseline with staging, if requested
- `missingness_metadata.json` — mined missingness information

## Main pipeline

The main components are:

```text
config.py              # dataset registry: loaders, categorical columns, mining thresholds, subgroups
categorical_encoder.py # shared categorical/numeric encoder (fit on real data, reused everywhere)
imputation.py           # simple median/mode imputation + MICE-based downstream regression
missingness/
    mining.py           # co-missingness block mining (RQ1)
    patterns.py         # per-row missingness pattern labeling
synthesis/
    bayesian_network.py # PyBNesian based synthesis (shared by S1/S2)
    shadow_columns.py   # S1: shadow column construction/reinsertion
    shadow_patterns.py  # S2: pattern driven synthesis branch
subgroups/
    discovery.py         # KMeans clustering + outlier rule fitting
    outlier_rule.py       # frozen OutlierRule: apply/explain
    privacy.py             # anonymeter attack-row helpers
staging/                # optional disease staging (eGFR/CKD stage, ALBI score)
    albi.py
    egfr.py
export_synthetic.py     # entry point: generate synthetic datasets
evaluate.py              # entry point: cross-validated R/S1/S2 evaluation
```

Dataset-specific settings are stored in `config.py`.

## Evaluation

`evaluate.py` runs stratified K-fold cross-validation three ways per fold — **R** (real data, MICE + logistic regression baseline), **S1** (shadow-columns synthetic data), and **S2** (shadow-patterns synthetic data) — and reports AUC, F1, sensitivity, specificity, precision, and Brier score for each.

```bash
python evaluate.py --dataset ckd
```

Optional flags (all default **ON**; disable individually with `--no-<flag>`):

- `--subgroup-metadata` — per-subgroup TPR/specificity gaps (outlier subgroup + dataset-specific hand-defined subgroups) and representation/retention ratios, pooled out-of-fold across all folds. Requires all three branches (`R`, `S1`, `S2`).
- `--privacy-checks` — anonymeter singling-out risk for S1/S2, broken down by subgroup if `--subgroup-metadata` is also on.
- `--pipeline-metadata` — aggregated R/S1/S2 performance across folds, plus whether the real missingness-pattern structure survives synthesis in S1/S2 (Dmis).
- `--metrics-history` — per-fold metrics appended to `outputs/metrics_history.csv`.


### Evaluation output

Written to `outputs/` (or `--output-dir`):

- `metrics_history.csv` — per-fold metrics, appended across runs
- `subgroup_metadata.json` — per-subgroup fairness/privacy/representation results, overwritten per run
- `pipeline_metadata.json` — aggregated performance + missingness-structure preservation, overwritten per run


## Subgroup discovery

Subgroups (KMeans-based outlier detection) are fit once on real data and frozen as artifacts, so both the export and evaluation pipelines apply the exact same rule to real and synthetic patients.

```bash
python -m subgroups.discovery
```

This writes `artifacts/outlier_rule_<dataset>.joblib`, which `evaluate.py` and `export_synthetic.py` load via `OutlierRule.load(dataset_name)`. K is fixed by hand, chosen via a manual silhouette/stability comparison outside this pipeline; see `subgroups/discovery.py` for details and reported diagnostics (silhouette width, seed stability/ARI).
