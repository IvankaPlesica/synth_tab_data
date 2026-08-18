# Synthetic Data Pipeline

This project generates synthetic datasets for **CKD** and **hepatitis**.

It includes:

- missingness pattern mining
- S1 synthesis using shadow columns
- S2 synthesis using missingness patterns
- a real data baseline
- subgroup and outlier analysis
- optional disease staging

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
config.py
imputation.py
missingness/
synthesis/
subgroups/
staging/
export_synthetic.py
```

Dataset-specific settings are stored in `config.py`.

## Evaluation

The evaluation pipeline can be run with the dataset-specific configuration and is used to compare the real and synthetic data.

