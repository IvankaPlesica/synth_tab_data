'''
R/S1/S2 evaluation.
 --subgroup-metadata : per-subgroup TPR/specificity gaps and representation/retention ratios, pooled out-of-fold across all folds
  --privacy-checks     : anonymeter singling-out risk for S1/S2, broken down by subgroup if --subgroup-metadata is also on
  --pipeline-metadata  : aggregated R/S1/S2 performance across folds, plus whether the real missingness pattern structure  survives synthesis in S1/S2 (Dmis)
  --metrics-history    : per-fold metrics appended to outputs/metrics_history.csv
default: ON, (e.g. --no-subgroup-metadata to turn one off)
'''

import argparse
import csv
import json
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from statsmodels.genmod.generalized_linear_model import PerfectSeparationWarning
from config import DATASET_REGISTRY
from imputation import simple_impute, mice_and_regression
from missingness.mining import MissingnessModel
from missingness.patterns import assign_patterns
from synthesis.shadow_columns import build_shadow_columns, reinsert_from_shadow_columns
from synthesis.shadow_patterns import run_s2_branch
from synthesis.bayesian_network import synthesize
from subgroups.outlier_rule import OutlierRule
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=PerfectSeparationWarning)

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
METRICS_FIELDS = ["dataset", "condition", "fold", "seed", "auc", "f1", "sensitivity", "specificity", "precision", "brier"]



def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(DATASET_REGISTRY), help="Which dataset to evaluate.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for CV splitting, synthesis, and MICE.")
    p.add_argument("--branches", nargs="+", default=["R", "S1", "S2"], choices=["R", "S1", "S2"], help="Which branch(es) to run. Must be all three if --subgroup-metadata is on since subgroup gaps compare across branches.")
    p.add_argument("--output-dir", default=OUTPUTS_DIR, help="Directory for metrics_history.csv / *_metadata.json.")
    p.add_argument("--subgroup-metadata", action=argparse.BooleanOptionalAction, default=True, help="Per-subgroup TPR/specificity gaps + representation, written to subgroup_metadata.json.")
    p.add_argument("--privacy-checks", action=argparse.BooleanOptionalAction, default=True, help="Anonymeter singling-out risk for S1/S2.")
    p.add_argument("--pipeline-metadata", action=argparse.BooleanOptionalAction, default=True, help="Aggregated performance + missingness-structure (Dmis) preservation, written to pipeline_metadata.json.")
    p.add_argument("--metrics-history", action=argparse.BooleanOptionalAction, default=True, help="Append per-fold metrics to metrics_history.csv.")
    args = p.parse_args()

    if args.subgroup_metadata and set(args.branches) != {"R", "S1", "S2"}:
        p.error("--subgroup-metadata requires all three branches (R S1 S2); "
                "either drop --branches or turn off --no-subgroup-metadata")
    return args


def append_metrics_csv(records, path):
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(records)

def update_subgroup_metadata(dataset_name, subgroup_name, record, path):
    # per run
    all_medatada = {}
    if os.path.exists(path):
        with open(path) as f:
            all_medatada = json.load(f)
    all_medatada.setdefault(dataset_name, {})[subgroup_name] = record
    with open(path, 'w') as f:
        json.dump(all_medatada, f, indent=2)

# redundancy with update_subgroup_metadata
def write_pipeline_metadata(dataset_name, record, path):
    all_medatada = {}
    if os.path.exists(path):
        with open(path) as f:
            all_medatada = json.load(f)
    all_medatada[dataset_name] = record
    with open(path, 'w') as f:
        json.dump(all_medatada, f, indent=2)

# short readable profile for subgroups
def characterize_subgroup(df, mask, categorical_cols):
    num_cols = [c for c in df.columns if c not in categorical_cols and c != "class"]
    chars = {}
    for col in num_cols:
        vals = df.loc[mask, col].dropna()
        if len(vals):
            chars[col] = f"{vals.mean():.2f} +/- {vals.std():.2f}"
    for col in categorical_cols:
        if col not in df.columns:
            continue
        vals = df.loc[mask, col].dropna()
        if len(vals):
            mode = vals.mode().iloc[0]
            chars[col] = f"{mode} ({(vals == mode).mean():.0%})"
    return chars

# compute dmis
def compute_dmis(real_labels, synth_labels):
    #L1 divergence between real and synthetic pattern-label rate distributions (Wang, Asif & Vaidya 2023, Eq. 1)
    real_rates = real_labels.value_counts(normalize=True)
    synth_rates = synth_labels.value_counts(normalize=True)
    all_labels = set(real_rates.index) | set(synth_rates.index)
    return sum(abs(real_rates.get(m,0.0) - synth_rates.get(m, 0.0)) for m in all_labels)

# TPR restricted to subgroup rows
def group_tpr(y_true, y_pred, mask):
    actual_pos = (y_true == 1) & mask
    if actual_pos.sum() == 0:
        return None
    return (y_pred[actual_pos] == 1).mean()

def group_tnr(y_true, y_pred, mask):
    actual_neg = (y_true == 0) & mask
    if actual_neg.sum() == 0:
        return None
    return (y_pred[actual_neg] == 0).mean()


def main():
    #set up parsed and documents
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_csv_path = os.path.join(args.output_dir, "metrics_history.csv")
    subgroup_json_path = os.path.join(args.output_dir, "subgroup_metadata.json")
    pipeline_json_path = os.path.join(args.output_dir, "pipeline_metadata.json")

    dataset_name = args.dataset
    config = DATASET_REGISTRY[dataset_name]
    run_branches = set(args.branches)

    outlier_rule = None
    if args.subgroup_metadata:
        outlier_rule = OutlierRule.load(dataset_name)
    if args.privacy_checks:
        from anonymeter.evaluators import SinglingOutEvaluator
        from subgroups.privacy import attacked_rows
    
    # set up dataset and parameters
    df = config.loader()
    n_splits = config.n_splits
    positive_label = config.positive_label
    categorical_cols = config.categorical_cols
    min_support = config.min_support
    min_any_confidence = config.min_any_confidence


    X = df.drop(columns=["class"])
    y = df["class"]
    for c in categorical_cols:
        if c in X.columns:
            X[c] = X[c].astype("object")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    print(f"\nDataset: {dataset_name}  |  folds: {skf.get_n_splits()}  |  branches: {sorted(run_branches)} |  seed: {args.seed}")


    # Experiment scores

    scores = {b: [] for b in run_branches}

    if args.subgroup_metadata:
        oof_proba = {b: np.full(len(df), np.nan) for b in ["R", "S1", "S2"]}
        subgroup_defs = [("outlier", outlier_rule.apply)] + list(config.extra_subgroups)
        synth_subgroup_counts = {name: {"S1": 0, "S2": 0} for name, _ in subgroup_defs}
        synth_total_counts = {"S1": 0, "S2": 0}

    if args.privacy_checks:
        so_risk = {"S1": [], "S2": []}
        so_attacked_mask = {
            "S1": pd.Series(False, index=df.index),
            "S2": pd.Series(False, index=df.index),
        }
    
    if args.pipeline_metadata:
        pipeline_fold_metrics = {b: [] for b in run_branches}
        s1_features_all, s2_features_all = [], []
    
    # set up sets
    for fold_num, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold_num + 1} in progress.")
        X_train = X.iloc[train_index].copy()
        X_test = X.iloc[test_index].copy()
        y_train = (y.iloc[train_index] == positive_label).astype(int)
        y_test = (y.iloc[test_index] == positive_label).astype(int)
    
        fold_metrics_records = []

        # Baseline R
        if "R" in run_branches:
            if args.subgroup_metadata:
                score_r, proba_r, metrics_r = mice_and_regression(X_train, y_train, X_test, y_test, seed=args.seed, return_proba=True, return_metrics=True)
                oof_proba["R"][test_index] = proba_r
            else:
                score_r, metrics_r = mice_and_regression(X_train, y_train, X_test, y_test, seed=args.seed, return_metrics=True)

            scores["R"].append(score_r)
            fold_metrics_records.append({"dataset": dataset_name, "condition": "R", "fold": fold_num, "seed": args.seed, **metrics_r})
        
            if args.pipeline_metadata:
                pipeline_fold_metrics["R"].append(metrics_r)
            print(f"Fold {fold_num}: Score R = {score_r}")

        # simple impute for train set
        X_train_filled = simple_impute(X_train)

        # Shadow columns S1
        if "S1" in run_branches:
            shadow_cols = build_shadow_columns(X_train)
            synth_features_with_shadows, synth_label_s1 = synthesize(X_train_filled, y_train, categorical_cols, shadow_cols=shadow_cols, seed=args.seed)
            s1_features = reinsert_from_shadow_columns(synth_features_with_shadows, X_train.columns)

            if args.subgroup_metadata:
                for subgroup_name, mask_fn in subgroup_defs:
                    synth_subgroup_counts[subgroup_name]["S1"] += int(mask_fn(s1_features).sum())
                synth_total_counts["S1"] += len(s1_features)
                score_s1, proba_s1, metrics_s1 = mice_and_regression(s1_features, synth_label_s1, X_test, y_test, seed=args.seed,return_proba=True, return_metrics=True)
                oof_proba["S1"][test_index] = proba_s1
            else:
                score_s1, metrics_s1 = mice_and_regression(s1_features, synth_label_s1, X_test, y_test, seed=args.seed, return_metrics=True)
            scores["S1"].append(score_s1)
            fold_metrics_records.append({"dataset": dataset_name, "condition": "S1", "fold": fold_num,"seed": args.seed, **metrics_s1})
        
            if args.pipeline_metadata:
                pipeline_fold_metrics["S1"].append(metrics_s1)
                s1_features_all.append(s1_features)
            print(f"Fold {fold_num}: Score S1 = {score_s1}")

            if args.privacy_checks:
                n_attacks_s1 = min(500, len(X_train))
                so_eval_s1 = SinglingOutEvaluator(ori=X_train, syn=s1_features, control=X_test, n_attacks=n_attacks_s1)
                so_eval_s1.evaluate(mode="multivariate")
                so_risk["S1"].append(so_eval_s1.risk().value)
                hits_s1 = attacked_rows(X_test, so_eval_s1.queries())
                if hits_s1:
                    so_attacked_mask["S1"].loc[list(hits_s1)] = True     

        # Shadow patterns S2
        if "S2" in run_branches:
            mining_model = MissingnessModel().fit(X_train, min_support=min_support, min_any_confidence=min_any_confidence)
            print(f"Fold {fold_num}: {len(mining_model.blocks)} missingness blocks "
                    f"(min_support={min_support}, min_any_confidence={min_any_confidence}): "
                    + ", ".join("+".join(b["itemset"]) for b in mining_model.blocks))

            s2_features, synth_label_s2 = run_s2_branch(X_train, X_train_filled, y_train, categorical_cols, mining_model, seed=args.seed)

            if args.subgroup_metadata:
                for subgroup_name, mask_fn in subgroup_defs:
                    synth_subgroup_counts[subgroup_name]["S2"] += int(mask_fn(s2_features).sum())
                synth_total_counts["S2"] += len(s2_features)
                score_s2, proba_s2, metrics_s2 = mice_and_regression(s2_features, synth_label_s2, X_test, y_test, seed=args.seed,return_proba=True, return_metrics=True)
                oof_proba["S2"][test_index] = proba_s2
            else:
                score_s2, metrics_s2 = mice_and_regression(s2_features, synth_label_s2, X_test, y_test, seed=args.seed, return_metrics=True)
            scores["S2"].append(score_s2)
            fold_metrics_records.append({"dataset": dataset_name, "condition": "S2", "fold": fold_num,"seed": args.seed, **metrics_s2})
        
            if args.pipeline_metadata:
                pipeline_fold_metrics["S2"].append(metrics_s2)
                s2_features_all.append(s2_features)
            print(f"Fold {fold_num}: Score S2 = {score_s2}")

            if args.privacy_checks:
                n_attacks_s2 = min(500, len(X_train))
                so_eval_s2 = SinglingOutEvaluator(ori=X_train, syn=s2_features, control=X_test, n_attacks=n_attacks_s2)
                so_eval_s2.evaluate(mode="multivariate")
                so_risk["S2"].append(so_eval_s2.risk().value)
                hits_s2 = attacked_rows(X_test, so_eval_s2.queries())
                if hits_s2:
                    so_attacked_mask["S2"].loc[list(hits_s2)] = True

        if args.metrics_history:
            append_metrics_csv(fold_metrics_records, metrics_csv_path)

    for b in sorted(run_branches):
        print(f"\nMean Score {b}: {np.mean(scores[b]):.4f} (std: {np.std(scores[b]):.4f})")
    

    # Pipeline metadata: aggregated performance, Dmis preservation
    if args.pipeline_metadata:
        def aggregate_fold_metrics(records):
            agg = {}
            for key in ["auc", "f1", "sensitivity", "specificity", "precision", "brier"]:
                vals = [r[key] for r in records]
                agg[f"{key}_mean"] = float(np.mean(vals))
                agg[f"{key}_std"] = float(np.std(vals))
            return agg

        performance = {branch: aggregate_fold_metrics(records) for branch, records in pipeline_fold_metrics.items()}

        missingness_structure = None
        if "S1" in run_branches and "S2" in run_branches:
            structure_model = MissingnessModel().fit(X, min_support=min_support, min_any_confidence=min_any_confidence)
            real_labels = assign_patterns(X, structure_model)
            s1_pooled = pd.concat(s1_features_all, ignore_index=True)
            s2_pooled = pd.concat(s2_features_all, ignore_index=True)
            s1_labels = assign_patterns(s1_pooled, structure_model)
            s2_labels = assign_patterns(s2_pooled, structure_model)

            block_cols = {c for b in structure_model.blocks for c in b["itemset"]}
            independent_columns = {
                col: round(float(rate), 4)
                for col, rate in structure_model.base_rates.items()
                if col not in block_cols
            }
            blocks_metadata = [
                {
                    "itemset": b["itemset"],
                    "any_confidence": b["any_confidence"],
                    "all_confidence": b["all_confidence"],
                    "cross_support": b["cross_support"],
                    "conditionals": {
                        col: round(float(p), 4)
                        for col, p in structure_model.block_conditionals[tuple(b["itemset"])].items()
                    },
                }
                for b in structure_model.blocks
            ]

            missingness_structure = {
                "blocks": blocks_metadata,
                "independent_columns": independent_columns,
                "real_pattern_label_rates": real_labels.value_counts(normalize=True).round(4).to_dict(),
                "S1": {
                    "dmis": float(compute_dmis(real_labels, s1_labels)),
                    "pattern_label_rates": s1_labels.value_counts(normalize=True).round(4).to_dict(),
                    "n_pooled": len(s1_pooled),
                },
                "S2": {
                    "dmis": float(compute_dmis(real_labels, s2_labels)),
                    "pattern_label_rates": s2_labels.value_counts(normalize=True).round(4).to_dict(),
                    "n_pooled": len(s2_pooled),
                },
            }
            print(f"\nDmis (lower = closer to real): S1={missingness_structure['S1']['dmis']:.4f}, "
                  f"S2={missingness_structure['S2']['dmis']:.4f}")

        pipeline_record = {
            "n_splits": n_splits,
            "min_support": min_support,
            "min_any_confidence": min_any_confidence,
            "seed": args.seed,
            "performance": performance,
            "missingness_structure": missingness_structure,
        }
        write_pipeline_metadata(dataset_name, pipeline_record, pipeline_json_path)

    # TPR gap and privacy checks
    if args.subgroup_metadata:
        for branch, proba in oof_proba.items():
            assert not np.isnan(proba).any(), f"every row should get exactly one OOF prediction ({branch})"
        y_true = (y == positive_label).astype(int).to_numpy()

        subgroups = [("outlier", outlier_rule.apply(df))] + [(name, mask_fn(df)) for name, mask_fn in config.extra_subgroups]

        for subgroup_name, mask in subgroups:
            print(f"\n{'=' * 60}\nSubgroup: {subgroup_name} (n={mask.sum()} / {len(df)})\n{'=' * 60}")

            record = {
                "n": int(mask.sum()),
                "total": int(len(df)),
                "definition": (
                    "distance-to-cluster-centroid rule, fit on real data (subgroups/discovery.py); "
                    "explanation_tree below is a ~97%-fidelity decision tree"
                    if subgroup_name == "outlier" else subgroup_name
                ),
                "explanation_tree": outlier_rule.explain() if subgroup_name == "outlier" else None,
                "characterization": characterize_subgroup(df, mask, categorical_cols),
                "metrics": {},
                "privacy": {},
                "representation": {},
            }

            real_prevalence = mask.sum() / len(df)
            for branch in ["S1", "S2"]:
                n_pooled = synth_subgroup_counts[subgroup_name][branch]
                total_pooled = synth_total_counts[branch]
                synth_prevalence = n_pooled / total_pooled if total_pooled else None
                retention_ratio = (synth_prevalence / real_prevalence) if (synth_prevalence is not None and real_prevalence > 0) else None
                record["representation"][branch] = {
                    "n_pooled": n_pooled,
                    "total_pooled": total_pooled,
                    "synthetic_prevalence": synth_prevalence,
                    "retention_ratio": retention_ratio,
                }

            n_neg_subgroup = int(((y_true == 0) & mask).sum())
            n_neg_non_subgroup = int(((y_true == 0) & ~mask).sum())
            if n_neg_subgroup == 0:
                print(f"(NOTE: 0 actual negative-class patients in subgroup '{subgroup_name}' -- "
                      f"specificity/FPR-gap is undefined for this subgroup)")

            tpr_gaps, spec_gaps = {}, {}
            for branch, proba in oof_proba.items():
                y_pred = (proba >= 0.5).astype(int)

                tpr_out = group_tpr(y_true, y_pred, mask)
                tpr_non = group_tpr(y_true, y_pred, ~mask)
                tpr_gap = tpr_non - tpr_out
                tpr_gaps[branch] = tpr_gap

                tnr_out = group_tnr(y_true, y_pred, mask)
                tnr_non = group_tnr(y_true, y_pred, ~mask)
                spec_gap = (tnr_non - tnr_out) if (tnr_out is not None and tnr_non is not None) else None
                spec_gaps[branch] = spec_gap

                print(f"{branch}: tpr_subgroup={tpr_out:.3f} tpr_non_subgroup={tpr_non:.3f} tpr_gap={tpr_gap:.3f}")

                record["metrics"][branch] = {
                    "tpr_subgroup": float(tpr_out) if tpr_out is not None else None,
                    "tpr_non_subgroup": float(tpr_non) if tpr_non is not None else None,
                    "tpr_gap": float(tpr_gap) if tpr_gap is not None else None,
                    "n_neg_subgroup": n_neg_subgroup,
                    "n_neg_non_subgroup": n_neg_non_subgroup,
                    "specificity_subgroup": float(tnr_out) if tnr_out is not None else None,
                    "specificity_non_subgroup": float(tnr_non) if tnr_non is not None else None,
                    "specificity_gap": float(spec_gap) if spec_gap is not None else None,
                }
                if branch != "R":
                    tpr_delta = tpr_gap - tpr_gaps["R"]
                    record["metrics"][branch]["tpr_delta_vs_R"] = float(tpr_delta)
                    if spec_gap is not None and spec_gaps.get("R") is not None:
                        record["metrics"][branch]["specificity_delta_vs_R"] = float(spec_gap - spec_gaps["R"])
                    else:
                        record["metrics"][branch]["specificity_delta_vs_R"] = None

            if args.privacy_checks:
                print(f"\nSingling-out privacy risk for subgroup '{subgroup_name}'")
                for branch in ["S1", "S2"]:
                    mean_risk = np.mean(so_risk[branch])
                    attacked_arr = so_attacked_mask[branch].to_numpy()
                    out_rate = attacked_arr[mask].mean() if mask.sum() else float("nan")
                    non_rate = attacked_arr[~mask].mean() if (~mask).sum() else float("nan")
                    print(f"{branch}: mean risk={mean_risk:.4f}  subgroup attack rate={out_rate:.3f}  "
                          f"non-subgroup attack rate={non_rate:.3f}")
                    record["privacy"][branch] = {
                        "mean_risk_across_folds": float(mean_risk),
                        "std_risk_across_folds": float(np.std(so_risk[branch])),
                        "subgroup_attack_rate": float(out_rate) if not np.isnan(out_rate) else None,
                        "non_subgroup_attack_rate": float(non_rate) if not np.isnan(non_rate) else None,
                        "gap": float(out_rate - non_rate) if not (np.isnan(out_rate) or np.isnan(non_rate)) else None,
                        "note": "anonymeter's rng is unseeded -- not reproducible run-to-run",
                    }

            update_subgroup_metadata(dataset_name, subgroup_name, record, subgroup_json_path)
            print(f"(subgroup metadata for '{subgroup_name}' written to {subgroup_json_path})")

    elif args.privacy_checks:
        print("\nSingling-out privacy risk (S1 vs S2):")
        for branch in ["S1", "S2"]:
            if branch in run_branches:
                print(f"{branch}: mean risk across folds = {np.mean(so_risk[branch]):.4f} "
                      f"(std={np.std(so_risk[branch]):.4f})")


if __name__ == "__main__":
    main()

