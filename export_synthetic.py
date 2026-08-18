'''
generate synth datasets as described in thesis
'''

import argparse
import json
import os
from config import DATASET_REGISTRY
from imputation import simple_impute
from missingness.mining import MissingnessModel
from synthesis.shadow_columns import build_shadow_columns, reinsert_from_shadow_columns
from synthesis.shadow_patterns import run_s2_branch
from synthesis.bayesian_network import synthesize

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

def write_missingness_metadata(dataset_name, mining_model, path):
    block_columns = {c for b in mining_model.blocks for c in b['itemset']}
    independent_columns ={
        column: round(float(rate), 4)
        for column, rate in mining_model.base_rates.items()
        if column not in block_columns
    }

    blocks_metadata = [
        {
            "itemset": b["itemset"],
            "any_confidence": b["any_confidence"],
            "all_confidence": b["all_confidence"],
            "cross_support": b["cross_support"],
            "conditionals": {
                col: round(float(p), 4)
                for col, p in mining_model.block_conditionals[tuple(b["itemset"])].items()
            },
        }
        for b in mining_model.blocks
    ]

    record = {
        "min_support": mining_model.min_support,
        "min_any_confidence": mining_model.min_any_confidence,
        "blocks": blocks_metadata,
        "independent_columns": independent_columns,
    }

    all_metadata = {}
    if os.path.exists(path):
        with open(path) as f:
            all_metadata = json.load(f)
    all_metadata[dataset_name] = record
    with open(path, 'w') as f:
        json.dump(all_metadata, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(DATASET_REGISTRY))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=OUTPUTS_DIR)
    p.add_argument("--with-staging", action="store_true",
                    help="Also write a disease staged CSV per branch "
                         "(eGFR for CKD, ALBI for hepatitis).")
    return p.parse_args()

def print_sanity_checks(name, out_df):
    print(f"\n--- {name} sanity checks ---")
    print(f"shape: {out_df.shape}")
    print("missing values per column:")
    print(out_df.isnull().sum())
    print("class distribution:")
    print(out_df["class"].value_counts())

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    dataset_name = args.dataset
    config = DATASET_REGISTRY[dataset_name]
    df = config.loader()
    positive_label = config.positive_label
    categorical_cols = config.categorical_cols
    min_support = config.min_support
    min_any_confidence = config.min_any_confidence
    negative_label = next(v for v in df["class"].unique() if v != positive_label)

    def restore_class_labels(synth_label):
        return synth_label.astype(str).map({"1": positive_label, "0": negative_label})
    
    X = df.drop(columns=["class"])
    y = df["class"]
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype('object')
    y_encoded = (y == positive_label).astype(int)

    #missingness on full data
    mining_model = MissingnessModel().fit(X, min_support=min_support, min_any_confidence=min_any_confidence)
    print(f"Global missingness blocks: {len(mining_model.blocks)} blocks "
          f"(min_support={min_support}, min_any_confidence={min_any_confidence}):")
    for i, block in enumerate(mining_model.blocks):
        print(f"    Block_{i}: {block['itemset']}")

    missingness_json_path = os.path.join(args.output_dir, "missingness_metadata.json")
    write_missingness_metadata(dataset_name, mining_model, missingness_json_path)

    X_filled = simple_impute(X)

    # S1
    shadow_cols = build_shadow_columns(X)
    synth_features_with_shadows, synth_label_s1 = synthesize(X_filled, y_encoded, categorical_cols, shadow_cols=shadow_cols, seed=args.seed)
    s1_features = reinsert_from_shadow_columns(synth_features_with_shadows, X.columns)
    s1_df = s1_features.copy()
    s1_df["class"] = restore_class_labels(synth_label_s1).values
    s1_path = os.path.join(args.output_dir, f"{dataset_name}_synthetic_s1.csv")
    s1_df.to_csv(s1_path, index=False)

    # S2
    s2_features, synth_label_s2 = run_s2_branch(X, X_filled, y_encoded, categorical_cols, mining_model, seed=args.seed)
    s2_df = s2_features.copy()
    s2_df["class"] = restore_class_labels(synth_label_s2).values
    s2_path = os.path.join(args.output_dir, f"{dataset_name}_synthetic_s2.csv")
    s2_df.to_csv(s2_path, index=False)

    # B
    b_path = os.path.join(args.output_dir, f"{dataset_name}_synthetic_b.csv")
    df.to_csv(b_path, index=False)

if __name__ == "__main__":
    main()


    if args.with_staging:
        for name, out_df, path in [("S1", s1_df, s1_path), ("S2", s2_df, s2_path), ("B", df, b_path)]:
            staged_path = path.replace(".csv", "_staged.csv")
            config.staging_fn(out_df, export_path=staged_path, **config.staging_kwargs)
