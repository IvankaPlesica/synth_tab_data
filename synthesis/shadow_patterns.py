'''
S2 branch missingness representation
'''
import numpy as np
from missingness.patterns import assign_patterns
from synthesis.bayesian_network import synthesize, extract_synth_pattern_labels

def expand_label(label):
    if label in {'mp0', 'mp_other'}:
        return set()
    columns = set()
    for block in label.split(' & '):
        columns.update(block.split('+'))
    return columns

def build_column_to_block(blocks):
    # maps every member to its block
    final_block = {}
    mark = set()

    for block in blocks:
        itemset = set(block['itemset'])
        overlap = mark & itemset
        assert not overlap, f"blocks overlap on columns {overlap}:{blocks}"
        mark.update(itemset)
        for column in itemset:
            final_block[column] = itemset
        return final_block
    

def get_residual_rates(X_train, blocks, real_pattern_labels, y_train):
    # residual missingness rate for every column, per class
    residual_rates_class = {}

    for value in y_train.unique():
        mask = (y_train == value)
        X_class = X_train.loc[mask]
        labels = real_pattern_labels.loc[mask]
        residual_rates = X_class.isnull().mean().to_dict()

        for block in blocks:
            itemset = block['itemset']
            block_columns = set(itemset)
            triggered = labels.apply(lambda l: block_columns.issubset(expand_label(l)))
            subset = X_class.loc[~triggered]
            for column in itemset:
                residual_rates[column] = subset[column].isnull().mean() if len(subset) else 0.0
            
        residual_rates_class[str(value)] = residual_rates

    return residual_rates_class


def apply_deterministic_patterns(synth_features, synth_pattern_labels):
    # 1 to 1 part of assigning patterns
    features = synth_features.copy()
    labels = [expand_label(label) for label in synth_pattern_labels]

    for i, columns in enumerate(labels):
        if columns:
            features.loc[features.index[i], list(columns)] = np.nan
    
    return features, labels



def apply_residual_noise(out, synth_pattern_labels, parsed_labels, column_to_block,
                          residual_rates_by_class, synth_class, rng):
    eligible_base = np.asarray(synth_pattern_labels) != "mp0"  # excludes true complete rows only
    synth_class_arr = np.asarray(synth_class).astype(str)

    for col in out.columns:
        if col in column_to_block:
            block_cols = column_to_block[col]
            eligible = np.array([
                eligible_base[i] and not block_cols.issubset(parsed_labels[i])
                for i in range(len(out))
            ])
        else:
            eligible = eligible_base  # non-block columns: eligible on every non-mp0 row
        rate = np.array([
            residual_rates_by_class.get(synth_class_arr[i], {}).get(col, 0.0)
            for i in range(len(out))
        ])
        missing = rng.random(len(out)) < rate
        out.loc[missing & eligible, col] = np.nan
    return out


def reinsert_pattern_missingness(synth_features, synth_labels, column_to_block, residual_rates, synth_class, seed=42):
    rng = np.random.default_rng(seed)
    output, parsed_labels = apply_deterministic_patterns(synth_features, synth_labels)
    output = apply_residual_noise(output, synth_labels, parsed_labels, column_to_block, residual_rates, synth_class, rng)
    
    return output


def run_s2_branch(X, X_filled, y_train, categorical_cols, mining_model, seed=42):
    real_pattern_labels = assign_patterns(X, mining_model)
    column_to_block = build_column_to_block(mining_model.blocks)
    residual_rates_by_class = get_residual_rates(X, mining_model.blocks, real_pattern_labels, y_train)

    synth_features_with_pattern, synth_label = synthesize(
        X_filled, y_train, categorical_cols, pattern_labels=real_pattern_labels, seed=seed
    )
    synth_features, synth_pattern_labels = extract_synth_pattern_labels(synth_features_with_pattern)

    s2_features = reinsert_pattern_missingness(
        synth_features, synth_pattern_labels, column_to_block,
        residual_rates_by_class, synth_label, seed=seed,
    )
    return s2_features, synth_label