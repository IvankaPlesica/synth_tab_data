'''
wrapper around semsynth PyBNesian
'''

import pandas as pd
from semsynth.backends.pybnesian import learn_bn
from semsynth.utils import  coerce_continuous_to_float, coerce_discrete_to_category, rename_categorical_categories_to_str

def synthesize(X_train_filled, y_train, categorical_cols, shadow_cols=None, pattern_labels=None, seed=42):
    working = X_train_filled.copy()
    working['class'] = y_train.values
    extra_columns = []
    if shadow_cols is not None:
        working = pd.concat([working, shadow_cols], axis=1)
        extra_columns += list(shadow_cols.columns)
    if pattern_labels is not None:
        working['missingness_pattern'] = pattern_labels
        extra_columns.append('missingness_pattern')
    disc_columns = [c for c in categorical_cols if c in working.columns] + ['class'] + extra_columns
    cont_columns = [c for c in working.columns if c not in disc_columns]
    working = coerce_discrete_to_category(working, disc_columns)
    working = rename_categorical_categories_to_str(working, disc_columns)
    working = coerce_continuous_to_float(working, cont_columns)
    
    bn_output = learn_bn(working, bn_type='semiparametric', random_state=seed, arc_blacklist=[],max_indegree=2)

    synth = bn_output.model.sample(len(working), seed=seed)
    synth_df = synth.to_pandas().reindex(columns=working.columns)
    for column in disc_columns:
        if column in synth_df.columns:
            synth_df[column] = synth_df[column].astype('category')
    synth_label = synth_df['class']
    synth_features_shadows = synth_df.drop(columns=['class'])
    return synth_features_shadows, synth_label


def extract_synth_pattern_labels(synth_features_with_pattern):
    output = synth_features_with_pattern.copy()
    pattern_labels = output['missingness_patterns'].astype(str)
    output = output.drop(columns=['missingness_pattern'])
    return output, pattern_labels