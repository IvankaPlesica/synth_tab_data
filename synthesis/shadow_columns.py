'''
S1 branch for missingness
'''
import numpy as np

def build_shadow_columns(X_train):
    return X_train.isnull().astype(int).add_suffix("_NA")

def reinsert_from_shadow_columns(synth_features_with_shadows, original_cols):
    output = synth_features_with_shadows.copy()
    for column in original_cols:
        na_column = f"{column}_NA"
        if na_column in output.columns:
            mask = output[na_column].astype(str) == '1'
            output.loc[mask, column] = np.nan
    output = output.drop(columns = [c for c in output.columns if c.endswith('_NA')])
    return output