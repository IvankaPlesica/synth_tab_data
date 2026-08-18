'''
eGFR* approximation and CKD staging; with limitations, as described in the thesis.
'''

import numpy as np
import pandas as pd

KAPPA = 0.8
ALPHA = -0.27
MIN_ADULT_AGE = 18

STAGE_BINS = [0,15,30,45,60,90,np.inf]
STAGE_LABELS = ['G5','G4','G3b','G3a','G2','G1']

BAND_BINS = [0,15,60,90,np.inf]
BAND_LABELS = ['Failure','Disease','Early','Normal']

def add_egfr_columns(df, sc_col='sc', age_col='age', export_path=None):
    df = df.copy()
    sc = df[sc_col]
    age = df[age_col]

    invalid = sc.isna() | age.isna() | (age < MIN_ADULT_AGE)

    ratio = sc / KAPPA
    age_factor = 0.9938 ** age

    egfr = np.where (invalid, 
                     np.nan, 
                     np.where (
                         ratio < 1, 
                         142 * (ratio ** ALPHA) * age_factor,
                         142 * (ratio ** -1.200) * age_factor
                         ),
                     )
    
    df['egfr_star'] = np.round(egfr, 1)
    df['ckd_stage'] = pd.cut (df['egfr_star'], bins=STAGE_BINS, labels=STAGE_LABELS)

    band = pd.cut(df['egfr_star'], bins=BAND_BINS, labels=BAND_LABELS)
    cat = band.astype(object)
    is_pediatric = age < MIN_ADULT_AGE
    is_unstageable = df['ckd_stage'].isna() & ~is_pediatric
    cat[is_pediatric] = 'Pediatric'
    cat[is_unstageable] = 'Unstageable'
    df['ckd_stage_cat'] = cat

    if export_path:
        df.to_csv(export_path, index=False)

    return df                    

