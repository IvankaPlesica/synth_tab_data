'''
ALBI score and liver disease severity grade; with limitations, as described in the thesis.
'''

import numpy as np
import pandas as pd

BILIRUBIN_MGDL_TO_UMOLL = 17.11
ALBUMIN_GDL_TO_GL = 10
ALBI_COEFF_BILIRUBIN = 0.66
ALBI_COEF_ALBUMIN = -0.0852
MIN_ADULT_AGE = 18

GRADE_BINS = [-np.inf,-2.60,-1.39,np.inf]
GRADE_LABELS = ['1','2','3']


def add_albi_columns(df, bilirubin_col='bilirubin', albumin_col='albumin', age_col='age', export_path=None):
    df = df.copy()
    bilirubin = df[bilirubin_col]
    albumin = df[albumin_col]
    age = df[age_col]

    invalid = bilirubin.isna() | albumin.isna() | (age < MIN_ADULT_AGE)

    bilirubin_umoll = bilirubin * BILIRUBIN_MGDL_TO_UMOLL
    albumin_gl =  albumin * ALBUMIN_GDL_TO_GL

    albi = np.where (invalid,
                     np.nan,
                     np.log10(bilirubin_umoll) * ALBI_COEF_ALBUMIN + albumin_gl * ALBI_COEF_ALBUMIN,
                     )
    
    df['albi_score'] = np.round(albi, 1)
    df['albi_grade'] = pd.cut (df['albi_score'], bins=GRADE_BINS, labels=GRADE_LABELS)

    cat = df['albi_grade'].astype(object)
    is_pediatric = age < MIN_ADULT_AGE
    is_unstageable = df['albi_grade'].isna() & ~is_pediatric
    cat[is_pediatric] = 'Pediatric'
    cat[is_unstageable] = 'Unscorable'
    df['albi_grade_cat'] = cat

    if export_path:
        df.to_csv(export_path, index=False)
    return df                    

