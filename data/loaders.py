'''
data loaders, UCI CKD and Hepatitis
'''

import gzip
import os
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

def load_ckd(path=None):
    path = path or os.path.join(CACHE_DIR, '336.csv.gz')
    with gzip.open(path) as f:
        df = pd.read_csv(f)

    df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)
    df["dm"] = df["dm"].replace("\tno", "no")
    df["class"] = df["class"].replace("ckd\t", "ckd")

    return df

def load_hepatitis(path=None):
    path = path or os.path.join(CACHE_DIR, '46.csv.gz')
    with gzip.open(path) as f:
        df = pd.read_csv(f)

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    return df

