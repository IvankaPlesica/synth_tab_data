'''
dataset configuration
set up for CKD and Hepatitis
data loader, columns defiition, subgroup definition, disease staging
'''

from dataclasses import dataclass
from typing import Callable, Optional
import pandas as pd
from data.loaders import load_ckd, load_hepatitis
from staging.egfr import add_egfr_columns
from staging.albi import add_albi_columns

@dataclass(frozen=True)
class DatasetConfig:
    name:str
    loader: Callable[[], pd.DataFrame]
    n_splits: int
    positive_label: object
    categorical_cols: list
    min_support: float
    min_any_confidence: float
    extra_subgroups: list
    staging_fn: Callable
    staging_kwargs: dict
    discovery_restrict_to_label: Optional[object] = None

def _mp0_mask(df):
    features = df.drop(columns="class", errors="ignore")
    return features.isna().sum(axis=1).eq(0).to_numpy()

DATASET_REGISTRY = {
    'ckd': DatasetConfig(
        name="ckd",
        loader=load_ckd,
        n_splits=5,
        positive_label="ckd",
        categorical_cols=["sg", "al", "su", "rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"],
        min_support=0.1,
        min_any_confidence=0.7,
        extra_subgroups=[
            ("age>=65", lambda df: (df["age"] >= 65).to_numpy()),
            ("htn==yes", lambda df: (df["htn"] == "yes").to_numpy()),
            ("dm==yes", lambda df: (df["dm"] == "yes").to_numpy()),
            ("mp0", _mp0_mask),
        ],
        staging_fn=add_egfr_columns,
        staging_kwargs={"sc_col": "sc", "age_col": "age"},
        discovery_restrict_to_label=None,
    ),
    'hepatitis': DatasetConfig(
        name="hepatitis",
        loader=load_hepatitis,
        n_splits=3,
        positive_label=1,
        categorical_cols=["sex", "steroid", "antivirals", "fatigue", "malaise", "anorexia", "liver_big", "liver_firm", "spleen_palpable", "spiders", "ascites", "varices", "histology"],
        min_support=0.06,
        min_any_confidence=0.6,
        extra_subgroups=[
            ("age>=50", lambda df: (df["age"] >= 50).to_numpy()),
            ("mp0", _mp0_mask),
        ],
        staging_fn=add_albi_columns,
        staging_kwargs={"bilirubin_col": "bilirubin", "albumin_col": "albumin", "age_col": "age"},
        discovery_restrict_to_label="__not_positive__",
    ),
}