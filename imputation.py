'''
simple impute
prediction for R, S1 and S2 conditions
mice and regression for all three branches
10 imputations, 20 iterations poooled via Rubin
'''


import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score, precision_score,
    brier_score_loss, confusion_matrix
)
from statsmodels.imputation.mice import MICE, MICEData

from categorical_encoder import CategoricalEncoder

def simple_impute(df):
    imputed = df.copy()
    num_cols = imputed.select_dtypes(include='number').columns
    cat_cols = imputed.select_dtypes(include=['object', 'category']).columns

    if len(num_cols) > 0:
        num_imputer = SimpleImputer(strategy='median').set_output(transform='pandas')
        imputed[num_cols] = num_imputer.fit_transform(imputed[num_cols])
    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent').set_output(transform='pandas')
        imputed[cat_cols]=cat_imputer.fit_transform(imputed[cat_cols])
    return imputed

def mice_and_regression(X_train, y_train, X_test, y_test, seed=42, return_proba=False, return_metrics=False):
    np.random.seed(seed)
    
    fml = "y ~ " + " + ".join(c for c in X_train.columns)
    encoder = CategoricalEncoder()

    train_data = encoder.fit_transform(X_train)
    X_test_filled = simple_impute(X_test)
    test_data = encoder.transform(X_test_filled)

    y_train = pd.Categorical(y_train).codes.astype(float)
    train_data['y']=y_train

    imp = MICEData(train_data)
    mice_model = MICE(fml, sm.GLM, imp, init_kwds={"family":sm.families.Binomial()})
    results = mice_model.fit(10,20)

    design_test = sm.GLM.from_formula(
        fml.replace("y ~", "y_dummy ~"),
        data=test_data.assign(y_dummy=0)
    ).exog                                 

    linear_predictor = design_test @ results.params
    proba = 1 / (1 + np.exp(-linear_predictor))

    auc = round(roc_auc_score(y_test,proba),4)

    metrics=None
    if return_metrics:
        y_pred = (proba >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        metrics = {
            "auc": auc,
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "sensitivity": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "specificity": round(specificity, 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "brier": round(brier_score_loss(y_test, proba), 4),
        }

    if return_proba and return_metrics:
        return auc, proba, metrics
    if return_metrics:
        return auc, metrics
    if return_proba:
        return auc, proba
    return auc