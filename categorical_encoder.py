'''
used in imputation, clustering, outliers
fit on train data only (fit and transform separate steps to preserve categories)
'''

import pandas as pd

class CategoricalEncoder:
    '''
    numeric strings to numeric data type
    categorical strings to integer codes
    learned on fit, transformed to train
    '''

    def __init__(self):
        self.columns_={}

    def fit(self, X):
        self.columns_={}
        cat_columns = X.select_dtypes(include=['object','category']).columns
        
        for column in cat_columns:
            values = X[column].astype("string")
            not_null = values.dropna()
            numeric = pd.to_numeric(not_null,errors='coerce')

            if len(not_null) > 0 and numeric.notna().all():
                self.columns_[column] = {'type':'numeric_string'}
            else:
                categories = sorted(not_null.unique())
                mapping = {category: code for code, category in enumerate(categories)}
                self.columns_[column] = {'type':'categorical','mapping':mapping}
            
        return self
        
    def transform(self, X):
        X = X.copy()
        for column, info in self.columns_.items():
            if info['type'] == 'numeric_string':
                X[column] = pd.to_numeric(X[column], errors='coerce')
            else:
                X[column] = X[column].astype('string').map(info['mapping'])
        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)