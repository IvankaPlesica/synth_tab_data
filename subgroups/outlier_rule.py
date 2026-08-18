from pathlib import Path

import joblib
import numpy as np
from sklearn.tree import export_text

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


class OutlierRule:
    def __init__(self, saved):
        self.encoder = saved["encoder"]
        self.cat_cols = saved["cat_cols"]
        self.kmeans = saved["kmeans"]
        self.scaler = saved["scaler"]
        self.medians = saved["medians"]
        self.modes = saved["modes"]
        self.num_cols = saved["num_cols"]
        self.cluster_cutoffs = saved["cluster_cutoffs"]
        self.outlier_tree = saved["outlier_tree"]

    @classmethod
    def load(cls, dataset_name, artifacts_dir=ARTIFACTS_DIR):
        path = artifacts_dir / f"outlier_rule_{dataset_name}.joblib"
        return cls(joblib.load(path))

    def apply(self, df):
        df = df.drop(columns=["class"], errors="ignore").copy()
        df[self.cat_cols] = df[self.cat_cols].astype("object")
        df[self.num_cols] = df[self.num_cols].fillna(self.medians)
        df[self.cat_cols] = df[self.cat_cols].fillna(self.modes)
        scaled = self.scaler.transform(self.encoder.transform(df))
        labels = self.kmeans.predict(scaled)
        distances = np.linalg.norm(scaled - self.kmeans.cluster_centers_[labels], axis=1)
        cutoffs = np.array([self.cluster_cutoffs[label] for label in labels])
        return distances > cutoffs

    def explain(self):
        return export_text(self.outlier_tree, feature_names=self.encoder.feature_names_,)