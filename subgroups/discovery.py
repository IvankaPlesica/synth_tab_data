'''
KMeans clustering, K=2 subgroups
a distance-to-centroid outlier rule
depth 3 decision trees that explain the clusters and the outlier flag.
'''

import itertools

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from categorical_encoder import CategoricalEncoder

RANDOM_STATE = 42
DEFAULT_K = 2
OUTLIER_STD = 2.0
TREE_DEPTH = 3


# Restrict clustering to the configured class, if specified
def _restrict(df, config):
    label = config.discovery_restrict_to_label

    if label is None:
        return df

    if label == "__not_positive__":
        label = next(
            v for v in df["class"].unique()
            if v != config.positive_label
        )

    return df[df["class"] == label].reset_index(drop=True)


def fit_default_subgroups(
    dataset_name,
    config,
    k=DEFAULT_K,
    random_state=RANDOM_STATE,
):
    df = _restrict(config.loader(), config)

    cat_cols = config.categorical_cols
    real = df.drop(columns="class")
    real[cat_cols] = real[cat_cols].astype("object")
    num_cols = [c for c in real.columns if c not in cat_cols]

    # Impute, encode, and standardize
    medians = real[num_cols].median()
    modes = real[cat_cols].mode().iloc[0]

    imputed = real.copy()
    imputed[num_cols] = imputed[num_cols].fillna(medians)
    imputed[cat_cols] = imputed[cat_cols].fillna(modes)

    encoder = CategoricalEncoder().fit(imputed)
    encoded = encoder.transform(imputed)

    scaler = StandardScaler().fit(encoded)
    scaled = scaler.transform(encoded)

    # KMeans
    kmeans = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=random_state,
    ).fit(scaled)

    cluster_labels = kmeans.labels_
    clusters = sorted(set(cluster_labels))

    print(
        f"[{dataset_name}] cluster sizes: "
        f"{pd.Series(cluster_labels).value_counts().sort_index().to_dict()}"
    )

    print(
        f"[{dataset_name}] silhouette: "
        f"{silhouette_score(scaled, cluster_labels):.3f} "
        "(reported only -- K is fixed above, not chosen by this score)"
    )

    # Stability across 10 random seeds
    runs = [
        KMeans(
            n_clusters=k,
            n_init=10,
            random_state=s,
        ).fit_predict(scaled)
        for s in range(10)
    ]

    aris = [
        adjusted_rand_score(a, b)
        for a, b in itertools.combinations(runs, 2)
    ]

    print(
        f"[{dataset_name}] seed stability (10 seeds): "
        f"mean ARI={np.mean(aris):.3f}"
    )

    # Descriptive statistics for the resulting clusters
    cluster_stats = {}

    for c in clusters:
        cluster = real.loc[cluster_labels == c]

        cluster_stats[c] = {
            "n": len(cluster),
            "numeric_medians": cluster[num_cols].median().to_dict(),
            "categorical_modes": cluster[cat_cols].mode().iloc[0].to_dict(),
        }

        print(f"[{dataset_name}] cluster {c} descriptive statistics:")
        print(f"  n={cluster_stats[c]['n']}")
        print(
            f"  numeric medians: "
            f"{cluster_stats[c]['numeric_medians']}"
        )
        print(
            f"  categorical modes: "
            f"{cluster_stats[c]['categorical_modes']}"
        )

    # Decision tree describing the cluster assignments
    tree_features = encoder.transform(real)

    cluster_tree = DecisionTreeClassifier(
        max_depth=TREE_DEPTH,
        random_state=random_state,
    ).fit(tree_features, cluster_labels)

    print(
        f"[{dataset_name}] cluster explanation tree fidelity: "
        f"{cluster_tree.score(tree_features, cluster_labels):.1%}"
    )

    # Distance-based outlier rule
    distances = np.linalg.norm(
        scaled - kmeans.cluster_centers_[cluster_labels],
        axis=1,
    )

    outlier_mask = np.zeros(len(scaled), dtype=bool)
    cluster_cutoffs = {}

    for c in clusters:
        mask = cluster_labels == c
        cluster_distances = distances[mask]

        cutoff = (
            cluster_distances.mean()
            + OUTLIER_STD * cluster_distances.std()
        )

        cluster_cutoffs[c] = cutoff
        outlier_mask[mask] = cluster_distances > cutoff

    print(
        f"[{dataset_name}] flagged "
        f"{outlier_mask.sum()} outliers out of {len(scaled)}"
    )

    # Decision tree describing the outlier rule
    outlier_tree = DecisionTreeClassifier(
        max_depth=TREE_DEPTH,
        random_state=random_state,
    ).fit(
        tree_features,
        outlier_mask.astype(int),
    )

    print(
        f"[{dataset_name}] outlier-rule tree fidelity: "
        f"{outlier_tree.score(tree_features, outlier_mask.astype(int)):.1%}"
    )

    return {
        "outlier_tree": outlier_tree,
        "encoder": encoder,
        "cat_cols": cat_cols,
        "kmeans": kmeans,
        "scaler": scaler,
        "medians": medians,
        "modes": modes,
        "num_cols": num_cols,
        "cluster_cutoffs": cluster_cutoffs,
        "cluster_tree": cluster_tree,
        "cluster_stats": cluster_stats,
    }


def save_outlier_rule(dataset_name, config, out_path, k=DEFAULT_K):
    saved = fit_default_subgroups(dataset_name, config, k=k)
    joblib.dump(saved, out_path)
    print(f"Saved {out_path}")