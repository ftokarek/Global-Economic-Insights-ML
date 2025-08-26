from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

FEATURES = [
    "GDP",
    "Interest Rate",
    "Inflation Rate",
    "Jobless Rate",
    "Gov. Budget",
    "Debt/GDP",
    "Current Account",
    "Population",
]

@dataclass
class ClusterResult:
    labels: np.ndarray
    centers: pd.DataFrame
    pca_coords: pd.DataFrame
    profile: pd.DataFrame


def fit_kmeans_pca(df: pd.DataFrame, k: int = 5, random_state: int = 42) -> ClusterResult:
    data = df.dropna(subset=FEATURES).copy()
    if data.empty:
        return ClusterResult(
            labels=np.array([]),
            centers=pd.DataFrame(columns=[*FEATURES, "cluster"]),
            pca_coords=pd.DataFrame(columns=["name", "region", "subregion", "PC1", "PC2", "cluster"]),
            profile=pd.DataFrame(columns=["cluster", *FEATURES]),
        )

    X = data[FEATURES].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(Xs)
    pca_df = pd.DataFrame(coords, columns=["PC1", "PC2"], index=data.index)

    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(Xs)

    centers_scaled = km.cluster_centers_
    centers_unscaled = pd.DataFrame(scaler.inverse_transform(centers_scaled), columns=FEATURES)
    centers_unscaled["cluster"] = np.arange(k)

    prof = (
        data.assign(cluster=labels)
        .groupby("cluster")[FEATURES]
        .median()
        .reset_index()
        .sort_values("cluster")
    )

    pca_coords = data[["name", "region", "subregion"]].join(pca_df)
    pca_coords["cluster"] = labels

    return ClusterResult(labels=labels, centers=centers_unscaled, pca_coords=pca_coords, profile=prof)