from typing import Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report


def perform_pca(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    n_components: Optional[int] = None,
) -> dict:
    cols = columns or df.select_dtypes(include="number").columns.tolist()
    if len(cols) < 2:
        raise ValueError("Need at least 2 numeric columns for PCA")

    data = df[cols].dropna()
    if len(data) < 2:
        raise ValueError("Not enough rows after dropping NaN")

    n = min(n_components or len(cols), len(data), len(cols))
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    pca = PCA(n_components=n)
    coords = pca.fit_transform(scaled)

    return {
        "explained_variance_pct": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance_pct": np.cumsum(pca.explained_variance_ratio_).tolist(),
        "loadings": pd.DataFrame(
            pca.components_.T,
            index=cols,
            columns=[f"PC{i+1}" for i in range(n)],
        ).to_dict(),
        "coordinates": pd.DataFrame(
            coords,
            columns=[f"PC{i+1}" for i in range(n)],
        ).to_dict(orient="records"),
        "n_components": n,
        "columns": cols,
    }


def perform_clustering(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    n_clusters: int = 3,
) -> dict:
    cols = columns or df.select_dtypes(include="number").columns.tolist()
    if len(cols) < 2:
        raise ValueError("Need at least 2 numeric columns for clustering")

    data = df[cols].dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled)

    centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=cols,
    )

    from sklearn.metrics import silhouette_score
    silhouette = float(silhouette_score(scaled, labels))

    return {
        "labels": labels.tolist(),
        "cluster_centers": centers.to_dict(orient="records"),
        "inertia": float(kmeans.inertia_),
        "silhouette_score": silhouette,
        "n_clusters": n_clusters,
        "columns": cols,
    }


def detect_anomalies(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    contamination: float = 0.05,
) -> dict:
    cols = columns or df.select_dtypes(include="number").columns.tolist()
    data = df[cols].dropna()

    iso = IsolationForest(contamination=contamination, random_state=42)
    predictions = iso.fit_predict(data)
    scores = iso.decision_function(data)

    anomaly_mask = predictions == -1
    anomaly_indices = data.index[anomaly_mask].tolist()

    return {
        "anomaly_count": int(anomaly_mask.sum()),
        "anomaly_pct": round(anomaly_mask.mean() * 100, 2),
        "anomaly_indices": anomaly_indices,
        "scores": scores.tolist(),
        "columns": cols,
    }


def train_regression_model(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: Optional[list[str]] = None,
    test_size: float = 0.2,
) -> dict:
    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column!r}")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    features = feature_columns or [c for c in num_cols if c != target_column]
    if not features:
        raise ValueError("No feature columns available")

    data = df[features + [target_column]].dropna()
    X, y = data[features], data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "r2": round(float(r2_score(y_test, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        "feature_importance": dict(
            zip(features, model.feature_importances_.round(4).tolist())
        ),
        "target_column": target_column,
        "feature_columns": features,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }


def train_classification_model(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: Optional[list[str]] = None,
    test_size: float = 0.2,
) -> dict:
    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column!r}")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    features = feature_columns or [c for c in num_cols if c != target_column]
    if not features:
        raise ValueError("No feature columns available")

    data = df[features + [target_column]].dropna()
    X, y = data[features], data[target_column]
    unique_classes = np.unique(y)
    stratify = y if len(unique_classes) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "feature_importance": dict(
            zip(features, model.feature_importances_.round(4).tolist())
        ),
        "target_column": target_column,
        "feature_columns": features,
        "classes": unique_classes.tolist(),
    }
