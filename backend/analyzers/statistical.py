from typing import Optional
import pandas as pd
import numpy as np


def descriptive_stats(df: pd.DataFrame, columns: Optional[list[str]] = None) -> dict:
    cols = columns or df.select_dtypes(include="number").columns.tolist()
    if not cols:
        return {}
    result = df[cols].describe().T
    result["skew"] = df[cols].skew()
    result["kurtosis"] = df[cols].kurtosis()
    return result.to_dict()


def group_by_stats(
    df: pd.DataFrame,
    group_column: str,
    agg_column: str,
    agg_func: str = "mean",
) -> dict:
    if group_column not in df.columns or agg_column not in df.columns:
        raise ValueError(f"Column not found: {group_column!r} or {agg_column!r}")
    grouped = (
        df.groupby(group_column)[agg_column]
        .agg(agg_func)
        .sort_values(ascending=False)
        .reset_index()
    )
    return grouped.to_dict(orient="records")


def correlation_matrix(df: pd.DataFrame, columns: Optional[list[str]] = None) -> dict:
    cols = columns or df.select_dtypes(include="number").columns.tolist()
    if len(cols) < 2:
        raise ValueError("Need at least 2 numeric columns for correlation")
    return df[cols].corr().to_dict()


def value_counts(
    df: pd.DataFrame,
    column: str,
    normalize: bool = False,
    limit: int = 20,
) -> dict:
    if column not in df.columns:
        raise ValueError(f"Column not found: {column!r}")
    counts = df[column].value_counts(normalize=normalize).head(limit)
    return counts.to_dict()


def missing_values_summary(df: pd.DataFrame) -> list[dict]:
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    return (
        pd.DataFrame({"column": missing.index, "missing": missing.values, "pct": pct.values})
        .query("missing > 0")
        .sort_values("missing", ascending=False)
        .to_dict(orient="records")
    )


def outliers_summary(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    method: str = "iqr",
    threshold: float = 1.5,
) -> list[dict]:
    cols = columns or df.select_dtypes(include="number").columns.tolist()
    rows = []
    for col in cols:
        series = df[col].dropna()
        if method == "iqr":
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            mask = (series < q1 - threshold * iqr) | (series > q3 + threshold * iqr)
        else:
            z = (series - series.mean()) / series.std()
            mask = z.abs() > threshold
        outliers = series[mask]
        if len(outliers):
            rows.append({
                "column": col,
                "count": int(len(outliers)),
                "pct": round(len(outliers) / len(df) * 100, 2),
                "min": float(outliers.min()),
                "max": float(outliers.max()),
            })
    return rows
