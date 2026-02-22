import pandas as pd
import numpy as np
import pytest
from backend.analyzers import statistical


@pytest.fixture
def df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "a": rng.normal(100, 15, 100),
        "b": rng.normal(50, 10, 100),
        "cat": rng.choice(["x", "y", "z"], 100),
    })


def test_descriptive_stats_returns_expected_keys(df):
    result = statistical.descriptive_stats(df)
    assert "a" in result
    assert "mean" in result["a"]
    assert "skew" in result["a"]


def test_descriptive_stats_empty_columns(df):
    result = statistical.descriptive_stats(df, columns=[])
    assert result == {}


def test_group_by_stats(df):
    result = statistical.group_by_stats(df, "cat", "a", "mean")
    assert isinstance(result, list)
    assert len(result) == 3
    assert "cat" in result[0]
    assert "a" in result[0]


def test_group_by_stats_invalid_column(df):
    with pytest.raises(ValueError):
        statistical.group_by_stats(df, "nonexistent", "a")


def test_correlation_matrix(df):
    result = statistical.correlation_matrix(df)
    assert "a" in result
    assert "b" in result["a"]
    assert abs(result["a"]["a"] - 1.0) < 1e-9


def test_correlation_matrix_insufficient_columns(df):
    with pytest.raises(ValueError):
        statistical.correlation_matrix(df, columns=["a"])


def test_value_counts(df):
    result = statistical.value_counts(df, "cat")
    assert isinstance(result, dict)
    assert sum(result.values()) == 100


def test_missing_values_summary_no_missing(df):
    result = statistical.missing_values_summary(df)
    assert result == []


def test_missing_values_summary_with_missing():
    df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, 3]})
    result = statistical.missing_values_summary(df)
    cols = [r["column"] for r in result]
    assert "b" in cols


def test_outliers_summary_iqr(df):
    # Inject obvious outlier
    df.loc[0, "a"] = 9999
    result = statistical.outliers_summary(df, columns=["a"])
    assert any(r["column"] == "a" for r in result)


def test_outliers_summary_zscore(df):
    df.loc[0, "b"] = -9999
    result = statistical.outliers_summary(df, columns=["b"], method="zscore", threshold=3.0)
    assert any(r["column"] == "b" for r in result)
