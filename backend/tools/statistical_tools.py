from typing import Optional
from langchain_core.tools import tool
from ..schemas import DescriptiveStatsInput, GroupByInput, CorrelationInput, ValueCountsInput, OutliersInput
from ..analyzers import statistical
from .guards import require_df


@tool("descriptive_stats", args_schema=DescriptiveStatsInput)
def descriptive_stats(session_id: Optional[str] = None, columns: list[str] | None = None) -> dict:
    """Get mean, std, min, max, count, skew, and kurtosis for numeric columns."""
    df, err = require_df(session_id)
    if err:
        return err
    return statistical.descriptive_stats(df, columns)


@tool("group_by_stats", args_schema=GroupByInput)
def group_by_stats(
    session_id: Optional[str] = None,
    group_column: Optional[str] = None,
    agg_column: Optional[str] = None,
    agg_func: str = "mean",
) -> dict:
    """Aggregate a numeric column grouped by a categorical column."""
    if not group_column or not agg_column:
        return {"error": "group_column and agg_column are both required"}

    df, err = require_df(session_id)
    if err:
        return err
    try:
        return {"result": statistical.group_by_stats(df, group_column, agg_column, agg_func)}
    except ValueError as e:
        return {"error": str(e)}


@tool("correlation_matrix", args_schema=CorrelationInput)
def correlation_matrix(session_id: Optional[str] = None, columns: list[str] | None = None) -> dict:
    """Compute Pearson correlation matrix for numeric columns."""
    df, err = require_df(session_id)
    if err:
        return err
    try:
        return statistical.correlation_matrix(df, columns)
    except ValueError as e:
        return {"error": str(e)}


@tool("value_counts", args_schema=ValueCountsInput)
def value_counts(
    session_id: Optional[str] = None,
    column: Optional[str] = None,
    normalize: bool = False,
    limit: int = 20,
) -> dict:
    """Get the top N most frequent values in a column."""
    if not column:
        return {"error": "column is required"}

    df, err = require_df(session_id)
    if err:
        return err
    try:
        return statistical.value_counts(df, column, normalize=normalize, limit=limit)
    except ValueError as e:
        return {"error": str(e)}


@tool("outliers_summary", args_schema=OutliersInput)
def outliers_summary(
    session_id: Optional[str] = None,
    columns: list[str] | None = None,
    method: str = "iqr",
    threshold: float = 1.5,
) -> dict:
    """Detect outliers in numeric columns using IQR or Z-score method."""
    df, err = require_df(session_id)
    if err:
        return err
    return {"outliers": statistical.outliers_summary(df, columns, method=method, threshold=threshold)}
