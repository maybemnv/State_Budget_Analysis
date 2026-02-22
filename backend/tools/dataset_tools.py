from langchain_core.tools import tool
from ..schemas import DescribeDatasetInput, GenerateChartSpecInput
from ..session import get_df, get_session
from ..analyzers.statistical import missing_values_summary
from .guards import require_df
from typing import Optional


@tool("describe_dataset", args_schema=DescribeDatasetInput)
def describe_dataset(session_id: str) -> dict:
    """Always call this first. Returns schema, dtypes, null counts, sample rows, and column summary."""
    session = get_session(session_id)
    if session is None:
        return {"error": f"Session {session_id!r} not found"}
    df = session["df"]
    meta = session["metadata"]
    return {
        "filename": meta["filename"],
        "shape": meta["shape"],
        "columns": meta["columns"],
        "numeric_columns": meta["numeric_columns"],
        "categorical_columns": meta["categorical_columns"],
        "dtypes": meta["dtypes"],
        "missing_values_total": meta["missing_values"],
        "missing_by_column": missing_values_summary(df),
        "sample": df.head(5).to_dict(orient="records"),
        "numeric_summary": df.describe().round(4).to_dict() if meta["numeric_columns"] else {},
    }


@tool("generate_chart_spec", args_schema=GenerateChartSpecInput)
def generate_chart_spec(
    session_id: str,
    chart_type: Optional[str] = None,
    x_column: Optional[str] = None,
    y_column: Optional[str] = None,
    color_column: Optional[str] = None,
    title: Optional[str] = None,
) -> dict:
    """Generate a Vega-Lite chart spec for the frontend. chart_type: scatter, line, bar, histogram, box."""
    if not chart_type:
        return {"error": "chart_type is required. Choose from: scatter, line, bar, histogram, box"}

    df, err = require_df(session_id)
    if err:
        return err

    for col in (x_column, y_column):
        if col and col not in df.columns:
            return {"error": f"Column not found: {col!r}"}

    mark_map = {
        "scatter": "point",
        "line": "line",
        "bar": "bar",
        "histogram": "bar",
        "box": "boxplot",
        "heatmap": "rect",
    }

    enc: dict = {}
    if x_column:
        col_type = (
            "temporal" if "date" in x_column.lower()
            else "quantitative" if df[x_column].dtype.kind in "ifc"
            else "nominal"
        )
        enc["x"] = {"field": x_column, "type": col_type}

    if y_column:
        enc["y"] = {"field": y_column, "type": "quantitative"}
    elif chart_type == "histogram" and x_column:
        enc["y"] = {"aggregate": "count", "type": "quantitative"}

    if chart_type == "heatmap":
        enc["color"] = {"aggregate": "count", "type": "quantitative"}
    elif color_column and color_column in df.columns:
        enc["color"] = {"field": color_column, "type": "nominal"}

    data_cols = [c for c in (x_column, y_column, color_column) if c]
    return {
        "chart_spec": {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "title": title or f"{chart_type.capitalize()}: {x_column or ''}/{y_column or ''}",
            "mark": mark_map.get(chart_type, "point"),
            "encoding": enc,
            "data": {"values": df[data_cols].head(2000).to_dict(orient="records")},
        },
        "chart_type": chart_type,
    }
