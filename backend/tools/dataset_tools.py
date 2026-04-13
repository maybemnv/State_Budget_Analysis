from langchain_core.tools import tool
import pandas as pd
from ..schemas import DescribeDatasetInput, GenerateChartSpecInput
from ..session import get_session
from ..db.database import get_db
from ..analyzers.statistical import missing_values_summary
from .guards import require_df
from typing import Optional

MAX_SAMPLE_ROWS = 3
MAX_CHART_DATA_ROWS = 2000


@tool("describe_dataset", args_schema=DescribeDatasetInput)
async def describe_dataset(session_id: Optional[str] = None) -> dict:
    """Always call this first. Returns schema, dtypes, null counts, sample rows, and column summary."""
    async with get_db() as db:
        session = await get_session(session_id, db)

    if session is None:
        return {"error": f"Session {session_id!r} not found"}

    df, err = await require_df(session_id, db)
    if err:
        return err

    num_summary = {}
    for col in session["schema"]["numeric_columns"]:
        if col not in df.columns:
            continue
        s = df[col].describe()
        # Handle all-NaN case where describe() returns empty or NaN
        if len(s) == 0 or pd.isna(s.get("count", 0)):
            continue
        num_summary[col] = {
            "count": int(s["count"]),
            "mean": round(float(s.get("mean", 0)), 2),
            "std": round(float(s.get("std", 0)), 2),
            "min": round(float(s.get("min", 0)), 2),
            "max": round(float(s.get("max", 0)), 2),
        }

    missing = missing_values_summary(df)
    non_zero_missing = {item["column"]: item["missing"] for item in missing}

    return {
        "filename": session["filename"],
        "shape": session["schema"]["shape"],
        "columns": session["schema"]["columns"],
        "numeric_columns": session["schema"]["numeric_columns"],
        "categorical_columns": session["schema"]["categorical_columns"],
        "dtypes": session["schema"]["dtypes"],
        "missing_values_total": session["schema"]["missing_values"],
        "missing_by_column": non_zero_missing,
        "sample": df.head(MAX_SAMPLE_ROWS).to_dict(orient="records"),
        "numeric_summary": num_summary,
    }


@tool("generate_chart_spec", args_schema=GenerateChartSpecInput)
async def generate_chart_spec(
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

    df, err = await require_df(session_id)
    if err:
        return err

    for col in (x_column, y_column, color_column):
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
            "data": {"values": df[data_cols].head(MAX_CHART_DATA_ROWS).to_dict(orient="records")},
        },
        "chart_type": chart_type,
    }
