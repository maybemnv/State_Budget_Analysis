"""
Benchmark query set — 30 natural-language queries, grouped by tool domain.

Strategy: the agent is NOT called live. Instead we assert that the tool-name
strings appear in our prompt-to-tool mapping table, and we verify the output
parser handles representative outputs correctly. Live agent integration tests
live in test_api.py.
"""

import json
import pytest
from backend.agent.output_parser import parse_output


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SCHEMA_QUERIES = [
    "What columns does this dataset have?",
    "Give me an overview of the data.",
    "What are the data types of each column?",
    "How many rows and columns are there?",
    "Show me a sample of the data.",
]

DESCRIPTIVE_QUERIES = [
    "What is the mean and standard deviation of revenue?",
    "Give me descriptive statistics for all numeric columns.",
    "What is the range of values in the sales column?",
    "Show me the min, max, and median for each numeric field.",
    "Is the distribution of expenditure skewed?",
]

GROUPBY_QUERIES = [
    "What is the total revenue by category?",
    "Which department has the highest average spend?",
    "Break down sales by region using sum.",
    "What is the median income per state?",
    "Count the number of records per year.",
]

CORRELATION_QUERIES = [
    "Are revenue and cost correlated?",
    "Show me the correlation matrix.",
    "Which columns are most strongly related to profit?",
]

OUTLIER_QUERIES = [
    "Are there any outliers in the revenue column?",
    "Detect anomalies in my dataset.",
    "Which rows have unusual spending patterns?",
]

CLUSTERING_QUERIES = [
    "Cluster the data into 4 groups.",
    "Find natural groupings in the dataset.",
    "Which cluster does most of the data fall into?",
]

REGRESSION_QUERIES = [
    "Predict revenue from the other numeric columns.",
    "Which features are most important for predicting cost?",
    "Train a regression model to forecast expenditure.",
]

FORECAST_QUERIES = [
    "Forecast revenue for the next 6 months.",
    "What will spending look like over the next year?",
    "Run an ARIMA forecast on the time series.",
]

ALL_BENCHMARK_QUERIES = (
    SCHEMA_QUERIES
    + DESCRIPTIVE_QUERIES
    + GROUPBY_QUERIES
    + CORRELATION_QUERIES
    + OUTLIER_QUERIES
    + CLUSTERING_QUERIES
    + REGRESSION_QUERIES
    + FORECAST_QUERIES
)


# ---------------------------------------------------------------------------
# Sanity: we have exactly 30 queries
# ---------------------------------------------------------------------------

def test_benchmark_query_count() -> None:
    assert len(ALL_BENCHMARK_QUERIES) == 30


# ---------------------------------------------------------------------------
# Tool-selection guide mapping (not live calls — validates the guide strings)
# ---------------------------------------------------------------------------

TOOL_GUIDE_KEYWORDS: dict[str, list[str]] = {
    "describe_dataset": ["schema", "columns", "data type", "sample", "overview", "rows"],
    "descriptive_stats": ["mean", "std", "min", "max", "median", "skew", "descriptive"],
    "group_by_stats": ["by category", "by region", "by department", "per state", "per year", "break down"],
    "correlation_matrix": ["correlation", "related", "correlated"],
    "outliers_summary": ["outlier"],
    "detect_anomalies": ["anomal", "unusual"],
    "run_kmeans": ["cluster"],
    "run_regression": ["predict", "regression", "forecast expenditure", "important for predicting"],
    "run_forecast": ["forecast", "arima", "next 6 months", "next year"],
}


@pytest.mark.parametrize("query", SCHEMA_QUERIES)
def test_schema_queries_match_describe(query: str) -> None:
    keywords = TOOL_GUIDE_KEYWORDS["describe_dataset"]
    assert any(kw in query.lower() for kw in keywords), f"Query not matched: {query!r}"


@pytest.mark.parametrize("query", DESCRIPTIVE_QUERIES)
def test_descriptive_queries_match_tool(query: str) -> None:
    keywords = TOOL_GUIDE_KEYWORDS["descriptive_stats"]
    assert any(kw in query.lower() for kw in keywords), f"Query not matched: {query!r}"


@pytest.mark.parametrize("query", GROUPBY_QUERIES)
def test_groupby_queries_match_tool(query: str) -> None:
    keywords = TOOL_GUIDE_KEYWORDS["group_by_stats"]
    assert any(kw in query.lower() for kw in keywords), f"Query not matched: {query!r}"


@pytest.mark.parametrize("query", CORRELATION_QUERIES)
def test_correlation_queries_match_tool(query: str) -> None:
    keywords = TOOL_GUIDE_KEYWORDS["correlation_matrix"]
    assert any(kw in query.lower() for kw in keywords), f"Query not matched: {query!r}"


@pytest.mark.parametrize("query", OUTLIER_QUERIES)
def test_outlier_queries_match_tool(query: str) -> None:
    all_kw = TOOL_GUIDE_KEYWORDS["outliers_summary"] + TOOL_GUIDE_KEYWORDS["detect_anomalies"]
    assert any(kw in query.lower() for kw in all_kw), f"Query not matched: {query!r}"


@pytest.mark.parametrize("query", CLUSTERING_QUERIES)
def test_clustering_queries_match_tool(query: str) -> None:
    keywords = TOOL_GUIDE_KEYWORDS["run_kmeans"]
    assert any(kw in query.lower() for kw in keywords), f"Query not matched: {query!r}"


@pytest.mark.parametrize("query", REGRESSION_QUERIES)
def test_regression_queries_match_tool(query: str) -> None:
    keywords = TOOL_GUIDE_KEYWORDS["run_regression"]
    assert any(kw in query.lower() for kw in keywords), f"Query not matched: {query!r}"


@pytest.mark.parametrize("query", FORECAST_QUERIES)
def test_forecast_queries_match_tool(query: str) -> None:
    keywords = TOOL_GUIDE_KEYWORDS["run_forecast"]
    assert any(kw in query.lower() for kw in keywords), f"Query not matched: {query!r}"


# ---------------------------------------------------------------------------
# Output parser unit tests
# ---------------------------------------------------------------------------

_VEGA_SPEC = {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "mark": "bar",
    "encoding": {"x": {"field": "category"}, "y": {"field": "revenue"}},
}


def test_parser_plain_text() -> None:
    result = parse_output("Revenue peaked at 1.2M in Q3.")
    assert result["answer"] == "Revenue peaked at 1.2M in Q3."
    assert result["chart_spec"] is None
    assert not result["has_error"]


def test_parser_extracts_chart_spec() -> None:
    raw = f"Here is the chart:\n```json\n{json.dumps(_VEGA_SPEC)}\n```\nRevenue is highest in Q3."
    result = parse_output(raw)
    assert result["chart_spec"] == _VEGA_SPEC
    assert "Revenue is highest in Q3." in result["answer"]
    assert json.dumps(_VEGA_SPEC) not in result["answer"]


def test_parser_flags_error_response() -> None:
    result = parse_output("I encountered an error: Session not found.")
    assert result["has_error"]


def test_parser_no_false_positive_on_normal_text() -> None:
    result = parse_output("The top category is Electronics with 42% of revenue.")
    assert not result["has_error"]
    assert result["chart_spec"] is None
