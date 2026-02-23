import json
from typing import Any, Optional
from pydantic import BaseModel, Field, model_validator


class SessionUnwrapper(BaseModel):
    session_id: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _unwrap_payload(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        sid = data.get("session_id")
        if isinstance(sid, str) and sid.strip().startswith("{"):
            try:
                nested = json.loads(sid)
                if isinstance(nested, dict):
                    # Merge nested keys into the top-level dict
                    # This fixes cases where the LLM does:
                    # {"session_id": "{\"session_id\": \"...\", \"group_column\": \"...\"}"}
                    return {**nested, **data, "session_id": nested.get("session_id", sid)}
            except json.JSONDecodeError:
                pass
        return data


# --- Statistical tool schemas ---

class DescriptiveStatsInput(SessionUnwrapper):
    columns: Optional[list[str]] = None


class GroupByInput(SessionUnwrapper):
    group_column: Optional[str] = None
    agg_column: Optional[str] = None
    agg_func: str = Field(default="mean", pattern="^(mean|median|sum|min|max|count|std)$")


class CorrelationInput(SessionUnwrapper):
    columns: Optional[list[str]] = None


class ValueCountsInput(SessionUnwrapper):
    column: Optional[str] = None
    normalize: bool = False
    limit: int = Field(default=20, ge=1, le=200)


class OutliersInput(SessionUnwrapper):
    columns: Optional[list[str]] = None
    method: str = Field(default="iqr", pattern="^(iqr|zscore)$")
    threshold: float = 1.5


# --- ML tool schemas ---

class PCAInput(SessionUnwrapper):
    columns: Optional[list[str]] = None
    n_components: Optional[int] = Field(default=None, ge=2)


class ClusteringInput(SessionUnwrapper):
    columns: Optional[list[str]] = None
    n_clusters: int = Field(default=3, ge=2, le=20)


class RegressionInput(SessionUnwrapper):
    target_column: Optional[str] = None
    feature_columns: Optional[list[str]] = None
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)


class ClassificationInput(SessionUnwrapper):
    target_column: Optional[str] = None
    feature_columns: Optional[list[str]] = None
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)


class AnomalyDetectionInput(SessionUnwrapper):
    columns: Optional[list[str]] = None
    contamination: float = Field(default=0.05, gt=0.0, lt=0.5)


# --- Time series tool schemas ---

class TimeSeriesInput(SessionUnwrapper):
    date_column: Optional[str] = None
    value_column: Optional[str] = None


class ForecastInput(SessionUnwrapper):
    date_column: Optional[str] = None
    value_column: Optional[str] = None
    steps: int = Field(default=12, ge=1)
    model: str = Field(default="arima", pattern="^(arima|prophet)$")


class StationarityInput(SessionUnwrapper):
    date_column: Optional[str] = None
    value_column: Optional[str] = None


# --- Dataset + chart tool schemas ---

class DescribeDatasetInput(SessionUnwrapper):
    pass


class GenerateChartSpecInput(SessionUnwrapper):
    chart_type: Optional[str] = Field(default=None, description="One of: scatter, line, bar, histogram, heatmap, box")
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    color_column: Optional[str] = None
    title: Optional[str] = None


# --- API request / response models ---

class UploadResponse(BaseModel):
    session_id: str
    filename: str
    rows: int
    columns: int
    column_names: list[str]


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    chart_spec: Optional[dict[str, Any]] = None
    has_error: bool = False
    steps: list[dict[str, Any]] = []
