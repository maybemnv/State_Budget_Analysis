import json
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


class SessionUnwrapper(BaseModel):
    """Mixin to handle the common 'Action Input: {"session_id": "{JSON}"}' LLM error.

    If the LLM puts the whole JSON object string into the session_id field,
    this validator extracts the actual UUID before Pydantic fails validation.
    """
    session_id: str

    @field_validator("session_id", mode="before")
    @classmethod
    def _unwrap_session_id(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip().startswith("{"):
            try:
                data = json.loads(v)
                if isinstance(data, dict):
                    return data.get("session_id", v)
            except json.JSONDecodeError:
                pass
        return v


# --- Statistical tool schemas ---

class DescriptiveStatsInput(SessionUnwrapper):
    columns: Optional[list[str]] = None


class GroupByInput(SessionUnwrapper):
    group_column: str
    agg_column: str
    agg_func: str = Field(default="mean", pattern="^(mean|median|sum|min|max|count|std)$")


class CorrelationInput(SessionUnwrapper):
    columns: Optional[list[str]] = None


class ValueCountsInput(SessionUnwrapper):
    column: str
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
    target_column: str
    feature_columns: Optional[list[str]] = None
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)


class ClassificationInput(SessionUnwrapper):
    target_column: str
    feature_columns: Optional[list[str]] = None
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)


class AnomalyDetectionInput(SessionUnwrapper):
    columns: Optional[list[str]] = None
    contamination: float = Field(default=0.05, gt=0.0, lt=0.5)


# --- Time series tool schemas ---

class TimeSeriesInput(SessionUnwrapper):
    date_column: str
    value_column: str


class ForecastInput(SessionUnwrapper):
    date_column: str
    value_column: str
    steps: int = Field(default=12, ge=1)
    model: str = Field(default="arima", pattern="^(arima|prophet)$")


class StationarityInput(SessionUnwrapper):
    date_column: str
    value_column: str


# --- Dataset + chart tool schemas ---

class DescribeDatasetInput(SessionUnwrapper):
    pass


class GenerateChartSpecInput(SessionUnwrapper):
    chart_type: str = Field(description="One of: scatter, line, bar, histogram, heatmap, box")
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
