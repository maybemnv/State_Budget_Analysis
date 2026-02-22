from typing import Any, Optional
from pydantic import BaseModel, Field


class DescriptiveStatsInput(BaseModel):
    session_id: str
    columns: Optional[list[str]] = None


class GroupByInput(BaseModel):
    session_id: str
    group_column: str
    agg_column: str
    agg_func: str = Field(default="mean", pattern="^(mean|median|sum|min|max|count|std)$")


class CorrelationInput(BaseModel):
    session_id: str
    columns: Optional[list[str]] = None


class ValueCountsInput(BaseModel):
    session_id: str
    column: str
    normalize: bool = False
    limit: int = Field(default=20, ge=1, le=200)


class OutliersInput(BaseModel):
    session_id: str
    columns: Optional[list[str]] = None
    method: str = Field(default="iqr", pattern="^(iqr|zscore)$")
    threshold: float = 1.5


class PCAInput(BaseModel):
    session_id: str
    columns: Optional[list[str]] = None
    n_components: Optional[int] = Field(default=None, ge=2)


class ClusteringInput(BaseModel):
    session_id: str
    columns: Optional[list[str]] = None
    n_clusters: int = Field(default=3, ge=2, le=20)


class RegressionInput(BaseModel):
    session_id: str
    target_column: str
    feature_columns: Optional[list[str]] = None
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)


class ClassificationInput(BaseModel):
    session_id: str
    target_column: str
    feature_columns: Optional[list[str]] = None
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)


class AnomalyDetectionInput(BaseModel):
    session_id: str
    columns: Optional[list[str]] = None
    contamination: float = Field(default=0.05, gt=0.0, lt=0.5)


class TimeSeriesInput(BaseModel):
    session_id: str
    date_column: str
    value_column: str


class ForecastInput(BaseModel):
    session_id: str
    date_column: str
    value_column: str
    steps: int = Field(default=12, ge=1)
    model: str = Field(default="arima", pattern="^(arima|prophet)$")


class StationarityInput(BaseModel):
    session_id: str
    date_column: str
    value_column: str


class DescribeDatasetInput(BaseModel):
    session_id: str


class GenerateChartSpecInput(BaseModel):
    session_id: str
    chart_type: str = Field(description="One of: scatter, line, bar, histogram, box")
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    color_column: Optional[str] = None
    title: Optional[str] = None


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
