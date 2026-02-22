# DataLens AI

An autonomous data analysis platform powered by a LangChain ReAct agent and Google Gemini. Upload a structured dataset and ask questions in plain English — the agent selects and chains the appropriate analytical tools, returns precise results, and can render Vega-Lite chart specifications for the frontend.

## Architecture

```
backend/
  main.py              FastAPI application entry point
  config.py            Settings loaded from .env via pydantic-settings
  session.py           In-memory session store (upload → session_id)
  schemas.py           Pydantic request / response / tool-input models
  streaming.py         WebSocket streaming callback for LangChain events
  agent/
    analyst_agent.py   ReAct agent construction (Gemini LLM + tools + prompt)
  routes/
    upload.py          POST /upload, GET /sessions/{session_id}
    chat.py            POST /chat/{session_id}, WS /ws/{session_id}
  tools/
    guards.py          Shared session guard utility
    dataset_tools.py   describe_dataset, generate_chart_spec
    statistical_tools.py  descriptive_stats, group_by_stats, correlation_matrix, value_counts, outliers_summary
    ml_tools.py        run_pca, run_kmeans, detect_anomalies, run_regression, run_classification
    time_series_tools.py  check_stationarity, run_forecast, decompose_time_series
  analyzers/
    statistical.py     Core statistical computation functions
    ml.py              PCA, clustering, regression, classification, anomaly detection
    time_series/       Preprocessing, decomposition, stationarity, forecasting
```

## Requirements

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager
- Google Gemini API key

## Setup

```bash
# Create and activate virtual environment
uv venv
.venv\Scripts\Activate   # Windows
source .venv/bin/activate # Unix

# Install dependencies
uv sync
```

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

## Running

```bash
uv run uvicorn backend.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`. Interactive docs at `/docs`.

## API Reference

### Upload a dataset

```
POST /upload
Content-Type: multipart/form-data

file: <CSV | XLSX | XLS | Parquet>
```

Returns a `session_id` that is used for all subsequent requests.

### Chat (HTTP)

```
POST /chat/{session_id}
Content-Type: application/json

{ "message": "What are the top spending categories?" }
```

Returns the agent's final answer and the intermediate tool steps.

### Chat (WebSocket)

```
WS /ws/{session_id}
```

Streams agent events as JSON frames: `thought`, `tool_call`, `tool_result`, `answer`, `error`, `done`.

### Session info

```
GET /sessions/{session_id}
```

Returns dataset metadata (shape, columns, dtypes) without the raw data.

### Health check

```
GET /health
```

## Supported File Types

| Format  | Extension       |
| ------- | --------------- |
| CSV     | `.csv`          |
| Excel   | `.xlsx`, `.xls` |
| Parquet | `.parquet`      |

Maximum upload size is configurable via `MAX_UPLOAD_MB` (default: 100 MB).

## Agent Tools

| Tool                    | Description                                                             |
| ----------------------- | ----------------------------------------------------------------------- |
| `describe_dataset`      | Schema, dtypes, null counts, sample rows, numeric summary               |
| `generate_chart_spec`   | Vega-Lite v5 specification for scatter, line, bar, histogram, box plots |
| `descriptive_stats`     | Mean, std, min, max, skew, kurtosis per column                          |
| `group_by_stats`        | Aggregation (mean / sum / count / etc.) grouped by a categorical column |
| `correlation_matrix`    | Pearson correlation matrix                                              |
| `value_counts`          | Top-N most frequent values in a column                                  |
| `outliers_summary`      | Outlier detection via IQR or Z-score                                    |
| `run_pca`               | PCA with explained variance and 2D/3D projection coordinates            |
| `run_kmeans`            | K-means clustering with silhouette score                                |
| `detect_anomalies`      | Isolation Forest anomaly detection                                      |
| `run_regression`        | Random Forest regression — R², RMSE, feature importance                 |
| `run_classification`    | Random Forest classification — accuracy, per-class metrics              |
| `check_stationarity`    | ADF + KPSS stationarity tests                                           |
| `run_forecast`          | ARIMA or Prophet forecast with confidence intervals                     |
| `decompose_time_series` | Trend / seasonal / residual decomposition                               |

## Running Tests

```bash
uv run pytest
```
