# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

**CSV Data Analyzer** - A comprehensive data analysis application with both Streamlit web interface and Flask API capabilities. The application performs data visualization, statistical analysis, machine learning insights, AI-powered analysis (using Google's Gemini), and time series forecasting.

The repository also contains a state budget analysis project for Haryana with data processing and visualization scripts.

## Common Commands

### Running the Application

**Streamlit Web Application (Primary Interface):**
```powershell
streamlit run app.py
```

**Flask API Mode (Alternative Interface):**
```powershell
python cli.py --mode web --port 5000 --host 0.0.0.0
```

**CLI Mode (Non-interactive analysis):**
```powershell
python cli.py --mode cli --input path/to/data.csv --analysis all --output-dir output
```

**Batch Processing Multiple Files:**
```powershell
python cli.py --mode batch --input-dir path/to/csvs --analysis all --output-dir output
```

### Development Commands

**Install Dependencies:**
```powershell
pip install -r requirements.txt
```

**Running Specific Analysis Types:**
```powershell
# Overview only
python cli.py --mode cli --input data.csv --analysis overview

# Visualizations only
python cli.py --mode cli --input data.csv --analysis visualize

# Statistical analysis only
python cli.py --mode cli --input data.csv --analysis stats

# Machine learning only
python cli.py --mode cli --input data.csv --analysis ml
```

**With Gemini AI Analysis:**
```powershell
python cli.py --mode cli --input data.csv --analysis all --gemini-key YOUR_API_KEY
```

### Budget Haryana Data Processing

**Process Raw Budget Data:**
```powershell
cd budget_hry/scripts
python process_data.py <filename>
```
This generates:
- Processed CSV files (total and percentage) in `processed_data/`
- Line plots in `plots/`

**Generate Plots:**
```powershell
cd budget_hry/scripts
python gen_plot.py
```

## Architecture Overview

### Core Application Structure

The application follows a **modular analyzer pattern** where each analysis capability is isolated into its own class:

```
DataLoader (Central Data Management)
    ↓
    ├── Visualizer (Chart/Plot Generation)
    ├── StatisticalAnalyzer (Descriptive Stats, Outliers, Missing Values)
    ├── MLAnalyzer (PCA, Clustering, Regression, Classification)
    ├── GeminiAnalyzer (AI-Powered Insights)
    └── TimeSeriesAnalyzer (Decomposition, Forecasting, Stationarity Tests)
```

### Key Architectural Patterns

1. **Shared State Management**: All analyzer modules depend on `DataLoader` which manages the data state. In Streamlit, this uses `st.session_state` to maintain data across interactions.

2. **Analyzer Independence**: Each analyzer (Visualizer, StatisticalAnalyzer, MLAnalyzer, etc.) is self-contained and doesn't depend on other analyzers.

3. **Dual Interface Support**: 
   - **Streamlit (`app.py`)**: Interactive web UI with session state
   - **Flask API (`api_handler.py`)**: RESTful API for programmatic access
   - **CLI (`cli.py`)**: Command-line batch processing

4. **Output Management**: Different execution modes write outputs to different locations:
   - Streamlit: Displays inline
   - Flask API: Saves to `static/` subdirectories
   - CLI mode: Saves to user-specified `--output-dir`

### Data Flow

1. **Upload/Load**: User uploads CSV/Excel → `DataLoader.load_data()` → Stored in session state
2. **Analysis Request**: User selects analysis → Corresponding analyzer invoked → Results generated
3. **Visualization**: Matplotlib figures created → Returned to interface → Displayed/saved
4. **ML Models**: Data prepared → Scaled → Model trained → Results + plots returned

### Critical Dependencies Between Modules

- **All analyzers require DataLoader**: They call `data_loader.get_data()`, `data_loader.get_numeric_columns()`, etc.
- **Time Series Analyzer is standalone**: Does not inherit from DataLoader pattern, operates on passed DataFrame
- **Gemini Analyzer needs API configuration**: Must call `configure(api_key)` before `analyze_data()`
- **ML Analyzer uses sklearn pipelines**: StandardScaler → PCA/KMeans/RandomForest → Results

### State Management in Streamlit

The application uses `st.session_state` for:
- `data`: The loaded pandas DataFrame
- `filename`: Name of uploaded file
- `gemini_api_key`: User's Gemini API key
- `gemini_configured`: Boolean flag for Gemini API status

**Important**: When modifying DataLoader, ensure session state variables are properly initialized in `__init__()`.

### Budget Haryana Workflow

Separate data processing pipeline for state budget analysis:

1. **Raw Data**: CSV files in `budget_hry/raw_data/`
2. **Processing**: `process_data.py` calculates totals and percentages
3. **Output**: 
   - Processed CSVs → `budget_hry/processed_data/`
   - Plots → `budget_hry/plots/`

This is a **separate workflow** from the main CSV analyzer application.

## File Organization Logic

### Main Application Files

- `app.py`: Streamlit UI entry point (primary interface)
- `cli.py`: Command-line interface and Flask server
- `api_handler.py`: Flask API request handlers
- `data_loader.py`: Data loading and preprocessing
- `visualizer.py`: Chart and plot generation
- `statistical_analyzer.py`: Statistical analysis methods
- `ml_analyzer.py`: Machine learning algorithms
- `gemini_analyzer.py`: Google Gemini AI integration
- `time_series_analyzer.py`: Time series analysis and forecasting

### Budget Haryana Files

- `budget_hry/raw_data/`: Source CSV files
- `budget_hry/scripts/`: Processing and plotting scripts
- `budget_hry/processed_data/`: Generated CSVs with totals/percentages
- `budget_hry/plots/`: Generated chart images

### Web Scraper (Separate Experiment)

- `web scrapper/`: CollegeDunia scraping experiment (unsuccessful due to bot detection)
- See `web scrapper/COLLEGEDUNIA_SCRAPING_REPORT.md` for detailed analysis

## Important Implementation Details

### Adding New Analysis Features

When adding a new analyzer:

1. Create a new class that accepts `data_loader` in `__init__()`
2. Import it in `app.py` and initialize: `new_analyzer = NewAnalyzer(data_loader)`
3. Add a new tab in the Streamlit UI
4. Add corresponding handlers in `api_handler.py` if API support is needed
5. Add CLI support in `cli.py` under the appropriate `--analysis` type

### Working with DataLoader

DataLoader provides these key methods:
- `load_data(uploaded_file)`: Load CSV/Excel from Streamlit upload
- `get_data()`: Returns the DataFrame from session state
- `get_numeric_columns()`: Returns list of numeric column names
- `get_categorical_columns()`: Returns list of categorical column names
- `get_data_info()`: Returns dict with rows, columns, memory usage, missing values

### Machine Learning Pipeline

The ML Analyzer follows this pattern for all algorithms:

1. **Select columns** to analyze
2. **Drop NaN values** from selected columns
3. **Standardize** using `StandardScaler`
4. **Apply algorithm** (PCA, KMeans, RandomForest, etc.)
5. **Generate plots** using matplotlib/seaborn
6. **Return results dict** with model, transformed data, plots, metrics

### Time Series Analysis

Time series module requires:
- Date column (must be convertible to datetime)
- Numeric value column
- Supports ARIMA and Prophet forecasting methods
- Uses Plotly for interactive visualizations (not matplotlib like other modules)

### Gemini AI Integration

Gemini analyzer generates prompts with:
- Dataset structure (`.info()`)
- Data sample (`.head(5)`)
- Statistics (`.describe()`)

Pre-defined analysis types:
- "Data Summary and Insights"
- "Correlation Analysis"
- "Trend Identification"
- "Anomaly Detection"
- "Custom Analysis" (requires custom_question parameter)

## Windows-Specific Notes

This project runs on Windows (PowerShell). Path handling:
- Use raw strings or escaped backslashes: `r"C:\path"` or `"C:\\path"`
- The CLI defaults to `--host 0.0.0.0` for Flask server
- Budget Haryana scripts use relative paths with forward slashes in code but are run from specific directories

## Testing the Application

To test the full application:

1. **Prepare test data**: Use `sample_data.csv` or any CSV file
2. **Run Streamlit**: `streamlit run app.py`
3. **Upload file** via sidebar
4. **Test each tab**:
   - Overview: Verify data preview and column info
   - Visualizations: Generate at least one chart
   - Statistical Analysis: Check descriptive stats
   - ML Insights: Run PCA or clustering
   - Gemini AI: (Requires API key)
   - Time Series: (Requires datetime column)

## Common Issues

### Streamlit Session State

If data seems to "disappear" between interactions, ensure:
- DataLoader initializes session state variables in `__init__`
- All modules call `data_loader.get_data()` instead of storing local copies

### ML Model Errors

If ML analysis fails:
- Check for sufficient numeric columns (PCA/Clustering need ≥2)
- Ensure no all-NaN columns after selecting subset
- Verify data types are actually numeric

### Time Series Issues

Time series analysis requires:
- At least 24 data points for seasonal decomposition (period=12)
- Date column must be parseable by `pd.to_datetime()`
- Data should be regularly spaced (monthly/daily frequency)

### Gemini API

Common Gemini errors:
- Invalid API key: Check key format and permissions
- Rate limiting: Add delays between requests
- Response parsing: Gemini returns markdown text, not structured data

## Dependencies

Key packages (see `requirements.txt`):
- `streamlit`: Web UI framework
- `pandas`: Data manipulation
- `matplotlib`, `seaborn`, `plotly`: Visualizations
- `scikit-learn`: Machine learning
- `google-generativeai`: Gemini AI
- `statsmodels`: Time series statistics
- `prophet`: Facebook Prophet forecasting
- `openpyxl`: Excel file support

## Project Context

This repository contains two distinct projects:

1. **CSV Data Analyzer** (main): General-purpose data analysis tool
2. **Budget Haryana**: Specific analysis of Haryana state budget data
3. **Web Scraper** (experimental): Unsuccessful attempt at CollegeDunia scraping

The main application is designed to be data-agnostic and can analyze any CSV/Excel file with appropriate structure.
