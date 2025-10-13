import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model configurations
MODEL_CONFIG = {
    'arima': {
        'default_order': (1, 1, 1),
        'default_seasonal_order': (1, 1, 1, 12)
    },
    'prophet': {
        'yearly_seasonality': 'auto',
        'weekly_seasonality': 'auto',
        'daily_seasonality': 'auto',
        'seasonality_mode': 'additive'
    }
}

# Visualization settings
PLOT_CONFIG = {
    'template': 'plotly_white',
    'colors': {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'accent': '#2ca02c',
        'error': '#d62728',
        'forecast': '#ff7f0e',
        'confidence': 'rgba(255, 127, 14, 0.2)'
    },
    'margin': dict(l=50, r=50, t=80, b=50),
    'height': 600
}

# Application settings
APP_CONFIG = {
    'max_upload_size_mb': 100,
    'allowed_extensions': ['.csv', '.xlsx', '.xls', '.parquet'],
    'default_date_format': '%Y-%m-%d',
    'timezone': 'UTC'
}
