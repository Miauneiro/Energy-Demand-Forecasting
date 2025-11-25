"""
Configuration settings for Energy Demand Forecasting Platform.

Contains API endpoints, data parameters, and model configurations.
"""

from datetime import datetime, timedelta

# ============================================================================
# API CONFIGURATION
# ============================================================================

# EIA (Energy Information Administration) API
EIA_BASE_URL = "https://api.eia.gov/v2"
EIA_ELECTRICITY_ENDPOINT = "/electricity/rto/region-data/data/"

# Supported regions
REGIONS = {
    'US48': {
        'name': 'United States Lower 48',
        'description': 'Contiguous United States',
        'timezone': 'US/Eastern'
    },
    'CAL': {
        'name': 'California ISO',
        'description': 'California Independent System Operator',
        'timezone': 'US/Pacific'
    },
    'TEX': {
        'name': 'Electric Reliability Council of Texas',
        'description': 'ERCOT - Texas Grid',
        'timezone': 'US/Central'
    }
}

# Data type codes
DATA_TYPES = {
    'D': 'Demand',
    'NG': 'Net Generation',
    'TI': 'Total Interchange'
}

# ============================================================================
# DATA FETCHING PARAMETERS
# ============================================================================

# Default date ranges
DEFAULT_YEARS = 3
DEFAULT_START_DATE = datetime.now() - timedelta(days=3*365)
DEFAULT_END_DATE = datetime.now()

# API rate limiting
MAX_RECORDS_PER_REQUEST = 5000  # EIA API limit
REQUEST_DELAY = 0.5  # Seconds between requests (be nice to API)

# Data validation
MIN_DEMAND_MW = 100000  # Sanity check - US demand never below this
MAX_DEMAND_MW = 700000  # Sanity check - US demand never above this

# ============================================================================
# FILE PATHS
# ============================================================================

# Data directories
DATA_DIR = "data"
RAW_DATA_DIR = f"{DATA_DIR}/raw"
PROCESSED_DATA_DIR = f"{DATA_DIR}/processed"

# Default output files
DEFAULT_RAW_FILE = f"{RAW_DATA_DIR}/us_demand_historical.csv"
DEFAULT_PROCESSED_FILE = f"{PROCESSED_DATA_DIR}/us_demand_processed.csv"

# Model directories
MODEL_DIR = "models"
DEFAULT_MODEL_FILE = f"{MODEL_DIR}/lstm_energy_forecaster.pth"

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Forecast configurations
FORECAST_CONFIGS = {
    'short_term': {
        'name': 'Short-Term (1-hour ahead)',
        'lookback': 24,      # hours
        'horizon': 1,        # hours
        'hidden_units': 64,
        'num_layers': 2,
        'dropout': 0.2,
        'use_case': 'Real-time operations, immediate load balancing'
    },
    'daily': {
        'name': 'Day-Ahead (24-hour ahead)',
        'lookback': 48,
        'horizon': 24,
        'hidden_units': 128,
        'num_layers': 3,
        'dropout': 0.3,
        'use_case': 'Day-ahead planning, staff scheduling, generator commitment'
    },
    'weekly': {
        'name': 'Weekly (7-day ahead)',
        'lookback': 168,     # 1 week
        'horizon': 168,
        'hidden_units': 256,
        'num_layers': 4,
        'dropout': 0.3,
        'use_case': 'Weekly operations, maintenance planning, fuel procurement'
    }
}

# Training parameters
TRAINING_DEFAULTS = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 100,
    'early_stopping_patience': 10,
    'validation_split': 0.2,
    'test_split': 0.1
}

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

# Time features to extract
TIME_FEATURES = [
    'hour',          # 0-23
    'day_of_week',   # 0-6 (Monday=0)
    'day_of_month',  # 1-31
    'month',         # 1-12
    'is_weekend',    # 0 or 1
    'is_holiday',    # 0 or 1 (if holiday calendar available)
]

# Normalization method
NORMALIZATION_METHOD = 'minmax'  # or 'standard'

# ============================================================================
# VISUALIZATION
# ============================================================================

# Plot styling
PLOT_STYLE = {
    'figure_size': (14, 6),
    'color_actual': '#2E86AB',
    'color_forecast': '#A23B72',
    'color_baseline': '#F18F01',
    'line_width': 2,
    'grid_alpha': 0.3
}
