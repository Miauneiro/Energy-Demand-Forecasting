"""
Energy Demand Forecasting Platform

A production-ready LSTM-based electricity demand forecasting system.
"""

__version__ = "0.1.0"
__author__ = "João Manero"

from .data_fetcher import EIADataFetcher
from .config import FORECAST_CONFIGS, REGIONS

__all__ = ['EIADataFetcher', 'FORECAST_CONFIGS', 'REGIONS']
