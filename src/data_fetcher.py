"""
Data fetcher for US electricity demand from EIA API.

Handles downloading, caching, and validation of historical electricity data.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time
from tqdm import tqdm
import json

from .config import (
    EIA_BASE_URL,
    EIA_ELECTRICITY_ENDPOINT,
    MAX_RECORDS_PER_REQUEST,
    REQUEST_DELAY,
    MIN_DEMAND_MW,
    MAX_DEMAND_MW,
    RAW_DATA_DIR,
    DEFAULT_RAW_FILE
)


class EIADataFetcher:
    """
    Fetches electricity demand data from EIA API.
    
    Features:
    - Handles pagination for large date ranges
    - Caches downloaded data
    - Validates data quality
    - Progress bars for long downloads
    """
    
    def __init__(self, api_key):
        """
        Initialize the data fetcher.
        
        Args:
            api_key (str): EIA API key
        """
        self.api_key = api_key
        self.base_url = EIA_BASE_URL + EIA_ELECTRICITY_ENDPOINT
        
        # Ensure data directory exists
        Path(RAW_DATA_DIR).mkdir(parents=True, exist_ok=True)
    
    def fetch_demand_data(
        self,
        start_date,
        end_date,
        region='US48',
        save_path=None,
        show_progress=True
    ):
        """
        Fetch electricity demand data for a date range.
        
        Args:
            start_date (datetime): Start date
            end_date (datetime): End date
            region (str): Region code (default: 'US48')
            save_path (str): Path to save CSV (default: DEFAULT_RAW_FILE)
            show_progress (bool): Show progress bar
            
        Returns:
            pd.DataFrame: Electricity demand data with columns:
                - timestamp: datetime
                - demand_mw: float (megawatts)
                - region: str
        """
        
        if save_path is None:
            save_path = DEFAULT_RAW_FILE
        
        # Check cache first
        if Path(save_path).exists():
            print(f"📦 Found cached data at {save_path}")
            response = input("Use cached data? (y/n): ")
            if response.lower() == 'y':
                return pd.read_csv(save_path, parse_dates=['timestamp'])
        
        print(f"🌐 Fetching data from EIA API...")
        print(f"   Region: {region}")
        print(f"   Date range: {start_date.date()} to {end_date.date()}")
        
        # Calculate expected records
        hours = int((end_date - start_date).total_seconds() / 3600)
        print(f"   Expected records: ~{hours:,} hours")
        print()
        
        # Fetch data in chunks (API has limits)
        all_data = []
        offset = 0
        
        # Progress bar
        if show_progress:
            pbar = tqdm(total=hours, desc="Downloading", unit=" hours")
        
        while True:
            # Build request
            params = {
                'api_key': self.api_key,
                'frequency': 'hourly',
                'data[0]': 'value',
                'facets[respondent][]': region,
                'facets[type][]': 'D',  # D = Demand
                'start': start_date.strftime('%Y-%m-%dT%H'),
                'end': end_date.strftime('%Y-%m-%dT%H'),
                'sort[0][column]': 'period',
                'sort[0][direction]': 'asc',
                'offset': offset,
                'length': MAX_RECORDS_PER_REQUEST
            }
            
            # Make request
            try:
                response = requests.get(self.base_url, params=params)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                print(f"\n❌ API request failed: {e}")
                raise
            
            # Parse response
            try:
                data = response.json()
                records = data['response']['data']
            except (KeyError, json.JSONDecodeError) as e:
                print(f"\n❌ Failed to parse API response: {e}")
                raise
            
            # No more data
            if not records:
                break
            
            all_data.extend(records)
            
            # Update progress
            if show_progress:
                pbar.update(len(records))
            
            # Check if we got all data
            if len(records) < MAX_RECORDS_PER_REQUEST:
                break  # Last batch
            
            offset += MAX_RECORDS_PER_REQUEST
            
            # Be nice to the API
            time.sleep(REQUEST_DELAY)
        
        if show_progress:
            pbar.close()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Clean and format
        df = self._clean_data(df)
        
        # Validate
        self._validate_data(df, start_date, end_date)
        
        # Save
        df.to_csv(save_path, index=False)
        print(f"\n💾 Saved to: {save_path}")
        
        return df
    
    def _clean_data(self, df):
        """
        Clean and format raw API data.
        
        Args:
            df (pd.DataFrame): Raw data from API
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        # Rename columns
        df = df.rename(columns={
            'period': 'timestamp',
            'value': 'demand_mw',
            'respondent': 'region'
        })
        
        # Keep only what we need
        df = df[['timestamp', 'demand_mw', 'region']].copy()
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['demand_mw'] = pd.to_numeric(df['demand_mw'], errors='coerce')
        
        # Sort by time
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        
        return df
    
    def _validate_data(self, df, start_date, end_date):
        """
        Validate data quality.
        
        Args:
            df (pd.DataFrame): Data to validate
            start_date (datetime): Expected start
            end_date (datetime): Expected end
        """
        print("\n🔍 Validating data...")
        
        # Check for missing values
        missing = df['demand_mw'].isna().sum()
        if missing > 0:
            print(f"   ⚠️  {missing} missing values ({missing/len(df)*100:.1f}%)")
        
        # Check for outliers
        outliers = ((df['demand_mw'] < MIN_DEMAND_MW) | 
                   (df['demand_mw'] > MAX_DEMAND_MW)).sum()
        if outliers > 0:
            print(f"   ⚠️  {outliers} outlier values")
        
        # Check for gaps
        df_sorted = df.sort_values('timestamp')
        time_diffs = df_sorted['timestamp'].diff()
        expected_diff = pd.Timedelta(hours=1)
        gaps = (time_diffs > expected_diff * 1.5).sum()
        
        if gaps > 0:
            print(f"   ⚠️  {gaps} time gaps detected")
        
        # Check completeness
        expected_hours = (end_date - start_date).total_seconds() / 3600
        actual_hours = len(df)
        completeness = (actual_hours / expected_hours) * 100
        
        if completeness < 95:
            print(f"   ⚠️  Only {completeness:.1f}% complete")
        else:
            print(f"   ✅ {completeness:.1f}% complete")
        
        # Summary stats
        print(f"\n📊 Data Summary:")
        print(f"   Records: {len(df):,}")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Demand range: {df['demand_mw'].min():,.0f} - {df['demand_mw'].max():,.0f} MW")
        print(f"   Mean demand: {df['demand_mw'].mean():,.0f} MW")
        print(f"   Std dev: {df['demand_mw'].std():,.0f} MW")


def quick_fetch(api_key, days=7, region='US48'):
    """
    Quick utility to fetch recent data.
    
    Args:
        api_key (str): EIA API key
        days (int): Number of days to fetch
        region (str): Region code
        
    Returns:
        pd.DataFrame: Recent demand data
    """
    fetcher = EIADataFetcher(api_key)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    return fetcher.fetch_demand_data(
        start_date=start_date,
        end_date=end_date,
        region=region,
        save_path=f"{RAW_DATA_DIR}/recent_{days}days.csv"
    )
