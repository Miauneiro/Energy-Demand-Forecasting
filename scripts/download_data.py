"""
Download historical US electricity demand data from EIA API.

This script fetches 2-3 years of hourly electricity demand data
for the US Lower 48 states and saves it to data/raw/

Usage:
    python scripts/download_data.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_fetcher import EIADataFetcher
from dotenv import load_dotenv


def main():
    """Main download function."""
    
    print("=" * 70)
    print("  US ELECTRICITY DEMAND DATA DOWNLOADER")
    print("=" * 70)
    print()
    
    # Load API key from .env
    load_dotenv()
    api_key = os.getenv('EIA_API_KEY')
    
    if not api_key:
        print("❌ ERROR: EIA_API_KEY not found in .env file!")
        print()
        print("Please create a .env file in the project root with:")
        print("EIA_API_KEY=your_key_here")
        print()
        return
    
    print(f"✅ API Key loaded: {api_key[:10]}...")
    print()
    
    # Initialize fetcher
    fetcher = EIADataFetcher(api_key)
    
    # Define date range (last 3 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365)  # ~3 years
    
    print(f"📅 Configuration:")
    print(f"   Start date: {start_date.date()}")
    print(f"   End date: {end_date.date()}")
    print(f"   Duration: ~3 years")
    print(f"   Region: US Lower 48")
    print(f"   Frequency: Hourly")
    print(f"   Expected records: ~{3*365*24:,}")
    print()
    
    input("Press Enter to start download (or Ctrl+C to cancel)...")
    print()
    
    # Fetch data
    try:
        df = fetcher.fetch_demand_data(
            start_date=start_date,
            end_date=end_date,
            region='US48',
            show_progress=True
        )
        
        print()
        print("=" * 70)
        print("  ✅ DOWNLOAD COMPLETE!")
        print("=" * 70)
        print()
        print("📁 File saved to: data/raw/us_demand_historical.csv")
        print()
        print("🎉 Next steps:")
        print("   1. Explore the data in notebooks/01_data_exploration.ipynb")
        print("   2. Run preprocessing: python scripts/preprocess_data.py")
        print("   3. Train model: python scripts/train_model.py")
        print()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Download cancelled by user")
        sys.exit(0)
        
    except Exception as e:
        print()
        print("=" * 70)
        print("  ❌ ERROR DURING DOWNLOAD")
        print("=" * 70)
        print()
        print(f"Error: {str(e)}")
        print()
        print("Troubleshooting:")
        print("1. Check your API key is correct in .env")
        print("2. Verify internet connection")
        print("3. Try reducing date range (fewer years)")
        print("4. Check EIA API status: https://www.eia.gov/opendata/")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
