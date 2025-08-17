#!/usr/bin/env python3
"""
Fetch fresh NVIDIA historical data and populate database
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import data_fetcher
from datetime import datetime, timedelta

def fetch_nvidia_historical_data():
    """Fetch fresh NVIDIA historical data"""
    print("Fetching NVIDIA historical data...")
    
    # Calculate date range - get 2 years of data
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    
    print(f"Date range: {start_date} to {end_date}")
    
    try:
        # Fetch NVIDIA historical data
        nvidia_data = data_fetcher.get_historical_data('NVDA', start_date, end_date)
        
        if nvidia_data is not None and len(nvidia_data) > 0:
            print(f"Successfully fetched {len(nvidia_data)} days of NVIDIA data")
            print(f"Date range: {nvidia_data.index.min()} to {nvidia_data.index.max()}")
            print(f"Latest NVIDIA price: ${nvidia_data['Close'].iloc[-1]:.2f}")
            
            # Also get current price for verification
            current_price = data_fetcher.get_current_price('NVDA')
            print(f"Current NVIDIA price from API: ${current_price:.2f}")
            
            return True
        else:
            print("Failed to fetch NVIDIA data")
            return False
            
    except Exception as e:
        print(f"Error fetching NVIDIA data: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    fetch_nvidia_historical_data()