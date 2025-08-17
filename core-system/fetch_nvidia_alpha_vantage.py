#!/usr/bin/env python3
"""
Fetch NVIDIA data directly from Alpha Vantage API
"""
import requests
import pandas as pd
import sqlite3
import os
from datetime import datetime

def fetch_nvidia_from_alpha_vantage():
    """Fetch NVIDIA data from Alpha Vantage"""
    
    # Get API key
    api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        print("ERROR: ALPHA_VANTAGE_API_KEY environment variable not found")
        return False
    
    print(f"Fetching NVIDIA data from Alpha Vantage...")
    
    # Alpha Vantage TIME_SERIES_DAILY endpoint for NVDA
    url = f'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': 'NVDA',
        'outputsize': 'full',  # Get full historical data
        'apikey': api_key
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if 'Error Message' in data:
            print(f"Alpha Vantage Error: {data['Error Message']}")
            return False
            
        if 'Note' in data:
            print(f"Alpha Vantage Rate Limit: {data['Note']}")
            return False
        
        if 'Time Series (Daily)' not in data:
            print("No time series data found in Alpha Vantage response")
            print(f"Response keys: {list(data.keys())}")
            return False
        
        # Convert to DataFrame
        time_series = data['Time Series (Daily)']
        print(f"Received {len(time_series)} days of NVIDIA data from Alpha Vantage")
        
        # Convert to pandas DataFrame
        df_data = []
        for date_str, values in time_series.items():
            df_data.append({
                'date': date_str,
                'open': float(values['1. open']),
                'high': float(values['2. high']),
                'low': float(values['3. low']),
                'close': float(values['4. close']),
                'volume': int(values['5. volume'])
            })
        
        df = pd.DataFrame(df_data)
        df = df.sort_values('date')  # Sort by date ascending
        
        print(f"NVIDIA data range: {df['date'].min()} to {df['date'].max()}")
        print(f"Latest NVIDIA price: ${df['close'].iloc[-1]:.2f}")
        
        # Insert into database
        conn = sqlite3.connect('models_dashboard.db')
        cursor = conn.cursor()
        
        # Delete existing NVIDIA data
        cursor.execute('DELETE FROM price_history WHERE symbol = ?', ('NVDA',))
        print('Deleted existing NVIDIA data')
        
        # Insert new data
        records_inserted = 0
        for _, row in df.iterrows():
            cursor.execute('''
                INSERT OR REPLACE INTO price_history 
                (symbol, date, open, high, low, close, volume) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                'NVDA', row['date'], row['open'], row['high'], 
                row['low'], row['close'], row['volume']
            ))
            records_inserted += 1
        
        conn.commit()
        conn.close()
        print(f'Successfully inserted {records_inserted} NVIDIA records to database')
        return True
        
    except Exception as e:
        print(f"Error fetching from Alpha Vantage: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    fetch_nvidia_from_alpha_vantage()