#!/usr/bin/env python3
"""
Fetch NVIDIA data directly from YFinance with rate limiting
"""
import yfinance as yf
import pandas as pd
import sqlite3
import time
from datetime import datetime, timedelta

def fetch_nvidia_yfinance():
    """Fetch NVIDIA data from YFinance"""
    
    print("Fetching NVIDIA data from YFinance...")
    
    try:
        # Create ticker object
        nvda = yf.Ticker("NVDA")
        
        # Add delay to avoid rate limiting
        time.sleep(2)
        
        # Get 1 year of historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        print(f"Fetching data from {start_date.date()} to {end_date.date()}")
        
        # Fetch historical data
        hist = nvda.history(start=start_date, end=end_date)
        
        if hist.empty:
            print("No data returned from YFinance")
            return False
        
        print(f"Successfully fetched {len(hist)} days of NVIDIA data")
        print(f"Date range: {hist.index.min().date()} to {hist.index.max().date()}")
        print(f"Latest NVIDIA close price: ${hist['Close'].iloc[-1]:.2f}")
        
        # Get current price info
        info = nvda.info
        current_price = info.get('currentPrice', hist['Close'].iloc[-1])
        print(f"Current NVIDIA price from info: ${current_price:.2f}")
        
        # Prepare data for database
        hist_reset = hist.reset_index()
        hist_reset['Date'] = hist_reset['Date'].dt.strftime('%Y-%m-%d')
        
        # Insert into database
        conn = sqlite3.connect('reports_tracking.db')
        cursor = conn.cursor()
        
        # Delete existing NVIDIA data
        cursor.execute('DELETE FROM price_history WHERE symbol = ?', ('NVDA',))
        print('Deleted existing NVIDIA data')
        
        # Insert new data
        records_inserted = 0
        for _, row in hist_reset.iterrows():
            cursor.execute('''
                INSERT OR REPLACE INTO price_history 
                (symbol, date, open, high, low, close, volume) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                'NVDA', row['Date'], row['Open'], row['High'], 
                row['Low'], row['Close'], row['Volume']
            ))
            records_inserted += 1
        
        conn.commit()
        conn.close()
        print(f'Successfully inserted {records_inserted} NVIDIA records to database')
        
        return True
        
    except Exception as e:
        print(f"Error fetching NVIDIA data from YFinance: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    fetch_nvidia_yfinance()