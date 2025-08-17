#!/usr/bin/env python3
"""Quick check of NVIDIA's actual current price"""

import yfinance as yf
import time
from datetime import datetime

print("Checking NVIDIA (NVDA) actual price...")
print("=" * 50)

# Wait a bit to avoid rate limiting
time.sleep(2)

try:
    # Get NVIDIA ticker
    nvda = yf.Ticker("NVDA")
    
    # Get info
    info = nvda.info
    current_price = info.get('currentPrice') or info.get('regularMarketPrice')
    
    if current_price:
        print(f"NVIDIA Current Price (from info): ${current_price:.2f}")
    
    # Also try history
    hist = nvda.history(period="5d")
    if not hist.empty:
        latest_close = hist['Close'].iloc[-1]
        print(f"NVIDIA Latest Close Price: ${latest_close:.2f}")
        print(f"Date: {hist.index[-1].strftime('%Y-%m-%d')}")
        
        # Show last 5 days
        print("\nLast 5 days of NVDA prices:")
        for date, row in hist.iterrows():
            print(f"  {date.strftime('%Y-%m-%d')}: ${row['Close']:.2f}")
    
    # Check if there's a stock split issue
    print("\n" + "=" * 50)
    print("IMPORTANT: NVIDIA had a 10-for-1 stock split on June 7, 2024")
    print("Pre-split price ~$1200 became ~$120 post-split")
    print("If seeing $600+ prices, the data source may be using pre-split prices")
    
except Exception as e:
    print(f"Error: {e}")
    print("\nTrying alternative method...")
    
    import requests
    # Try a simple web API
    try:
        # Using Yahoo Finance query
        url = "https://query1.finance.yahoo.com/v8/finance/chart/NVDA"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            current = data['chart']['result'][0]['meta']['regularMarketPrice']
            print(f"NVIDIA Current Price (from API): ${current:.2f}")
    except:
        print("Alternative method also failed")