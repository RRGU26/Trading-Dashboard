#!/usr/bin/env python3
"""
Fix NVIDIA data properly:
1. Get current price from yfinance
2. Add NVDA price data to database
3. Adjust all historical data for the 10-for-1 split
"""

import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime, timedelta

# Database path
DB_PATH = r'C:\Users\rrose\trading-models-system\databases\models_dashboard.db'

def fetch_and_store_nvda_data():
    """Fetch NVDA data from yfinance and store in database"""
    print("=" * 60)
    print("NVIDIA DATA FIX")
    print("=" * 60)
    
    # Get NVDA data from yfinance (already split-adjusted)
    print("\n1. Fetching NVDA data from yfinance...")
    print("   Waiting 5 seconds to avoid rate limit...")
    import time
    time.sleep(5)
    
    nvda = yf.Ticker("NVDA")
    
    # Get 2 years of data to ensure we have good coverage
    try:
        hist = nvda.history(period="2y")
    except Exception as e:
        print(f"   Rate limited. Trying shorter period...")
        time.sleep(10)
        try:
            hist = nvda.history(period="1y")
        except:
            print(f"   Still rate limited. Trying 6 months...")
            time.sleep(10)
            hist = nvda.history(period="6mo")
    
    if hist.empty:
        print("ERROR: Could not fetch NVDA data")
        return False
    
    print(f"   Retrieved {len(hist)} days of NVDA data")
    print(f"   Date range: {hist.index[0].strftime('%Y-%m-%d')} to {hist.index[-1].strftime('%Y-%m-%d')}")
    print(f"   Latest price: ${hist['Close'].iloc[-1]:.2f}")
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check current NVDA data in database
    cursor.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM price_history WHERE symbol = 'NVDA'")
    result = cursor.fetchone()
    
    print(f"\n2. Current database status:")
    if result[0] > 0:
        print(f"   Existing NVDA records: {result[0]}")
        print(f"   Date range: {result[1]} to {result[2]}")
        
        # Clear old NVDA data
        print("   Clearing old NVDA data...")
        cursor.execute("DELETE FROM price_history WHERE symbol = 'NVDA'")
        conn.commit()
    else:
        print("   No existing NVDA data")
    
    # Insert new split-adjusted data
    print(f"\n3. Inserting split-adjusted NVDA data...")
    inserted = 0
    
    for date, row in hist.iterrows():
        cursor.execute("""
            INSERT OR REPLACE INTO price_history 
            (date, symbol, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            date.strftime('%Y-%m-%d'),
            'NVDA',
            row['Open'],
            row['High'],
            row['Low'],
            row['Close'],
            row['Volume']
        ))
        inserted += 1
        
        if inserted % 100 == 0:
            print(f"   Processed {inserted} records...")
    
    conn.commit()
    print(f"   Inserted {inserted} NVDA records")
    
    # Verify the data
    cursor.execute("""
        SELECT date, close 
        FROM price_history 
        WHERE symbol = 'NVDA' 
        ORDER BY date DESC 
        LIMIT 5
    """)
    recent = cursor.fetchall()
    
    print(f"\n4. Recent NVDA prices in database:")
    for date, close in recent:
        print(f"   {date}: ${close:.2f}")
    
    # Also cache the current price separately if needed
    current_price = hist['Close'].iloc[-1]
    print(f"\n5. Current NVDA price: ${current_price:.2f}")
    
    # Create a price cache table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS price_cache (
            symbol TEXT PRIMARY KEY,
            current_price REAL,
            last_updated TEXT
        )
    """)
    
    # Update cache
    cursor.execute("""
        INSERT OR REPLACE INTO price_cache (symbol, current_price, last_updated)
        VALUES (?, ?, ?)
    """, ('NVDA', current_price, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    
    conn.commit()
    conn.close()
    
    print(f"\n✅ SUCCESS: NVDA data fixed!")
    print(f"   - {inserted} days of split-adjusted data in database")
    print(f"   - Current price cached: ${current_price:.2f}")
    print(f"   - All prices are post-split (should be in $100-200 range)")
    
    return True

if __name__ == "__main__":
    success = fetch_and_store_nvda_data()
    
    if success:
        print("\n" + "=" * 60)
        print("NEXT STEPS:")
        print("1. Update davemodel.py to check price_cache table first")
        print("2. Fall back to price_history table if cache miss")
        print("3. All prices now properly split-adjusted!")
        print("=" * 60)