#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup database and fetch QQQ historical data
"""
import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime
import os

# Database path
DB_PATH = r'C:\Users\rrose\trading-models-system\database\trading_models.db'

def create_database():
    """Create the database and stock_data table if they don't exist"""
    print("Creating database and tables...")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create stock_data table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            UNIQUE(date, symbol)
        )
    """)
    
    # Create index for faster queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_stock_date_symbol 
        ON stock_data(date, symbol)
    """)
    
    conn.commit()
    conn.close()
    print("Database created successfully!")

def fetch_and_save_qqq_data(start_date='2014-01-01', end_date='2022-08-07'):
    """
    Fetch QQQ historical data and save to database
    """
    print("\nQQQ Historical Data Fetcher")
    print("=" * 60)
    print(f"Target period: {start_date} to {end_date}")
    
    try:
        # Create database if needed
        create_database()
        
        # Check current data in database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT MIN(date) as min_date, MAX(date) as max_date, COUNT(*) as count
            FROM stock_data
            WHERE symbol = 'QQQ'
        """)
        result = cursor.fetchone()
        
        if result and result[2] > 0:
            print(f"\nCurrent database status for QQQ:")
            print(f"  - Existing records: {result[2]}")
            print(f"  - Date range: {result[0]} to {result[1]}")
        else:
            print("\nNo existing QQQ data in database")
        
        # Fetch data using yfinance
        print(f"\nFetching QQQ data from Yahoo Finance...")
        print("This may take a moment...")
        
        qqq = yf.Ticker("QQQ")
        
        # Get historical data
        hist_data = qqq.history(start=start_date, end=end_date)
        
        if hist_data.empty:
            print("ERROR: No data retrieved from Yahoo Finance")
            return False
        
        print(f"Retrieved {len(hist_data)} records")
        
        # Reset index to make Date a column
        hist_data = hist_data.reset_index()
        
        # Prepare data for database
        hist_data['Date'] = pd.to_datetime(hist_data['Date']).dt.strftime('%Y-%m-%d')
        hist_data['Symbol'] = 'QQQ'
        
        # Rename columns to match database schema
        hist_data = hist_data.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Select only required columns
        data_to_insert = hist_data[['date', 'Symbol', 'open', 'high', 'low', 'close', 'volume']]
        
        print(f"\nInserting data into database...")
        
        # Insert data using REPLACE to handle duplicates
        inserted = 0
        for _, row in data_to_insert.iterrows():
            cursor.execute("""
                REPLACE INTO stock_data (date, symbol, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (row['date'], row['Symbol'], row['open'], row['high'], row['low'], row['close'], row['volume']))
            inserted += 1
            
            if inserted % 100 == 0:
                print(f"  Processed {inserted}/{len(data_to_insert)} records...")
        
        conn.commit()
        
        # Verify final state
        cursor.execute("""
            SELECT MIN(date) as min_date, MAX(date) as max_date, COUNT(*) as count
            FROM stock_data
            WHERE symbol = 'QQQ'
        """)
        final_result = cursor.fetchone()
        
        print(f"\nSuccessfully updated QQQ data!")
        print(f"Final database status:")
        print(f"  - Total records: {final_result[2]}")
        print(f"  - Date range: {final_result[0]} to {final_result[1]}")
        
        # Calculate years of data
        if final_result[0] and final_result[1]:
            start = datetime.strptime(final_result[0], '%Y-%m-%d')
            end = datetime.strptime(final_result[1], '%Y-%m-%d')
            years = (end - start).days / 365.25
            print(f"  - Total years of data: {years:.1f} years")
        
        # Show sample of data
        cursor.execute("""
            SELECT date, open, high, low, close, volume
            FROM stock_data
            WHERE symbol = 'QQQ'
            ORDER BY date DESC
            LIMIT 5
        """)
        recent_data = cursor.fetchall()
        
        print(f"\nMost recent 5 records:")
        print(f"{'Date':<12} {'Open':>8} {'High':>8} {'Low':>8} {'Close':>8} {'Volume':>12}")
        print("-" * 60)
        for row in recent_data:
            print(f"{row[0]:<12} {row[1]:>8.2f} {row[2]:>8.2f} {row[3]:>8.2f} {row[4]:>8.2f} {row[5]:>12}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # First, let's fetch ALL available data to see what we can get
    print("First attempting to fetch all available QQQ data...")
    success = fetch_and_save_qqq_data(start_date='2010-01-01', end_date='2025-01-01')
    
    if success:
        print("\nQQQ historical data successfully loaded!")
        print("All QQQ models now have access to extended historical data")
    else:
        print("\nFailed to load data. Please check the error and try again.")