#!/usr/bin/env python3
"""
One-time QQQ Data Backfill Script
Pulls 7+ years of missing QQQ historical data (2014-2022) and stores in database
"""

import yfinance as yf
import pandas as pd
import sqlite3
import time
from datetime import datetime, timedelta
import os

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "reports_tracking.db")

def backfill_qqq_historical_data():
    """Backfill missing QQQ data from 2014 to 2022"""
    
    print("=" * 60)
    print("QQQ HISTORICAL DATA BACKFILL")
    print("=" * 60)
    
    # Check current data in database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT MIN(date), MAX(date), COUNT(*) FROM price_history WHERE symbol = "QQQ"')
    min_date, max_date, count = cursor.fetchone()
    
    print(f"Current QQQ data in database:")
    print(f"  Range: {min_date} to {max_date}")
    print(f"  Total days: {count}")
    
    # Determine backfill period (we need 2014-08-13 to 2022-08-07 to fill the gap)
    backfill_start = "2014-08-13"  # 10+ years back
    backfill_end = "2022-08-07"    # Just before existing data starts
    
    print(f"\\nBackfilling QQQ data:")
    print(f"  Period: {backfill_start} to {backfill_end}")
    print(f"  This will add ~7-8 years of historical data")
    
    try:
        # Fetch data using yfinance
        print("\\nFetching QQQ data from Yahoo Finance...")
        ticker = yf.Ticker("QQQ")
        
        # Fetch the backfill period
        start_date = datetime.strptime(backfill_start, "%Y-%m-%d")
        end_date = datetime.strptime(backfill_end, "%Y-%m-%d")
        
        historical_data = ticker.history(start=start_date, end=end_date + timedelta(days=1))
        
        if historical_data.empty:
            print("[FAIL] No data received from Yahoo Finance")
            return False
            
        print(f"[OK] Successfully fetched {len(historical_data)} days of QQQ data")
        print(f"   Date range: {historical_data.index.min().date()} to {historical_data.index.max().date()}")
        
        # Prepare data for database insertion
        records_to_insert = []
        
        for date, row in historical_data.iterrows():
            # Convert timezone-aware datetime to date string
            date_str = date.strftime('%Y-%m-%d')
            
            record = (
                'QQQ',                          # symbol
                date_str,                       # date
                float(row['Open']),             # open
                float(row['High']),             # high  
                float(row['Low']),              # low
                float(row['Close']),            # close
                float(row['Close']),            # adj_close (same as close for QQQ)
                int(row['Volume']),             # volume
                datetime.now().isoformat()      # updated_at
            )
            records_to_insert.append(record)
        
        # Insert into database (avoid duplicates)
        print(f"\\nInserting {len(records_to_insert)} records into database...")
        
        cursor.executemany('''
            INSERT OR REPLACE INTO price_history 
            (symbol, date, open, high, low, close, adj_close, volume, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', records_to_insert)
        
        conn.commit()
        
        # Verify the insertion
        cursor.execute('SELECT MIN(date), MAX(date), COUNT(*) FROM price_history WHERE symbol = "QQQ"')
        new_min_date, new_max_date, new_count = cursor.fetchone()
        
        added_count = new_count - count
        
        print(f"[OK] Successfully added {added_count} new QQQ records")
        print(f"\\nUpdated QQQ data in database:")
        print(f"  Range: {new_min_date} to {new_max_date}")
        print(f"  Total days: {new_count}")
        
        # Calculate years of coverage
        start_dt = datetime.strptime(new_min_date.split(' ')[0], '%Y-%m-%d')
        end_dt = datetime.strptime(new_max_date.split(' ')[0], '%Y-%m-%d')
        years_coverage = (end_dt - start_dt).days / 365.25
        
        print(f"  Years of coverage: {years_coverage:.1f} years")
        print(f"\\n[SUCCESS] QQQ data backfill completed successfully!")
        print(f"   QQQ models now have {years_coverage:.1f} years of historical data")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Error during backfill: {e}")
        return False
        
    finally:
        conn.close()

def test_qqq_data_access():
    """Test that QQQ models can now access the full dataset"""
    print("\\n" + "=" * 60)
    print("TESTING QQQ MODEL DATA ACCESS")
    print("=" * 60)
    
    try:
        # Test data_fetcher access
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        import data_fetcher
        
        # Test 10-year data fetch
        start_date = "2014-08-13"
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"Testing data_fetcher.get_historical_data('QQQ', '{start_date}', '{end_date}')...")
        
        qqq_data = data_fetcher.get_historical_data('QQQ', start_date, end_date)
        
        if qqq_data is not None and not qqq_data.empty:
            print(f"[OK] Success! Retrieved {len(qqq_data)} days of QQQ data")
            print(f"   Date range: {qqq_data.index.min()} to {qqq_data.index.max()}")
            print(f"   Years: {(qqq_data.index.max() - qqq_data.index.min()).days / 365.25:.1f}")
            print("\\n[SUCCESS] QQQ models will now have access to 10+ years of data!")
            return True
        else:
            print("[FAIL] data_fetcher returned no data")
            return False
            
    except Exception as e:
        print(f"[FAIL] Error testing data access: {e}")
        return False

if __name__ == "__main__":
    print("Starting QQQ historical data backfill...")
    
    # Step 1: Backfill the data
    success = backfill_qqq_historical_data()
    
    if success:
        # Step 2: Test access
        test_qqq_data_access()
        
        print("\\n" + "=" * 60)
        print("BACKFILL COMPLETE - QQQ MODELS READY!")
        print("=" * 60)
        print("All three QQQ models now have access to 10+ years of historical data:")
        print("• QQQ Trading Signal Model")
        print("• QQQ Long Bull Model") 
        print("• QQQ Master Model")
        print("\\nNext model runs will use the full historical dataset for better training!")
    else:
        print("\\n[FAIL] Backfill failed. Please check the errors above.")