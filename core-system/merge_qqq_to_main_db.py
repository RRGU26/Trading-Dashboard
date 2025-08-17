#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge QQQ data from temporary database to main models_dashboard.db
"""
import sqlite3
from datetime import datetime

# Database paths
TEMP_DB = r'C:\Users\rrose\trading-models-system\database\trading_models.db'
MAIN_DB = r'C:\Users\rrose\trading-models-system\databases\models_dashboard.db'

def merge_qqq_data():
    """Merge QQQ data from temporary database to main database"""
    print("QQQ Data Merger")
    print("=" * 60)
    
    try:
        # Connect to both databases
        temp_conn = sqlite3.connect(TEMP_DB)
        main_conn = sqlite3.connect(MAIN_DB)
        
        temp_cursor = temp_conn.cursor()
        main_cursor = main_conn.cursor()
        
        # First, check what's in the temporary database
        temp_cursor.execute("""
            SELECT COUNT(*) as count, MIN(date) as min_date, MAX(date) as max_date
            FROM stock_data
            WHERE symbol = 'QQQ'
        """)
        temp_result = temp_cursor.fetchone()
        print(f"\nSource database (temporary):")
        print(f"  - QQQ records: {temp_result[0]}")
        print(f"  - Date range: {temp_result[1]} to {temp_result[2]}")
        
        # Check current QQQ data in main database
        main_cursor.execute("""
            SELECT COUNT(*) as count, MIN(date) as min_date, MAX(date) as max_date
            FROM price_history
            WHERE symbol = 'QQQ'
        """)
        main_result = main_cursor.fetchone()
        
        if main_result[0] > 0:
            print(f"\nTarget database (main) - existing QQQ data:")
            print(f"  - QQQ records: {main_result[0]}")
            print(f"  - Date range: {main_result[1]} to {main_result[2]}")
        else:
            print(f"\nTarget database (main) has no existing QQQ data")
        
        # Get all QQQ data from temporary database
        temp_cursor.execute("""
            SELECT date, symbol, open, high, low, close, volume
            FROM stock_data
            WHERE symbol = 'QQQ'
            ORDER BY date
        """)
        qqq_data = temp_cursor.fetchall()
        
        print(f"\nMerging {len(qqq_data)} QQQ records into main database...")
        
        # Insert or replace data in main database
        merged = 0
        for row in qqq_data:
            date, symbol, open_price, high, low, close, volume = row
            
            # Use REPLACE to handle duplicates
            main_cursor.execute("""
                REPLACE INTO price_history (date, symbol, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (date, symbol, open_price, high, low, close, volume))
            
            merged += 1
            if merged % 500 == 0:
                print(f"  Processed {merged}/{len(qqq_data)} records...")
        
        # Commit the changes
        main_conn.commit()
        
        # Verify the merge
        main_cursor.execute("""
            SELECT COUNT(*) as count, MIN(date) as min_date, MAX(date) as max_date
            FROM price_history
            WHERE symbol = 'QQQ'
        """)
        final_result = main_cursor.fetchone()
        
        print(f"\nMerge completed successfully!")
        print(f"Final QQQ data in main database:")
        print(f"  - Total records: {final_result[0]}")
        print(f"  - Date range: {final_result[1]} to {final_result[2]}")
        
        # Calculate years of data
        if final_result[1] and final_result[2]:
            start = datetime.strptime(final_result[1], '%Y-%m-%d')
            end = datetime.strptime(final_result[2], '%Y-%m-%d')
            years = (end - start).days / 365.25
            print(f"  - Total years of data: {years:.1f} years")
        
        # Show sample of recent data
        main_cursor.execute("""
            SELECT date, open, high, low, close, volume
            FROM price_history
            WHERE symbol = 'QQQ'
            ORDER BY date DESC
            LIMIT 5
        """)
        recent_data = main_cursor.fetchall()
        
        print(f"\nMost recent 5 QQQ records in main database:")
        print(f"{'Date':<12} {'Open':>8} {'High':>8} {'Low':>8} {'Close':>8} {'Volume':>12}")
        print("-" * 60)
        for row in recent_data:
            print(f"{row[0]:<12} {row[1]:>8.2f} {row[2]:>8.2f} {row[3]:>8.2f} {row[4]:>8.2f} {row[5]:>12}")
        
        # Close connections
        temp_conn.close()
        main_conn.close()
        
        print(f"\nSUCCESS! QQQ data has been merged into the main database.")
        print(f"All trading models can now access 15 years of QQQ historical data!")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = merge_qqq_data()
    
    if not success:
        print("\nFailed to merge data. Please check the error above.")