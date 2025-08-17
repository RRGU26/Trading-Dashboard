#!/usr/bin/env python3
"""
Load NVIDIA data from the existing CSV cache file
"""
import pandas as pd
import sqlite3
import os

def load_nvidia_from_csv():
    """Load NVIDIA data from CSV cache"""
    
    csv_file = r"C:\Users\rrose\OneDrive\Desktop\Trading System\qqq_data_cache\NVDA_data.csv"
    
    print(f"Loading NVIDIA data from: {csv_file}")
    
    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}")
        return False
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        print(f"Loaded CSV with {len(df)} records")
        print(f"Columns: {list(df.columns)}")
        
        if len(df) > 0:
            # Show data sample
            print(f"Date range: {df.iloc[0]['Date'] if 'Date' in df.columns else 'N/A'} to {df.iloc[-1]['Date'] if 'Date' in df.columns else 'N/A'}")
            
            # Check if we have Close price
            if 'Close' in df.columns:
                print(f"Latest NVIDIA price in CSV: ${df['Close'].iloc[-1]:.2f}")
            elif 'close' in df.columns:
                print(f"Latest NVIDIA price in CSV: ${df['close'].iloc[-1]:.2f}")
            
            # Insert into database
            conn = sqlite3.connect('models_dashboard.db')
            cursor = conn.cursor()
            
            # Delete existing NVIDIA data
            cursor.execute('DELETE FROM price_history WHERE symbol = ?', ('NVDA',))
            print('Deleted existing NVIDIA data')
            
            # Insert new data
            records_inserted = 0
            for _, row in df.iterrows():
                # Handle different column name formats
                date = row.get('Date', row.get('date', ''))
                open_price = row.get('Open', row.get('open', 0))
                high_price = row.get('High', row.get('high', 0))
                low_price = row.get('Low', row.get('low', 0))
                close_price = row.get('Close', row.get('close', 0))
                volume = row.get('Volume', row.get('volume', 0))
                
                cursor.execute('''
                    INSERT OR REPLACE INTO price_history 
                    (symbol, date, open, high, low, close, volume) 
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    'NVDA', date, open_price, high_price, 
                    low_price, close_price, volume
                ))
                records_inserted += 1
            
            conn.commit()
            conn.close()
            print(f'Successfully inserted {records_inserted} NVIDIA records to database')
            return True
        else:
            print("CSV file is empty")
            return False
            
    except Exception as e:
        print(f"Error loading NVIDIA CSV: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    load_nvidia_from_csv()