#!/usr/bin/env python3
"""
Restore NVIDIA historical data from backup CSV file
"""
import pandas as pd
import sqlite3
from datetime import datetime

def restore_nvidia_data():
    """Restore NVIDIA data from backup CSV"""
    # Read the backup CSV file to find NVIDIA data
    backup_file = r'C:\Users\rrose\OneDrive\Desktop\Trading System\data\price_history_backup.csv'
    print(f'Reading backup file: {backup_file}')

    try:
        df = pd.read_csv(backup_file)
        print(f'Backup file loaded: {len(df)} total records')
        
        # Filter for NVIDIA data
        nvidia_data = df[df['symbol'] == 'NVDA'].copy()
        print(f'NVIDIA records in backup: {len(nvidia_data)}')
        
        if len(nvidia_data) > 0:
            print(f'NVIDIA date range: {nvidia_data["date"].min()} to {nvidia_data["date"].max()}')
            print(f'Latest NVIDIA price in backup: ${nvidia_data["close"].iloc[-1]:.2f}')
            
            # Insert into database
            conn = sqlite3.connect('models_dashboard.db')
            cursor = conn.cursor()
            
            # Delete existing NVIDIA data first
            cursor.execute('DELETE FROM price_history WHERE symbol = ?', ('NVDA',))
            print('Deleted existing NVIDIA data')
            
            # Insert backup data
            records_inserted = 0
            for _, row in nvidia_data.iterrows():
                cursor.execute('''
                    INSERT OR REPLACE INTO price_history 
                    (symbol, date, open, high, low, close, volume) 
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    row['symbol'], row['date'], row['open'], row['high'], 
                    row['low'], row['close'], row['volume']
                ))
                records_inserted += 1
            
            conn.commit()
            conn.close()
            print(f'Successfully restored {records_inserted} NVIDIA records to database')
            return True
            
        else:
            print('No NVIDIA data found in backup file')
            return False
            
    except Exception as e:
        print(f'Error: {e}')
        return False

if __name__ == '__main__':
    restore_nvidia_data()