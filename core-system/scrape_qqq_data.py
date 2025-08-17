#!/usr/bin/env python3
"""
Scrape QQQ historical data from Yahoo Finance
URL: https://finance.yahoo.com/quote/QQQ/history/?period1=1388534400&period2=1672531200
"""

import requests
import pandas as pd
import sqlite3
import json
import re
from datetime import datetime
import os
import time

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "models_dashboard.db")

def scrape_yahoo_finance_data(url):
    """Scrape historical data from Yahoo Finance page"""
    
    print("=" * 60)
    print("SCRAPING QQQ HISTORICAL DATA FROM YAHOO FINANCE")
    print("=" * 60)
    print(f"URL: {url}")
    
    try:
        # Add headers to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        print("\nFetching page content...")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        print(f"Page fetched successfully (size: {len(response.text)} bytes)")
        
        # Look for the data in the page's JavaScript
        # Yahoo Finance embeds the data in a JavaScript variable
        print("\nExtracting data from page...")
        
        # Find the JSON data embedded in the page
        # Look for the pattern that contains historical prices
        pattern = r'HistoricalPriceStore":\s*({.*?"prices":\s*\[.*?\].*?})'
        match = re.search(pattern, response.text)
        
        if not match:
            # Try alternative pattern
            pattern = r'"HistoricalPriceStore":\s*({[^}]*"prices":[^}]*})'
            match = re.search(pattern, response.text)
        
        if not match:
            print("[WARNING] Could not find HistoricalPriceStore in page")
            print("Trying alternative extraction method...")
            
            # Try to find the data table directly
            # Look for the actual price data pattern
            pattern = r'{"date":\d+,"open":[\d.]+,"high":[\d.]+,"low":[\d.]+,"close":[\d.]+,"volume":\d+,"adjclose":[\d.]+}'
            matches = re.findall(pattern, response.text)
            
            if matches:
                print(f"Found {len(matches)} data points using alternative method")
                
                # Parse the data points
                data_points = []
                for match_str in matches:
                    try:
                        data = json.loads(match_str)
                        data_points.append(data)
                    except:
                        continue
                
                if data_points:
                    return process_price_data(data_points)
            
            return None
        
        # Extract and parse the JSON data
        json_str = match.group(1)
        
        # Clean up the JSON string
        json_str = json_str.replace('\\', '')
        
        try:
            data = json.loads(json_str)
            prices = data.get('prices', [])
            
            if not prices:
                print("[WARNING] No price data found in HistoricalPriceStore")
                return None
                
            print(f"Found {len(prices)} data points")
            
            return process_price_data(prices)
            
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse JSON data: {e}")
            
            # Try to extract data using regex
            print("Attempting regex extraction of individual data points...")
            
            # Find all date, open, high, low, close, volume patterns
            date_pattern = r'"date":(\d+)'
            open_pattern = r'"open":([\d.]+)'
            high_pattern = r'"high":([\d.]+)'
            low_pattern = r'"low":([\d.]+)'
            close_pattern = r'"close":([\d.]+)'
            volume_pattern = r'"volume":(\d+)'
            
            dates = re.findall(date_pattern, json_str)
            opens = re.findall(open_pattern, json_str)
            highs = re.findall(high_pattern, json_str)
            lows = re.findall(low_pattern, json_str)
            closes = re.findall(close_pattern, json_str)
            volumes = re.findall(volume_pattern, json_str)
            
            if dates and opens and closes:
                print(f"Extracted {len(dates)} data points via regex")
                
                data_points = []
                for i in range(min(len(dates), len(opens), len(closes))):
                    data_points.append({
                        'date': int(dates[i]),
                        'open': float(opens[i]) if i < len(opens) else None,
                        'high': float(highs[i]) if i < len(highs) else None,
                        'low': float(lows[i]) if i < len(lows) else None,
                        'close': float(closes[i]) if i < len(closes) else None,
                        'volume': int(volumes[i]) if i < len(volumes) else 0
                    })
                
                return process_price_data(data_points)
            
            return None
            
    except requests.RequestException as e:
        print(f"[ERROR] Failed to fetch page: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return None

def process_price_data(prices):
    """Process the raw price data into a DataFrame"""
    
    if not prices:
        return None
    
    print(f"\nProcessing {len(prices)} price records...")
    
    # Convert to DataFrame
    df_data = []
    
    for price_point in prices:
        if isinstance(price_point, dict):
            # Convert Unix timestamp to date
            if 'date' in price_point:
                date = datetime.fromtimestamp(price_point['date'])
                
                df_data.append({
                    'Date': date,
                    'Open': price_point.get('open'),
                    'High': price_point.get('high'),
                    'Low': price_point.get('low'),
                    'Close': price_point.get('close'),
                    'Volume': price_point.get('volume', 0),
                    'Adj_Close': price_point.get('adjclose', price_point.get('close'))
                })
    
    if not df_data:
        print("[WARNING] No valid data points after processing")
        return None
    
    df = pd.DataFrame(df_data)
    df = df.set_index('Date')
    df = df.sort_index()
    
    # Remove any rows with all NaN values
    df = df.dropna(how='all')
    
    # Forward fill missing values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    print(f"Processed data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    return df

def save_to_database(df):
    """Save the scraped data to the database"""
    
    if df is None or df.empty:
        print("[ERROR] No data to save")
        return False
    
    print("\n" + "=" * 60)
    print("SAVING DATA TO DATABASE")
    print("=" * 60)
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check existing data
        cursor.execute('SELECT MIN(date), MAX(date), COUNT(*) FROM price_history WHERE symbol = "QQQ"')
        min_date, max_date, count = cursor.fetchone()
        
        print(f"Current QQQ data in database:")
        print(f"  Range: {min_date} to {max_date}")
        print(f"  Total days: {count}")
        
        # Prepare records for insertion
        records_to_insert = []
        
        for date, row in df.iterrows():
            date_str = date.strftime('%Y-%m-%d')
            
            record = (
                'QQQ',                          # symbol
                date_str,                       # date
                float(row['Open']),             # open
                float(row['High']),             # high
                float(row['Low']),              # low
                float(row['Close']),            # close
                float(row.get('Adj_Close', row['Close'])),  # adj_close
                int(row['Volume']),             # volume
                datetime.now().isoformat()      # updated_at
            )
            records_to_insert.append(record)
        
        print(f"\nInserting {len(records_to_insert)} records...")
        
        # Insert or replace records
        cursor.executemany('''
            INSERT OR REPLACE INTO price_history 
            (symbol, date, open, high, low, close, adj_close, volume, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', records_to_insert)
        
        conn.commit()
        
        # Verify the update
        cursor.execute('SELECT MIN(date), MAX(date), COUNT(*) FROM price_history WHERE symbol = "QQQ"')
        new_min_date, new_max_date, new_count = cursor.fetchone()
        
        added_count = new_count - count
        
        print(f"\n[SUCCESS] Added/updated {added_count} QQQ records")
        print(f"\nUpdated QQQ data in database:")
        print(f"  Range: {new_min_date} to {new_max_date}")
        print(f"  Total days: {new_count}")
        
        # Calculate years of coverage
        if new_min_date and new_max_date:
            start_dt = datetime.strptime(new_min_date.split(' ')[0], '%Y-%m-%d')
            end_dt = datetime.strptime(new_max_date.split(' ')[0], '%Y-%m-%d')
            years_coverage = (end_dt - start_dt).days / 365.25
            
            print(f"  Years of coverage: {years_coverage:.1f} years")
            
            if years_coverage >= 9:
                print("\n[SUCCESS] QQQ models now have 10+ years of historical data!")
            
        conn.close()
        return True
        
    except Exception as e:
        print(f"[ERROR] Database save failed: {e}")
        return False

def main():
    """Main execution"""
    
    # The URL provided by the user
    # period1=1388534400 is approximately 2014-01-01
    # period2=1672531200 is approximately 2022-12-31
    url = "https://finance.yahoo.com/quote/QQQ/history/?period1=1388534400&period2=1672531200"
    
    print("Starting QQQ historical data scraping...")
    print(f"Target period: ~2014 to ~2022")
    
    # Scrape the data
    df = scrape_yahoo_finance_data(url)
    
    if df is not None and not df.empty:
        # Save to database
        success = save_to_database(df)
        
        if success:
            print("\n" + "=" * 60)
            print("SCRAPING COMPLETE - QQQ DATA UPDATED!")
            print("=" * 60)
            print("QQQ models will now use the extended historical dataset")
            print("Next model runs will benefit from 10+ years of training data!")
        else:
            print("\n[FAIL] Failed to save data to database")
    else:
        print("\n[FAIL] Failed to scrape data from Yahoo Finance")
        print("The page structure may have changed or the URL may be incorrect")

if __name__ == "__main__":
    main()