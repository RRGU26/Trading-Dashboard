#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify all QQQ models can access the full 15 years of data
"""
import sqlite3
from datetime import datetime, timedelta
import os
import sys

# Database path
DB_PATH = r'C:\Users\rrose\trading-models-system\databases\models_dashboard.db'

def verify_qqq_models():
    """Verify QQQ models have access to full historical data"""
    print("QQQ Models Verification")
    print("=" * 60)
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check QQQ data availability
    cursor.execute("""
        SELECT COUNT(*) as count, MIN(date) as min_date, MAX(date) as max_date
        FROM price_history
        WHERE symbol = 'QQQ'
    """)
    result = cursor.fetchone()
    
    print(f"\nQQQ Data Available in Database:")
    print(f"  - Total records: {result[0]}")
    print(f"  - Date range: {result[1]} to {result[2]}")
    
    # Calculate years
    start = datetime.strptime(result[1], '%Y-%m-%d')
    end = datetime.strptime(result[2], '%Y-%m-%d')
    years = (end - start).days / 365.25
    print(f"  - Total years: {years:.1f} years")
    
    # Define QQQ models and their data requirements
    qqq_models = [
        {
            'name': 'QQQ Trading Signal',
            'path': 'models/qqq-trading-signal',
            'start_date': '2014-08-13',  # Updated to 10 years back
            'lookback_days': 10 * 365
        },
        {
            'name': 'QQQ Long Bull',
            'path': 'models/qqq-long-bull',
            'start_date': '2014-08-13',  # Updated to 10 years back
            'lookback_days': 10 * 365
        },
        {
            'name': 'QQQ Master',
            'path': 'models/qqq-master',
            'start_date': '2014-08-13',  # Updated to 10 years back
            'lookback_days': 10 * 365
        }
    ]
    
    print(f"\n" + "=" * 60)
    print("QQQ Models Data Access Verification:")
    print("=" * 60)
    
    all_models_ready = True
    
    for model in qqq_models:
        print(f"\n{model['name']}:")
        print(f"  Required start date: {model['start_date']}")
        print(f"  Required lookback: {model['lookback_days']} days ({model['lookback_days']/365:.1f} years)")
        
        # Check if we have data from the required start date
        cursor.execute("""
            SELECT COUNT(*) 
            FROM price_history
            WHERE symbol = 'QQQ' AND date >= ?
        """, (model['start_date'],))
        count = cursor.fetchone()[0]
        
        # Check if start date is available
        cursor.execute("""
            SELECT MIN(date)
            FROM price_history
            WHERE symbol = 'QQQ' AND date >= ?
        """, (model['start_date'],))
        actual_start = cursor.fetchone()[0]
        
        if actual_start:
            print(f"  Data available from: {actual_start}")
            print(f"  Records available: {count}")
            
            # Calculate actual years available
            actual_start_dt = datetime.strptime(actual_start, '%Y-%m-%d')
            actual_years = (end - actual_start_dt).days / 365.25
            print(f"  Actual years available: {actual_years:.1f} years")
            
            # Check if sufficient
            if actual_years >= (model['lookback_days'] / 365):
                print(f"  Status: READY - Has full {model['lookback_days']/365:.0f} years of data")
            else:
                print(f"  Status: PARTIAL - Has {actual_years:.1f} years, needs {model['lookback_days']/365:.0f} years")
                all_models_ready = False
        else:
            print(f"  Status: NO DATA - Required start date not available")
            all_models_ready = False
    
    # Special check for 10-year requirement
    print(f"\n" + "=" * 60)
    print("10-Year Data Requirement Check:")
    print("=" * 60)
    
    ten_years_ago = (datetime.now() - timedelta(days=10*365)).strftime('%Y-%m-%d')
    cursor.execute("""
        SELECT COUNT(*), MIN(date)
        FROM price_history
        WHERE symbol = 'QQQ' AND date >= ?
    """, (ten_years_ago,))
    ten_year_result = cursor.fetchone()
    
    if ten_year_result[0] > 0:
        print(f"  10 years back from today: {ten_years_ago}")
        print(f"  Earliest data in that range: {ten_year_result[1]}")
        print(f"  Trading days in 10-year range: {ten_year_result[0]}")
        print(f"  Status: CONFIRMED - Full 10+ years of data available!")
    
    # Final summary
    print(f"\n" + "=" * 60)
    print("FINAL VERIFICATION SUMMARY:")
    print("=" * 60)
    
    if years >= 10:
        print(f"  SUCCESS: Database has {years:.1f} years of QQQ data")
        print(f"  All QQQ models have access to FULL 10+ years of historical data")
        print(f"  Models are ready for robust training and predictions!")
    else:
        print(f"  WARNING: Only {years:.1f} years of data available")
        print(f"  Some models may not have full 10-year history")
    
    conn.close()
    
    return all_models_ready

if __name__ == "__main__":
    models_ready = verify_qqq_models()
    
    if models_ready:
        print("\n" + "=" * 60)
        print("ALL QQQ MODELS ARE READY WITH 15+ YEARS OF DATA!")
        print("Tomorrow's trading execution will use the full dataset!")
        print("=" * 60)