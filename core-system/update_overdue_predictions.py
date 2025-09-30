#!/usr/bin/env python3
"""
Update overdue predictions with actual prices from price_history
"""

import sqlite3
import pandas as pd
from datetime import date, datetime
import yfinance as yf

DB_PATH = r'C:\Users\rrose\trading-models-system\core-system\reports_tracking.db'

def get_actual_price(symbol, target_date):
    """Get actual price for a symbol on target date"""
    
    # First check price_history table
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check price_history first  
    cursor.execute("""
        SELECT close FROM price_history 
        WHERE symbol = ? AND date = ?
    """, (symbol, target_date))
    
    print(f"Checking price_history for {symbol} on {target_date}")
    
    result = cursor.fetchone()
    if result:
        conn.close()
        return result[0]
    
    conn.close()
    
    # If not in price_history, try to fetch from yfinance
    try:
        if symbol in ['QQQ', 'NVDA']:
            ticker = yf.Ticker(symbol)
            # For recent dates, get a range of data
            from datetime import datetime, timedelta
            target_dt = datetime.strptime(target_date, '%Y-%m-%d')
            start_date = target_dt - timedelta(days=5)  # Get a few days of data
            end_date = target_dt + timedelta(days=1)
            
            hist = ticker.history(start=start_date, end=end_date)
            if not hist.empty:
                # Find the exact date or closest trading day
                hist_dates = hist.index.date
                target_date_obj = target_dt.date()
                
                for i, date in enumerate(hist_dates):
                    if date == target_date_obj:
                        return hist['Close'].iloc[i]
                
                # If exact date not found, get the closest trading day before
                before_dates = [d for d in hist_dates if d <= target_date_obj]
                if before_dates:
                    closest_date = max(before_dates)
                    closest_idx = list(hist_dates).index(closest_date)
                    print(f"Using closest trading day {closest_date} for {target_date}")
                    return hist['Close'].iloc[closest_idx]
        elif symbol == 'BTC-USD':
            # For crypto, try CoinGecko or yfinance BTC-USD
            ticker = yf.Ticker('BTC-USD')
            hist = ticker.history(start=target_date, end=target_date)
            if not hist.empty:
                return hist['Close'].iloc[0]
        elif symbol == 'ALGO-USD':
            # For ALGO, might need special handling
            pass
    except Exception as e:
        print(f"Error fetching {symbol} for {target_date}: {e}")
    
    return None

def update_overdue_predictions():
    """Update all overdue predictions with actual prices"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("Updating overdue predictions with actual prices...")
    print("=" * 60)
    
    # Get all overdue predictions without actual prices
    cursor.execute("""
        SELECT id, model, symbol, prediction_date, target_date, predicted_price, current_price
        FROM model_predictions 
        WHERE target_date <= date('now') 
        AND (actual_price IS NULL OR actual_price = '')
        ORDER BY target_date DESC
    """)
    
    overdue = cursor.fetchall()
    print(f"Found {len(overdue)} overdue predictions to update")
    
    updated = 0
    for row in overdue:
        pred_id, model, symbol, pred_date, target_date, predicted_price, current_price = row
        
        # Skip corrupted Bitcoin predictions
        if isinstance(predicted_price, bytes):
            print(f"  SKIP: {model} {symbol} {target_date} - corrupted data")
            continue
        
        # Get actual price
        actual_price = get_actual_price(symbol, target_date)
        
        if actual_price:
            # Calculate metrics
            prediction_error = predicted_price - actual_price
            error_pct = (prediction_error / actual_price * 100) if actual_price != 0 else 0
            
            # Direction correctness
            predicted_up = predicted_price > current_price
            actual_up = actual_price > current_price
            direction_correct = predicted_up == actual_up
            
            # Update the prediction
            cursor.execute("""
                UPDATE model_predictions 
                SET actual_price = ?, 
                    error_pct = ?,
                    direction_correct = ?
                WHERE id = ?
            """, (actual_price, error_pct, direction_correct, pred_id))
            
            print(f"  OK {model}: {symbol} {target_date}")
            print(f"      Predicted: ${predicted_price:.2f}, Actual: ${actual_price:.2f}, Error: {error_pct:.1f}%")
            updated += 1
        else:
            print(f"  FAIL {model}: {symbol} {target_date} - no price data found")
    
    conn.commit()
    conn.close()
    
    print(f"\nOK Updated {updated} predictions with actual prices")
    return updated

def clean_corrupted_data():
    """Clean up corrupted predictions"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("\nCleaning corrupted predictions...")
    
    # Find Bitcoin predictions with bytes data
    cursor.execute("""
        SELECT id, model, symbol, predicted_price 
        FROM model_predictions 
        WHERE symbol = 'BTC-USD'
    """)
    
    bitcoin_preds = cursor.fetchall()
    cleaned = 0
    
    for pred_id, model, symbol, predicted_price in bitcoin_preds:
        if isinstance(predicted_price, bytes):
            print(f"  Removing corrupted Bitcoin prediction ID {pred_id}")
            cursor.execute("DELETE FROM model_predictions WHERE id = ?", (pred_id,))
            cleaned += 1
    
    conn.commit()
    conn.close()
    
    print(f"OK Cleaned {cleaned} corrupted predictions")
    return cleaned

if __name__ == "__main__":
    print("OVERDUE PREDICTIONS UPDATER")
    print("=" * 60)
    
    # Clean corrupted data first
    clean_corrupted_data()
    
    # Update overdue predictions
    updated = update_overdue_predictions()
    
    if updated > 0:
        print(f"\nSUCCESS: Updated {updated} overdue predictions")
        print("Dashboard will now show actual vs predicted comparisons!")
    else:
        print(f"\nWARNING: No predictions were updated - check data availability")