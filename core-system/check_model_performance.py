#!/usr/bin/env python3
"""
Check how models performed by comparing predictions to actual results
"""

import sqlite3
import json
from datetime import datetime, timedelta

def check_model_performance():
    # Connect to database
    conn = sqlite3.connect('models_dashboard.db')
    cursor = conn.cursor()
    
    # Get yesterday's predictions (2025-08-15)
    yesterday = '2025-08-15'
    today = '2025-08-16'
    
    # Query for yesterday's predictions
    cursor.execute('''
        SELECT model, prediction_date, prediction_data 
        FROM model_predictions 
        WHERE prediction_date = ?
        ORDER BY model
    ''', (yesterday,))
    
    yesterday_predictions = cursor.fetchall()
    
    # Query for today's predictions (which contain today's actual prices)
    cursor.execute('''
        SELECT model, prediction_date, prediction_data 
        FROM model_predictions 
        WHERE prediction_date = ?
        ORDER BY model
    ''', (today,))
    
    today_predictions = cursor.fetchall()
    
    print(f"MODELS PERFORMANCE CHECK - Comparing {yesterday}'s predictions to {today}'s actuals")
    print("=" * 80)
    
    # Create dictionaries for easy lookup
    yesterday_data = {}
    for model, date, data in yesterday_predictions:
        try:
            yesterday_data[model] = json.loads(data)
        except:
            yesterday_data[model] = {}
    
    today_data = {}
    for model, date, data in today_predictions:
        try:
            today_data[model] = json.loads(data)
        except:
            today_data[model] = {}
    
    # Check each model's performance
    results = []
    
    # QQQ Models
    qqq_models = ['QQQ Long Bull Model', 'QQQ Master Model', 'QQQ Trading Signal', 'Wishing Well QQQ Model']
    
    for model in qqq_models:
        if model in yesterday_data and model in today_data:
            print(f"\n{model}:")
            yesterday_pred = yesterday_data[model]
            today_actual = today_data[model]
            
            # Get current price from today's data
            current_price = today_actual.get('current_price', 577.34)
            print(f"  Today's QQQ Price: ${current_price}")
            
            # Check yesterday's prediction
            if 'target_price' in yesterday_pred:
                target = yesterday_pred['target_price']
                print(f"  Yesterday's Target Price: ${target}")
                error = ((current_price - target) / target) * 100
                print(f"  Prediction Error: {error:.2f}%")
                
            if 'expected_return' in yesterday_pred:
                expected = yesterday_pred['expected_return']
                print(f"  Expected Return was: {expected}%")
                
            if 'signal' in yesterday_pred:
                signal = yesterday_pred['signal']
                print(f"  Yesterday's Signal: {signal}")
                
                # Check if signal was correct
                if model in today_data and 'current_price' in today_actual:
                    # Assuming we have yesterday's price stored
                    if 'current_price' in yesterday_pred:
                        yesterday_price = yesterday_pred['current_price']
                        actual_return = ((current_price - yesterday_price) / yesterday_price) * 100
                        print(f"  Actual Return: {actual_return:.2f}%")
                        
                        if signal == 'BUY' and actual_return > 0:
                            print(f"  ✅ Signal was CORRECT (BUY and price went up)")
                        elif signal == 'SELL' and actual_return < 0:
                            print(f"  ✅ Signal was CORRECT (SELL and price went down)")
                        elif signal == 'HOLD':
                            print(f"  ✅ HOLD signal (neutral)")
                        else:
                            print(f"  ❌ Signal was INCORRECT")
    
    # NVIDIA Model
    if 'NVIDIA Bull Momentum' in yesterday_data:
        print(f"\nNVIDIA Bull Momentum:")
        yesterday_pred = yesterday_data['NVIDIA Bull Momentum']
        
        if 'predicted_1_day_price' in yesterday_pred:
            predicted = yesterday_pred['predicted_1_day_price']
            print(f"  Predicted 1-Day Price: ${predicted}")
            
            # Get actual NVDA price from today
            if 'NVIDIA Bull Momentum' in today_data:
                actual = today_data['NVIDIA Bull Momentum'].get('current_price', 180.45)
                print(f"  Actual Price Today: ${actual}")
                error = ((actual - predicted) / predicted) * 100
                print(f"  Prediction Error: {error:.2f}%")
                
                if abs(error) < 2:
                    print(f"  ✅ Good prediction (within 2%)")
                else:
                    print(f"  ❌ Poor prediction (error > 2%)")
    
    # Algorand Model
    if 'Algorand Model' in yesterday_data:
        print(f"\nAlgorand Model:")
        yesterday_pred = yesterday_data['Algorand Model']
        
        if 'predicted_price' in yesterday_pred:
            predicted = yesterday_pred['predicted_price']
            print(f"  Predicted Price: ${predicted}")
            
            if 'Algorand Model' in today_data:
                actual = today_data['Algorand Model'].get('current_price', 0.2854)
                print(f"  Actual Price Today: ${actual}")
                error = ((actual - predicted) / predicted) * 100
                print(f"  Prediction Error: {error:.2f}%")
    
    # Bitcoin Model
    if 'Bitcoin Model' in yesterday_data:
        print(f"\nBitcoin Model:")
        yesterday_pred = yesterday_data['Bitcoin Model']
        
        if 'predicted_1_day_price' in yesterday_pred:
            predicted = yesterday_pred['predicted_1_day_price']
            print(f"  Predicted 1-Day Price: ${predicted}")
            
            if 'Bitcoin Model' in today_data:
                actual = today_data['Bitcoin Model'].get('current_price', 117776.0)
                print(f"  Actual Price Today: ${actual}")
                error = ((actual - predicted) / predicted) * 100
                print(f"  Prediction Error: {error:.2f}%")
                
                if abs(error) < 2:
                    print(f"  ✅ Good prediction (within 2%)")
                else:
                    print(f"  ❌ Poor prediction (error > 2%)")
    
    print("\n" + "=" * 80)
    print("SUMMARY: Performance analysis complete")
    
    conn.close()

if __name__ == "__main__":
    check_model_performance()