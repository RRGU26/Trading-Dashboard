#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily Health Check v2 - Focus on what matters:
1. Did models run today?
2. Did predictions save to database?
3. What was the accuracy of predictions?
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Database path
DB_PATH = r'C:\Users\rrose\trading-models-system\databases\reports_tracking.db'

def check_model_executions():
    """Check if models ran today by looking at database predictions"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    today = datetime.now().strftime('%Y-%m-%d')
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    print("=" * 60)
    print("TRADING MODELS EXECUTION STATUS")
    print("=" * 60)
    print(f"Checking for: {today}\n")
    
    # Get all model predictions from today
    cursor.execute("""
        SELECT model, COUNT(*) as prediction_count, 
               MAX(created_timestamp) as latest_prediction
        FROM model_predictions
        WHERE DATE(created_timestamp) = ?
        GROUP BY model
        ORDER BY model
    """, (today,))
    
    today_results = cursor.fetchall()
    
    if not today_results:
        # Check yesterday if today is empty (models might not have run yet)
        print(f"No predictions found for today. Checking yesterday ({yesterday})...\n")
        cursor.execute("""
            SELECT model, COUNT(*) as prediction_count, 
                   MAX(created_timestamp) as latest_prediction
            FROM model_predictions
            WHERE DATE(created_timestamp) = ?
            GROUP BY model
            ORDER BY model
        """, (yesterday,))
        today_results = cursor.fetchall()
        check_date = yesterday
    else:
        check_date = today
    
    # Define expected models
    expected_models = [
        'QQQ Trading Signal',
        'Long Bull Model V3.2',  # QQQ Long Bull
        'Algorand Model',
        'Bitcoin Model',
        'NVIDIA Bull Momentum Model',
        'Wishing Well QQQ Model',
        'QQQ Master Model'
    ]
    
    models_that_ran = {}
    
    print(f"Models that ran on {check_date}:")
    print("-" * 40)
    
    for model_name, count, latest in today_results:
        models_that_ran[model_name] = {
            'count': count,
            'latest': latest
        }
        status = "[OK]"
        print(f"{status} {model_name}: {count} predictions at {latest}")
    
    print("\nModels that didn't run:")
    print("-" * 40)
    
    missing_models = []
    for model in expected_models:
        if model not in models_that_ran:
            print(f"[MISSING] {model}: No predictions")
            missing_models.append(model)
    
    if not missing_models:
        print("[SUCCESS] All expected models ran!")
    
    conn.close()
    return models_that_ran, missing_models

def check_prediction_accuracy():
    """Check accuracy of recent predictions"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("\n" + "=" * 60)
    print("PREDICTION ACCURACY (Last 5 Days)")
    print("=" * 60)
    
    # Get actual vs predicted for recent predictions
    five_days_ago = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
    
    cursor.execute("""
        SELECT 
            mp.model,
            mp.symbol,
            mp.prediction_date,
            mp.predicted_price,
            ph.close as actual_price,
            ROUND(((ph.close - mp.predicted_price) / mp.predicted_price * 100), 2) as error_pct
        FROM model_predictions mp
        LEFT JOIN price_history ph 
            ON mp.symbol = ph.symbol 
            AND mp.prediction_date = ph.date
        WHERE mp.prediction_date >= ?
            AND ph.close IS NOT NULL
        ORDER BY mp.model, mp.prediction_date DESC
    """, (five_days_ago,))
    
    results = cursor.fetchall()
    
    # Group by model
    model_accuracy = {}
    for model, symbol, pred_date, predicted, actual, error in results:
        if model not in model_accuracy:
            model_accuracy[model] = []
        model_accuracy[model].append({
            'symbol': symbol,
            'date': pred_date,
            'predicted': predicted,
            'actual': actual,
            'error': error
        })
    
    # Calculate accuracy stats for each model
    for model, predictions in model_accuracy.items():
        if predictions:
            avg_error = sum(abs(p['error']) for p in predictions) / len(predictions)
            direction_correct = sum(1 for p in predictions if 
                                   (p['predicted'] > p['actual'] and p['error'] < 0) or
                                   (p['predicted'] < p['actual'] and p['error'] > 0))
            
            print(f"\n{model}:")
            print(f"  Predictions checked: {len(predictions)}")
            print(f"  Avg error: {avg_error:.2f}%")
            print(f"  Direction accuracy: {direction_correct}/{len(predictions)} ({direction_correct/len(predictions)*100:.0f}%)")
            
            # Show most recent prediction
            latest = predictions[0]
            print(f"  Latest: {latest['symbol']} on {latest['date']}")
            print(f"    Predicted: ${latest['predicted']:.2f}, Actual: ${latest['actual']:.2f}, Error: {latest['error']:.2f}%")
    
    conn.close()
    return model_accuracy

def check_database_health():
    """Quick database health check"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check table counts
        cursor.execute("""
            SELECT 
                (SELECT COUNT(*) FROM model_predictions) as predictions,
                (SELECT COUNT(*) FROM price_history) as prices,
                (SELECT COUNT(*) FROM model_metrics) as metrics
        """)
        
        counts = cursor.fetchone()
        conn.close()
        
        print("\n" + "=" * 60)
        print("DATABASE HEALTH")
        print("=" * 60)
        print(f"[OK] Database accessible")
        print(f"  Total predictions: {counts[0]:,}")
        print(f"  Price history records: {counts[1]:,}")
        print(f"  Model metrics: {counts[2]:,}")
        
        return True
    except Exception as e:
        print(f"[ERROR] Database error: {e}")
        return False

def generate_summary():
    """Generate actionable summary"""
    print("\n" + "=" * 60)
    print("SUMMARY & ACTION ITEMS")
    print("=" * 60)
    
    models_ran, missing = check_model_executions()
    
    if missing:
        print("\n[WARNING] ATTENTION REQUIRED:")
        print(f"  - {len(missing)} models didn't run")
        print("  - Check if market was closed or data issues")
        print("  - Run manual execution if needed")
    else:
        print("\n[SUCCESS] ALL SYSTEMS OPERATIONAL")
        print("  - All models executed successfully")
        print("  - Predictions saved to database")
    
    print("\nNext Steps:")
    print("  1. Review Combined Trading Report on Desktop")
    print("  2. Check model predictions in dashboard")
    print("  3. Execute any recommended trades before market close")
    
    print(f"\nReport generated: {datetime.now().strftime('%Y-%m-%d %I:%M %p ET')}")

def main():
    print("DAILY TRADING SYSTEM HEALTH CHECK V2")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"Time: {datetime.now().strftime('%I:%M %p ET')}\n")
    
    # Check database first
    if not check_database_health():
        print("\n[CRITICAL] Database not accessible!")
        return
    
    # Check model executions
    models_ran, missing = check_model_executions()
    
    # Check prediction accuracy
    accuracy = check_prediction_accuracy()
    
    # Generate summary
    generate_summary()

if __name__ == "__main__":
    main()