#!/usr/bin/env python3
"""
Daily Model Performance Tracker
Runs at 9 AM to update and display rolling model performance
Integrates with daily health check process
"""

import sqlite3
from datetime import datetime, timedelta
import os

def update_missing_prices():
    """Update any missing actual prices from price_history"""
    
    conn = sqlite3.connect('reports_tracking.db')
    cursor = conn.cursor()
    
    # Check for missing prices
    cursor.execute("""
        SELECT p.id, p.symbol, p.target_date
        FROM model_predictions p
        WHERE p.actual_price IS NULL 
        AND date(p.target_date) <= date('now', '-1 day')
        LIMIT 50
    """)
    
    missing = cursor.fetchall()
    
    if missing:
        print(f"  Updating {len(missing)} missing prices...")
        updated = 0
        
        for pred_id, symbol, target_date in missing:
            cursor.execute("""
                SELECT CAST(close AS REAL)
                FROM price_history 
                WHERE symbol = ? 
                AND date(date) = date(?)
            """, (symbol, target_date))
            
            result = cursor.fetchone()
            if result and result[0]:
                cursor.execute("""
                    UPDATE model_predictions 
                    SET actual_price = ?
                    WHERE id = ?
                """, (float(result[0]), pred_id))
                updated += 1
        
        conn.commit()
        print(f"  Updated {updated} prices")
    
    conn.close()

def calculate_model_performance():
    """Calculate and return current model performance metrics - FIXED for signal-based models"""

    conn = sqlite3.connect('reports_tracking.db')
    cursor = conn.cursor()

    results = []

    # First, handle signal-based models (check if signal_predictions table exists)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='signal_predictions'")
    if cursor.fetchone():
        # Get Wishing Well signal performance
        cursor.execute("""
            SELECT
                'Wishing Well QQQ Model' as model,
                COUNT(*) as total_signals,
                SUM(CASE WHEN signal_correct = 1 THEN 1 ELSE 0 END) as correct_signals,
                AVG(CASE WHEN actual_return IS NOT NULL THEN ABS(actual_return) ELSE 0 END) as avg_return,
                COUNT(CASE WHEN actual_return IS NOT NULL THEN 1 END) as resolved_signals
            FROM signal_predictions
            WHERE model = 'Wishing Well QQQ Model'
        """)

        signal_row = cursor.fetchone()
        if signal_row and signal_row[1] > 0:
            model, total, correct, avg_return, resolved = signal_row
            signal_accuracy = (correct / resolved * 100) if resolved > 0 else 0

            # Show signal info even if no resolved signals yet
            results.append({
                'model': model,
                'accuracy': signal_accuracy,
                'correct': correct or 0,
                'total': total,
                'resolved': resolved,
                'avg_error': abs(avg_return) or 0,
                'is_signal_model': True
            })

    # Then handle price prediction models (exclude Wishing Well)
    cursor.execute("""
        SELECT
            model,
            COUNT(*) as total_resolved,
            SUM(CASE
                WHEN CAST(predicted_price AS REAL) > CAST(current_price AS REAL)
                     AND CAST(actual_price AS REAL) > CAST(current_price AS REAL) THEN 1
                WHEN CAST(predicted_price AS REAL) < CAST(current_price AS REAL)
                     AND CAST(actual_price AS REAL) < CAST(current_price AS REAL) THEN 1
                ELSE 0
            END) as correct_predictions,
            AVG(ABS((CAST(predicted_price AS REAL) - CAST(actual_price AS REAL)) /
                    CAST(actual_price AS REAL) * 100)) as avg_error
        FROM model_predictions
        WHERE actual_price IS NOT NULL
        AND predicted_price IS NOT NULL
        AND current_price IS NOT NULL
        AND model != 'Wishing Well QQQ Model'  -- Exclude signal-based models
        GROUP BY model
        ORDER BY (SUM(CASE
            WHEN CAST(predicted_price AS REAL) > CAST(current_price AS REAL)
                 AND CAST(actual_price AS REAL) > CAST(current_price AS REAL) THEN 1
            WHEN CAST(predicted_price AS REAL) < CAST(current_price AS REAL)
                 AND CAST(actual_price AS REAL) < CAST(current_price AS REAL) THEN 1
            ELSE 0
        END) * 1.0 / COUNT(*)) DESC
    """)

    for row in cursor.fetchall():
        model, total, correct, avg_error = row
        accuracy = (correct / total * 100) if total > 0 else 0
        results.append({
            'model': model,
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'avg_error': avg_error or 0,
            'is_signal_model': False
        })

    # Sort all results by accuracy
    results.sort(key=lambda x: x['accuracy'], reverse=True)

    conn.close()
    return results

def check_yesterday_status():
    """Check if each model made predictions and saved prices yesterday - FIXED for signal models"""

    conn = sqlite3.connect('reports_tracking.db')
    cursor = conn.cursor()

    yesterday = (datetime.now() - timedelta(days=1)).date()
    status = {}

    # Check for price prediction models
    cursor.execute("""
        SELECT model,
               COUNT(CASE WHEN date(prediction_date) = date(?) THEN 1 END) as pred_yesterday,
               COUNT(CASE WHEN date(target_date) = date(?) AND actual_price IS NOT NULL THEN 1 END) as price_saved
        FROM model_predictions
        WHERE model != 'Wishing Well QQQ Model'
        GROUP BY model
    """, (yesterday, yesterday))

    for model, pred_count, price_count in cursor.fetchall():
        status[model] = {
            'prediction_made': pred_count > 0,
            'price_saved': price_count > 0
        }

    # Check for signal-based models (if table exists)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='signal_predictions'")
    if cursor.fetchone():
        cursor.execute("""
            SELECT model,
                   COUNT(CASE WHEN date(prediction_date) = date(?) THEN 1 END) as signals_yesterday,
                   COUNT(CASE WHEN date(target_date) = date(?) AND actual_return IS NOT NULL THEN 1 END) as returns_saved
            FROM signal_predictions
            WHERE model = 'Wishing Well QQQ Model'
            GROUP BY model
        """, (yesterday, yesterday))

        for model, signal_count, return_count in cursor.fetchall():
            status[model] = {
                'prediction_made': signal_count > 0,
                'price_saved': return_count > 0  # For signals, this represents return data
            }

    conn.close()
    return status

def display_performance_table(results):
    """Display the performance table in a formatted way with daily status checks"""
    
    # Get yesterday's status for each model
    yesterday_status = check_yesterday_status()
    
    print("\n  MODEL PERFORMANCE RANKINGS (Updated Daily):")
    print("  " + "=" * 85)
    print(f"  {'Rank':<6} {'Model':<25} {'Accuracy':<10} {'Sample':<8} {'Avg Error':<10} {'Status':<8} {'Yesterday':<12}")
    print(f"  {'':6} {'':25} {'':10} {'':8} {'':10} {'':8} {'Pred | Price':<12}")
    print("  " + "-" * 85)
    
    for rank, result in enumerate(results, 1):
        model_name = result['model'][:24]

        # Handle signal-based models differently
        if result.get('is_signal_model', False):
            resolved = result.get('resolved', 0)
            total = result['total']
            if resolved > 0:
                accuracy = f"GMI {result['accuracy']:.0f}%"
                avg_error = f"{result['avg_error']:.1f}% ret"
                sample = f"{resolved}/{total}"
            else:
                accuracy = "Pending"
                avg_error = "N/A"
                sample = f"0/{total}"
        else:
            accuracy = f"{result['accuracy']:.1f}%"
            avg_error = f"{result['avg_error']:.1f}%"
            sample = str(result['total'])

        # Add performance status indicator (adjusted for signal models)
        if result.get('is_signal_model', False):
            resolved = result.get('resolved', 0)
            if resolved == 0:
                status = "NEW"  # New signal model with no resolved predictions yet
            elif result['accuracy'] >= 60:
                status = "GOOD"
            elif result['accuracy'] >= 40:
                status = "OK"
            else:
                status = "POOR"
        else:
            # For price prediction models
            if result['accuracy'] >= 65:
                status = "GOOD"
            elif result['accuracy'] >= 50:
                status = "OK"
            else:
                status = "POOR"
        
        # Check yesterday's activity
        model_status = yesterday_status.get(result['model'], {})
        pred_check = "Y" if model_status.get('prediction_made', False) else "N"
        price_check = "Y" if model_status.get('price_saved', False) else "N"
        yesterday_checks = f"{pred_check:^5}|{price_check:^5}"
        
        print(f"  {rank:<6} {model_name:<25} {accuracy:<10} {sample:<8} {avg_error:<10} {status:<8} {yesterday_checks:<12}")
    
    print("  " + "=" * 85)
    print("  Legend: Yesterday Pred = Prediction made yesterday | Price = Price history saved")

def save_performance_history(results):
    """Save performance history to track trends"""
    
    conn = sqlite3.connect('reports_tracking.db')
    cursor = conn.cursor()
    
    # Create history table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS performance_history (
            date DATE,
            model TEXT,
            accuracy REAL,
            sample_size INTEGER,
            avg_error REAL,
            PRIMARY KEY (date, model)
        )
    """)
    
    today = datetime.now().date()
    
    for result in results:
        cursor.execute("""
            INSERT OR REPLACE INTO performance_history 
            (date, model, accuracy, sample_size, avg_error)
            VALUES (?, ?, ?, ?, ?)
        """, (today, result['model'], result['accuracy'], 
              result['total'], result['avg_error']))
    
    conn.commit()
    conn.close()

def show_performance_trend():
    """Show 7-day performance trend for top models"""
    
    conn = sqlite3.connect('reports_tracking.db')
    cursor = conn.cursor()
    
    # Check if history table exists
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='performance_history'
    """)
    
    if not cursor.fetchone():
        conn.close()
        return
    
    # Get 7-day trend for top 3 models
    cursor.execute("""
        SELECT model, date, accuracy
        FROM performance_history
        WHERE date >= date('now', '-7 days')
        AND model IN (
            SELECT model 
            FROM performance_history 
            WHERE date = date('now')
            ORDER BY accuracy DESC
            LIMIT 3
        )
        ORDER BY model, date
    """)
    
    trends = {}
    for model, date, accuracy in cursor.fetchall():
        if model not in trends:
            trends[model] = []
        trends[model].append((date, accuracy))
    
    if trends:
        print("\n  7-DAY PERFORMANCE TRENDS (Top 3 Models):")
        print("  " + "-" * 50)
        
        for model, data in trends.items():
            if len(data) > 1:
                first_acc = data[0][1]
                last_acc = data[-1][1]
                change = last_acc - first_acc
                trend = "UP" if change > 0 else "DOWN" if change < 0 else "FLAT"
                print(f"  {model[:24]:<25} {trend} {change:+.1f}% over 7 days")
    
    conn.close()

def run_daily_performance_update():
    """Main function to run the daily performance update"""
    
    print("\n" + "="*70)
    print("  DAILY MODEL PERFORMANCE UPDATE")
    print(f"  Run Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Step 1: Update missing prices
    print("\n  Step 1: Updating missing actual prices...")
    update_missing_prices()
    
    # Step 2: Calculate current performance
    print("  Step 2: Calculating model performance...")
    results = calculate_model_performance()
    
    # Step 3: Display performance table
    display_performance_table(results)
    
    # Step 4: Save history
    print("\n  Step 3: Saving performance history...")
    save_performance_history(results)
    
    # Step 5: Show trends
    show_performance_trend()
    
    # Summary statistics
    conn = sqlite3.connect('reports_tracking.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            COUNT(DISTINCT model) as models,
            COUNT(*) as total_predictions,
            COUNT(actual_price) as resolved,
            COUNT(CASE WHEN actual_price IS NULL AND date(target_date) <= date('now') THEN 1 END) as missing
        FROM model_predictions
    """)
    
    stats = cursor.fetchone()
    
    print("\n  DATABASE STATISTICS:")
    print("  " + "-" * 30)
    print(f"  Total Models: {stats[0]}")
    print(f"  Total Predictions: {stats[1]}")
    print(f"  Resolved: {stats[2]} ({stats[2]/stats[1]*100:.1f}%)")
    print(f"  Pending Resolution: {stats[3]}")
    
    conn.close()
    
    print("\n" + "="*70)
    print("  Daily performance update complete!")
    print("="*70)

if __name__ == "__main__":
    run_daily_performance_update()