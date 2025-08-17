#!/usr/bin/env python3
"""
Daily Trading System Health Report
Generates comprehensive status report at 9am daily
"""

import sqlite3
import pandas as pd
import yfinance as yf
import os
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
import requests
import subprocess

# Configuration
DB_PATH = r'C:\Users\rrose\trading-models-system\core-system\models_dashboard.db'
DASHBOARD_PATH = r'C:\Users\rrose\trading-models-system\dashboard_improved.py'
REPORTS_DIR = r'C:\Users\rrose\trading-models-system\core-system\reports'

def print_header(title):
    """Print formatted section header"""
    print(f"\n{'='*60}")
    print(f"=== {title.upper()} ===")
    print(f"{'='*60}")

def print_status(item, status, details=""):
    """Print formatted status line"""
    status_icon = "OK" if status else "FAIL"
    print(f"[{status_icon}] {item}")
    if details:
        print(f"    {details}")

def check_database_health():
    """Check database connectivity and integrity"""
    print_header("Database Health Check")
    
    # Check if database file exists
    db_exists = os.path.exists(DB_PATH)
    print_status("Database File Exists", db_exists, DB_PATH if db_exists else "File not found")
    
    if not db_exists:
        return False
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check core tables exist
        tables = ['model_predictions', 'price_history', 'model_metrics']
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print_status(f"Table '{table}'", True, f"{count:,} records")
        
        # Check TODAY's predictions specifically
        cursor.execute("""
            SELECT COUNT(*) FROM model_predictions 
            WHERE prediction_date = date('now')
        """)
        today_predictions = cursor.fetchone()[0]
        
        # Check yesterday's predictions for context
        cursor.execute("""
            SELECT model, COUNT(*) as count
            FROM model_predictions 
            WHERE prediction_date = date('now', '-1 day')
            GROUP BY model
            ORDER BY count DESC
        """)
        yesterday_models = cursor.fetchall()
        
        # More nuanced status for today's predictions
        if today_predictions > 0:
            print_status("Today's New Predictions", True, f"{today_predictions} predictions made today")
        elif yesterday_models:
            print_status("Today's New Predictions", False, 
                        f"0 predictions today (but {len(yesterday_models)} models ran yesterday)")
        else:
            print_status("Today's New Predictions", False, "No recent model activity")
        
        # Check recent predictions (last 3 days)
        cursor.execute("""
            SELECT COUNT(*) FROM model_predictions 
            WHERE prediction_date >= date('now', '-3 days')
        """)
        recent_predictions = cursor.fetchone()[0]
        print_status("Recent Activity (3 days)", recent_predictions > 0, 
                    f"{recent_predictions} predictions")
        
        # Show yesterday's model activity for context
        if yesterday_models and today_predictions == 0:
            print("    Yesterday's Model Activity:")
            for model, count in yesterday_models:
                print(f"      {model}: {count} predictions on 2025-08-14")
        
        # Check overdue predictions with details
        cursor.execute("""
            SELECT model, symbol, target_date, predicted_price
            FROM model_predictions 
            WHERE target_date <= date('now') AND (actual_price IS NULL OR actual_price = '')
            ORDER BY target_date DESC
        """)
        overdue = cursor.fetchall()
        overdue_count = len(overdue)
        print_status("Overdue Predictions", overdue_count == 0, 
                    f"{overdue_count} predictions need actual prices" if overdue_count > 0 else "All current")
        
        if overdue_count > 0 and overdue_count <= 5:
            print("    Overdue Details:")
            for pred in overdue:
                print(f"      {pred[0]}: {pred[1]} target {pred[2]}")
        
        # Check models that ran today
        cursor.execute("""
            SELECT model, COUNT(*) as count
            FROM model_predictions 
            WHERE prediction_date = date('now')
            GROUP BY model
            ORDER BY count DESC
        """)
        todays_models = cursor.fetchall()
        active_models_today = len(todays_models)
        
        if active_models_today > 0:
            print_status("Active Models Today", True, f"{active_models_today} models generated predictions")
            print("    Models Active Today:")
            for model, count in todays_models:
                print(f"      {model}: {count} predictions")
        else:
            # Check if models are scheduled to run or should have run
            from datetime import datetime
            current_hour = datetime.now().hour
            if current_hour < 9:
                print_status("Active Models Today", True, "Models typically run after 9 AM")
            else:
                print_status("Active Models Today", False, 
                            f"No models ran today (last activity: yesterday with {len(yesterday_models)} models)")
        
        conn.close()
        return True
        
    except Exception as e:
        print_status("Database Connection", False, str(e))
        return False

def check_price_data_currency():
    """Check if price data is current"""
    print_header("Price Data Currency Check")
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check latest price data dates
        symbols = ['QQQ', 'NVDA', 'BTC-USD']
        today = date.today()
        
        for symbol in symbols:
            cursor.execute("""
                SELECT MAX(date) FROM price_history WHERE symbol = ?
            """, (symbol,))
            result = cursor.fetchone()
            
            if result and result[0]:
                latest_date = datetime.strptime(result[0], '%Y-%m-%d').date()
                days_old = (today - latest_date).days
                is_current = days_old <= 5  # Allow up to 5 days for weekends
                
                print_status(f"{symbol} Price Data", is_current, 
                           f"Latest: {latest_date} ({days_old} days ago)")
            else:
                print_status(f"{symbol} Price Data", False, "No data found")
        
        conn.close()
        
        # Test live API connectivity
        print("\nAPI Connectivity Test:")
        try:
            # Test yfinance
            ticker = yf.Ticker('QQQ')
            info = ticker.info
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            print_status("yfinance API", current_price is not None, 
                        f"QQQ current price: ${current_price}" if current_price else "No price data")
        except Exception as e:
            print_status("yfinance API", False, str(e))
            
    except Exception as e:
        print_status("Price Data Check", False, str(e))

def check_model_performance():
    """Check recent model performance and predictions"""
    print_header("Model Performance & Trading Signals Check")
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check today's signals first
        cursor.execute("""
            SELECT model, symbol, predicted_price, current_price,
                   CASE WHEN predicted_price > current_price THEN 'BUY' ELSE 'SELL' END as signal,
                   target_date, confidence
            FROM model_predictions 
            WHERE prediction_date = date('now')
            ORDER BY model
        """)
        
        todays_signals = cursor.fetchall()
        print_status("Today's Trading Signals", len(todays_signals) > 0,
                    f"{len(todays_signals)} new signals generated")
        
        if todays_signals:
            print("    Today's Signals:")
            for signal in todays_signals:
                model, symbol, pred_price, curr_price, signal_type, target, conf = signal
                conf_text = f", {conf}% confidence" if conf else ""
                print(f"      {model}: {symbol} {signal_type} ${pred_price:.2f} (from ${curr_price:.2f}) -> {target}{conf_text}")
        
        # Check recent performance with better accuracy calculation  
        cursor.execute("""
            SELECT 
                model,
                COUNT(*) as total_predictions,
                SUM(CASE WHEN actual_price IS NOT NULL AND actual_price != '' THEN 1 ELSE 0 END) as with_actuals,
                SUM(CASE WHEN direction_correct = 1 THEN 1 ELSE 0 END) as correct_direction,
                AVG(ABS(error_pct)) as avg_error,
                MAX(prediction_date) as latest_prediction
            FROM model_predictions 
            WHERE prediction_date >= date('now', '-7 days')
            GROUP BY model
            ORDER BY latest_prediction DESC
        """)
        
        recent_performance = cursor.fetchall()
        
        print("\nModel Performance (Last 7 Days):")
        for perf in recent_performance:
            model, total, with_actuals, correct, avg_error, latest = perf
            
            if with_actuals > 0:
                accuracy = (correct / with_actuals) * 100
                performance_good = accuracy >= 60
                status_text = f"{total} predictions, {with_actuals} completed, {accuracy:.1f}% accurate"
                if avg_error:
                    status_text += f", {avg_error:.1f}% avg error"
            else:
                performance_good = total > 0  # Good if making predictions
                status_text = f"{total} predictions, 0 completed (too recent)"
                
            status_text += f" (latest: {latest})"
            print_status(f"  {model}", performance_good, status_text)
        
        # Check for models that should be running but aren't
        cursor.execute("""
            SELECT DISTINCT model FROM model_predictions 
            WHERE prediction_date >= date('now', '-30 days')
        """)
        all_models = [row[0] for row in cursor.fetchall()]
        
        cursor.execute("""
            SELECT DISTINCT model FROM model_predictions 
            WHERE prediction_date >= date('now', '-2 days')
        """)
        recent_models = [row[0] for row in cursor.fetchall()]
        
        inactive_models = set(all_models) - set(recent_models)
        if inactive_models:
            print_status("Inactive Models", False, 
                        f"{len(inactive_models)} models haven't run recently: {', '.join(inactive_models)}")
        
        conn.close()
        
    except Exception as e:
        print_status("Model Performance Check", False, str(e))

def check_dashboard_status():
    """Check if dashboard is accessible and functional"""
    print_header("Dashboard Status Check")
    
    # Check if dashboard file exists
    dashboard_exists = os.path.exists(DASHBOARD_PATH)
    print_status("Dashboard File", dashboard_exists, DASHBOARD_PATH if dashboard_exists else "File not found")
    
    if not dashboard_exists:
        return False
    
    # Try to check if streamlit is running on port 8502
    try:
        response = requests.get('http://localhost:8502', timeout=5)
        dashboard_running = response.status_code == 200
        print_status("Dashboard Running", dashboard_running, 
                    "http://localhost:8502" if dashboard_running else "Not accessible")
    except:
        print_status("Dashboard Running", False, "Not accessible on port 8502")
    
    return True

def check_system_resources():
    """Check system resources and disk space"""
    print_header("System Resources Check")
    
    try:
        # Check disk space (Windows compatible)
        import shutil
        free_space_gb = shutil.disk_usage(os.path.dirname(DB_PATH)).free / (1024**3)
        disk_ok = free_space_gb > 1.0  # At least 1GB free
        
        print_status("Disk Space", disk_ok, f"{free_space_gb:.1f} GB available")
        
        # Check if reports directory exists and is writable
        reports_exist = os.path.exists(REPORTS_DIR)
        if reports_exist:
            reports_writable = os.access(REPORTS_DIR, os.W_OK)
            print_status("Reports Directory", reports_writable, 
                        REPORTS_DIR if reports_writable else "Not writable")
        else:
            print_status("Reports Directory", False, "Directory not found")
            
    except Exception as e:
        print_status("System Resources Check", False, str(e))

def check_recent_model_runs():
    """Check when models last ran successfully"""
    print_header("Recent Model Execution Check")
    
    try:
        # Check for recent log files
        log_files = []
        if os.path.exists(REPORTS_DIR):
            for file in os.listdir(REPORTS_DIR):
                if file.endswith('.log') or 'Report' in file:
                    file_path = os.path.join(REPORTS_DIR, file)
                    mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    log_files.append((file, mod_time))
        
        if log_files:
            log_files.sort(key=lambda x: x[1], reverse=True)
            latest_log = log_files[0]
            hours_ago = (datetime.now() - latest_log[1]).total_seconds() / 3600
            
            recent_run = hours_ago <= 24  # Model ran within 24 hours
            print_status("Recent Model Execution", recent_run,
                        f"Latest: {latest_log[0]} ({hours_ago:.1f} hours ago)")
        else:
            print_status("Recent Model Execution", False, "No recent log files found")
            
        # Check database for recent predictions
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT model, MAX(prediction_date) as latest_prediction
            FROM model_predictions
            GROUP BY model
            ORDER BY latest_prediction DESC
        """)
        
        results = cursor.fetchall()
        if results:
            print("\nLatest Predictions by Model:")
            for model, latest_date in results:
                if latest_date:
                    pred_date = datetime.strptime(latest_date, '%Y-%m-%d').date()
                    days_ago = (date.today() - pred_date).days
                    recent = days_ago <= 1
                    print_status(f"  {model}", recent, f"{latest_date} ({days_ago} days ago)")
        
        conn.close()
        
    except Exception as e:
        print_status("Model Execution Check", False, str(e))

def generate_summary_report():
    """Generate overall system health summary with actionable insights"""
    print_header("Daily Health Report Summary")
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Report Generated: {current_time}")
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # TODAY's specific metrics
        cursor.execute("SELECT COUNT(*) FROM model_predictions WHERE prediction_date = date('now')")
        todays_predictions = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(DISTINCT model) FROM model_predictions 
            WHERE prediction_date = date('now')
        """)
        active_models_today = cursor.fetchone()[0]
        
        # Overall system metrics
        cursor.execute("SELECT COUNT(*) FROM model_predictions")
        total_predictions = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*) FROM model_predictions 
            WHERE actual_price IS NOT NULL AND actual_price != ''
        """)
        predictions_with_results = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*) FROM model_predictions 
            WHERE target_date <= date('now') AND (actual_price IS NULL OR actual_price = '')
        """)
        overdue_predictions = cursor.fetchone()[0]
        
        # Accuracy calculation with better filtering
        cursor.execute("""
            SELECT 
                COUNT(*) as total_with_results,
                SUM(CASE WHEN direction_correct = 1 THEN 1 ELSE 0 END) as correct_directions,
                AVG(ABS(error_pct)) as avg_error
            FROM model_predictions 
            WHERE actual_price IS NOT NULL AND actual_price != '' AND direction_correct IS NOT NULL
        """)
        result = cursor.fetchone()
        total_with_results, correct_directions, avg_error = result
        overall_accuracy = (correct_directions / total_with_results * 100) if total_with_results > 0 else 0
        
        # Recent trading signals
        cursor.execute("""
            SELECT COUNT(*) FROM model_predictions 
            WHERE prediction_date >= date('now', '-3 days')
        """)
        recent_signals = cursor.fetchone()[0]
        
        conn.close()
        
        print(f"\nTODAY'S ACTIVITY:")
        print(f"  New Predictions: {todays_predictions}")
        print(f"  Active Models: {active_models_today}")
        
        print(f"\nSYSTEM PERFORMANCE:")
        print(f"  Total Predictions: {total_predictions:,}")
        print(f"  Completed Predictions: {predictions_with_results:,}")
        print(f"  Overall Direction Accuracy: {overall_accuracy:.1f}% ({correct_directions}/{total_with_results})")
        if avg_error:
            print(f"  Average Price Error: {avg_error:.1f}%")
        print(f"  Recent Activity (3 days): {recent_signals} signals")
        
        print(f"\nISSUES TO ADDRESS:")
        print(f"  Overdue Predictions: {overdue_predictions}")
        
        # Calculate health score with better criteria
        health_score = 100
        
        # Penalize for system issues
        if overdue_predictions > 5:
            health_score -= 25
        elif overdue_predictions > 0:
            health_score -= 10
            
        if overall_accuracy < 50 and total_with_results >= 5:
            health_score -= 30
        elif overall_accuracy < 60 and total_with_results >= 5:
            health_score -= 15
            
        if todays_predictions == 0:
            health_score -= 20  # No activity today
            
        if predictions_with_results < 5:
            health_score -= 15  # Need more completed predictions for evaluation
            
        print(f"\nSYSTEM HEALTH SCORE: {health_score}/100")
        
        if health_score >= 85:
            status = "EXCELLENT"
        elif health_score >= 70:
            status = "GOOD" 
        elif health_score >= 50:
            status = "FAIR"
        else:
            status = "NEEDS ATTENTION"
            
        print(f"OVERALL STATUS: {status}")
        
        # Check if models should have run today
        current_hour = datetime.now().hour
        
        # Specific recommendations
        print(f"\nRECOMMENDATIONS:")
        if todays_predictions == 0:
            if current_hour < 9:
                print("- INFO: Models typically run after 9 AM market open")
            else:
                print("- CHECK: No new predictions today - verify model scheduling and execution")
                print("- ACTION: Check if models are configured to run daily vs other intervals")
        
        if overdue_predictions > 0:
            print(f"- UPDATE: {overdue_predictions} predictions need actual prices - run update_overdue_predictions.py")
        
        if overall_accuracy < 60 and total_with_results >= 5:
            print(f"- REVIEW: Model accuracy at {overall_accuracy:.1f}% - investigate underperforming models")
        
        if predictions_with_results < total_predictions * 0.3:
            print("- MONITOR: Many predictions still pending - ensure actual price updates are working")
            
        # Get yesterday activity (already have this data from above)
        yesterday_activity = recent_signals
        
        # Trading readiness assessment - be more realistic
        recent_models_working = yesterday_activity > 0 or todays_predictions > 0
        data_current = overdue_predictions < 5
        accuracy_acceptable = overall_accuracy >= 50 or total_with_results < 5
        
        trading_ready = recent_models_working and data_current and accuracy_acceptable
        
        if trading_ready:
            if todays_predictions > 0:
                status_detail = "All systems operational with fresh predictions"
            else:
                status_detail = "System functional (last predictions: yesterday)"
        else:
            issues = []
            if not recent_models_working:
                issues.append("no recent model activity")
            if not data_current:
                issues.append(f"{overdue_predictions} overdue predictions")  
            if not accuracy_acceptable:
                issues.append(f"low accuracy ({overall_accuracy:.1f}%)")
            status_detail = f"Issues: {', '.join(issues)}"
            
        print(f"\nTRADING SYSTEM STATUS: {'READY' if trading_ready else 'NEEDS ATTENTION'}")
        print(f"  {status_detail}")
            
    except Exception as e:
        print(f"Error generating summary: {e}")

def main():
    """Run complete daily health report"""
    print("TRADING SYSTEM DAILY HEALTH REPORT")
    print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all health checks
    check_database_health()
    check_price_data_currency()
    check_model_performance()
    check_dashboard_status()
    check_system_resources()
    check_recent_model_runs()
    generate_summary_report()
    
    print(f"\n{'='*60}")
    print("Health report complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()