#!/usr/bin/env python3
"""
DAILY 9 AM HEALTH CHECK PROCESS
Comprehensive daily health check including model performance tracking
Should be scheduled to run daily at 9:00 AM
"""

import subprocess
import sys
import os
from datetime import datetime

def run_health_check():
    """Main daily health check orchestrator"""
    
    print("\n" + "="*80)
    print("  DAILY 9 AM HEALTH CHECK")
    print(f"  Run Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Step 1: Run model performance tracker
    print("\n[1/4] Running Model Performance Tracker...")
    print("-" * 60)
    try:
        result = subprocess.run(
            [sys.executable, "daily_model_performance_tracker.py"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        # Display the output
        if result.stdout:
            print(result.stdout)
        
        if result.returncode != 0 and result.stderr:
            print(f"Warning: {result.stderr}")
    except Exception as e:
        print(f"Error running performance tracker: {e}")
    
    # Step 2: Check database integrity
    print("\n[2/4] Checking Database Integrity...")
    print("-" * 60)
    try:
        import sqlite3
        conn = sqlite3.connect('reports_tracking.db')
        cursor = conn.cursor()
        
        # Check for data consistency
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(CASE WHEN actual_price IS NOT NULL AND direction_correct IS NULL THEN 1 END) as missing_direction,
                COUNT(CASE WHEN actual_price IS NULL AND date(target_date) <= date('now', '-2 days') THEN 1 END) as overdue
            FROM model_predictions
        """)
        
        total, missing_dir, overdue = cursor.fetchone()
        
        print(f"  Database Check Results:")
        print(f"    Total Predictions: {total}")
        print(f"    Missing Direction Flags: {missing_dir}")
        print(f"    Overdue Price Updates: {overdue}")
        
        if missing_dir > 0 or overdue > 10:
            print("    STATUS: ATTENTION NEEDED")
        else:
            print("    STATUS: OK")
        
        conn.close()
    except Exception as e:
        print(f"  Error checking database: {e}")
    
    # Step 3: Check system processes
    print("\n[3/4] Checking System Processes...")
    print("-" * 60)
    
    processes_to_check = [
        "wrapper.py",
        "data_fetcher.py",
        "model_db_integrator.py"
    ]
    
    for process in processes_to_check:
        try:
            # Check if process file exists
            if os.path.exists(process):
                print(f"  {process}: FOUND")
            else:
                print(f"  {process}: NOT FOUND")
        except:
            pass
    
    # Step 4: Generate daily summary
    print("\n[4/4] Daily Summary...")
    print("-" * 60)
    
    try:
        conn = sqlite3.connect('reports_tracking.db')
        cursor = conn.cursor()
        
        # Get yesterday's activity
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT model) as active_models,
                COUNT(*) as predictions_made,
                AVG(confidence) as avg_confidence
            FROM model_predictions
            WHERE date(prediction_date) = date('now', '-1 day')
        """)
        
        active, preds, conf = cursor.fetchone()
        
        print(f"  Yesterday's Activity:")
        print(f"    Active Models: {active or 0}")
        print(f"    Predictions Made: {preds or 0}")
        print(f"    Average Confidence: {conf:.1f}%" if conf else "    Average Confidence: N/A")
        
        conn.close()
    except Exception as e:
        print(f"  Error generating summary: {e}")
    
    print("\n" + "="*80)
    print("  DAILY HEALTH CHECK COMPLETE")
    print("  Next run scheduled for tomorrow at 9:00 AM")
    print("="*80)

if __name__ == "__main__":
    run_health_check()