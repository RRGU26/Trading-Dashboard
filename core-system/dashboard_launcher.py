#!/usr/bin/env python3
"""
DASHBOARD LAUNCHER
==================
Easy launcher for the trading models dashboard with automatic setup
"""

import subprocess
import time
import webbrowser
import os
import sys

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    print("TRADING MODELS DASHBOARD LAUNCHER")
    print("=" * 50)
    
    # Check if we're in the right directory
    dashboard_file = "dashboard.py"
    if not os.path.exists(dashboard_file):
        print(f"[ERROR] {dashboard_file} not found in current directory")
        print(f"Current directory: {os.getcwd()}")
        return False
    
    # Check database
    db_file = "reports_tracking.db"
    if not os.path.exists(db_file):
        print(f"[WARNING] {db_file} not found - dashboard may show limited data")
    else:
        print(f"[OK] Database found: {db_file}")
    
    try:
        print("[INFO] Starting Streamlit dashboard...")
        print("[INFO] Dashboard will be available at: http://localhost:8501")
        print("[INFO] Press Ctrl+C to stop the dashboard")
        print()
        
        # Launch streamlit
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", dashboard_file,
            "--server.port", "8501",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ])
        
        # Wait a moment for startup
        time.sleep(3)
        
        # Try to open browser
        try:
            webbrowser.open("http://localhost:8501")
            print("[OK] Dashboard opened in browser")
        except:
            print("[INFO] Could not auto-open browser. Manually go to: http://localhost:8501")
        
        # Keep running
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n[INFO] Shutting down dashboard...")
            process.terminate()
            
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to launch dashboard: {e}")
        return False

if __name__ == "__main__":
    success = launch_dashboard()