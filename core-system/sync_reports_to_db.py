#!/usr/bin/env python3
"""
Sync Reports to Database
Scans report files and ensures they're tracked in reports_tracking.db
Should be run after models complete to ensure health check can find reports
"""

import os
import sys
from datetime import datetime

def sync_reports_to_database():
    """Scan reports and sync to database"""
    try:
        # Add current directory to path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(script_dir)
        
        from trading_reports_database import get_database
        
        # Scan the reports directory
        reports_path = os.path.join(script_dir, 'reports')
        
        if not os.path.exists(reports_path):
            print(f"Reports directory not found: {reports_path}")
            return False
        
        print(f"[INFO] Scanning reports in: {reports_path}")
        
        # Initialize database and scan
        db = get_database('reports_tracking.db')
        total, new = db.scan_report_files([reports_path])
        
        print(f"[OK] Scanned {total} files, added {new} new files to database")
        
        if new > 0:
            print(f"[INFO] Added {new} new report files to tracking database")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to sync reports to database: {e}")
        return False

if __name__ == "__main__":
    success = sync_reports_to_database()
    sys.exit(0 if success else 1)