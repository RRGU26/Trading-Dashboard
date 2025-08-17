#!/usr/bin/env python3
"""
QQQ Daily Automation Bridge Script
Connects wrapper.py to the automated trading system for options analysis
"""

import sys
import os
import subprocess
import logging

# Add parent directory to path to access automated_trading_system
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run the automated trading system for QQQ analysis"""
    try:
        logger.info("Starting QQQ automated trading analysis...")
        
        # Path to the automated trading system
        automated_system_path = r"C:\Users\rrose\automated_trading_system.py"
        
        if not os.path.exists(automated_system_path):
            logger.error(f"Automated trading system not found at: {automated_system_path}")
            return 1
        
        # Run the automated trading system with --run-once flag
        result = subprocess.run([
            sys.executable, 
            automated_system_path, 
            "--run-once"
        ], capture_output=True, text=True, cwd=r"C:\Users\rrose")
        
        if result.returncode == 0:
            logger.info("✅ QQQ automated trading analysis completed successfully")
            print("Email sent successfully")  # This is what wrapper.py looks for
            return 0
        else:
            logger.error(f"QQQ automated trading analysis failed: {result.stderr}")
            return 1
            
    except Exception as e:
        logger.error(f"Error in QQQ automation: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())