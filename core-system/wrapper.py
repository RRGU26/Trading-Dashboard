import subprocess
import os
import logging
import datetime
import sys
import time

# Unicode handling for Windows console compatibility
if sys.platform.startswith('win'):
    try:
        # Set environment encoding
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        
        # Reconfigure stdout/stderr to handle Unicode properly
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
            
    except (AttributeError, OSError):
        # Fallback for older Python versions or if reconfigure fails
        import codecs
        try:
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')
        except (AttributeError, OSError):
            pass

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Set up logging to a file in the same directory as the script
log_filename = os.path.join(script_dir, "wrapper_log.txt")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("wrapper")

# Use the full path to the Python executable
python_path = r"C:\Users\rrose\AppData\Local\Programs\Python\Python311\python.exe"
if not os.path.exists(python_path):
    logger.warning(f"Specified Python path not found: {python_path}")
    if sys.executable:
        python_path = sys.executable
        logger.info(f"Using system Python path: {python_path}")
    else:
        python_path = "python"
        logger.warning("Using default 'python' command from PATH")

# Define paths for all models and scripts
longhorn_model_path = os.path.join(script_dir, "QQQ Long Horn Bull Model.py")
trading_signal_path = os.path.join(script_dir, "QQQ Trading Signal.py")
algorand_model_path = os.path.join(script_dir, "Algorand Price Prediction Model.py")
bitcoin_model_path = os.path.join(script_dir, "new bitcoin model updated.py")
wishing_well_path = r"C:\Users\rrose\OneDrive\Desktop\wishing well.py"
nvidia_model_path = os.path.join(script_dir, "davemodel.py")
qqq_master_model_path = os.path.join(script_dir, "unified_qqq_master_model_optimized.py")
send_report_path = os.path.join(script_dir, "send_report.py")
dashboard_data_path = os.path.join(script_dir, "dashboard.data.py")
database_check_path = os.path.join(script_dir, "database check.py")

# API key for market data
ALPHA_VANTAGE_API_KEY = "HMHALLINAHS2FF4Z"

def setup_environment():
    """Set up environment variables consistently"""
    os.environ["AV_API_KEY"] = ALPHA_VANTAGE_API_KEY
    os.environ["TWELVEDATA_API_KEY"] = ALPHA_VANTAGE_API_KEY
    
    # Email configuration (matches send_report.py)
    EMAIL_RECIPIENTS = [
        "RRGU26@gmail.com",
        "timbarney62@gmail.com", 
        "rebeccalynnrosenthal@gmail.com",
        "samkest419@gmail.com",
        "georgelaffey@gmail.com", 
        "gmrosenthal1@gmail.com",
        "david.worldco@gmail.com"
    ]
    os.environ['EMAIL_RECIPIENTS'] = ','.join(EMAIL_RECIPIENTS)
    os.environ['RECIPIENT_EMAIL'] = EMAIL_RECIPIENTS[0]

def run_model(model_path, model_name):
    """Run a model and log the results - FIXED for Unicode"""
    try:
        # Set environment variables for API keys
        setup_environment()
        
        logger.info(f"Running {model_name}...")
        
        # FIXED: Add encoding and error handling for Unicode
        process = subprocess.run(
            [python_path, model_path],
            check=False,
            capture_output=True,
            text=True,
            encoding='utf-8',        # ADDED: Explicit UTF-8 encoding
            errors='replace'         # ADDED: Replace invalid characters
        )
        
        # Print output to console for visibility with safe encoding
        if process.stdout:
            try:
                print(process.stdout)
            except UnicodeEncodeError:
                safe_output = process.stdout.encode('ascii', errors='replace').decode('ascii')
                print(safe_output)
                
        if process.stderr:
            try:
                print(process.stderr)
            except UnicodeEncodeError:
                safe_error = process.stderr.encode('ascii', errors='replace').decode('ascii')
                print(safe_error)
        
        if process.returncode == 0:
            logger.info(f"Successfully ran {model_name}")
            return True
        else:
            logger.error(f"Error running {model_name}: Exit code {process.returncode}")
            return False
            
    except Exception as e:
        logger.error(f"Exception running {model_name}: {str(e)}")
        return False

def verify_dashboard_data_collection():
    """Verify that dashboard data was actually collected today"""
    try:
        import sqlite3
        from datetime import datetime
        
        db_path = os.path.join(script_dir, "models_dashboard.db")
        
        if not os.path.exists(db_path):
            logger.warning("[WARN] Database file not found - no data was collected")
            return False
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        today = datetime.now().date().strftime('%Y-%m-%d')
        
        # Check for predictions made today
        cursor.execute("""
        SELECT COUNT(*), model FROM model_predictions 
        WHERE prediction_date = ?
        GROUP BY model
        """, (today,))
        
        predictions_today = cursor.fetchall()
        
        if predictions_today:
            total_predictions = sum(count for count, model in predictions_today)
            logger.info(f"[OK] Verified {total_predictions} predictions collected for today:")
            for count, model in predictions_today:
                logger.info(f"   - {model}: {count} predictions")
            
            # Also check for current prices
            cursor.execute("""
            SELECT symbol, close FROM price_history 
            WHERE date = ?
            """, (today,))
            
            prices_today = cursor.fetchall()
            if prices_today:
                logger.info(f"[OK] Current prices also updated for {len(prices_today)} symbols")
                for symbol, price in prices_today:
                    logger.info(f"   - {symbol}: ${price:.4f}")
            
            conn.close()
            return True
        else:
            logger.warning("[WARN] No predictions found in database for today - data collection may have failed")
            conn.close()
            return False
            
    except Exception as e:
        logger.error(f"Error verifying dashboard data: {e}")
        return False

def run_dashboard_data_collection():
    """Enhanced dashboard data collection with better error handling"""
    try:
        logger.info("="*50)
        logger.info("Starting dashboard data collection...")
        
        if not os.path.exists(dashboard_data_path):
            logger.error(f"Dashboard data script not found: {dashboard_data_path}")
            return False
        
        # Set environment variables
        setup_environment()
        
        # Run dashboard data collection with detailed logging
        process = subprocess.run(
            [python_path, dashboard_data_path],
            check=False,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=300  # 5 minute timeout
        )
        
        # Log all output
        if process.stdout:
            logger.info("Dashboard data collection output:")
            for line in process.stdout.split('\n'):
                if line.strip():
                    logger.info(f"  {line}")
        
        if process.stderr:
            logger.warning("Dashboard data collection errors/warnings:")
            for line in process.stderr.split('\n'):
                if line.strip():
                    logger.warning(f"  {line}")
        
        if process.returncode == 0:
            logger.info("[OK] Dashboard data collection completed successfully")
            return verify_dashboard_data_collection()
        else:
            logger.error(f"[FAIL] Dashboard data collection failed with exit code: {process.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("[FAIL] Dashboard data collection timed out after 5 minutes")
        return False
    except Exception as e:
        logger.error(f"[FAIL] Exception during dashboard data collection: {e}")
        return False

def run_daily_trading_analysis():
    """Run comprehensive daily trading analysis and send professional email"""
    try:
        logger.info("="*60)
        logger.info("[ANALYSIS] STARTING DAILY TRADING ANALYSIS")
        logger.info("="*60)
        
        # Check if daily_trading_analysis.py exists
        analysis_script_path = os.path.join(script_dir, "daily_trading_analysis.py")
        if not os.path.exists(analysis_script_path):
            logger.error(f"[FAIL] Daily trading analysis script not found: {analysis_script_path}")
            return False
        
        setup_environment()
        
        start_time = time.time()
        process = subprocess.run(
            [python_path, analysis_script_path],
            check=False,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=300  # 5 minute timeout
        )
        
        duration = time.time() - start_time
        
        # Log all output
        if process.stdout:
            logger.info("[ANALYSIS] Daily trading analysis output:")
            for line in process.stdout.split('\n'):
                if line.strip():
                    logger.info(f"  {line}")
        
        if process.stderr:
            logger.warning("[ANALYSIS] Daily trading analysis warnings/errors:")
            for line in process.stderr.split('\n'):
                if line.strip():
                    logger.warning(f"  {line}")
        
        if process.returncode == 0:
            logger.info(f"[OK] Daily trading analysis completed successfully in {duration:.1f} seconds")
            
            # Look for success indicators in output
            if "analysis email sent successfully" in process.stdout:
                logger.info("[EMAIL] ✅ Trading analysis email confirmed sent")
            elif "Personal Trading Analysis" in process.stdout:
                logger.info("[EMAIL] ✅ Trading analysis generated")
            
            return True
        else:
            logger.error(f"[FAIL] Daily trading analysis failed with exit code: {process.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("[FAIL] Daily trading analysis timed out after 5 minutes")
        return False
    except Exception as e:
        logger.error(f"[FAIL] Exception during daily trading analysis: {e}")
        return False

def run_database_health_check():
    """Run comprehensive database health check"""
    try:
        logger.info("="*60)
        logger.info("[SEARCH] STARTING DATABASE HEALTH CHECK")
        logger.info("="*60)
        
        if not os.path.exists(database_check_path):
            logger.error(f"[FAIL] Database health check script not found: {database_check_path}")
            return False
        
        setup_environment()
        
        start_time = time.time()
        process = subprocess.run(
            [python_path, database_check_path],
            check=False,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=180  # 3 minute timeout
        )
        
        duration = time.time() - start_time
        
        # Log all output with proper formatting
        if process.stdout:
            logger.info("[CLIPBOARD] Database health check output:")
            for line in process.stdout.split('\n'):
                if line.strip():
                    if any(indicator in line for indicator in ['[OK]', '[WARN]', '[FAIL]']):
                        logger.info(f"{line}")
                    else:
                        logger.info(f"   {line}")
        
        if process.stderr:
            logger.warning("[WARN] Database health check warnings/errors:")
            for line in process.stderr.split('\n'):
                if line.strip():
                    logger.warning(f"   {line}")
        
        if process.returncode == 0:
            logger.info(f"[OK] Database health check completed successfully in {duration:.1f} seconds")
            
            # Try to read the health report for summary
            try:
                health_report_path = os.path.join(script_dir, "daily_health_report.json")
                if os.path.exists(health_report_path):
                    import json
                    with open(health_report_path, 'r') as f:
                        health_data = json.load(f)
                    
                    overall_status = health_data.get('overall_status', 'UNKNOWN')
                    checks_passed = health_data.get('checks_passed', 0)
                    checks_failed = health_data.get('checks_failed', 0)
                    warnings = health_data.get('warnings', 0)
                    critical_issues = health_data.get('critical_issues', 0)
                    
                    status_emoji = "[OK]" if overall_status == 'HEALTHY' else "[WARN]" if overall_status == 'WARNING' else "[FAIL]"
                    
                    logger.info("="*50)
                    logger.info(f"[CHART] DATABASE HEALTH SUMMARY:")
                    logger.info(f"{status_emoji} Overall Status: {overall_status}")
                    logger.info(f"   Checks Passed: {checks_passed}")
                    logger.info(f"   Checks Failed: {checks_failed}")
                    logger.info(f"   Warnings: {warnings}")
                    logger.info(f"   Critical Issues: {critical_issues}")
                    
                    recommendations = health_data.get('recommendations', [])
                    if recommendations:
                        logger.info("[IDEA] RECOMMENDATIONS:")
                        for rec in recommendations:
                            priority = rec.get('priority', 'MEDIUM')
                            category = rec.get('category', 'General')
                            issue = rec.get('issue', 'Unknown issue')
                            action = rec.get('action', 'No action specified')
                            
                            priority_emoji = "[ALERT]" if priority == 'CRITICAL' else "[WARN]" if priority == 'HIGH' else "[THOUGHT]"
                            logger.info(f"{priority_emoji} [{priority}] {category}: {issue}")
                            logger.info(f"   Action: {action}")
                    
                    logger.info("="*50)
                    return overall_status in ['HEALTHY', 'WARNING']
                else:
                    logger.warning("[WARN] Health report file not found, but health check completed")
                    return True
                    
            except Exception as e:
                logger.warning(f"[WARN] Could not read health report details: {e}")
                return True
                
        else:
            logger.error(f"[FAIL] Database health check failed with exit code: {process.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("[FAIL] Database health check timed out after 3 minutes")
        return False
    except Exception as e:
        logger.error(f"[FAIL] Exception during database health check: {e}")
        return False

def check_report_files():
    """Check if report files were generated today"""
    try:
        logger.info("Checking for generated report files...")
        
        import glob
        from datetime import datetime
        
        # Synchronized search locations (matches send_report.py)
        search_locations = [
            r"C:\Users\rrose\OneDrive\Desktop",  # Primary
            r"C:\Users\rrose\Desktop",           # Secondary
            script_dir                           # Script directory
        ]
        
        report_patterns = [
            "*Long_Bull_Report*.txt",
            "*Trading_Signal*.txt", 
            "*Bitcoin_Prediction_Report*.txt",
            "*algorand*report*.txt",
            "*WishingWealthQQQ_signal*.txt",
            "*NVIDIA*Report*.txt"
        ]
        
        found_reports = []
        today = datetime.now().date()
        
        for location in search_locations:
            if os.path.exists(location):
                for pattern in report_patterns:
                    files = glob.glob(os.path.join(location, pattern))
                    for file_path in files:
                        file_time = datetime.fromtimestamp(os.path.getmtime(file_path)).date()
                        if file_time == today:
                            found_reports.append(file_path)
        
        if found_reports:
            logger.info(f"[OK] Found {len(found_reports)} report files created today:")
            for report in found_reports:
                logger.info(f"   - {os.path.basename(report)}")
        else:
            logger.warning("[WARN] No report files found that were created today")
        
        return len(found_reports) > 0
        
    except Exception as e:
        logger.error(f"Error checking report files: {e}")
        return False

def main():
    logger.info(f"[START] Starting enhanced wrapper process at {datetime.datetime.now()}")
    
    # Set up environment variables early
    setup_environment()
    
    # Verify script files exist
    missing_scripts = []
    script_info = [
        (longhorn_model_path, "QQQ Long Horn Bull Model.py"),
        (trading_signal_path, "QQQ Trading Signal.py"),
        (algorand_model_path, "Algorand Price Prediction Model.py"),
        (bitcoin_model_path, "new bitcoin model updated.py"),
        (wishing_well_path, "wishing well.py"),
        (nvidia_model_path, "davemodel.py"),
        (qqq_master_model_path, "unified_qqq_master_model_optimized.py")
    ]
    
    for script_path, script_name in script_info:
        if not os.path.exists(script_path):
            missing_scripts.append(f"{script_name} - {script_path}")
            logger.error(f"Script not found: {script_path}")
    
    if missing_scripts:
        logger.error("Some scripts are missing:")
        for script in missing_scripts:
            logger.error(f"  - {script}")
        logger.error("Continuing with available scripts...")
    
    # Run all models sequentially
    models_successful = True
    models_run = []
    
    # Run Algorand Price Prediction Model
    if os.path.exists(algorand_model_path):
        success = run_model(algorand_model_path, "Algorand Price Prediction Model.py")
        models_run.append(("Algorand Model", success))
        if not success:
            models_successful = False
        time.sleep(2)
    else:
        logger.warning("Skipping Algorand model - script not found")
    
    # Run Bitcoin Prediction Model
    if os.path.exists(bitcoin_model_path):
        success = run_model(bitcoin_model_path, "new bitcoin model updated.py")
        models_run.append(("Bitcoin Model", success))
        if not success:
            models_successful = False
        time.sleep(2)
    else:
        logger.warning("Skipping Bitcoin model - script not found")
    
    # Run QQQ Long Horn Bull Model
    if os.path.exists(longhorn_model_path):
        success = run_model(longhorn_model_path, "QQQ Long Horn Bull Model.py")
        models_run.append(("QQQ Long Horn Bull Model", success))
        if not success:
            models_successful = False
        time.sleep(2)
    else:
        logger.warning("Skipping QQQ Long Horn Bull model - script not found")
    
    # Run QQQ Trading Signal
    if os.path.exists(trading_signal_path):
        success = run_model(trading_signal_path, "QQQ Trading Signal.py")
        models_run.append(("QQQ Trading Signal", success))
        if not success:
            models_successful = False
        time.sleep(2)
    else:
        logger.warning("Skipping QQQ Trading Signal - script not found")
    
    # Run NVIDIA Bull Momentum Model
    if os.path.exists(nvidia_model_path):
        success = run_model(nvidia_model_path, "NVIDIA Bull Momentum Model")
        models_run.append(("NVIDIA Bull Momentum Model", success))
        if not success:
            models_successful = False
        time.sleep(2)
    else:
        logger.warning("Skipping NVIDIA model - script not found")

    # Run Wishing Well model
    if os.path.exists(wishing_well_path):
        success = run_model(wishing_well_path, "Wishing Well QQQ Model")
        models_run.append(("Wishing Well QQQ Model", success))
        if not success:
            models_successful = False
        time.sleep(2)
    else:
        logger.warning("Skipping Wishing Well model - script not found")
    
    # Run QQQ Master Model
    if os.path.exists(qqq_master_model_path):
        success = run_model(qqq_master_model_path, "QQQ Master Model")
        models_run.append(("QQQ Master Model", success))
        if not success:
            models_successful = False
        else:
            # If QQQ Master Model succeeded, trigger its email system
            logger.info("[EMAIL] QQQ Master Model succeeded - triggering email system...")
            try:
                qqq_email_script = os.path.join(script_dir, "daily_qqq_automation.py")
                if os.path.exists(qqq_email_script):
                    email_process = subprocess.run(
                        [python_path, qqq_email_script, "--email-only"],
                        check=False,
                        capture_output=True,
                        text=True,
                        encoding='utf-8',
                        errors='replace',
                        timeout=120
                    )
                    
                    if email_process.returncode == 0:
                        logger.info("[EMAIL] QQQ email sent successfully")
                        if "Email sent successfully" in email_process.stdout:
                            logger.info("[EMAIL] ✅ QQQ email confirmation found")
                    else:
                        logger.warning(f"[EMAIL] QQQ email failed: {email_process.stderr}")
                else:
                    logger.warning("[EMAIL] QQQ email script not found")
            except Exception as e:
                logger.warning(f"[EMAIL] QQQ email system error: {e}")
        time.sleep(2)
    else:
        logger.warning("Skipping QQQ Master Model - script not found")
    
    # Log model summary
    logger.info("="*50)
    logger.info("[CHART_UP] MODEL EXECUTION SUMMARY:")
    for model_name, success in models_run:
        status = "[OK] SUCCESS" if success else "[FAIL] FAILED"
        logger.info(f"  {model_name}: {status}")
    logger.info("="*50)
    
    # Only run send_report.py if all models completed successfully
    if models_successful:
        try:
            logger.info("[EMAIL] All models completed successfully. Running send_report.py...")
            
            send_report_success = False
            if os.path.exists(send_report_path):
                # IMPROVED: Enhanced send_report.py call with proper Unicode handling
                process = subprocess.run(
                    [python_path, send_report_path], 
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',     # ADDED: Consistent encoding
                    errors='replace'      # ADDED: Error handling
                )
                
                # Print output for visibility
                if process.stdout:
                    print(process.stdout)
                if process.stderr:
                    print(process.stderr)
                
                if process.returncode == 0:
                    logger.info("[OK] Successfully ran send_report.py")
                    send_report_success = True
                    
                    # Look for success indicators in output
                    if "Email sent successfully" in process.stdout:
                        logger.info("[EMAIL] ✅ Email confirmation found in output")
                    elif "Reports processed successfully" in process.stdout:
                        logger.info("[EMAIL] ✅ Report processing confirmed")
                else:
                    logger.error(f"[FAIL] send_report.py failed with exit code: {process.returncode}")
                    if process.stderr:
                        logger.error(f"Error output: {process.stderr}")
            else:
                logger.warning(f"Send report script not found: {send_report_path}")
            
            # Dashboard data collection with verification
            logger.info("="*50)
            logger.info("[CHART] Starting enhanced dashboard data collection...")
            
            reports_found = check_report_files()
            if not reports_found:
                logger.warning("[WARN] Few or no report files found")
            
            dashboard_success = run_dashboard_data_collection()
            
            # Run database health check
            logger.info("="*50)
            logger.info("[SEARCH] Running database health check...")
            health_check_success = run_database_health_check()
            
            # Final workflow summary
            logger.info("="*60)
            logger.info("[FINISH] COMPLETE WORKFLOW SUMMARY")
            logger.info("="*60)
            
            # Run daily trading analysis (the comprehensive email)
            logger.info("="*50)
            logger.info("[ANALYSIS] Running daily trading analysis...")
            trading_analysis_success = run_daily_trading_analysis()
            
            workflow_steps = [
                ("Model Execution", models_successful, "All prediction models ran successfully"),
                ("Report Generation", send_report_success, "Email reports sent successfully"),
                ("Data Collection", dashboard_success, "Dashboard data collected and verified"),
                ("Health Check", health_check_success, "Database health validated"),
                ("Trading Analysis", trading_analysis_success, "Comprehensive trading analysis email sent")
            ]
            
            all_successful = True
            for step_name, success, description in workflow_steps:
                status_emoji = "[OK]" if success else "[FAIL]"
                logger.info(f"{status_emoji} {step_name}: {'SUCCESS' if success else 'FAILED'}")
                logger.info(f"   {description}")
                if not success:
                    all_successful = False
            
            if all_successful:
                logger.info("[CELEBRATION] COMPLETE SUCCESS! All workflow steps completed successfully!")
                logger.info("   - Models generated predictions")
                logger.info("   - Reports were sent via email")
                logger.info("   - Data was collected and stored in database")
                logger.info("   - Database health check passed")
            else:
                logger.error("[WARN] PARTIAL SUCCESS - Some workflow steps failed")
                logger.error("   Please review the logs above for specific issues")
                
                if not send_report_success:
                    logger.error("   [EMAIL] Report sending failed - check email configuration")
                if not dashboard_success:
                    logger.error("   [CHART] Data collection failed - check report file generation")
                if not health_check_success:
                    logger.error("   [SEARCH] Health check failed - check database integrity")
            
            logger.info("="*60)
                
        except Exception as e:
            logger.error(f"Error in post-model processing: {e}")
    else:
        logger.error("[FAIL] One or more models failed. Not running downstream processes")
        
        # Still run health check to diagnose issues
        logger.info("="*50)
        logger.info("[SEARCH] Running health check to diagnose issues...")
        run_database_health_check()
    
    logger.info("="*50)
    logger.info(f"[FINISH] Wrapper process completed at {datetime.datetime.now()}")
    logger.info("="*50)

if __name__ == "__main__":
    main()