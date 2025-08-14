#!/usr/bin/env python3
"""
Daily Health Check Email System
Sends 9 AM health check summary email
"""

import sys
import os
import subprocess
import logging
import json
from datetime import datetime, date, timedelta

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import email system from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from trading_reports_email import EmailManager
except ImportError:
    from send_report import send_email_with_reports  # Fallback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_database_predictions():
    """Check if predictions are being stored properly in the database"""
    import sqlite3
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, "models_dashboard.db")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        today = date.today().isoformat()
        yesterday = (datetime.now() - timedelta(days=2)).date().isoformat()
        
        # Check predictions for today and yesterday
        cursor.execute("""
            SELECT model, COUNT(*) as count, MAX(prediction_date) as latest
            FROM model_predictions 
            WHERE prediction_date >= ? 
            GROUP BY model
        """, (yesterday,))
        
        recent_predictions = cursor.fetchall()
        
        # Check reports table in reports_tracking.db
        conn.close()
        
        # Connect to reports_tracking.db for reports data
        reports_db_path = os.path.join(script_dir, "reports_tracking.db")
        reports_conn = sqlite3.connect(reports_db_path)
        reports_cursor = reports_conn.cursor()
        
        reports_cursor.execute("""
            SELECT model_type, COUNT(*) as count, MAX(generated_date) as latest
            FROM report_files 
            WHERE generated_date >= ? 
            GROUP BY model_type
        """, (yesterday,))
        
        recent_reports = reports_cursor.fetchall()
        reports_conn.close()
        
        # Expected models
        expected_models = [
            'QQQ Long Bull Model',
            'QQQ Trading Signal', 
            'Algorand Model',
            'Bitcoin Model',
            'NVIDIA Bull Momentum',
            'Wishing Well QQQ',
            'QQQ Master Model'
        ]
        
        # conn.close()  # Already closed above
        
        return {
            'predictions': recent_predictions,
            'reports': recent_reports,
            'expected_models': expected_models,
            'database_accessible': True
        }
        
    except Exception as e:
        logger.error(f"Database check failed: {e}")
        return {
            'database_accessible': False,
            'error': str(e)
        }

def check_model_execution():
    """Check if models have executed recently by looking at wrapper logs"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wrapper_log = os.path.join(script_dir, "wrapper_log.txt")
    
    try:
        if not os.path.exists(wrapper_log):
            return {'log_accessible': False, 'error': 'Wrapper log not found'}
        
        # Read last 100 lines of wrapper log
        with open(wrapper_log, 'r', encoding='utf-8') as f:
            lines = f.readlines()[-100:]
        
        # Look for recent model executions (last 48 hours to account for weekend/holiday gaps)
        yesterday = datetime.now() - timedelta(days=2)
        
        recent_executions = []
        model_statuses = {}
        
        for line in lines:
            if 'Successfully ran' in line and '2025-' in line:
                try:
                    # Extract date from log line
                    date_str = line.split(' - ')[0]
                    log_date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S,%f')
                    
                    if log_date >= yesterday:
                        # More flexible model matching - skip send_report.py
                        if 'send_report.py' in line:
                            continue  # Skip the report sending, not a trading model
                        elif 'Algorand' in line:
                            model_statuses['Algorand Model'] = 'SUCCESS'
                        elif 'bitcoin' in line.lower():
                            model_statuses['Bitcoin Model'] = 'SUCCESS'
                        elif 'QQQ Long Horn' in line or 'QQQ Long Bull' in line:
                            model_statuses['QQQ Long Bull Model'] = 'SUCCESS'
                        elif 'QQQ Trading Signal' in line:
                            model_statuses['QQQ Trading Signal'] = 'SUCCESS'
                        elif 'NVIDIA' in line:
                            model_statuses['NVIDIA Bull Momentum'] = 'SUCCESS'
                        elif 'Wishing Well' in line:
                            model_statuses['Wishing Well QQQ'] = 'SUCCESS'
                        elif 'QQQ Master' in line:
                            model_statuses['QQQ Master Model'] = 'SUCCESS'
                        else:
                            # Log unmatched successful runs for debugging
                            logger.info(f"Unmatched successful run: {line.strip()}")
                        
                        recent_executions.append({
                            'timestamp': log_date.isoformat(),
                            'message': line.strip()
                        })
                except Exception as e:
                    # Debug: log parsing errors
                    logger.warning(f"Error parsing log line: {line[:50]}... - {e}")
                    continue
        
        return {
            'log_accessible': True,
            'recent_executions': recent_executions,
            'model_statuses': model_statuses,
            'last_run_count': len(recent_executions)
        }
        
    except Exception as e:
        logger.error(f"Log check failed: {e}")
        return {
            'log_accessible': False,
            'error': str(e)
        }

def run_health_check():
    """Run comprehensive health check including database and model execution"""
    try:
        logger.info("Running comprehensive system health check...")
        
        # Check database predictions
        db_check = check_database_predictions()
        
        # Check model execution
        execution_check = check_model_execution()
        
        # Run the original database health check too
        script_dir = os.path.dirname(os.path.abspath(__file__))
        health_check_script = os.path.join(script_dir, "database check.py")
        
        system_health = {}
        if os.path.exists(health_check_script):
            result = subprocess.run([
                sys.executable, 
                health_check_script
            ], capture_output=True, text=True, cwd=script_dir)
            
            if result.returncode == 0:
                health_report_file = os.path.join(script_dir, "daily_health_report.json")
                if os.path.exists(health_report_file):
                    with open(health_report_file, 'r') as f:
                        system_health = json.load(f)
        
        # Combine all checks
        return {
            'database_check': db_check,
            'execution_check': execution_check,
            'system_health': system_health,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error running comprehensive health check: {e}")
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def create_health_email_html(health_data):
    """Create HTML email content from comprehensive health check results"""
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Analyze overall system health
    if not health_data or 'error' in health_data:
        status_color = "#dc3545"
        status_text = "SYSTEM ERROR"
        overall_health = "CRITICAL"
    else:
        db_check = health_data.get('database_check', {})
        exec_check = health_data.get('execution_check', {})
        
        # Determine overall status
        issues = 0
        if not db_check.get('database_accessible', False):
            issues += 1
        if not exec_check.get('log_accessible', False):
            issues += 1
        if exec_check.get('last_run_count', 0) == 0:
            issues += 1
            
        if issues == 0:
            status_color = "#28a745"
            status_text = "HEALTHY"
            overall_health = "GOOD"
        elif issues == 1:
            status_color = "#ffc107"
            status_text = "WARNINGS"
            overall_health = "CAUTION"
        else:
            status_color = "#dc3545"
            status_text = "ISSUES FOUND"
            overall_health = "CRITICAL"
    
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: {status_color}; color: white; padding: 15px; border-radius: 5px; }}
            .status {{ font-size: 24px; font-weight: bold; }}
            .section {{ margin: 15px 0; padding: 15px; border-left: 4px solid #007bff; background-color: #f8f9fa; border-radius: 5px; }}
            .success {{ color: #28a745; font-weight: bold; }}
            .warning {{ color: #ffc107; font-weight: bold; }}
            .error {{ color: #dc3545; font-weight: bold; }}
            .metric {{ margin: 8px 0; padding: 5px 0; }}
            .model-status {{ display: inline-block; margin: 3px; padding: 4px 8px; border-radius: 3px; font-size: 12px; }}
            .model-success {{ background-color: #d4edda; color: #155724; }}
            .model-missing {{ background-color: #f8d7da; color: #721c24; }}
            table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <div class="status">🔍 Daily System Health Check - {today}</div>
            <div>Overall Status: {status_text}</div>
        </div>
    """
    
    if not health_data or 'error' in health_data:
        html += f"""
        <div class="section">
            <h3>❌ System Error</h3>
            <div class="error">Health check failed to complete: {health_data.get('error', 'Unknown error')}</div>
        </div>
        """
    else:
        # Model Execution Status
        exec_check = health_data.get('execution_check', {})
        if exec_check.get('log_accessible'):
            html += '<div class="section"><h3>🤖 Trading Models Status</h3>'
            
            model_statuses = exec_check.get('model_statuses', {})
            expected_models = ['QQQ Long Bull Model', 'QQQ Trading Signal', 'Algorand Model', 'Bitcoin Model', 'NVIDIA Bull Momentum', 'Wishing Well QQQ', 'QQQ Master Model']
            
            for model in expected_models:
                if model in model_statuses:
                    html += f'<span class="model-status model-success">✅ {model}</span>'
                else:
                    html += f'<span class="model-status model-missing">❌ {model}</span>'
            
            last_run_count = exec_check.get('last_run_count', 0)
            html += f'<div class="metric">Recent Executions (24h): <strong>{last_run_count}</strong></div>'
            html += '</div>'
        else:
            html += '<div class="section"><h3>🤖 Trading Models Status</h3>'
            html += '<div class="error">❌ Cannot access wrapper logs</div></div>'
        
        # Database Status
        db_check = health_data.get('database_check', {})
        if db_check.get('database_accessible'):
            html += '<div class="section"><h3>💾 Database Status</h3>'
            html += '<div class="success">✅ Database accessible</div>'
            
            # Predictions stored
            predictions = db_check.get('predictions', [])
            reports = db_check.get('reports', [])
            
            html += '<h4>Recent Predictions:</h4><table>'
            html += '<tr><th>Model</th><th>Predictions</th><th>Latest</th></tr>'
            
            if predictions:
                for model, count, latest in predictions:
                    html += f'<tr><td>{model}</td><td>{count}</td><td>{latest}</td></tr>'
            else:
                html += '<tr><td colspan="3" class="warning">⚠️ No predictions found in last 24h</td></tr>'
            html += '</table>'
            
            html += '<h4>Recent Reports:</h4><table>'
            html += '<tr><th>Model</th><th>Reports</th><th>Latest</th></tr>'
            
            if reports:
                for model, count, latest in reports:
                    html += f'<tr><td>{model}</td><td>{count}</td><td>{latest}</td></tr>'
            else:
                html += '<tr><td colspan="3" class="warning">⚠️ No reports found in last 24h</td></tr>'
            html += '</table>'
            
            html += '</div>'
        else:
            html += '<div class="section"><h3>💾 Database Status</h3>'
            html += f'<div class="error">❌ Database inaccessible: {db_check.get("error", "Unknown error")}</div></div>'
        
        # System Health (from original check)
        system_health = health_data.get('system_health', {})
        if system_health:
            summary = system_health.get('summary', {})
            if summary:
                html += '<div class="section"><h3>⚙️ System Health Summary</h3>'
                html += f'<div class="metric">Database Integrity: {"✅" if summary.get("checks_passed", 0) > 0 else "❌"}</div>'
                html += f'<div class="metric">Price Data: {"✅" if summary.get("checks_passed", 0) > 0 else "❌"}</div>'
                html += f'<div class="metric">Critical Issues: <span class="error">{summary.get("critical_issues", 0)}</span></div>'
                html += '</div>'
    
    # Action Items
    html += '<div class="section"><h3>🎯 Action Items</h3>'
    
    if overall_health == "CRITICAL":
        html += '<div class="error">🚨 IMMEDIATE ATTENTION REQUIRED</div>'
        html += '<div class="metric">• Check system logs immediately</div>'
        html += '<div class="metric">• Verify database connectivity</div>'
        html += '<div class="metric">• Restart trading system if needed</div>'
    elif overall_health == "CAUTION":
        html += '<div class="warning">⚠️ MONITORING REQUIRED</div>'
        html += '<div class="metric">• Review missing models</div>'
        html += '<div class="metric">• Check for data gaps</div>'
    else:
        html += '<div class="success">✅ SYSTEM OPERATING NORMALLY</div>'
        html += '<div class="metric">• Next trading run: Today 3:40 PM</div>'
        html += '<div class="metric">• All systems functioning</div>'
    
    html += '</div>'
    
    html += f"""
        <div class="section">
            <h3>📅 Schedule</h3>
            <div class="metric">• Next trading models run: Today 3:40 PM</div>
            <div class="metric">• Next health check: Tomorrow 9:00 AM</div>
            <div class="metric">• Dashboard: <a href="https://trading-models-dashboard.streamlit.app">View Live Dashboard</a></div>
        </div>
        
        <div style="margin-top: 20px; padding: 10px; background-color: #e9ecef; border-radius: 5px; font-size: 12px; color: #6c757d;">
            Generated: {datetime.now().strftime('%Y-%m-%d %I:%M %p ET')}<br>
            System: Comprehensive Trading System Health Check v3.0
        </div>
    </body>
    </html>
    """
    
    return html

def send_health_check_email():
    """Send the daily health check email"""
    try:
        logger.info("Starting daily health check email...")
        
        # Run health check
        health_data = run_health_check()
        
        # Create email content
        html_content = create_health_email_html(health_data)
        
        # Create plain text version
        today = datetime.now().strftime('%Y-%m-%d')
        if health_data:
            status = health_data.get('overall_status', 'UNKNOWN')
            summary = health_data.get('summary', {})
            plain_text = f"""Daily System Health Check - {today}
            
Overall Status: {status}
Total Checks: {summary.get('total_checks', 'N/A')}
Checks Passed: {summary.get('checks_passed', 'N/A')}
Checks Failed: {summary.get('checks_failed', 'N/A')}

Next trading models run: Today 3:40 PM
Dashboard: https://trading-models-dashboard.streamlit.app

Generated: {datetime.now().strftime('%Y-%m-%d %I:%M %p ET')}
"""
        else:
            plain_text = f"""Daily System Health Check - {today}
            
Status: ERROR - Health check failed to run
Next trading models run: Today 3:40 PM

Generated: {datetime.now().strftime('%Y-%m-%d %I:%M %p ET')}
"""
        
        # Send email using existing email system
        try:
            email_manager = EmailManager()
            email_manager.initialize()
            
            # TESTING ONLY - single recipient
            recipients = [
                "RRGU26@gmail.com"
            ]
            
            subject = f"📊 Daily System Health Check - {today}"
            
            success = email_manager.send_email(
                sender_email=email_manager.sender_email,
                sender_password=email_manager.sender_password,
                recipient_emails=recipients,
                subject=subject,
                html_body=html_content,
                plain_text_body=plain_text
            )
            
            if success:
                logger.info("✅ Health check email sent successfully")
                print("✅ Health check email sent successfully")
                return True
            else:
                logger.error("❌ Failed to send health check email")
                return False
                
        except Exception as email_error:
            logger.error(f"Email system error: {email_error}")
            return False
            
    except Exception as e:
        logger.error(f"Error in health check email system: {e}")
        return False

def main():
    """Main function"""
    try:
        success = send_health_check_email()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())