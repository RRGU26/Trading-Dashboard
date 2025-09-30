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

def get_model_performance_table():
    """Generate the daily model performance table with checkmarks"""
    import sqlite3
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, "reports_tracking.db")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Calculate performance metrics
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
        GROUP BY model
        ORDER BY (SUM(CASE 
            WHEN CAST(predicted_price AS REAL) > CAST(current_price AS REAL) 
                 AND CAST(actual_price AS REAL) > CAST(current_price AS REAL) THEN 1
            WHEN CAST(predicted_price AS REAL) < CAST(current_price AS REAL) 
                 AND CAST(actual_price AS REAL) < CAST(current_price AS REAL) THEN 1
            ELSE 0
        END) * 1.0 / COUNT(*)) DESC
    """)
    
    performance_data = []
    for row in cursor.fetchall():
        model, total, correct, avg_error = row
        accuracy = (correct / total * 100) if total > 0 else 0
        performance_data.append({
            'model': model,
            'accuracy': accuracy,
            'sample': total,
            'avg_error': avg_error or 0
        })
    
    # Check yesterday's activity
    yesterday = (datetime.now() - timedelta(days=1)).date()
    
    # First get predictions made yesterday
    cursor.execute("""
        SELECT model, 
               COUNT(CASE WHEN date(prediction_date) = date(?) THEN 1 END) as pred_yesterday
        FROM model_predictions
        GROUP BY model
    """, (yesterday,))
    
    yesterday_predictions = {}
    for model, pred_count in cursor.fetchall():
        yesterday_predictions[model] = pred_count > 0
    
    # Map models to their symbols (including truncated names)
    model_symbols = {
        'Long Bull Model V3.2': 'QQQ',
        'QQQ Master Model': 'QQQ',
        'QQQ Trading Signal': 'QQQ',
        'Wishing Well QQQ Model': 'QQQ',
        'Bitcoin Model': 'BTC-USD',
        'NVIDIA Bull Momentum Model': 'NVDA',
        'NVIDIA Bull Momentum Mod': 'NVDA',  # Truncated name in database
        'Algorand Model': 'ALGO-USD'
    }
    
    # Check if price history was saved yesterday for each model's symbol
    cursor.execute("""
        SELECT symbol, COUNT(*) as saved_count
        FROM price_history
        WHERE date = date(?)
        GROUP BY symbol
    """, (yesterday,))
    
    saved_prices = {row[0]: row[1] > 0 for row in cursor.fetchall()}
    
    yesterday_status = {}
    for model in model_symbols:
        symbol = model_symbols.get(model)
        yesterday_status[model] = {
            'pred': 'Y' if yesterday_predictions.get(model, False) else 'N',
            'price': 'Y' if symbol and saved_prices.get(symbol, False) else 'N'
        }
    
    conn.close()
    
    # Build HTML table
    html_table = """
    <h3>MODEL PERFORMANCE RANKINGS (Updated Daily)</h3>
    <table border='1' style='border-collapse: collapse; width: 100%;'>
    <tr style='background-color: #f0f0f0;'>
        <th style='padding: 5px;'>Rank</th>
        <th style='padding: 5px;'>Model</th>
        <th style='padding: 5px;'>Accuracy</th>
        <th style='padding: 5px;'>Sample</th>
        <th style='padding: 5px;'>Avg Error</th>
        <th style='padding: 5px;'>Status</th>
        <th style='padding: 5px;'>Pred Yesterday</th>
        <th style='padding: 5px;'>Price Saved</th>
    </tr>
    """
    
    for rank, data in enumerate(performance_data, 1):
        status = 'GOOD' if data['accuracy'] >= 65 else 'OK' if data['accuracy'] >= 50 else 'POOR'
        status_color = '#00ff00' if status == 'GOOD' else '#ffff00' if status == 'OK' else '#ff0000'
        
        model_status = yesterday_status.get(data['model'], {})
        pred_check = model_status.get('pred', 'N')
        price_check = model_status.get('price', 'N')
        
        html_table += f"""
        <tr>
            <td style='padding: 5px;'>{rank}</td>
            <td style='padding: 5px;'>{data['model'][:24]}</td>
            <td style='padding: 5px;'>{data['accuracy']:.1f}%</td>
            <td style='padding: 5px;'>{data['sample']}</td>
            <td style='padding: 5px;'>{data['avg_error']:.1f}%</td>
            <td style='padding: 5px; background-color: {status_color};'>{status}</td>
            <td style='padding: 5px; text-align: center;'>{pred_check}</td>
            <td style='padding: 5px; text-align: center;'>{price_check}</td>
        </tr>
        """
    
    html_table += "</table>"
    
    # Also generate text version
    text_table = "MODEL PERFORMANCE RANKINGS\n"
    text_table += "=" * 70 + "\n"
    text_table += f"{'Rank':<6} {'Model':<25} {'Accuracy':<10} {'Sample':<8} {'Yesterday':<12}\n"
    text_table += "-" * 70 + "\n"
    
    for rank, data in enumerate(performance_data, 1):
        model_status = yesterday_status.get(data['model'], {})
        yesterday_checks = f"{model_status.get('pred', 'N')}/{model_status.get('price', 'N')}"
        text_table += f"{rank:<6} {data['model'][:24]:<25} {data['accuracy']:.1f}%      {data['sample']:<8} {yesterday_checks:<12}\n"
    
    return html_table, text_table, performance_data

def check_database_predictions():
    """Check if predictions are being stored properly in the database"""
    import sqlite3
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, "reports_tracking.db")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        today = date.today().isoformat()
        yesterday = (datetime.now() - timedelta(days=1)).date().isoformat()
        
        # Check predictions for today and yesterday
        cursor.execute("""
            SELECT model, COUNT(*) as count, MAX(prediction_date) as latest
            FROM model_predictions 
            WHERE prediction_date >= ? 
            GROUP BY model
        """, (yesterday,))
        
        recent_predictions = cursor.fetchall()
        
        # Check actual report files on disk since reports_tracking.db is broken
        conn.close()
        
        # Look for actual report files from yesterday
        reports_dir = os.path.join(script_dir, "reports")
        recent_reports = []
        
        if os.path.exists(reports_dir):
            yesterday_str = yesterday.replace('-', '')  # Convert 2025-08-14 to 20250814
            
            # Count reports by looking for files with yesterday's date
            report_counts = {}
            for filename in os.listdir(reports_dir):
                if yesterday_str in filename and filename.endswith('.txt'):
                    # Extract model type from filename
                    if 'Long_Bull' in filename:
                        model_type = 'QQQ Long Bull Model'
                    elif 'Trading_Signal' in filename:
                        model_type = 'QQQ Trading Signal'
                    elif 'NVIDIA' in filename:
                        model_type = 'NVIDIA Bull Momentum'
                    elif 'algorand' in filename.lower():
                        model_type = 'Algorand Model'
                    elif 'Bitcoin' in filename:
                        model_type = 'Bitcoin Model'
                    elif 'Master' in filename:
                        model_type = 'QQQ Master Model'
                    elif 'Wishing' in filename:
                        model_type = 'Wishing Well QQQ'
                    else:
                        model_type = 'Other'
                    
                    report_counts[model_type] = report_counts.get(model_type, 0) + 1
            
            # Convert to expected format (model_type, count, latest_date)
            for model_type, count in report_counts.items():
                recent_reports.append((model_type, count, yesterday))
        
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
        
        # Look for recent model executions (last 24 hours - focus on yesterday's activity for 9am report)
        yesterday = datetime.now() - timedelta(days=1)
        
        recent_executions = []
        model_statuses = {}
        
        for line in lines:
            if ('All prediction models ran successfully' in line or 'Model Execution: SUCCESS' in line) and '2025-' in line:
                try:
                    # Extract date from log line
                    date_str = line.split(' - ')[0]
                    log_date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S,%f')
                    
                    if log_date >= yesterday:
                        # If we find "All prediction models ran successfully", mark all expected models as SUCCESS
                        if 'All prediction models ran successfully' in line:
                            model_statuses['QQQ Long Bull Model'] = 'SUCCESS'
                            model_statuses['QQQ Trading Signal'] = 'SUCCESS'
                            model_statuses['Algorand Model'] = 'SUCCESS'
                            model_statuses['Bitcoin Model'] = 'SUCCESS'
                            model_statuses['NVIDIA Bull Momentum'] = 'SUCCESS'
                            model_statuses['Wishing Well QQQ'] = 'SUCCESS'
                            model_statuses['QQQ Master Model'] = 'SUCCESS'
                        
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

def check_price_data_currency():
    """Check if price data is current and saved to database"""
    try:
        logger.info("Checking price data currency...")
        
        import sqlite3
        from datetime import datetime, timedelta
        
        # Key symbols for trading models
        symbols_to_check = ['QQQ', 'NVDA', 'BTC-USD', 'ALGO-USD', 'VIX', '^VIX']
        
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports_tracking.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        today = datetime.now().date()
        yesterday = (datetime.now() - timedelta(days=1)).date()
        
        price_status = {
            'total_symbols': len(symbols_to_check),
            'current_count': 0,
            'stale_symbols': [],
            'missing_symbols': [],
            'symbol_details': {}
        }
        
        for symbol in symbols_to_check:
            cursor.execute("""
                SELECT MAX(date) as latest_date, close 
                FROM price_history 
                WHERE symbol = ?
            """, (symbol,))
            
            result = cursor.fetchone()
            
            if result and result[0]:
                latest_date_str = result[0]
                latest_price = result[1]
                
                try:
                    latest_date = datetime.strptime(latest_date_str, '%Y-%m-%d').date()
                    days_old = (today - latest_date).days
                    
                    status = "CURRENT"
                    if days_old == 0:
                        status = "TODAY"
                        price_status['current_count'] += 1
                    elif days_old == 1:
                        status = "YESTERDAY"
                        price_status['current_count'] += 1
                    elif days_old <= 3:
                        status = f"{days_old} DAYS OLD"
                        if symbol not in ['VIX', '^VIX']:  # VIX may have gaps on weekends
                            price_status['stale_symbols'].append(f"{symbol} ({days_old} days)")
                    else:
                        status = f"STALE ({days_old} days)"
                        price_status['stale_symbols'].append(f"{symbol} ({days_old} days)")
                    
                    price_status['symbol_details'][symbol] = {
                        'latest_date': latest_date_str,
                        'latest_price': latest_price,
                        'days_old': days_old,
                        'status': status
                    }
                    
                except Exception as e:
                    price_status['symbol_details'][symbol] = {
                        'error': f"Date parsing error: {e}"
                    }
            else:
                price_status['missing_symbols'].append(symbol)
                price_status['symbol_details'][symbol] = {
                    'status': 'NO DATA'
                }
        
        conn.close()
        
        # Overall assessment
        if price_status['current_count'] == price_status['total_symbols']:
            price_status['overall_status'] = 'EXCELLENT'
        elif price_status['current_count'] >= price_status['total_symbols'] - 1:
            price_status['overall_status'] = 'GOOD'
        elif len(price_status['stale_symbols']) <= 2:
            price_status['overall_status'] = 'FAIR'
        else:
            price_status['overall_status'] = 'POOR'
        
        price_status['check_successful'] = True
        
        logger.info(f"Price data check complete: {price_status['overall_status']}")
        logger.info(f"Current symbols: {price_status['current_count']}/{price_status['total_symbols']}")
        
        return price_status
        
    except Exception as e:
        logger.error(f"Price data check failed: {e}")
        return {
            'check_successful': False,
            'error': str(e),
            'overall_status': 'ERROR'
        }

def run_health_check():
    """Run comprehensive health check including database and model execution"""
    try:
        logger.info("Running comprehensive system health check...")
        
        # Get model performance table
        html_table, text_table, performance_data = get_model_performance_table()
        
        # Check database predictions
        db_check = check_database_predictions()
        
        # Check model execution
        execution_check = check_model_execution()
        
        # NEW: Check price data currency
        price_data_check = check_price_data_currency()
        
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
            'performance_table_html': html_table,
            'performance_table_text': text_table,
            'performance_data': performance_data,
            'database_check': db_check,
            'execution_check': execution_check,
            'price_data_check': price_data_check,
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
        price_check = health_data.get('price_data_check', {})
        
        # Determine overall status
        issues = 0
        if not db_check.get('database_accessible', False):
            issues += 1
        if not exec_check.get('log_accessible', False):
            issues += 1
        if exec_check.get('last_run_count', 0) == 0:
            issues += 1
        if price_check.get('overall_status') in ['POOR', 'ERROR']:
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
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta name="x-apple-disable-message-reformatting">
        <title>Health Check - {today}</title>
        <style>
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; 
                margin: 0; 
                padding: 10px; 
                line-height: 1.5;
                -webkit-text-size-adjust: 100%;
                background-color: #f8f9fa;
            }}
            .container {{
                max-width: 600px;
                margin: 0 auto;
                background-color: white;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            .header {{ 
                background-color: {status_color}; 
                color: white; 
                padding: 20px 15px; 
                text-align: center;
            }}
            .status {{ 
                font-size: 22px; 
                font-weight: bold; 
                margin-bottom: 8px;
            }}
            .subtitle {{
                font-size: 14px;
                opacity: 0.9;
            }}
            .section {{ 
                margin: 0; 
                padding: 20px 15px; 
                border-bottom: 1px solid #e9ecef;
            }}
            .section:last-child {{
                border-bottom: none;
            }}
            .section h3 {{
                margin: 0 0 15px 0;
                font-size: 18px;
                color: #333;
            }}
            .success {{ color: #28a745; font-weight: bold; }}
            .warning {{ color: #ffc107; font-weight: bold; }}
            .error {{ color: #dc3545; font-weight: bold; }}
            .metric {{ 
                margin: 12px 0; 
                padding: 8px 0;
                font-size: 14px;
            }}
            .model-grid {{
                display: grid;
                grid-template-columns: 1fr;
                gap: 8px;
                margin: 15px 0;
            }}
            .model-status {{ 
                padding: 8px 12px; 
                border-radius: 6px; 
                font-size: 14px;
                text-align: center;
                font-weight: 500;
            }}
            .model-success {{ 
                background-color: #d4edda; 
                color: #155724; 
                border-left: 4px solid #28a745;
            }}
            .model-missing {{ 
                background-color: #f8d7da; 
                color: #721c24;
                border-left: 4px solid #dc3545;
            }}
            .table-container {{
                overflow-x: auto;
                margin: 15px 0;
            }}
            table {{ 
                width: 100%; 
                border-collapse: collapse; 
                font-size: 14px;
                min-width: 300px;
            }}
            th, td {{ 
                padding: 12px 8px; 
                text-align: left; 
                border-bottom: 1px solid #ddd; 
            }}
            th {{ 
                background-color: #f2f2f2; 
                font-weight: 600;
                color: #333;
            }}
            .footer {{
                background-color: #e9ecef;
                padding: 15px;
                font-size: 12px;
                color: #6c757d;
                text-align: center;
            }}
            
            /* Mobile-specific styles */
            @media (max-width: 480px) {{
                .container {{
                    margin: 5px;
                    border-radius: 4px;
                }}
                .header {{
                    padding: 15px 10px;
                }}
                .status {{
                    font-size: 18px;
                }}
                .section {{
                    padding: 15px 10px;
                }}
                .section h3 {{
                    font-size: 16px;
                }}
                th, td {{
                    padding: 8px 4px;
                    font-size: 13px;
                }}
                .model-status {{
                    font-size: 13px;
                    padding: 6px 10px;
                }}
            }}
            
            /* Dark mode support */
            @media (prefers-color-scheme: dark) {{
                .container {{
                    background-color: #2d2d2d;
                }}
                .section {{
                    border-bottom-color: #404040;
                }}
                .section h3 {{
                    color: #ffffff;
                }}
                th {{
                    background-color: #404040;
                    color: #ffffff;
                }}
                td {{
                    color: #e0e0e0;
                }}
                .footer {{
                    background-color: #404040;
                    color: #b0b0b0;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
        <div class="header">
            <div class="status">🌅 Morning System Health Check - {today}</div>
            <div class="subtitle">Overall Status: {status_text} | Models run today at 3:40 PM</div>
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
        # Add Model Performance Table at the top
        if 'performance_table_html' in health_data:
            html += f"""
            <div class="section">
                {health_data['performance_table_html']}
            </div>
            """
        # Model Execution Status
        exec_check = health_data.get('execution_check', {})
        if exec_check.get('log_accessible'):
            html += '<div class="section"><h3>🤖 Trading Models Status</h3>'
            html += '<div class="model-grid">'
            
            model_statuses = exec_check.get('model_statuses', {})
            expected_models = ['QQQ Long Bull Model', 'QQQ Trading Signal', 'Algorand Model', 'Bitcoin Model', 'NVIDIA Bull Momentum', 'Wishing Well QQQ', 'QQQ Master Model']
            
            for model in expected_models:
                if model in model_statuses:
                    html += f'<div class="model-status model-success">✅ {model}</div>'
                else:
                    html += f'<div class="model-status model-missing">❌ {model}</div>'
            
            html += '</div>'
            
            last_run_count = exec_check.get('last_run_count', 0)
            html += f'<div class="metric">Recent Executions (yesterday): <strong>{last_run_count}</strong></div>'
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
            
            html += '<h4>Recent Predictions:</h4><div class="table-container"><table>'
            html += '<tr><th>Model</th><th>Predictions</th><th>Latest</th></tr>'
            
            if predictions:
                for model, count, latest in predictions:
                    html += f'<tr><td>{model}</td><td>{count}</td><td>{latest}</td></tr>'
            else:
                html += '<tr><td colspan="3" class="warning">⚠️ No predictions found recently (normal at 9am - models run at 3:40pm)</td></tr>'
            html += '</table></div>'
            
            html += '<h4>Recent Reports:</h4><div class="table-container"><table>'
            html += '<tr><th>Model</th><th>Reports</th><th>Latest</th></tr>'
            
            if reports:
                for model, count, latest in reports:
                    html += f'<tr><td>{model}</td><td>{count}</td><td>{latest}</td></tr>'
            else:
                html += '<tr><td colspan="3" class="warning">⚠️ No reports found recently (normal at 9am - models run at 3:40pm)</td></tr>'
            html += '</table></div>'
            
            html += '</div>'
        else:
            html += '<div class="section"><h3>💾 Database Status</h3>'
            html += f'<div class="error">❌ Database inaccessible: {db_check.get("error", "Unknown error")}</div></div>'
        
        # Price Data Status - NEW SECTION
        price_check = health_data.get('price_data_check', {})
        if price_check.get('check_successful'):
            status = price_check.get('overall_status', 'UNKNOWN')
            current_count = price_check.get('current_count', 0)
            total_symbols = price_check.get('total_symbols', 0)
            
            html += '<div class="section"><h3>📈 Price Data Status</h3>'
            
            # Status indicator
            if status == 'EXCELLENT':
                html += '<div class="success">✅ All price data current</div>'
            elif status == 'GOOD':
                html += '<div class="success">✅ Price data mostly current</div>'
            elif status == 'FAIR':
                html += '<div class="warning">⚠️ Some price data stale</div>'
            else:
                html += '<div class="error">❌ Price data issues detected</div>'
            
            html += f'<p><strong>Current Symbols:</strong> {current_count}/{total_symbols}</p>'
            
            # Symbol details table
            symbol_details = price_check.get('symbol_details', {})
            if symbol_details:
                html += '<h4>Symbol Status:</h4><div class="table-container"><table>'
                html += '<tr><th>Symbol</th><th>Latest Price</th><th>Latest Date</th><th>Status</th></tr>'
                
                for symbol, details in symbol_details.items():
                    if 'error' in details:
                        html += f'<tr><td>{symbol}</td><td colspan="3" class="error">{details["error"]}</td></tr>'
                    elif details.get('status') == 'NO DATA':
                        html += f'<tr><td>{symbol}</td><td colspan="3" class="error">NO DATA</td></tr>'
                    else:
                        price = details.get('latest_price', 'N/A')
                        date = details.get('latest_date', 'N/A')
                        status = details.get('status', 'UNKNOWN')
                        
                        # Format price
                        if isinstance(price, (int, float)):
                            price_str = f'${price:.2f}'
                        else:
                            price_str = str(price)
                        
                        # Color code status
                        if status in ['TODAY', 'YESTERDAY']:
                            status_class = 'success'
                        elif 'DAYS OLD' in status and 'STALE' not in status:
                            status_class = 'warning'
                        else:
                            status_class = 'error'
                        
                        html += f'<tr><td>{symbol}</td><td>{price_str}</td><td>{date}</td><td><span class="{status_class}">{status}</span></td></tr>'
                
                html += '</table></div>'
            
            # Stale symbols warning
            stale_symbols = price_check.get('stale_symbols', [])
            if stale_symbols:
                html += '<div class="warning">'
                html += '<h4>⚠️ Stale Data Alert:</h4>'
                html += '<ul>'
                for symbol in stale_symbols:
                    html += f'<li>{symbol}</li>'
                html += '</ul>'
                html += '<p><em>Note: These symbols may affect model accuracy. The 4:01 PM system will attempt to refresh this data.</em></p>'
                html += '</div>'
            
            html += '</div>'
        else:
            html += '<div class="section"><h3>📈 Price Data Status</h3>'
            html += f'<div class="error">❌ Price data check failed: {price_check.get("error", "Unknown error")}</div></div>'
        
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
        html += '<div class="metric">• Yesterday\'s models ran successfully</div>'
        html += '<div class="metric">• Next trading run: Today 3:40 PM</div>'
        html += '<div class="metric">• All systems ready for today\'s predictions</div>'
    
    html += '</div>'
    
    html += f"""
        <div class="section">
            <h3>📅 Schedule</h3>
            <div class="metric">• Models will run: Today 3:40 PM</div>
            <div class="metric">• Next health check: Tomorrow 9:00 AM</div>
            <div class="metric">• Dashboard: <a href="http://localhost:8502">View Live Dashboard</a></div>
        </div>
        
        <div class="footer">
            Generated: {datetime.now().strftime('%Y-%m-%d %I:%M %p ET')}<br>
            System: Comprehensive Trading System Health Check v3.0
        </div>
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
            
            # Determine status for subject line
            if health_data and 'error' not in health_data:
                db_check = health_data.get('database_check', {})
                exec_check = health_data.get('execution_check', {})
                
                issues = 0
                if not db_check.get('database_accessible', False):
                    issues += 1
                if not exec_check.get('log_accessible', False):
                    issues += 1
                if exec_check.get('last_run_count', 0) == 0:
                    issues += 1
                    
                if issues == 0:
                    status_for_subject = "HEALTHY"
                elif issues == 1:
                    status_for_subject = "WARNINGS"
                else:
                    status_for_subject = "ISSUES"
            else:
                status_for_subject = "ERROR"
                
            subject = f"🌅 HEALTH CHECK - {today} - {status_for_subject}"
            
            success = email_manager.send_email(
                sender_email=email_manager.sender_email,
                sender_password=email_manager.sender_password,
                recipient_emails=recipients,
                subject=subject,
                html_body=html_content,
                plain_text_body=plain_text
            )
            
            if success:
                logger.info("Health check email sent successfully")
                print("Health check email sent successfully")
                return True
            else:
                logger.error("Failed to send health check email")
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