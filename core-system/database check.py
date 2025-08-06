# database_health_check.py - Comprehensive Daily Database Health Check
import sqlite3
import os
import logging
import datetime
import traceback
import json
from collections import defaultdict
from datetime import datetime, timedelta, date
import pandas as pd

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "models_dashboard.db")
HEALTH_CHECK_LOG = os.path.join(SCRIPT_DIR, "database_health_check.log")
HEALTH_REPORT_FILE = os.path.join(SCRIPT_DIR, "daily_health_report.json")

# Set up logging with Windows-compatible encoding
import sys

# Configure UTF-8 encoding for Windows console if possible
if sys.platform.startswith('win'):
    try:
        # Try to set UTF-8 encoding for Windows console
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    except:
        pass

# Set up logging with fallback for Windows encoding issues
class SafeStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            # Replace emojis with text equivalents for Windows console
            msg = msg.replace('🔍', '[SCAN]')
            msg = msg.replace('✅', '[OK]')
            msg = msg.replace('⚠️', '[WARN]')
            msg = msg.replace('❌', '[FAIL]')
            msg = msg.replace('📊', '[CHART]')
            msg = msg.replace('💰', '[MONEY]')
            msg = msg.replace('📋', '[CLIPBOARD]')
            msg = msg.replace('🚨', '[ALERT]')
            msg = msg.replace('💡', '[IDEA]')
            msg = msg.replace('🎉', '[PARTY]')
            msg = msg.replace('🏁', '[FLAG]')
            msg = msg.replace('🔗', '[LINK]')
            
            self.stream.write(msg + self.terminator)
            self.flush()
        except UnicodeEncodeError:
            # Fallback: encode to ASCII with replacement
            safe_msg = self.format(record).encode('ascii', 'replace').decode('ascii')
            self.stream.write(safe_msg + self.terminator)
            self.flush()
        except Exception:
            super().emit(record)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(HEALTH_CHECK_LOG, encoding='utf-8'),
        SafeStreamHandler()
    ]
)
logger = logging.getLogger("db_health_check")

class DatabaseHealthChecker:
    """Comprehensive database health and completeness checker"""
    
    def __init__(self):
        self.today = datetime.now().date()
        self.yesterday = self.today - timedelta(days=1)
        self.week_ago = self.today - timedelta(days=7)
        self.month_ago = self.today - timedelta(days=30)
        
        # Expected symbols for monitoring
        self.PRICE_SYMBOLS = ["QQQ", "BTC-USD", "ALGO-USD", "VIX", "^VIX"]
        self.CRYPTO_SYMBOLS = ["BTC-USD", "ALGO-USD"]
        self.ONCHAIN_SYMBOLS = ["btc"]  # Only BTC onchain data needed
        self.MODEL_NAMES = [
            "Long Bull Model V3.2", 
            "QQQ Trading Signal", 
            "Algorand Model", 
            "Bitcoin Model", 
            "Wishing Well QQQ Model"
        ]
        
        # Health check results
        self.health_report = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'UNKNOWN',
            'checks_passed': 0,
            'checks_failed': 0,
            'warnings': 0,
            'critical_issues': 0,
            'details': {}
        }
    
    def connect_to_db(self):
        """Connect to the database with error handling"""
        try:
            if not os.path.exists(DB_PATH):
                logger.error(f"Database not found at {DB_PATH}")
                return None
            
            conn = sqlite3.connect(DB_PATH)
            return conn
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            return None
    
    def check_price_data_completeness(self):
        """Check if daily price data is being updated for all symbols"""
        logger.info("=== CHECKING PRICE DATA COMPLETENESS ===")
        
        try:
            conn = self.connect_to_db()
            if not conn:
                self.health_report['details']['price_data'] = {
                    'status': 'FAILED',
                    'error': 'Database connection failed'
                }
                self.health_report['checks_failed'] += 1
                return False
            
            cursor = conn.cursor()
            price_results = {}
            
            for symbol in self.PRICE_SYMBOLS:
                # Check today's price
                cursor.execute("""
                SELECT date, close FROM price_history 
                WHERE symbol = ? AND date = ?
                """, (symbol, self.today.strftime('%Y-%m-%d')))
                
                today_result = cursor.fetchone()
                
                # Check yesterday's price (fallback)
                cursor.execute("""
                SELECT date, close FROM price_history 
                WHERE symbol = ? AND date = ?
                """, (symbol, self.yesterday.strftime('%Y-%m-%d')))
                
                yesterday_result = cursor.fetchone()
                
                # Check recent data (last 7 days)
                cursor.execute("""
                SELECT COUNT(*) FROM price_history 
                WHERE symbol = ? AND date >= ?
                """, (symbol, self.week_ago.strftime('%Y-%m-%d')))
                
                recent_count = cursor.fetchone()[0]
                
                # Check data quality (no null prices)
                cursor.execute("""
                SELECT COUNT(*) FROM price_history 
                WHERE symbol = ? AND date >= ? AND (close IS NULL OR close = 0)
                """, (symbol, self.week_ago.strftime('%Y-%m-%d')))
                
                null_count = cursor.fetchone()[0]
                
                price_results[symbol] = {
                    'today_available': today_result is not None,
                    'today_price': today_result[1] if today_result else None,
                    'yesterday_available': yesterday_result is not None,
                    'yesterday_price': yesterday_result[1] if yesterday_result else None,
                    'recent_days_count': recent_count,
                    'null_price_count': null_count,
                    'status': 'OK' if (today_result or yesterday_result) and null_count == 0 else 'WARNING'
                }
                
                if today_result:
                    logger.info(f"✅ {symbol}: Today's price ${today_result[1]:.4f}")
                elif yesterday_result:
                    logger.warning(f"⚠️  {symbol}: No today's price, yesterday's: ${yesterday_result[1]:.4f}")
                else:
                    logger.error(f"❌ {symbol}: No recent price data")
                    price_results[symbol]['status'] = 'FAILED'
            
            conn.close()
            
            # Analyze results
            failed_symbols = [s for s, r in price_results.items() if r['status'] == 'FAILED']
            warning_symbols = [s for s, r in price_results.items() if r['status'] == 'WARNING']
            
            overall_status = 'OK'
            if failed_symbols:
                overall_status = 'FAILED'
                self.health_report['checks_failed'] += 1
                self.health_report['critical_issues'] += len(failed_symbols)
            elif warning_symbols:
                overall_status = 'WARNING'
                self.health_report['warnings'] += len(warning_symbols)
                self.health_report['checks_passed'] += 1
            else:
                self.health_report['checks_passed'] += 1
            
            self.health_report['details']['price_data'] = {
                'status': overall_status,
                'symbols': price_results,
                'failed_symbols': failed_symbols,
                'warning_symbols': warning_symbols,
                'summary': f"{len(self.PRICE_SYMBOLS) - len(failed_symbols)}/{len(self.PRICE_SYMBOLS)} symbols have recent data"
            }
            
            return overall_status != 'FAILED'
            
        except Exception as e:
            logger.error(f"Error checking price data: {e}")
            traceback.print_exc()
            self.health_report['details']['price_data'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            self.health_report['checks_failed'] += 1
            return False
    
    def check_onchain_data_completeness(self):
        """Check if onchain metrics are being updated for crypto symbols"""
        logger.info("=== CHECKING ONCHAIN DATA COMPLETENESS ===")
        
        try:
            conn = self.connect_to_db()
            if not conn:
                self.health_report['details']['onchain_data'] = {
                    'status': 'FAILED',
                    'error': 'Database connection failed'
                }
                self.health_report['checks_failed'] += 1
                return False
            
            cursor = conn.cursor()
            
            # Check if onchain_metrics table exists
            cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='onchain_metrics'
            """)
            
            if not cursor.fetchone():
                logger.info("[INFO] onchain_metrics table does not exist - skipping onchain data check")
                self.health_report['details']['onchain_data'] = {
                    'status': 'OK',
                    'message': 'Onchain data not required - skipping check'
                }
                self.health_report['checks_passed'] += 1
                conn.close()
                return True
            
            # Check if we have any onchain data at all
            cursor.execute("SELECT COUNT(*) FROM onchain_metrics")
            total_onchain_records = cursor.fetchone()[0]
            
            if total_onchain_records == 0:
                logger.info("[INFO] No onchain data found - treating as optional feature")
                self.health_report['details']['onchain_data'] = {
                    'status': 'OK',
                    'message': 'No onchain data found - treating as optional feature'
                }
                self.health_report['checks_passed'] += 1
                conn.close()
                return True
            
            # If we have onchain data, check its quality
            onchain_results = {}
            
            for asset in self.ONCHAIN_SYMBOLS:
                # Check recent onchain data (last 7 days)
                cursor.execute("""
                SELECT date, COUNT(DISTINCT metric) as metric_count
                FROM onchain_metrics 
                WHERE asset = ? AND date >= ?
                GROUP BY date
                ORDER BY date DESC
                """, (asset, self.week_ago.strftime('%Y-%m-%d')))
                
                recent_data = cursor.fetchall()
                
                # Check specific metrics availability
                cursor.execute("""
                SELECT DISTINCT metric FROM onchain_metrics 
                WHERE asset = ? AND date >= ?
                """, (asset, self.week_ago.strftime('%Y-%m-%d')))
                
                available_metrics = [row[0] for row in cursor.fetchall()]
                
                # Check data freshness
                cursor.execute("""
                SELECT MAX(date) FROM onchain_metrics 
                WHERE asset = ?
                """, (asset,))
                
                latest_date_result = cursor.fetchone()
                latest_date = latest_date_result[0] if latest_date_result[0] else None
                
                onchain_results[asset] = {
                    'recent_days_with_data': len(recent_data),
                    'latest_date': latest_date,
                    'available_metrics': available_metrics,
                    'metric_count': len(available_metrics),
                    'data_freshness_days': (self.today - datetime.strptime(latest_date, '%Y-%m-%d').date()).days if latest_date else None
                }
                
                if latest_date and (self.today - datetime.strptime(latest_date, '%Y-%m-%d').date()).days <= 3:
                    logger.info(f"[OK] {asset}: Recent onchain data available (latest: {latest_date})")
                    onchain_results[asset]['status'] = 'OK'
                elif latest_date:
                    logger.warning(f"[WARN] {asset}: Onchain data is stale (latest: {latest_date})")
                    onchain_results[asset]['status'] = 'WARNING'
                else:
                    logger.warning(f"[WARN] {asset}: No onchain data found")
                    onchain_results[asset]['status'] = 'WARNING'
            
            conn.close()
            
            # Analyze results - be more lenient since onchain data is optional
            warning_assets = [a for a, r in onchain_results.items() if r.get('status') == 'WARNING']
            ok_assets = [a for a, r in onchain_results.items() if r.get('status') == 'OK']
            
            # If we have ANY recent onchain data, consider it OK
            if ok_assets:
                status = 'OK'
                self.health_report['checks_passed'] += 1
                logger.info(f"[OK] Onchain data check passed - {len(ok_assets)} assets have recent data")
            elif onchain_results:
                status = 'WARNING'
                self.health_report['warnings'] += 1
                logger.warning(f"[WARN] Onchain data is stale for all assets")
            else:
                # No onchain data at all, but that's OK since it's optional
                status = 'OK'
                self.health_report['checks_passed'] += 1
                logger.info("[INFO] No onchain data configured - treating as optional")
            
            self.health_report['details']['onchain_data'] = {
                'status': status,
                'assets': onchain_results,
                'warning_assets': warning_assets,
                'ok_assets': ok_assets,
                'summary': f"Onchain data: {len(ok_assets)} assets current, {len(warning_assets)} stale" if onchain_results else "Onchain data not configured (optional)"
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking onchain data: {e}")
            traceback.print_exc()
            self.health_report['details']['onchain_data'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            self.health_report['checks_failed'] += 1
            return False
    
    def check_prediction_storage(self):
        """Check if model predictions are being stored daily"""
        logger.info("=== CHECKING PREDICTION STORAGE ===")
        
        try:
            conn = self.connect_to_db()
            if not conn:
                self.health_report['details']['prediction_storage'] = {
                    'status': 'FAILED',
                    'error': 'Database connection failed'
                }
                self.health_report['checks_failed'] += 1
                return False
            
            cursor = conn.cursor()
            prediction_results = {}
            
            # Check today's predictions
            cursor.execute("""
            SELECT model, COUNT(*) as prediction_count
            FROM model_predictions 
            WHERE prediction_date = ?
            GROUP BY model
            """, (self.today.strftime('%Y-%m-%d'),))
            
            today_predictions = dict(cursor.fetchall())
            
            # Check yesterday's predictions (fallback)
            cursor.execute("""
            SELECT model, COUNT(*) as prediction_count
            FROM model_predictions 
            WHERE prediction_date = ?
            GROUP BY model
            """, (self.yesterday.strftime('%Y-%m-%d'),))
            
            yesterday_predictions = dict(cursor.fetchall())
            
            # Check recent prediction frequency (last 7 days)
            cursor.execute("""
            SELECT model, COUNT(DISTINCT prediction_date) as active_days
            FROM model_predictions 
            WHERE prediction_date >= ?
            GROUP BY model
            """, (self.week_ago.strftime('%Y-%m-%d'),))
            
            recent_activity = dict(cursor.fetchall())
            
            # Check prediction data quality
            cursor.execute("""
            SELECT model, COUNT(*) as incomplete_predictions
            FROM model_predictions 
            WHERE prediction_date >= ? 
            AND (predicted_price IS NULL OR current_price IS NULL OR target_date IS NULL)
            GROUP BY model
            """, (self.week_ago.strftime('%Y-%m-%d'),))
            
            incomplete_predictions = dict(cursor.fetchall())
            
            for model in self.MODEL_NAMES:
                prediction_results[model] = {
                    'today_count': today_predictions.get(model, 0),
                    'yesterday_count': yesterday_predictions.get(model, 0),
                    'recent_active_days': recent_activity.get(model, 0),
                    'incomplete_count': incomplete_predictions.get(model, 0),
                    'status': 'OK'
                }
                
                # Determine status
                if today_predictions.get(model, 0) > 0:
                    logger.info(f"✅ {model}: {today_predictions[model]} predictions stored today")
                elif yesterday_predictions.get(model, 0) > 0:
                    logger.warning(f"⚠️  {model}: No predictions today, but {yesterday_predictions[model]} yesterday")
                    prediction_results[model]['status'] = 'WARNING'
                else:
                    logger.error(f"❌ {model}: No recent predictions found")
                    prediction_results[model]['status'] = 'FAILED'
                
                if incomplete_predictions.get(model, 0) > 0:
                    logger.warning(f"⚠️  {model}: {incomplete_predictions[model]} incomplete predictions in last week")
                    if prediction_results[model]['status'] == 'OK':
                        prediction_results[model]['status'] = 'WARNING'
            
            conn.close()
            
            # Analyze overall results
            failed_models = [m for m, r in prediction_results.items() if r['status'] == 'FAILED']
            warning_models = [m for m, r in prediction_results.items() if r['status'] == 'WARNING']
            
            overall_status = 'OK'
            if failed_models:
                overall_status = 'FAILED'
                self.health_report['checks_failed'] += 1
                self.health_report['critical_issues'] += len(failed_models)
            elif warning_models:
                overall_status = 'WARNING'
                self.health_report['warnings'] += len(warning_models)
                self.health_report['checks_passed'] += 1
            else:
                self.health_report['checks_passed'] += 1
            
            self.health_report['details']['prediction_storage'] = {
                'status': overall_status,
                'models': prediction_results,
                'failed_models': failed_models,
                'warning_models': warning_models,
                'summary': f"{len(self.MODEL_NAMES) - len(failed_models)}/{len(self.MODEL_NAMES)} models have recent predictions"
            }
            
            return overall_status != 'FAILED'
            
        except Exception as e:
            logger.error(f"Error checking prediction storage: {e}")
            traceback.print_exc()
            self.health_report['details']['prediction_storage'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            self.health_report['checks_failed'] += 1
            return False
    
    def check_actual_price_updates(self):
        """Check if actual prices are being updated for past predictions"""
        logger.info("=== CHECKING ACTUAL PRICE UPDATES ===")
        
        try:
            conn = self.connect_to_db()
            if not conn:
                self.health_report['details']['actual_price_updates'] = {
                    'status': 'FAILED',
                    'error': 'Database connection failed'
                }
                self.health_report['checks_failed'] += 1
                return False
            
            cursor = conn.cursor()
            
            # Check predictions that should have actual prices by now
            cursor.execute("""
            SELECT 
                model,
                COUNT(*) as total_due,
                COUNT(actual_price) as updated_count,
                COUNT(CASE WHEN actual_price IS NULL THEN 1 END) as missing_count
            FROM model_predictions 
            WHERE target_date <= ? AND target_date >= ?
            GROUP BY model
            """, (self.today.strftime('%Y-%m-%d'), self.month_ago.strftime('%Y-%m-%d')))
            
            actual_price_results = {}
            for row in cursor.fetchall():
                model, total_due, updated_count, missing_count = row
                actual_price_results[model] = {
                    'total_due': total_due,
                    'updated_count': updated_count,
                    'missing_count': missing_count,
                    'update_rate': (updated_count / total_due * 100) if total_due > 0 else 0
                }
            
            # Check recent update activity
            cursor.execute("""
            SELECT COUNT(*) FROM model_predictions 
            WHERE actual_price IS NOT NULL 
            AND target_date >= ? AND target_date <= ?
            """, (self.week_ago.strftime('%Y-%m-%d'), self.today.strftime('%Y-%m-%d')))
            
            recent_updates = cursor.fetchone()[0]
            
            # Check for stale predictions (target date passed but no actual price)
            cursor.execute("""
            SELECT model, symbol, target_date, COUNT(*) as stale_count
            FROM model_predictions 
            WHERE target_date < ? AND actual_price IS NULL
            GROUP BY model, symbol, target_date
            ORDER BY target_date DESC
            LIMIT 10
            """, (self.today.strftime('%Y-%m-%d'),))
            
            stale_predictions = cursor.fetchall()
            
            conn.close()
            
            # Analyze results
            total_missing = sum(r['missing_count'] for r in actual_price_results.values())
            total_due = sum(r['total_due'] for r in actual_price_results.values())
            overall_update_rate = (total_due - total_missing) / total_due * 100 if total_due > 0 else 0
            
            # Log results
            logger.info(f"Overall actual price update rate: {overall_update_rate:.1f}% ({total_due - total_missing}/{total_due})")
            logger.info(f"Recent updates in last week: {recent_updates}")
            
            for model, stats in actual_price_results.items():
                logger.info(f"{model}: {stats['update_rate']:.1f}% ({stats['updated_count']}/{stats['total_due']})")
            
            if stale_predictions:
                logger.warning(f"Found {len(stale_predictions)} groups of stale predictions:")
                for model, symbol, target_date, count in stale_predictions[:5]:  # Show first 5
                    logger.warning(f"  {model} ({symbol}): {count} predictions due {target_date}")
            
            # Determine status
            if overall_update_rate >= 90:
                status = 'OK'
                self.health_report['checks_passed'] += 1
            elif overall_update_rate >= 70:
                status = 'WARNING'
                self.health_report['warnings'] += 1
            else:
                status = 'FAILED'
                self.health_report['checks_failed'] += 1
                self.health_report['critical_issues'] += 1
            
            self.health_report['details']['actual_price_updates'] = {
                'status': status,
                'overall_update_rate': overall_update_rate,
                'total_due': total_due,
                'total_updated': total_due - total_missing,
                'total_missing': total_missing,
                'recent_updates': recent_updates,
                'models': actual_price_results,
                'stale_predictions_sample': stale_predictions[:5],
                'summary': f"{overall_update_rate:.1f}% of due predictions have actual prices"
            }
            
            return status != 'FAILED'
            
        except Exception as e:
            logger.error(f"Error checking actual price updates: {e}")
            traceback.print_exc()
            self.health_report['details']['actual_price_updates'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            self.health_report['checks_failed'] += 1
            return False
    
    def check_prediction_accuracy_assessment(self):
        """Check if prediction accuracy is being calculated and stored"""
        logger.info("=== CHECKING PREDICTION ACCURACY ASSESSMENT ===")
        
        try:
            conn = self.connect_to_db()
            if not conn:
                self.health_report['details']['accuracy_assessment'] = {
                    'status': 'FAILED',
                    'error': 'Database connection failed'
                }
                self.health_report['checks_failed'] += 1
                return False
            
            cursor = conn.cursor()
            
            # Check if accuracy columns exist
            cursor.execute("PRAGMA table_info(model_predictions)")
            columns = [col[1] for col in cursor.fetchall()]
            
            has_direction_correct = 'direction_correct' in columns
            has_error_pct = 'error_pct' in columns
            
            accuracy_results = {}
            
            # Check recent accuracy calculations
            if has_direction_correct:
                cursor.execute("""
                SELECT 
                    model,
                    COUNT(*) as total_evaluated,
                    COUNT(CASE WHEN direction_correct = 1 THEN 1 END) as correct_direction,
                    AVG(CASE WHEN direction_correct IS NOT NULL THEN direction_correct * 100.0 END) as direction_accuracy
                FROM model_predictions 
                WHERE actual_price IS NOT NULL 
                AND target_date >= ?
                GROUP BY model
                """, (self.month_ago.strftime('%Y-%m-%d'),))
                
                for row in cursor.fetchall():
                    model, total_evaluated, correct_direction, direction_accuracy = row
                    accuracy_results[model] = {
                        'total_evaluated': total_evaluated,
                        'correct_direction': correct_direction,
                        'direction_accuracy': direction_accuracy or 0,
                        'has_accuracy_data': True
                    }
            
            # Check for missing accuracy calculations
            cursor.execute("""
            SELECT model, COUNT(*) as missing_accuracy
            FROM model_predictions 
            WHERE actual_price IS NOT NULL 
            AND direction_correct IS NULL
            AND target_date >= ?
            GROUP BY model
            """, (self.month_ago.strftime('%Y-%m-%d'),))
            
            missing_accuracy = dict(cursor.fetchall())
            
            # Update accuracy results with missing data
            for model in self.MODEL_NAMES:
                if model not in accuracy_results:
                    accuracy_results[model] = {
                        'total_evaluated': 0,
                        'correct_direction': 0,
                        'direction_accuracy': 0,
                        'has_accuracy_data': False
                    }
                
                accuracy_results[model]['missing_accuracy_count'] = missing_accuracy.get(model, 0)
            
            conn.close()
            
            # Analyze results
            models_with_accuracy = [m for m, r in accuracy_results.items() if r['has_accuracy_data'] and r['total_evaluated'] > 0]
            models_missing_accuracy = [m for m, r in accuracy_results.items() if r.get('missing_accuracy_count', 0) > 0]
            
            # Log results
            for model, stats in accuracy_results.items():
                if stats['total_evaluated'] > 0:
                    logger.info(f"✅ {model}: {stats['direction_accuracy']:.1f}% accuracy ({stats['correct_direction']}/{stats['total_evaluated']})")
                else:
                    logger.warning(f"⚠️  {model}: No evaluated predictions found")
                
                if stats.get('missing_accuracy_count', 0) > 0:
                    logger.warning(f"⚠️  {model}: {stats.get('missing_accuracy_count', 0)} predictions missing accuracy calculation")
            
            # Determine status
            if not has_direction_correct:
                status = 'WARNING'
                message = 'direction_correct column missing - accuracy tracking not implemented'
                self.health_report['warnings'] += 1
            elif len(models_with_accuracy) >= len(self.MODEL_NAMES) * 0.8:  # 80% of models have accuracy data
                status = 'OK'
                self.health_report['checks_passed'] += 1
            elif len(models_with_accuracy) > 0:
                status = 'WARNING'
                self.health_report['warnings'] += 1
            else:
                status = 'FAILED'
                self.health_report['checks_failed'] += 1
                self.health_report['critical_issues'] += 1
            
            self.health_report['details']['accuracy_assessment'] = {
                'status': status,
                'has_direction_correct_column': has_direction_correct,
                'has_error_pct_column': has_error_pct,
                'models': accuracy_results,
                'models_with_accuracy': models_with_accuracy,
                'models_missing_accuracy': models_missing_accuracy,
                'summary': f"{len(models_with_accuracy)}/{len(self.MODEL_NAMES)} models have accuracy data"
            }
            
            return status != 'FAILED'
            
        except Exception as e:
            logger.error(f"Error checking accuracy assessment: {e}")
            traceback.print_exc()
            self.health_report['details']['accuracy_assessment'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            self.health_report['checks_failed'] += 1
            return False
    
    def check_database_integrity(self):
        """Check database integrity and schema"""
        logger.info("=== CHECKING DATABASE INTEGRITY ===")
        
        try:
            conn = self.connect_to_db()
            if not conn:
                self.health_report['details']['database_integrity'] = {
                    'status': 'FAILED',
                    'error': 'Database connection failed'
                }
                self.health_report['checks_failed'] += 1
                return False
            
            cursor = conn.cursor()
            integrity_results = {}
            
            # Check database integrity
            cursor.execute("PRAGMA integrity_check")
            integrity_check = cursor.fetchone()[0]
            integrity_results['integrity_check'] = integrity_check
            
            # Check table existence
            expected_tables = ['model_predictions', 'price_history', 'onchain_metrics']
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = [row[0] for row in cursor.fetchall()]
            
            integrity_results['expected_tables'] = expected_tables
            integrity_results['existing_tables'] = existing_tables
            integrity_results['missing_tables'] = [t for t in expected_tables if t not in existing_tables]
            
            # Check database size
            db_size = os.path.getsize(DB_PATH) / (1024 * 1024)  # MB
            integrity_results['database_size_mb'] = round(db_size, 2)
            
            # Check recent database activity
            cursor.execute("SELECT COUNT(*) FROM model_predictions WHERE prediction_date >= ?", (self.week_ago.strftime('%Y-%m-%d'),))
            recent_predictions = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM price_history WHERE date >= ?", (self.week_ago.strftime('%Y-%m-%d'),))
            recent_prices = cursor.fetchone()[0]
            
            integrity_results['recent_activity'] = {
                'predictions_last_week': recent_predictions,
                'price_updates_last_week': recent_prices
            }
            
            conn.close()
            
            # Determine status
            issues = []
            if integrity_check != 'ok':
                issues.append(f"Database integrity check failed: {integrity_check}")
            
            if integrity_results['missing_tables']:
                issues.append(f"Missing tables: {integrity_results['missing_tables']}")
            
            if db_size == 0:
                issues.append("Database is empty")
            
            if recent_predictions == 0 and recent_prices == 0:
                issues.append("No recent database activity")
            
            if issues:
                status = 'FAILED'
                self.health_report['checks_failed'] += 1
                self.health_report['critical_issues'] += len(issues)
                logger.error(f"❌ Database integrity issues: {'; '.join(issues)}")
            else:
                status = 'OK'
                self.health_report['checks_passed'] += 1
                logger.info(f"✅ Database integrity check passed ({db_size:.1f}MB)")
            
            integrity_results['status'] = status
            integrity_results['issues'] = issues
            
            self.health_report['details']['database_integrity'] = integrity_results
            
            return status != 'FAILED'
            
        except Exception as e:
            logger.error(f"Error checking database integrity: {e}")
            traceback.print_exc()
            self.health_report['details']['database_integrity'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            self.health_report['checks_failed'] += 1
            return False
    
    def generate_health_summary(self):
        """Generate a comprehensive health summary"""
        logger.info("=== GENERATING HEALTH SUMMARY ===")
        
        # Determine overall status
        if self.health_report['critical_issues'] > 0 or self.health_report['checks_failed'] > 0:
            self.health_report['overall_status'] = 'FAILED'
        elif self.health_report['warnings'] > 0:
            self.health_report['overall_status'] = 'WARNING'
        else:
            self.health_report['overall_status'] = 'HEALTHY'
        
        # Generate summary text
        total_checks = self.health_report['checks_passed'] + self.health_report['checks_failed']
        
        summary_lines = [
            f"📊 DATABASE HEALTH CHECK SUMMARY - {self.today}",
            f"Overall Status: {self.health_report['overall_status']}",
            f"Total Checks: {total_checks} (Passed: {self.health_report['checks_passed']}, Failed: {self.health_report['checks_failed']})",
            f"Warnings: {self.health_report['warnings']}, Critical Issues: {self.health_report['critical_issues']}"
        ]
        
        # Add status for each check
        for check_name, details in self.health_report['details'].items():
            status_emoji = "✅" if details['status'] == 'OK' else "⚠️" if details['status'] == 'WARNING' else "❌"
            summary_lines.append(f"{status_emoji} {check_name.replace('_', ' ').title()}: {details['status']}")
            
            if 'summary' in details:
                summary_lines.append(f"   {details['summary']}")
        
        summary_text = '\n'.join(summary_lines)
        
        self.health_report['summary_text'] = summary_text
        
        # Log summary
        logger.info("=" * 60)
        logger.info(summary_text)
        logger.info("=" * 60)
        
        return summary_text
    
    def save_health_report(self):
        """Save the health report to a JSON file"""
        try:
            with open(HEALTH_REPORT_FILE, 'w') as f:
                json.dump(self.health_report, f, indent=2, default=str)
            logger.info(f"Health report saved to {HEALTH_REPORT_FILE}")
            return True
        except Exception as e:
            logger.error(f"Error saving health report: {e}")
            return False
    
    def get_recommendations(self):
        """Generate actionable recommendations based on health check results"""
        recommendations = []
        
        for check_name, details in self.health_report['details'].items():
            if details['status'] == 'FAILED':
                if check_name == 'price_data':
                    if 'failed_symbols' in details and details['failed_symbols']:
                        recommendations.append({
                            'priority': 'HIGH',
                            'category': 'Price Data',
                            'issue': f"Missing price data for: {', '.join(details['failed_symbols'])}",
                            'action': "Check data_fetcher.py connections and API keys. Verify symbols are being fetched correctly."
                        })
                
                elif check_name == 'prediction_storage':
                    if 'failed_models' in details and details['failed_models']:
                        recommendations.append({
                            'priority': 'HIGH',
                            'category': 'Predictions',
                            'issue': f"No predictions stored for: {', '.join(details['failed_models'])}",
                            'action': "Check model report files are being generated and parsed correctly. Verify send_report.py execution."
                        })
                
                elif check_name == 'actual_price_updates':
                    recommendations.append({
                        'priority': 'HIGH',
                        'category': 'Price Updates',
                        'issue': f"Low actual price update rate: {details.get('overall_update_rate', 0):.1f}%",
                        'action': "Check update_pending_actual_prices() function. Verify data_fetcher can access historical prices."
                    })
                
                elif check_name == 'database_integrity':
                    recommendations.append({
                        'priority': 'CRITICAL',
                        'category': 'Database',
                        'issue': "Database integrity issues detected",
                        'action': "Backup database and run PRAGMA integrity_check. Consider database repair or restoration."
                    })
            
            elif details['status'] == 'WARNING':
                if check_name == 'onchain_data':
                    recommendations.append({
                        'priority': 'MEDIUM',
                        'category': 'Onchain Data',
                        'issue': "Onchain data is stale or missing",
                        'action': "Check CoinMetrics API connection and onchain data fetching functions."
                    })
                
                elif check_name == 'accuracy_assessment':
                    recommendations.append({
                        'priority': 'MEDIUM',
                        'category': 'Accuracy Tracking',
                        'issue': "Some predictions missing accuracy calculations",
                        'action': "Run update_direction_accuracy() function to calculate missing accuracy metrics."
                    })
        
        self.health_report['recommendations'] = recommendations
        return recommendations
    
    def run_full_health_check(self):
        """Run all health checks and generate comprehensive report"""
        logger.info(f"🔍 Starting comprehensive database health check for {self.today}")
        
        # Run all checks
        checks = [
            ('Database Integrity', self.check_database_integrity),
            ('Price Data Completeness', self.check_price_data_completeness),
            ('Onchain Data Completeness', self.check_onchain_data_completeness),
            ('Prediction Storage', self.check_prediction_storage),
            ('Actual Price Updates', self.check_actual_price_updates),
            ('Accuracy Assessment', self.check_prediction_accuracy_assessment)
        ]
        
        for check_name, check_function in checks:
            try:
                logger.info(f"Running {check_name} check...")
                check_function()
            except Exception as e:
                logger.error(f"Error in {check_name} check: {e}")
                self.health_report['checks_failed'] += 1
                self.health_report['critical_issues'] += 1
        
        # Generate summary and recommendations
        self.generate_health_summary()
        self.get_recommendations()
        
        # Save report
        self.save_health_report()
        
        return self.health_report


def run_daily_health_check():
    """Main function to run the daily health check"""
    try:
        checker = DatabaseHealthChecker()
        health_report = checker.run_full_health_check()
        
        # Print final status
        print(f"\n{'='*60}")
        print(f"DAILY DATABASE HEALTH CHECK COMPLETE")
        print(f"Overall Status: {health_report['overall_status']}")
        print(f"Report saved to: {HEALTH_REPORT_FILE}")
        print(f"{'='*60}")
        
        # Return status for integration with other scripts
        return health_report['overall_status'] != 'FAILED'
        
    except Exception as e:
        logger.error(f"Critical error in health check: {e}")
        traceback.print_exc()
        return False


def get_health_status_for_email():
    """Get a concise health status summary for inclusion in email reports"""
    try:
        if os.path.exists(HEALTH_REPORT_FILE):
            with open(HEALTH_REPORT_FILE, 'r') as f:
                health_report = json.load(f)
            
            status = health_report.get('overall_status', 'UNKNOWN')
            total_checks = health_report.get('checks_passed', 0) + health_report.get('checks_failed', 0)
            
            # Create emoji status
            status_emoji = "✅" if status == 'HEALTHY' else "⚠️" if status == 'WARNING' else "❌"
            
            summary = f"{status_emoji} Database Health: {status}"
            
            if total_checks > 0:
                summary += f" ({health_report.get('checks_passed', 0)}/{total_checks} checks passed)"
            
            if health_report.get('warnings', 0) > 0:
                summary += f", {health_report['warnings']} warnings"
            
            if health_report.get('critical_issues', 0) > 0:
                summary += f", {health_report['critical_issues']} critical issues"
            
            return summary
        else:
            return "⚠️ Database Health: No recent health check data available"
            
    except Exception as e:
        logger.error(f"Error getting health status for email: {e}")
        return "❌ Database Health: Error reading health check data"


def get_detailed_health_report():
    """Get detailed health report for dashboard or detailed analysis"""
    try:
        if os.path.exists(HEALTH_REPORT_FILE):
            with open(HEALTH_REPORT_FILE, 'r') as f:
                return json.load(f)
        else:
            return None
    except Exception as e:
        logger.error(f"Error reading detailed health report: {e}")
        return None


# Integration functions for send_report.py
def integrate_health_check_with_send_report():
    """
    Integration function to be called from send_report.py
    Returns health status for inclusion in email
    """
    logger.info("Running database health check as part of daily report...")
    
    # Run the health check
    success = run_daily_health_check()
    
    # Get status for email
    email_status = get_health_status_for_email()
    
    return {
        'success': success,
        'email_status': email_status,
        'detailed_report': get_detailed_health_report()
    }


if __name__ == "__main__":
    # Run health check if script is executed directly
    run_daily_health_check()