"""
Main application for trading reports system.
Orchestrates all components and provides the main entry point.
FULLY SYNCHRONIZED: 2025-07-19 - Complete sync with trading_reports_parsers.py and fixed syntax errors
ATTACHMENT FIX: Fixed email attachment processing to flatten nested lists
SYNTAX FIXED: Fixed all unterminated string literals and incomplete code blocks - COMPLETE VERSION
"""

import sys
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Import our modules - FIXED imports to match what's actually available
from trading_reports_config import config
from trading_reports_database import db_manager
from trading_reports_parsers import parse_all_reports, find_report_files, get_latest_trading_signals, print_parsing_summary
from trading_reports_email import email_manager
from trading_reports_validation import validate_report_data

logger = logging.getLogger("trading_reports.main")

# Import options strategy analyzer
try:
    from options_strategy_analyzer import OptionsStrategyAnalyzer
    HAS_OPTIONS_ANALYZER = True
    logger.info("Options strategy analyzer available")
except ImportError:
    HAS_OPTIONS_ANALYZER = False
    logger.warning("Options strategy analyzer not available")

class HealthChecker:
    """Performs health checks on system components"""
    
    def __init__(self):
        self.checks = []
    
    def add_check(self, name: str, check_func):
        """Add a health check"""
        self.checks.append((name, check_func))
    
    def run_health_checks(self) -> Dict[str, str]:
        """Run all health checks and return results"""
        results = {}
        
        for name, check_func in self.checks:
            try:
                result = check_func()
                results[name] = "OK" if result else "FAILED"
            except Exception as e:
                results[name] = f"ERROR: {e}"
        
        return results
    
    def check_database_connection(self) -> bool:
        """Check database connectivity"""
        try:
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def check_email_configuration(self) -> bool:
        """Check email configuration"""
        return config.email.is_configured
    
    def check_required_directories(self) -> bool:
        """Check if required directories exist"""
        try:
            desktop_path = config.get_desktop_path()
            script_dir = config.get_script_dir()
            return desktop_path.exists() and script_dir.exists()
        except Exception as e:
            logger.error(f"Directory check failed: {e}")
            return False

class DataUpdater:
    """Handles daily data updates and maintenance"""
    
    def __init__(self):
        pass
    
    def run_daily_price_updates(self) -> int:
        """Run daily price updates using data_fetcher if available"""
        try:
            # Try to import data_fetcher
            from data_fetcher import (
                get_current_price, 
                clear_cache,
                save_current_price_to_db
            )
            data_fetcher_available = True
        except ImportError:
            logger.warning("data_fetcher.py not available. Price updates may be limited.")
            data_fetcher_available = False
            return 0
        
        if not data_fetcher_available:
            return 0
        
        logger.info("Running daily price updates...")
        
        # Clear cache to ensure fresh data
        clear_cache()
        
        updated_count = 0
        for symbol in config.reports.symbols_to_update:
            try:
                current_price = get_current_price(symbol)
                if current_price:
                    logger.info(f"Updated {symbol}: ${current_price:.2f}")
                    updated_count += 1
                else:
                    logger.warning(f"Failed to update price for {symbol}")
            except Exception as e:
                logger.error(f"Error updating price for {symbol}: {e}")
        
        logger.info(f"Daily price update completed: {updated_count}/{len(config.reports.symbols_to_update)} symbols updated")
        return updated_count
    
    def update_pending_predictions(self) -> int:
        """Update predictions with actual prices where target dates have passed"""
        try:
            # Try to import data_fetcher for historical prices
            try:
                from data_fetcher import get_historical_price
                data_fetcher_available = True
            except ImportError:
                data_fetcher_available = False
            
            pending_predictions = db_manager.get_pending_predictions()
            
            if not pending_predictions:
                logger.info("No predictions need actual price updates")
                return 0
            
            logger.info(f"Found {len(pending_predictions)} predictions to update with actual prices")
            
            updated_count = 0
            for prediction in pending_predictions:
                pred_id = prediction['id']
                symbol = prediction['symbol']
                target_date = prediction['target_date']
                
                try:
                    # Try to get actual price from database first
                    actual_price = db_manager.get_historical_price(symbol, target_date)
                    
                    # If not in database and data_fetcher available, try fetching
                    if not actual_price and data_fetcher_available:
                        from datetime import datetime
                        target_date_obj = datetime.strptime(target_date, '%Y-%m-%d').date()
                        actual_price = get_historical_price(symbol, target_date_obj)
                    
                    if actual_price:
                        if db_manager.update_prediction_actual(pred_id, actual_price):
                            updated_count += 1
                            logger.info(f"Updated prediction {pred_id}: {symbol} on {target_date} = ${actual_price:.2f}")
                    else:
                        logger.warning(f"No actual price found for {symbol} on {target_date}")
                        
                except Exception as e:
                    logger.warning(f"Error updating prediction {pred_id}: {e}")
                    continue
            
            logger.info(f"Updated {updated_count} predictions with actual prices")
            return updated_count
            
        except Exception as e:
            logger.error(f"Error in update_pending_predictions: {e}")
            return 0

class ReportGenerator:
    """Generates combined reports from model data"""
    
    def __init__(self):
        pass
    
    def create_combined_text_report(self, model_reports: Dict[str, Dict[str, Any]]) -> Optional[str]:
        """Create combined text report from all model data - FULLY SYNCHRONIZED with parser field names"""
        try:
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M')
            report_content = f"Trading Models Report - {current_time}\n\n"
            
            # Model configurations - EXACTLY matching parser output keys and field names
            model_configs = [
                {
                    'key': 'longhorn',  # Exact key from parser
                    'name': 'Long Bull Model',
                    'fields': [
                        ('current_price', 'Current Price'),
                        ('target_price', 'Target Price'),  # Exact field from longhorn parser
                        ('signal', 'Signal'),
                        ('suggested_action', 'Suggested Action'),
                        ('confidence', 'Confidence'),
                        ('risk_level', 'Risk Level'),
                        ('stop_loss', 'Stop Loss')
                    ]
                },
                {
                    'key': 'trading_signal',  # Exact key from parser
                    'name': 'QQQ Trading Signal',
                    'fields': [
                        ('current_price', 'Current Price'),
                        ('price_target', 'Price Target'),  # Exact field from trading_signal parser
                        ('signal', 'Signal'),
                        ('suggested_action', 'Suggested Action'),
                        ('confidence', 'Confidence'),
                        ('stop_loss', 'Stop Loss')
                    ]
                },
                {
                    'key': 'wishing_wealth',  # Exact key from parser
                    'name': 'Wishing Wealth QQQ Model',
                    'fields': [
                        ('current_price', 'Current Price'),
                        ('signal', 'Signal'),
                        ('suggested_action', 'Suggested Action'),
                        ('confidence', 'Confidence'),  # Computed field in parser
                        ('gmi_score', 'GMI Score'),
                        ('recommended_etf', 'Recommended ETF'),
                        ('timing_total_return', 'Strategy Total Return'),
                        ('win_rate', 'Win Rate'),
                        ('strategy', 'Strategy'),
                        ('qqq_trend', 'QQQ Trend')
                    ]
                },
                {
                    'key': 'nvidia',  # Exact key from parser
                    'name': 'NVIDIA Bull Momentum Model',
                    'fields': [
                        ('current_price', 'Current Price'),
                        ('predicted_1_day_price', 'Predicted 1-Day Price'),  # Exact field from nvidia parser
                        ('predicted_1_day_return', 'Predicted 1-Day Return'),
                        ('predicted_5_day_price', 'Predicted 5-Day Price'),
                        ('predicted_5_day_return', 'Predicted 5-Day Return'),
                        ('signal', 'Signal'),
                        ('suggested_action', 'Suggested Action'),
                        ('confidence', 'Confidence')
                    ]
                },
                {
                    'key': 'algorand',  # Exact key from parser
                    'name': 'Algorand Model',
                    'fields': [
                        ('current_price', 'Current Price'),
                        ('predicted_price', 'Predicted Price'),  # Exact field from algorand parser
                        ('predicted_1_day_price', 'Predicted 1-Day Price'),
                        ('expected_return', 'Expected Return'),
                        ('ensemble_1_day_return', 'Ensemble 1-Day Return'),
                        ('signal', 'Signal'),
                        ('suggested_action', 'Suggested Action'),
                        ('confidence', 'Confidence'),
                        ('market_regime', 'Market Regime'),
                        ('direction_accuracy', 'Direction Accuracy')
                    ]
                },
                {
                    'key': 'bitcoin',  # Exact key from parser
                    'name': 'Bitcoin Model',
                    'fields': [
                        ('current_price', 'Current Price'),
                        ('predicted_1_day_price', 'Predicted 1-Day Price'),  # Exact field from bitcoin parser
                        ('predicted_1_day_return', 'Predicted 1-Day Return'),
                        ('predicted_3_day_price', 'Predicted 3-Day Price'),
                        ('predicted_3_day_return', 'Predicted 3-Day Return'),
                        ('predicted_7_day_price', 'Predicted 7-Day Price'),
                        ('predicted_7_day_return', 'Predicted 7-Day Return'),
                        ('signal', 'Signal'),
                        ('suggested_action', 'Suggested Action')
                    ]
                }
            ]
            
            for model_config in model_configs:
                model_key = model_config['key']
                model_name = model_config['name']
                data = model_reports.get(model_key, {})
                
                if data:
                    # Get report date/timestamp - checking exact field names from parsers
                    report_date = data.get('report_timestamp', data.get('generated_date', 'N/A'))
                    report_content += f"{model_name} - {report_date}\n"
                    report_content += "=" * len(f"{model_name} - {report_date}") + "\n"
                    
                    # Add fields based on what's available
                    fields_added = 0
                    for field_key, field_name in model_config['fields']:
                        if field_key in data and data[field_key] is not None and str(data[field_key]).strip() != "" and str(data[field_key]) != "N/A":
                            value = data[field_key]
                            
                            # Format the value nicely
                            if isinstance(value, float):
                                if field_key.endswith('_price') or field_key == 'current_price' or field_key == 'target_price' or field_key == 'price_target':
                                    report_content += f"{field_name}: ${value:.2f}\n"
                                elif field_key.endswith('_return') or field_key.endswith('_rate') or 'percentage' in field_key or field_key == 'win_rate':
                                    report_content += f"{field_name}: {value:+.2f}%\n"
                                elif field_key == 'direction_accuracy':
                                    report_content += f"{field_name}: {value:.1f}%\n"
                                else:
                                    report_content += f"{field_name}: {value:.2f}\n"
                            else:
                                report_content += f"{field_name}: {value}\n"
                            
                            fields_added += 1
                    
                    if fields_added == 0:
                        report_content += "No data available\n"
                    
                    report_content += "\n"
                else:
                    report_content += f"{model_name} - No data found\n\n"
            
            # Add summary section using the exact field names from parsers
            report_content += "SUMMARY\n"
            report_content += "=" * 50 + "\n"
            
            signals_summary = []
            for model_config in model_configs:
                data = model_reports.get(model_config['key'], {})
                if data and 'signal' in data and data['signal']:
                    signals_summary.append(f"{model_config['name']}: {data['signal']}")
            
            if signals_summary:
                report_content += "Current Signals:\n"
                for signal in signals_summary:
                    report_content += f"  • {signal}\n"
            else:
                report_content += "No signals available\n"
            
            # Save to desktop
            desktop_path = config.get_desktop_path()
            filename = f'Combined_Trading_Report_{datetime.now().strftime("%Y%m%d_%H%M")}.txt'
            combined_report_path = desktop_path / filename
            
            with open(combined_report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"Combined report saved to: {combined_report_path}")
            return str(combined_report_path)
            
        except Exception as e:
            logger.error(f"Error creating combined report: {e}")
            return None

class TradingReportsApplication:
    """Main application class that orchestrates all components"""
    
    def __init__(self):
        self.health_checker = HealthChecker()
        self.data_updater = DataUpdater()
        self.report_generator = ReportGenerator()
        
        # Setup health checks
        self.health_checker.add_check("database", self.health_checker.check_database_connection)
        self.health_checker.add_check("email", self.health_checker.check_email_configuration)
        self.health_checker.add_check("directories", self.health_checker.check_required_directories)
    
    def run_startup_checks(self) -> bool:
        """Run startup health checks"""
        logger.info("=== RUNNING STARTUP CHECKS ===")
        
        # Validate configuration
        config_issues = config.validate_configuration()
        if config_issues:
            logger.warning("Configuration issues found:")
            for issue in config_issues:
                logger.warning(f"  - {issue}")
                print(f"Warning: {issue}")
        
        # Run health checks
        health_results = self.health_checker.run_health_checks()
        
        all_healthy = True
        for check_name, result in health_results.items():
            if result != "OK":
                logger.error(f"Health check failed - {check_name}: {result}")
                all_healthy = False
            else:
                logger.info(f"Health check passed - {check_name}: {result}")
        
        if not all_healthy:
            logger.warning("Some health checks failed - proceeding with caution")
        
        return True  # Continue even with warnings
    
    def run_data_maintenance(self) -> Dict[str, int]:
        """Run daily data maintenance tasks"""
        logger.info("=== RUNNING DAILY MAINTENANCE TASKS ===")
        
        # Backup database
        if not db_manager.backup_database():
            logger.error("Database backup failed")
        
        # Update current prices
        price_updates = self.data_updater.run_daily_price_updates()
        
        # Update prediction evaluations
        prediction_updates = self.data_updater.update_pending_predictions()
        
        return {
            'price_updates': price_updates,
            'prediction_updates': prediction_updates
        }
    
    def parse_all_reports(self) -> Dict[str, Dict[str, Any]]:
        """Find and parse all available reports - ENHANCED to search multiple directories"""
        logger.info("=== PARSING REPORTS ===")
        
        try:
            # Search multiple potential directories for reports
            search_directories = [
                config.get_desktop_path(),  # OneDrive Desktop (primary)
                Path.home() / 'Desktop',    # Regular Desktop  
                config.get_script_dir()     # Script directory
            ]
            
            all_model_reports = {}
            
            # Try each directory until we find reports
            for directory in search_directories:
                if not directory.exists():
                    continue
                    
                logger.info(f"Searching directory: {directory}")
                model_reports = parse_all_reports(directory=str(directory))
                
                if model_reports:
                    # Merge results, preferring newer reports
                    for model_type, data in model_reports.items():
                        if model_type not in all_model_reports or not all_model_reports[model_type]:
                            all_model_reports[model_type] = data
                        elif data:  # If we have data and existing is empty, replace
                            all_model_reports[model_type] = data
            
            # If we still don't have enough reports, try the default function without directory
            if len(all_model_reports) < 3:  # We expect at least 3-4 model types
                try:
                    default_reports = parse_all_reports()
                    for model_type, data in default_reports.items():
                        if model_type not in all_model_reports or not all_model_reports[model_type]:
                            all_model_reports[model_type] = data
                except Exception as e:
                    logger.warning(f"Default parse_all_reports failed: {e}")
            
            if all_model_reports:
                logger.info(f"Successfully parsed {len(all_model_reports)} reports from multiple directories")
                for report_type, data in all_model_reports.items():
                    if data:
                        # Count meaningful fields (not None, not empty string, not "N/A")
                        field_count = len([k for k, v in data.items() 
                                         if v is not None and str(v).strip() != "" and str(v) != "N/A"])
                        logger.info(f"  {report_type}: {field_count} meaningful fields extracted")
                        
                        # Log key fields for debugging
                        key_fields = ['current_price', 'signal', 'confidence', 'suggested_action']
                        for field in key_fields:
                            if field in data and data[field] is not None:
                                logger.info(f"    {field}: {data[field]}")
                    else:
                        logger.warning(f"  {report_type}: No data extracted")
                
                # Use the print_parsing_summary function from parsers
                print_parsing_summary(all_model_reports)
                        
            else:
                logger.warning("No reports were parsed successfully from any directory")
            
            return all_model_reports
            
        except Exception as e:
            logger.error(f"Error in parse_all_reports: {e}")
            return {}
    
    def store_predictions(self, model_reports: Dict[str, Dict[str, Any]]) -> int:
        """Store new predictions to database - FULLY SYNCHRONIZED with parser field names + VALUE CLEANING"""
        logger.info("=== STORING PREDICTIONS TO DATABASE ===")
        
        # Clean the model reports before processing
        cleaned_reports = {}
        for model_key, model_data in model_reports.items():
            cleaned_reports[model_key] = self.clean_extracted_values(model_data)
        
        # Model configurations with EXACT field names from each parser
        model_configs = {
            'longhorn': {  # From parse_longhorn_report
                'model_name': 'Long Bull Model', 
                'symbol': 'QQQ',
                'price_field': 'target_price',  # Exact field from longhorn parser
                'return_field': 'expected_return'  # FIXED: Now available from longhorn parser
            },
            'qqq_master': {  # From parse_qqq_master_report
                'model_name': 'QQQ Master Model',
                'symbol': 'QQQ',
                'price_field': 'predicted_price',  # Field name from qqq_master parser
                'return_field': 'expected_return'  # Field name from qqq_master parser
            },
            'trading_signal': {  # From parse_trading_signal_report
                'model_name': 'QQQ Trading Signal', 
                'symbol': 'QQQ',
                'price_field': 'price_target',  # Exact field from trading_signal parser
                'return_field': None  # No return field in trading_signal parser
            },
            'algorand': {  # From parse_algorand_report
                'model_name': 'Algorand Model', 
                'symbol': 'ALGO-USD',
                'price_field': 'predicted_price',  # Exact field from algorand parser
                'return_field': 'expected_return'  # Exact field from algorand parser
            },
            'bitcoin': {  # From parse_bitcoin_report
                'model_name': 'Bitcoin Model', 
                'symbol': 'BTC-USD',
                'price_field': 'predicted_1_day_price',  # Exact field from bitcoin parser
                'return_field': 'predicted_1_day_return'  # Exact field from bitcoin parser
            },
            'wishing_wealth': {  # From parse_wishing_wealth_report
                'model_name': 'Wishing Wealth QQQ Model', 
                'symbol': 'QQQ',
                'price_field': 'current_price',  # No price prediction in this parser, use current
                'return_field': 'timing_total_return'  # Use strategy return from this parser
            },
            'nvidia': {  # From parse_nvidia_report
                'model_name': 'NVIDIA Bull Momentum Model', 
                'symbol': 'NVDA',
                'price_field': 'predicted_1_day_price',  # Exact field from nvidia parser
                'return_field': 'predicted_1_day_return'  # Exact field from nvidia parser
            }
        }
        
        storage_success = 0
        
        for model_key, model_config in model_configs.items():
            data = cleaned_reports.get(model_key, {})  # Use cleaned data
            
            if not data:
                logger.warning(f"No data found for model: {model_key}")
                continue
                
            # Get current price (all parsers should have this)
            current_price_raw = data.get('current_price')
            if not current_price_raw or current_price_raw == 'N/A':
                logger.warning(f"No current price for {model_config['model_name']}")
                continue
                
            # Get predicted price using the correct field name for each model
            predicted_price_raw = data.get(model_config['price_field'])
            if not predicted_price_raw or predicted_price_raw == 'N/A':
                logger.warning(f"No predicted price for {model_config['model_name']} (looking for field: {model_config['price_field']})")
                # For debugging, show what fields are available
                available_fields = [k for k, v in data.items() if v is not None and str(v) != 'N/A']
                logger.info(f"  Available fields: {available_fields}")
                continue
                
            try:
                # Extract numeric values for storage - COMPLETE STRING CLEANING
                current_price = float(str(current_price_raw).replace('$', '').replace(',', ''))
                predicted_price = float(str(predicted_price_raw).replace('$', '').replace(',', ''))
                
                # Get expected return if available
                expected_return = 0.0
                if model_config['return_field'] and model_config['return_field'] in data:
                    return_raw = data[model_config['return_field']]
                    if return_raw and return_raw != 'N/A':
                        expected_return = float(str(return_raw).replace('%', '').replace('+', ''))
                
                # Store the prediction
                if db_manager.store_prediction(
                    model_config['model_name'], 
                    model_config['symbol'], 
                    current_price, 
                    predicted_price, 
                    expected_return, 
                    horizon_days=1  # Default to 1 day
                ):
                    storage_success += 1
                    logger.info(f"Stored prediction for {model_config['model_name']}: {model_config['symbol']} ${current_price:.2f} -> ${predicted_price:.2f} ({expected_return:+.2f}%)")
                else:
                    logger.error(f"Failed to store prediction for {model_config['model_name']}")
                    
            except (ValueError, TypeError) as e:
                logger.error(f"Error parsing values for {model_config['model_name']}: {e}")
                logger.error(f"  current_price_raw: {current_price_raw}")
                logger.error(f"  predicted_price_raw: {predicted_price_raw}")
                continue
        
        logger.info(f"Successfully stored {storage_success} predictions to database")
        return storage_success
    
    def generate_and_send_report(self, model_reports: Dict[str, Dict[str, Any]]) -> bool:
        """Generate combined report and send email - TODAY'S FILES ONLY + VALUE CLEANING"""
        logger.info("=== GENERATING AND SENDING REPORT ===")
        
        # Clean up parsed values before generating report
        cleaned_reports = {}
        for model_key, model_data in model_reports.items():
            cleaned_reports[model_key] = self.clean_extracted_values(model_data)
        
        # Generate combined text report with cleaned data
        combined_report = self.report_generator.create_combined_text_report(cleaned_reports)
        
        # Get report files for attachments - FILTERED TO TODAY ONLY
        try:
            desktop_path = config.get_desktop_path()
            report_files_result = find_report_files([str(desktop_path)])
            
            # FILTER TO TODAY'S FILES ONLY
            today_date = datetime.now().strftime('%Y%m%d')  # Format: 20250720
            today_alt = datetime.now().strftime('%Y-%m-%d')  # Format: 2025-07-20
            
            filtered_files = {}
            total_old_files = 0
            total_today_files = 0
            
            for model_type, file_list in report_files_result.items():
                today_files = []
                for file_path in file_list:
                    filename = os.path.basename(file_path)
                    # Check if file contains today's date in various formats
                    if (today_date in filename or 
                        today_alt in filename or
                        filename.endswith(f'{today_date}.txt') or
                        f'_{today_date}_' in filename):
                        today_files.append(file_path)
                        logger.info(f"Including today's file: {filename}")
                        total_today_files += 1
                    else:
                        logger.info(f"Skipping older file: {filename}")
                        total_old_files += 1
                
                if today_files:
                    filtered_files[model_type] = today_files
            
            logger.info(f"File filtering: {total_today_files} today's files, {total_old_files} older files skipped")
            report_files_result = filtered_files
            
        except Exception as e:
            logger.warning(f"Error finding report files: {e}")
            report_files_result = {}
        
        # Send email if configured
        email_sent = False
        try:
            if email_manager.is_initialized():
                # FIXED: Flatten nested list of file paths (TODAY'S FILES ONLY)
                attachments = [f for file_list in report_files_result.values() for f in file_list] if report_files_result else []
                
                # Add combined report to attachments
                if combined_report:
                    attachments.append(combined_report)
                
                # Generate options strategy analysis if available
                if HAS_OPTIONS_ANALYZER:
                    try:
                        options_analyzer = OptionsStrategyAnalyzer()
                        options_report_path = options_analyzer.save_strategy_report(cleaned_reports)
                        if options_report_path:
                            attachments.append(options_report_path)
                            logger.info("Options strategy analysis added to attachments")
                    except Exception as e:
                        logger.error(f"Failed to generate options analysis: {e}")
                
                logger.info(f"Sending email with {len(attachments)} today's attachments")
                email_sent = email_manager.send_trading_report(model_reports=cleaned_reports, attachments=attachments)
                
                if email_sent:
                    logger.info("Email report sent successfully")
                else:
                    logger.error("Failed to send email report")
            else:
                logger.warning("Email manager not initialized - cannot send report")
                email_sent = False
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            email_sent = False
        
        return email_sent
    
    def clean_extracted_values(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean up extracted values to remove parsing artifacts"""
        
        cleaned_data = data.copy()
        
        # Clean confidence values
        if 'confidence' in cleaned_data:
            confidence = str(cleaned_data['confidence']).strip()
            
            # Convert "VERY" to meaningful confidence levels
            if confidence.upper() == 'VERY':
                cleaned_data['confidence'] = '95%'  # Very high confidence
            elif confidence.upper() == 'VERY HIGH':
                cleaned_data['confidence'] = '95%'
            elif confidence.upper() == 'VERY LOW':
                cleaned_data['confidence'] = '15%'
            elif confidence.upper() == 'HIGH':
                cleaned_data['confidence'] = '85%'
            elif confidence.upper() == 'MEDIUM':
                cleaned_data['confidence'] = '65%'
            elif confidence.upper() == 'LOW':
                cleaned_data['confidence'] = '35%'
            elif 'CONFIDENCE' in confidence.upper():
                # Remove extra "CONFIDENCE" text
                cleaned_data['confidence'] = confidence.replace('CONFIDENCE', '').strip()
                if cleaned_data['confidence'].upper() == 'VERY':
                    cleaned_data['confidence'] = '95%'
        
        # Clean suggested_action values
        if 'suggested_action' in cleaned_data:
            action = str(cleaned_data['suggested_action']).strip()
            
            # Remove extra text artifacts
            if 'CONFIDENCE' in action.upper():
                action = action.replace('CONFIDENCE', '').strip()
            if 'CURRENT PRICE' in action.upper():
                action = action.replace('CURRENT PRICE', '').strip()
            
            # Standardize action values
            action_upper = action.upper()
            if action_upper in ['BUY', 'STRONG BUY']:
                cleaned_data['suggested_action'] = 'BUY'
            elif action_upper in ['SELL', 'STRONG SELL']:
                cleaned_data['suggested_action'] = 'SELL'
            elif action_upper in ['HOLD', 'NEUTRAL']:
                cleaned_data['suggested_action'] = 'HOLD'
            else:
                cleaned_data['suggested_action'] = action.upper()
        
        # Clean risk_level values
        if 'risk_level' in cleaned_data:
            risk = str(cleaned_data['risk_level']).strip()
            
            if risk.upper() == 'VERY':
                cleaned_data['risk_level'] = 'VERY HIGH'
            elif risk.upper() == 'VERY HIGH':
                cleaned_data['risk_level'] = 'VERY HIGH'
            elif risk.upper() == 'VERY LOW':
                cleaned_data['risk_level'] = 'VERY LOW'
            elif risk.upper() in ['HIGH', 'MEDIUM', 'LOW']:
                cleaned_data['risk_level'] = risk.upper()
        
        # Clean signal values
        if 'signal' in cleaned_data:
            signal = str(cleaned_data['signal']).strip()
            signal_upper = signal.upper()
            
            if signal_upper in ['BUY', 'STRONG BUY', 'BULLISH']:
                cleaned_data['signal'] = 'BUY'
            elif signal_upper in ['SELL', 'STRONG SELL', 'BEARISH']:
                cleaned_data['signal'] = 'SELL'
            elif signal_upper in ['HOLD', 'NEUTRAL']:
                cleaned_data['signal'] = 'HOLD'
        
        return cleaned_data
    
    def run(self) -> int:
        """Main application entry point"""
        try:
            logger.info("Starting Trading Reports System...")
            print("Starting Trading Reports System...")
            
            # Startup checks
            if not self.run_startup_checks():
                logger.error("Startup checks failed - exiting")
                return 1
            
            # Initialize email system
            email_configured = email_manager.initialize()
            if not email_configured:
                logger.warning("Email system not configured - reports will be generated but not sent")
            
            # Run data maintenance
            maintenance_results = self.run_data_maintenance()
            
            # Parse reports
            model_reports = self.parse_all_reports()
            
            if not model_reports:
                logger.error("No valid report data found - exiting")
                print("Error: No valid report data found. Exiting.")
                return 1
            
            # Store predictions
            storage_count = self.store_predictions(model_reports)
            
            # Generate and send report
            email_sent = self.generate_and_send_report(model_reports)
            
            # Print summary
            self.print_completion_summary(
                maintenance_results, storage_count, 
                len(model_reports), email_sent, email_configured
            )
            
            logger.info("Trading Reports System completed successfully")
            return 0
            
        except Exception as e:
            logger.error(f"Unexpected error in main application: {e}")
            print(f"Unexpected error: {e}")
            return 1
    
    def print_completion_summary(self, maintenance_results: Dict[str, int], 
                               storage_count: int, reports_parsed: int, 
                               email_sent: bool, email_configured: bool):
        """Print completion summary"""
        print("\n" + "="*50)
        print("TRADING REPORTS SYSTEM - COMPLETION SUMMARY")
        print("="*50)
        
        print(f"✓ Reports parsed: {reports_parsed}")
        print(f"✓ Price updates: {maintenance_results.get('price_updates', 0)}")
        print(f"✓ Predictions stored: {storage_count}")
        print(f"✓ Prediction evaluations: {maintenance_results.get('prediction_updates', 0)}")
        
        if email_configured:
            if email_sent:
                print("✓ Email report: SENT")
                try:
                    print(f"  Dashboard: {config.reports.dashboard_url}")
                except:
                    pass
            else:
                print("✗ Email report: FAILED")
        else:
            print("- Email report: DISABLED (not configured)")
        
        try:
            db_updated, db_status = db_manager.get_database_status()
            print(f"✓ Database status: {'OK' if db_updated else 'WARNING'}")
        except Exception as e:
            print(f"✗ Database status: ERROR - {e}")
        
        print("="*50)

def main():
    """Main entry point"""
    app = TradingReportsApplication()
    return app.run()

if __name__ == "__main__":
    sys.exit(main())