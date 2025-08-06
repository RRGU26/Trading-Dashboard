#!/usr/bin/env python3
"""
Trading Models Performance Monitoring System
============================================

This module implements comprehensive performance monitoring and optimization
for the trading models system. It provides:

1. Real-time performance tracking
2. Automated retraining triggers
3. Cross-model correlation analysis
4. Risk-adjusted performance metrics
5. Database optimization tools

Usage:
    python performance_monitoring_system.py --monitor
    python performance_monitoring_system.py --analyze-correlations
    python performance_monitoring_system.py --optimize-database
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("performance_monitor")
warnings.filterwarnings('ignore')

class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for trading models
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the performance monitor
        
        Args:
            db_path: Path to the main database file
        """
        self.db_path = db_path or "C:\\Users\\rrose\\models_dashboard.db"
        self.backup_db_path = self.db_path.replace('.db', '_backup.db')
        
        # Performance thresholds
        self.thresholds = {
            'accuracy_decline': 0.05,  # 5% drop in accuracy
            'sharpe_decline': 0.3,     # 30% drop in Sharpe ratio
            'max_drawdown': 0.15,      # 15% maximum drawdown
            'correlation_threshold': 0.8,  # High correlation threshold
            'min_predictions': 20,     # Minimum predictions for analysis
            'lookback_days': 30        # Days to look back for performance
        }
        
        # Model registry
        self.models = {
            'QQQ_Enhanced': {'asset': 'QQQ', 'type': 'classification'},
            'Bitcoin_Predictor': {'asset': 'BTC-USD', 'type': 'regression'},
            'Algorand_V2': {'asset': 'ALGO-USD', 'type': 'ensemble'},
            'NVIDIA_Model': {'asset': 'NVDA', 'type': 'regression'},
            'Long_Bull_Model': {'asset': 'MULTIPLE', 'type': 'trend'},
            'VEO_Model': {'asset': 'VEO', 'type': 'classification'},
            'Combined_Ensemble': {'asset': 'MULTIPLE', 'type': 'ensemble'}
        }
        
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize or create the performance monitoring database schema"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create performance monitoring tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name VARCHAR(50) NOT NULL,
                    asset_symbol VARCHAR(10) NOT NULL,
                    prediction_date DATE NOT NULL,
                    current_price DECIMAL(12,4),
                    predicted_price DECIMAL(12,4),
                    prediction_horizon INTEGER,
                    confidence_score DECIMAL(5,4),
                    actual_price DECIMAL(12,4),
                    accuracy_score DECIMAL(5,4),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name VARCHAR(50) NOT NULL,
                    asset_symbol VARCHAR(10) NOT NULL,
                    calculation_date DATE NOT NULL,
                    hit_rate DECIMAL(5,4),
                    sharpe_ratio DECIMAL(8,4),
                    max_drawdown DECIMAL(5,4),
                    total_return DECIMAL(8,4),
                    volatility DECIMAL(6,4),
                    num_predictions INTEGER,
                    avg_confidence DECIMAL(5,4),
                    sortino_ratio DECIMAL(8,4),
                    calmar_ratio DECIMAL(8,4),
                    information_ratio DECIMAL(8,4)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS retraining_triggers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name VARCHAR(50) NOT NULL,
                    trigger_type VARCHAR(50) NOT NULL,
                    trigger_date DATE NOT NULL,
                    trigger_value DECIMAL(8,4),
                    threshold_breached DECIMAL(8,4),
                    status VARCHAR(20) DEFAULT 'PENDING',
                    retraining_completed DATE,
                    notes TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_correlations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_1 VARCHAR(50) NOT NULL,
                    model_2 VARCHAR(50) NOT NULL,
                    correlation_coefficient DECIMAL(6,4),
                    calculation_date DATE NOT NULL,
                    period_days INTEGER,
                    significance_level DECIMAL(4,3)
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_model_date ON model_predictions(model_name, prediction_date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_asset_date ON model_predictions(asset_symbol, prediction_date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_performance_model_date ON model_performance_metrics(model_name, calculation_date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_triggers_model_date ON retraining_triggers(model_name, trigger_date)')
            
            conn.commit()
            conn.close()
            
            logger.info("Database schema initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def calculate_performance_metrics(self, model_name: str, days_back: int = 30) -> Dict:
        """
        Calculate comprehensive performance metrics for a model
        
        Args:
            model_name: Name of the model to analyze
            days_back: Number of days to look back
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get recent predictions with actual prices
            query = '''
                SELECT prediction_date, current_price, predicted_price, actual_price,
                       confidence_score, accuracy_score
                FROM model_predictions
                WHERE model_name = ? 
                AND prediction_date >= date('now', '-{} days')
                AND actual_price IS NOT NULL
                ORDER BY prediction_date
            '''.format(days_back)
            
            df = pd.read_sql_query(query, conn, params=[model_name])
            conn.close()
            
            if df.empty:
                logger.warning(f"No recent data found for model {model_name}")
                return {}
            
            # Calculate returns
            df['actual_return'] = (df['actual_price'] - df['current_price']) / df['current_price']
            df['predicted_return'] = (df['predicted_price'] - df['current_price']) / df['current_price']
            df['direction_correct'] = (np.sign(df['actual_return']) == np.sign(df['predicted_return'])).astype(int)
            
            # Performance metrics
            metrics = {
                'num_predictions': len(df),
                'hit_rate': df['direction_correct'].mean(),
                'avg_confidence': df['confidence_score'].mean(),
                'total_return': df['actual_return'].sum(),
                'volatility': df['actual_return'].std() * np.sqrt(252),  # Annualized
                'mean_return': df['actual_return'].mean(),
                'mean_absolute_error': np.abs(df['actual_return'] - df['predicted_return']).mean(),
                'rmse': np.sqrt(np.mean((df['actual_return'] - df['predicted_return'])**2))
            }
            
            # Calculate Sharpe ratio
            if metrics['volatility'] > 0:
                risk_free_rate = 0.02  # Assume 2% risk-free rate
                metrics['sharpe_ratio'] = (metrics['mean_return'] * 252 - risk_free_rate) / metrics['volatility']
            else:
                metrics['sharpe_ratio'] = 0
            
            # Calculate maximum drawdown
            cumulative_returns = (1 + df['actual_return']).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            metrics['max_drawdown'] = drawdown.min()
            
            # Calculate Sortino ratio (downside deviation)
            downside_returns = df['actual_return'][df['actual_return'] < 0]
            if len(downside_returns) > 0:
                downside_deviation = downside_returns.std() * np.sqrt(252)
                metrics['sortino_ratio'] = (metrics['mean_return'] * 252 - risk_free_rate) / downside_deviation
            else:
                metrics['sortino_ratio'] = float('inf')
            
            # Calculate Calmar ratio
            if metrics['max_drawdown'] < 0:
                metrics['calmar_ratio'] = (metrics['mean_return'] * 252) / abs(metrics['max_drawdown'])
            else:
                metrics['calmar_ratio'] = float('inf')
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics for {model_name}: {e}")
            return {}
    
    def check_retraining_triggers(self, model_name: str) -> List[Dict]:
        """
        Check if any retraining triggers have been activated
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            List of triggered conditions
        """
        triggers = []
        
        try:
            # Get current performance metrics
            current_metrics = self.calculate_performance_metrics(model_name, days_back=30)
            
            if not current_metrics:
                return triggers
            
            # Get historical performance for comparison
            historical_metrics = self.calculate_performance_metrics(model_name, days_back=90)
            
            # Check accuracy decline
            if (historical_metrics.get('hit_rate', 0) > 0 and 
                current_metrics.get('hit_rate', 0) < historical_metrics['hit_rate'] - self.thresholds['accuracy_decline']):
                
                triggers.append({
                    'type': 'accuracy_decline',
                    'current_value': current_metrics['hit_rate'],
                    'threshold': historical_metrics['hit_rate'] - self.thresholds['accuracy_decline'],
                    'severity': 'HIGH' if current_metrics['hit_rate'] < 0.5 else 'MEDIUM'
                })
            
            # Check Sharpe ratio decline
            if (historical_metrics.get('sharpe_ratio', 0) > 0 and 
                current_metrics.get('sharpe_ratio', 0) < historical_metrics['sharpe_ratio'] * (1 - self.thresholds['sharpe_decline'])):
                
                triggers.append({
                    'type': 'sharpe_decline',
                    'current_value': current_metrics['sharpe_ratio'],
                    'threshold': historical_metrics['sharpe_ratio'] * (1 - self.thresholds['sharpe_decline']),
                    'severity': 'HIGH' if current_metrics['sharpe_ratio'] < 0 else 'MEDIUM'
                })
            
            # Check maximum drawdown breach
            if current_metrics.get('max_drawdown', 0) < -self.thresholds['max_drawdown']:
                triggers.append({
                    'type': 'max_drawdown_breach',
                    'current_value': current_metrics['max_drawdown'],
                    'threshold': -self.thresholds['max_drawdown'],
                    'severity': 'CRITICAL'
                })
            
            # Log triggers to database
            if triggers:
                self._log_retraining_triggers(model_name, triggers)
            
            return triggers
            
        except Exception as e:
            logger.error(f"Error checking retraining triggers for {model_name}: {e}")
            return []
    
    def _log_retraining_triggers(self, model_name: str, triggers: List[Dict]):
        """Log retraining triggers to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for trigger in triggers:
                cursor.execute('''
                    INSERT INTO retraining_triggers 
                    (model_name, trigger_type, trigger_date, trigger_value, threshold_breached, notes)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    model_name,
                    trigger['type'],
                    datetime.now().date(),
                    trigger['current_value'],
                    trigger['threshold'],
                    f"Severity: {trigger['severity']}"
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Logged {len(triggers)} retraining triggers for {model_name}")
            
        except Exception as e:
            logger.error(f"Error logging triggers: {e}")
    
    def analyze_model_correlations(self, days_back: int = 60) -> pd.DataFrame:
        """
        Analyze correlations between different models' predictions
        
        Args:
            days_back: Number of days to analyze
            
        Returns:
            DataFrame with correlation matrix
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get prediction data for all models
            query = '''
                SELECT model_name, prediction_date, predicted_price, current_price,
                       (predicted_price - current_price) / current_price as predicted_return
                FROM model_predictions
                WHERE prediction_date >= date('now', '-{} days')
                AND predicted_price IS NOT NULL
                ORDER BY model_name, prediction_date
            '''.format(days_back)
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                logger.warning("No prediction data found for correlation analysis")
                return pd.DataFrame()
            
            # Pivot to get models as columns
            pivot_df = df.pivot_table(
                index='prediction_date', 
                columns='model_name', 
                values='predicted_return',
                aggfunc='mean'
            )
            
            # Calculate correlation matrix
            correlation_matrix = pivot_df.corr()
            
            # Save correlations to database
            self._save_correlations_to_db(correlation_matrix, days_back)
            
            # Identify highly correlated models
            high_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > self.thresholds['correlation_threshold']:
                        high_correlations.append({
                            'model_1': correlation_matrix.columns[i],
                            'model_2': correlation_matrix.columns[j],
                            'correlation': corr_value
                        })
            
            if high_correlations:
                logger.warning(f"Found {len(high_correlations)} highly correlated model pairs")
                for pair in high_correlations:
                    logger.warning(f"{pair['model_1']} <-> {pair['model_2']}: {pair['correlation']:.3f}")
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Error analyzing model correlations: {e}")
            return pd.DataFrame()
    
    def _save_correlations_to_db(self, correlation_matrix: pd.DataFrame, period_days: int):
        """Save correlation results to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Clear old correlations for this period
            cursor.execute('''
                DELETE FROM model_correlations 
                WHERE calculation_date = ? AND period_days = ?
            ''', (datetime.now().date(), period_days))
            
            # Insert new correlations
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    model_1 = correlation_matrix.columns[i]
                    model_2 = correlation_matrix.columns[j]
                    corr_value = correlation_matrix.iloc[i, j]
                    
                    if not pd.isna(corr_value):
                        cursor.execute('''
                            INSERT INTO model_correlations
                            (model_1, model_2, correlation_coefficient, calculation_date, period_days)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (model_1, model_2, corr_value, datetime.now().date(), period_days))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving correlations: {e}")
    
    def generate_performance_report(self) -> str:
        """
        Generate a comprehensive performance report for all models
        
        Returns:
            Path to the generated report
        """
        try:
            report_content = []
            report_content.append("# Trading Models Performance Report")
            report_content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_content.append("")
            
            # Overall system status
            report_content.append("## System Status Overview")
            report_content.append("")
            
            all_triggers = []
            model_summaries = []
            
            for model_name in self.models.keys():
                # Get performance metrics
                metrics = self.calculate_performance_metrics(model_name)
                
                if metrics:
                    model_summaries.append({
                        'model': model_name,
                        'asset': self.models[model_name]['asset'],
                        'hit_rate': metrics.get('hit_rate', 0),
                        'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                        'max_drawdown': metrics.get('max_drawdown', 0),
                        'num_predictions': metrics.get('num_predictions', 0)
                    })
                
                # Check triggers
                triggers = self.check_retraining_triggers(model_name)
                if triggers:
                    all_triggers.extend([(model_name, t) for t in triggers])
            
            # Create summary table
            if model_summaries:
                report_content.append("| Model | Asset | Hit Rate | Sharpe Ratio | Max Drawdown | Predictions |")
                report_content.append("|-------|-------|----------|--------------|--------------|-------------|")
                
                for summary in model_summaries:
                    report_content.append(
                        f"| {summary['model']} | {summary['asset']} | "
                        f"{summary['hit_rate']:.2%} | {summary['sharpe_ratio']:.3f} | "
                        f"{summary['max_drawdown']:.2%} | {summary['num_predictions']} |"
                    )
                report_content.append("")
            
            # Retraining alerts
            if all_triggers:
                report_content.append("## ⚠️ Retraining Alerts")
                report_content.append("")
                
                for model_name, trigger in all_triggers:
                    severity_emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡"}.get(trigger['severity'], "🔵")
                    report_content.append(
                        f"- {severity_emoji} **{model_name}**: {trigger['type']} "
                        f"(Current: {trigger['current_value']:.3f}, Threshold: {trigger['threshold']:.3f})"
                    )
                report_content.append("")
            
            # Correlation analysis
            correlation_matrix = self.analyze_model_correlations()
            if not correlation_matrix.empty:
                report_content.append("## Model Correlation Analysis")
                report_content.append("")
                report_content.append("High correlations (>0.8) may indicate redundant models:")
                report_content.append("")
                
                # Find high correlations
                high_corr_found = False
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        corr_value = correlation_matrix.iloc[i, j]
                        if abs(corr_value) > 0.8:
                            report_content.append(
                                f"- {correlation_matrix.columns[i]} ↔ {correlation_matrix.columns[j]}: {corr_value:.3f}"
                            )
                            high_corr_found = True
                
                if not high_corr_found:
                    report_content.append("- No high correlations detected ✅")
                
                report_content.append("")
            
            # Recommendations
            report_content.append("## Recommendations")
            report_content.append("")
            
            if all_triggers:
                critical_triggers = [t for _, t in all_triggers if t['severity'] == 'CRITICAL']
                if critical_triggers:
                    report_content.append("### Immediate Actions Required:")
                    for model_name, trigger in all_triggers:
                        if trigger['severity'] == 'CRITICAL':
                            report_content.append(f"- **{model_name}**: Immediate retraining required due to {trigger['type']}")
                    report_content.append("")
            
            # Performance optimization suggestions
            poor_performers = [s for s in model_summaries if s['hit_rate'] < 0.55]
            if poor_performers:
                report_content.append("### Models Requiring Attention:")
                for model in poor_performers:
                    report_content.append(f"- **{model['model']}**: Hit rate {model['hit_rate']:.2%} - consider feature engineering or retraining")
                report_content.append("")
            
            # Save report
            report_path = f"C:\\Users\\rrose\\performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(report_path, 'w') as f:
                f.write('\n'.join(report_content))
            
            logger.info(f"Performance report generated: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return ""
    
    def optimize_database(self):
        """Optimize database performance through indexing and cleanup"""
        try:
            # Create backup first
            import shutil
            shutil.copy2(self.db_path, self.backup_db_path)
            logger.info(f"Database backup created: {self.backup_db_path}")
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Analyze database
            cursor.execute("ANALYZE")
            
            # Vacuum database to reclaim space
            cursor.execute("VACUUM")
            
            # Update statistics
            cursor.execute("PRAGMA optimize")
            
            # Clean up old data (keep last 2 years)
            cutoff_date = (datetime.now() - timedelta(days=730)).date()
            
            cursor.execute("DELETE FROM model_predictions WHERE prediction_date < ?", (cutoff_date,))
            deleted_predictions = cursor.rowcount
            
            cursor.execute("DELETE FROM model_performance_metrics WHERE calculation_date < ?", (cutoff_date,))
            deleted_metrics = cursor.rowcount
            
            cursor.execute("DELETE FROM retraining_triggers WHERE trigger_date < ?", (cutoff_date,))
            deleted_triggers = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            logger.info("Database optimization completed:")
            logger.info(f"  - Deleted {deleted_predictions} old predictions")
            logger.info(f"  - Deleted {deleted_metrics} old metrics")
            logger.info(f"  - Deleted {deleted_triggers} old triggers")
            
        except Exception as e:
            logger.error(f"Error optimizing database: {e}")
            # Restore backup if error occurred
            if os.path.exists(self.backup_db_path):
                shutil.copy2(self.backup_db_path, self.db_path)
                logger.info("Database restored from backup due to error")

def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(description='Trading Models Performance Monitoring System')
    parser.add_argument('--monitor', action='store_true', help='Run performance monitoring')
    parser.add_argument('--analyze-correlations', action='store_true', help='Analyze model correlations')
    parser.add_argument('--optimize-database', action='store_true', help='Optimize database performance')
    parser.add_argument('--generate-report', action='store_true', help='Generate performance report')
    parser.add_argument('--db-path', type=str, help='Path to database file')
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = PerformanceMonitor(db_path=args.db_path)
    
    if args.monitor:
        logger.info("Running performance monitoring...")
        
        # Check all models for retraining triggers
        for model_name in monitor.models.keys():
            triggers = monitor.check_retraining_triggers(model_name)
            if triggers:
                logger.warning(f"Retraining triggers detected for {model_name}:")
                for trigger in triggers:
                    logger.warning(f"  - {trigger['type']}: {trigger['current_value']:.3f} (threshold: {trigger['threshold']:.3f})")
    
    if args.analyze_correlations:
        logger.info("Analyzing model correlations...")
        correlation_matrix = monitor.analyze_model_correlations()
        if not correlation_matrix.empty:
            print("\nModel Correlation Matrix:")
            print(correlation_matrix.round(3))
    
    if args.optimize_database:
        logger.info("Optimizing database...")
        monitor.optimize_database()
    
    if args.generate_report:
        logger.info("Generating performance report...")
        report_path = monitor.generate_performance_report()
        if report_path:
            print(f"\nPerformance report generated: {report_path}")
    
    # If no arguments provided, show help
    if not any([args.monitor, args.analyze_correlations, args.optimize_database, args.generate_report]):
        parser.print_help()
        print("\nExample usage:")
        print("  python performance_monitoring_system.py --monitor --generate-report")
        print("  python performance_monitoring_system.py --analyze-correlations")
        print("  python performance_monitoring_system.py --optimize-database")

if __name__ == '__main__':
    main()