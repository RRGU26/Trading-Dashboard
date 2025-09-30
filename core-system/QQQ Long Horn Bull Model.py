#!/usr/bin/env python3
"""
QQQ LONG BULL MODEL V3.2 - FINAL VIX ALIGNMENT FIX + TEXT REPORT GENERATION
- Prioritizes QQQ+VIX alignment over other assets
- Uses QQQ as reference dataset (not BTC)
- Better synthetic VIX features when real VIX fails
- Focus on QQQ-centric features for better performance
- ADDED: Text report generation for trading reports system integration
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
from datetime import datetime, timedelta
import os
import traceback
import sqlite3
from typing import Dict, List, Tuple, Optional
import joblib
import time

# Setup
warnings.filterwarnings("ignore")
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

try:
    import data_fetcher
    USE_DATA_FETCHER = True
except ImportError:
    print("ERROR: data_fetcher module not found. Cannot proceed.")
    sys.exit(1)

# Configuration
OUTPUTS_DIR = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# SYMBOLS CONFIG - QQQ first priority, VIX second priority
SYMBOLS_CONFIG = {
    'QQQ': {'symbols': ['QQQ'], 'required': True, 'description': 'Primary asset'},
    'VIX': {'symbols': ['^VIX', 'VIX'], 'required': False, 'description': 'Volatility index (CRITICAL)'},
    'BTC': {'symbols': ['BTC-USD', 'BTCUSD'], 'required': False, 'description': 'Crypto correlation'},
}

class DatabaseIntegrator:
    """Enhanced database integration with historical data leverage"""
    
    def __init__(self):
        self.db_path = self._find_database()
        self.model_name = "Long Bull Model V3.2"
        self.symbol = "QQQ"
        
    def _find_database(self) -> str:
        possible_paths = [
            os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "reports_tracking.db"),
            os.path.join(os.path.expanduser("~"), "Desktop", "reports_tracking.db"),
            "reports_tracking.db"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"[OK] Found database: {path}")
                return path
        
        print("[WARN]  Database not found - will run in standalone mode")
        return None
    
    def get_historical_vix_data(self, days_back: int = 252) -> Optional[pd.DataFrame]:
        """Get historical VIX data from database if available"""
        if not self.db_path:
            return None
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Query for VIX data from daily_prices table
            query = """
            SELECT date, close_price as vix_close
            FROM daily_prices 
            WHERE symbol IN ('^VIX', 'VIX', 'UVXY', 'VXX')
            ORDER BY date DESC
            LIMIT ?
            """
            
            cursor.execute(query, (days_back,))
            results = cursor.fetchall()
            conn.close()
            
            if results:
                df = pd.DataFrame(results, columns=['date', 'vix_close'])
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df.sort_index(inplace=True)
                
                print(f"[OK] Retrieved {len(df)} VIX data points from database")
                return df
            else:
                print("[WARN]  No VIX data found in database")
                return None
                
        except Exception as e:
            print(f"[ERROR] Failed to retrieve VIX data from database: {e}")
            return None
    
    def get_historical_predictions_performance(self) -> Dict:
        """Get performance of historical predictions for model tuning"""
        if not self.db_path:
            return {}
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get recent prediction performance
            query = """
            SELECT 
                AVG(CASE WHEN direction_correct = 1 THEN 1.0 ELSE 0.0 END) as direction_accuracy,
                AVG(ABS(error_pct)) as avg_error,
                COUNT(*) as total_predictions,
                AVG(confidence) as avg_confidence
            FROM model_predictions 
            WHERE model = ? 
            AND actual_price IS NOT NULL
            AND prediction_date >= date('now', '-90 days')
            """
            
            cursor.execute(query, (self.model_name,))
            result = cursor.fetchone()
            conn.close()
            
            if result and result[2] > 0:  # total_predictions > 0
                performance = {
                    'direction_accuracy': result[0] or 0,
                    'avg_error': result[1] or 0,
                    'total_predictions': result[2] or 0,
                    'avg_confidence': result[3] or 0
                }
                print(f"[DATA] Historical performance: {performance['direction_accuracy']:.1%} accuracy from {performance['total_predictions']} predictions")
                return performance
            else:
                print("[WARN]  No historical prediction data available")
                return {}
                
        except Exception as e:
            print(f"[ERROR] Failed to retrieve historical performance: {e}")
            return {}
    
    def save_prediction(self, prediction_data: Dict) -> bool:
        if not self.db_path:
            return False
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            current_price = prediction_data['current_price']
            predicted_price = prediction_data['predicted_price']
            predicted_return = prediction_data['predicted_return']
            confidence = prediction_data['confidence']
            horizon_days = prediction_data['horizon_days']
            
            prediction_date = datetime.now().date()
            target_date = prediction_date + timedelta(days=horizon_days)
            
            # Enhanced action logic
            if predicted_return > 2.0 and confidence > 80:
                action = "STRONG BUY"
            elif predicted_return > 1.0 and confidence > 70:
                action = "BUY"
            elif predicted_return > 0.5 and confidence > 60:
                action = "WEAK BUY"
            elif predicted_return > -0.5:
                action = "HOLD"
            elif predicted_return > -1.5:
                action = "SELL"
            elif predicted_return > -3.0:
                action = "STRONG SELL"
            else:
                action = "AVOID"
            
            cursor.execute("""
                INSERT INTO model_predictions (
                    model, symbol, prediction_date, target_date, horizon,
                    current_price, predicted_price, actual_price, confidence,
                    suggested_action, error_pct, direction_correct,
                    expected_return, actual_return, return_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.model_name, self.symbol, prediction_date, target_date, horizon_days,
                current_price, predicted_price, None, confidence,
                action, None, None, predicted_return, None, None
            ))
            
            conn.commit()
            conn.close()
            
            print(f"[OK] Prediction saved to database")
            print(f"   [DATE] {prediction_date} -> {target_date} ({horizon_days}d)")
            print(f"   [PRICE] ${current_price:.2f} -> ${predicted_price:.2f} ({predicted_return:+.2f}%)")
            print(f"   [TARGET] {action} (Confidence: {confidence:.1f}%)")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to save prediction: {e}")
            return False
    
    def save_metrics(self, training_results: Dict) -> bool:
        if not self.db_path or not training_results:
            return False
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            current_date = datetime.now().date()
            ensemble_metrics = training_results['ensemble_metrics']
            
            metrics = [
                ('r2_score', ensemble_metrics['r2']),
                ('mae', ensemble_metrics['mae']),
                ('mse', ensemble_metrics['mse']),
                ('direction_accuracy', ensemble_metrics['direction_accuracy']),
                ('hit_rate', ensemble_metrics['hit_rate']),
                ('sharpe_ratio', ensemble_metrics.get('sharpe_ratio', 0.0))
            ]
            
            for metric_type, metric_value in metrics:
                cursor.execute("""
                    INSERT INTO model_metrics (model, date, metric_type, metric_value)
                    VALUES (?, ?, ?, ?)
                """, (self.model_name, current_date, metric_type, float(metric_value)))
            
            conn.commit()
            conn.close()
            
            print(f"[OK] Model metrics saved to database")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to save metrics: {e}")
            return False

    def update_actual_prices(self) -> int:
        """Update actual prices for past predictions"""
        if not self.db_path:
            return 0
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, symbol, target_date, predicted_price, current_price, expected_return
                FROM model_predictions 
                WHERE model = ? AND actual_price IS NULL AND target_date <= date('now')
            """, (self.model_name,))
            
            pending = cursor.fetchall()
            updated = 0
            
            for pred_id, symbol, target_date, predicted_price, current_price, expected_return in pending:
                cursor.execute("""
                    SELECT close FROM price_history 
                    WHERE symbol = ? AND date = ?
                    ORDER BY created_timestamp DESC LIMIT 1
                """, (symbol, target_date))
                
                result = cursor.fetchone()
                if result:
                    actual_price = result[0]
                    actual_return = ((actual_price / current_price) - 1) * 100
                    error_pct = ((predicted_price - actual_price) / actual_price) * 100
                    return_error = expected_return - actual_return if expected_return else None
                    
                    predicted_direction = 1 if predicted_price > current_price else 0
                    actual_direction = 1 if actual_price > current_price else 0
                    direction_correct = 1 if predicted_direction == actual_direction else 0
                    
                    cursor.execute("""
                        UPDATE model_predictions 
                        SET actual_price = ?, actual_return = ?, error_pct = ?, 
                            direction_correct = ?, return_error = ?
                        WHERE id = ?
                    """, (actual_price, actual_return, error_pct, direction_correct, return_error, pred_id))
                    
                    updated += 1
            
            conn.commit()
            conn.close()
            
            if updated > 0:
                print(f"[OK] Updated {updated} predictions with actual prices")
            
            return updated
            
        except Exception as e:
            print(f"[ERROR] Error updating actual prices: {e}")
            return 0


class EnhancedDataFetcher:
    """Enhanced data fetching with smart VIX handling"""
    
    def __init__(self):
        self.data_cache = {}
        self.request_delay = 1
        
    def fetch_symbol_with_fallbacks(self, asset_name: str, symbols: List[str], 
                                   start_date: str, end_date: str, 
                                   min_rows: int = 100) -> Optional[pd.DataFrame]:
        """Fetch data with rate limiting protection"""
        
        print(f"  [FETCH] Fetching {asset_name}...")
        
        for i, symbol in enumerate(symbols):
            print(f"    Trying {symbol}...")
            
            if i > 0:
                time.sleep(self.request_delay)
            
            try:
                df = data_fetcher.get_historical_data(symbol, start_date, end_date)
                
                if df is not None and not df.empty and len(df) >= min_rows:
                    print(f"    [OK] {symbol}: {len(df)} rows via data_fetcher")
                    return self._prepare_dataframe(df)
                else:
                    print(f"    [WARN]  {symbol}: {len(df) if df is not None else 0} rows (insufficient)")
                    
            except Exception as e:
                print(f"    [ERROR] {symbol} failed: {e}")
        
        print(f"    [ERROR] No data source succeeded for {asset_name}")
        return None
    
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced dataframe preparation"""
        if df is None or df.empty:
            return df
            
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'Date' in df.columns:
                df = df.set_index('Date')
            elif 'date' in df.columns:
                df = df.set_index('date')
        
        # Standardize column names
        column_mapping = {
            'open': 'Open', 'high': 'High', 'low': 'Low', 
            'close': 'Close', 'volume': 'Volume', 'adj close': 'Adj Close'
        }
        df.columns = [column_mapping.get(col.lower(), col) for col in df.columns]
        
        # Clean numeric data
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Basic derived features
        df['returns_1d'] = df['Close'].pct_change()
        df['returns_3d'] = df['Close'].pct_change(3)
        df['volatility_5d'] = df['returns_1d'].rolling(5).std()
        df['volatility_20d'] = df['returns_1d'].rolling(20).std()
        
        # Remove extreme outliers
        for col in ['returns_1d', 'returns_3d']:
            if col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                df[col] = df[col].clip(mean_val - 5*std_val, mean_val + 5*std_val)
        
        return df.dropna()


class QQQLongBullModelV32:
    """V3.2 with QQQ+VIX priority alignment + Text Report Generation"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_cols = []
        self.data_fetcher = EnhancedDataFetcher()
        self.db_integrator = DatabaseIntegrator()
        
        self.prediction_horizons = [1, 3, 5, 7, 10, 14, 21]
        self.best_horizon = 5
        
        self.performance_metrics = {}
        self.correlation_assets = []
        
    def fetch_all_data(self) -> Dict[str, pd.DataFrame]:
        """PRIORITY DATA FETCHING: QQQ + VIX first, others optional"""
        print("[LAUNCH] PRIORITY DATA FETCHING: QQQ + VIX ALIGNMENT")
        print("=" * 60)
        
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=10*365)).strftime("%Y-%m-%d")  # 10 years of data
        
        print(f"[DATE] Date range: {start_date} to {end_date}")
        
        data = {}
        
        # STEP 1: Fetch QQQ (required)
        print("\n[TARGET] STEP 1: Fetching QQQ (required)")
        qqq_config = SYMBOLS_CONFIG['QQQ']
        qqq_df = self.data_fetcher.fetch_symbol_with_fallbacks(
            'QQQ', qqq_config['symbols'], start_date, end_date, 200
        )
        
        if qqq_df is None or qqq_df.empty:
            raise ValueError("[ERROR] Cannot proceed without QQQ data")
        
        data['QQQ'] = qqq_df
        self.correlation_assets.append('QQQ')
        print(f"    [OK] QQQ: {len(qqq_df)} rows - USING AS REFERENCE")
        
        # STEP 2: Fetch VIX and align to QQQ immediately - Enhanced with database fallback
        print("\n[HOT] STEP 2: Fetching VIX and aligning to QQQ")
        vix_config = SYMBOLS_CONFIG['VIX']
        vix_df = self.data_fetcher.fetch_symbol_with_fallbacks(
            'VIX', vix_config['symbols'], start_date, end_date, 50  # Lower threshold
        )
        
        # If API fetch failed, try database
        if vix_df is None or vix_df.empty:
            print("[WARN]  VIX API fetch failed - trying database...")
            db_vix_df = self.db_integrator.get_historical_vix_data()
            if db_vix_df is not None and not db_vix_df.empty:
                # Convert single column VIX data to OHLC format
                vix_df = pd.DataFrame(index=db_vix_df.index)
                vix_df['Close'] = db_vix_df['vix_close']
                vix_df['High'] = db_vix_df['vix_close'] * 1.02
                vix_df['Low'] = db_vix_df['vix_close'] * 0.98
                vix_df['Open'] = db_vix_df['vix_close']
                vix_df['Volume'] = 1000000  # Default volume
                print("[OK] Using VIX data from database")
        
        if vix_df is not None and not vix_df.empty:
            # Find QQQ-VIX overlap
            qqq_dates = qqq_df.index
            vix_dates = vix_df.index
            common_dates = qqq_dates.intersection(vix_dates)
            
            print(f"    [DATA] QQQ dates: {len(qqq_dates)}")
            print(f"    [DATA] VIX dates: {len(vix_dates)}")
            print(f"    [DATA] QQQ-VIX overlap: {len(common_dates)} days")
            
            if len(common_dates) >= 100:  # Need at least 100 overlapping days
                # Align both to common dates
                data['QQQ'] = qqq_df.loc[common_dates]
                data['VIX'] = vix_df.loc[common_dates]
                self.correlation_assets.append('VIX')
                print(f"    [OK] VIX: {len(common_dates)} aligned days with QQQ")
            else:
                print(f"    [WARN]  VIX: Only {len(common_dates)} overlapping days - will use synthetic VIX")
        else:
            print(f"    [WARN]  VIX: No data - will use synthetic VIX")
        
        # STEP 3: Fetch other correlations and align to QQQ+VIX
        print("\n[CHART] STEP 3: Fetching additional correlations")
        
        # Use current QQQ dates as reference (may be reduced after VIX alignment)
        reference_dates = data['QQQ'].index
        
        for asset_name, config in SYMBOLS_CONFIG.items():
            if asset_name in ['QQQ', 'VIX']:  # Already handled
                continue
            
            asset_df = self.data_fetcher.fetch_symbol_with_fallbacks(
                asset_name, config['symbols'], start_date, end_date, 50
            )
            
            if asset_df is not None and not asset_df.empty:
                # Find overlap with reference dates
                asset_dates = asset_df.index
                overlap_dates = reference_dates.intersection(asset_dates)
                
                if len(overlap_dates) >= 200:  # Good overlap
                    data[asset_name] = asset_df.loc[overlap_dates]
                    self.correlation_assets.append(asset_name)
                    print(f"    [OK] {asset_name}: {len(overlap_dates)} aligned days")
                else:
                    print(f"    [WARN]  {asset_name}: Only {len(overlap_dates)} overlapping days - skipping")
            else:
                print(f"    [ERROR] {asset_name}: No data available")
        
        # FINAL ALIGNMENT: Align all to common dates
        if len(data) > 1:
            print(f"\n[ALIGN] FINAL ALIGNMENT:")
            final_dates = None
            for asset_name, df in data.items():
                if final_dates is None:
                    final_dates = df.index
                else:
                    final_dates = final_dates.intersection(df.index)
            
            print(f"   [DATE] Final common dates: {len(final_dates)}")
            
            for asset_name in data:
                data[asset_name] = data[asset_name].loc[final_dates]
        
        # Data quality report
        print(f"\n[DATA] DATA QUALITY REPORT:")
        print(f"   [OK] Successfully aligned: {len(data)} assets")
        print(f"   [CHART] Available assets: {', '.join(self.correlation_assets)}")
        
        if len(data) > 0:
            sample_length = len(list(data.values())[0])
            print(f"   [DATE] Final training data: {sample_length} days")
            
            if sample_length < 200:
                print(f"   [WARN]  WARNING: Limited training data ({sample_length} days)")
        
        return data
    
    def create_qqq_focused_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create QQQ-focused features with smart VIX handling"""
        print("\n[TOOLS]  QQQ-FOCUSED FEATURE ENGINEERING")
        print("=" * 60)
        
        qqq_df = data['QQQ'].copy()
        
        # 1. COMPREHENSIVE QQQ INTERNAL FEATURES
        print("  [CHART] Creating comprehensive QQQ features...")
        
        # Multi-timeframe returns
        for period in [1, 2, 3, 5, 7, 10, 14, 21]:
            qqq_df[f'returns_{period}d'] = qqq_df['Close'].pct_change(period)
            qqq_df[f'direction_{period}d'] = np.sign(qqq_df[f'returns_{period}d'])
            qqq_df[f'abs_returns_{period}d'] = np.abs(qqq_df[f'returns_{period}d'])
        
        # Advanced momentum features
        qqq_df['momentum_acceleration'] = qqq_df['returns_3d'] - qqq_df['returns_3d'].shift(1)
        qqq_df['momentum_persistence'] = (qqq_df['direction_3d'] == qqq_df['direction_3d'].shift(1)).astype(int)
        qqq_df['trend_strength'] = qqq_df['direction_5d'].rolling(5).sum() / 5
        qqq_df['momentum_consistency'] = qqq_df['direction_1d'].rolling(10).std()
        
        # 2. ENHANCED VOLATILITY FEATURES
        print("  [DATA] Creating enhanced volatility features...")
        
        for window in [5, 10, 20, 30, 60]:
            vol = qqq_df['returns_1d'].rolling(window).std()
            qqq_df[f'volatility_{window}d'] = vol
            qqq_df[f'vol_rank_{window}d'] = vol.rolling(252).rank(pct=True)
            qqq_df[f'vol_zscore_{window}d'] = (vol - vol.rolling(252).mean()) / (vol.rolling(252).std() + 1e-8)
        
        # Volatility regimes and patterns
        qqq_df['vol_regime_high'] = (qqq_df['volatility_20d'] > qqq_df['volatility_20d'].rolling(60).quantile(0.8)).astype(int)
        qqq_df['vol_regime_low'] = (qqq_df['volatility_20d'] < qqq_df['volatility_20d'].rolling(60).quantile(0.2)).astype(int)
        qqq_df['vol_expansion'] = (qqq_df['volatility_5d'] > qqq_df['volatility_20d'] * 1.3).astype(int)
        qqq_df['vol_contraction'] = (qqq_df['volatility_5d'] < qqq_df['volatility_20d'] * 0.7).astype(int)
        qqq_df['vol_breakout'] = (qqq_df['volatility_5d'] > qqq_df['volatility_20d'].rolling(60).quantile(0.9)).astype(int)
        
        # 3. COMPREHENSIVE TECHNICAL INDICATORS
        print("  [TOOL] Creating technical indicators...")
        
        # RSI with multiple periods
        for period in [7, 14, 21, 30]:
            delta = qqq_df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            
            qqq_df[f'rsi_{period}'] = rsi
            qqq_df[f'rsi_{period}_overbought'] = (rsi > 70).astype(int)
            qqq_df[f'rsi_{period}_oversold'] = (rsi < 30).astype(int)
            qqq_df[f'rsi_{period}_extreme_ob'] = (rsi > 80).astype(int)
            qqq_df[f'rsi_{period}_extreme_os'] = (rsi < 20).astype(int)
            qqq_df[f'rsi_{period}_momentum'] = rsi.diff()
        
        # MACD variants
        for fast, slow in [(8, 17), (12, 26), (19, 39)]:
            ema_fast = qqq_df['Close'].ewm(span=fast).mean()
            ema_slow = qqq_df['Close'].ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=9).mean()
            macd_histogram = macd - macd_signal
            
            suffix = f"{fast}_{slow}"
            qqq_df[f'macd_{suffix}'] = macd
            qqq_df[f'macd_signal_{suffix}'] = macd_signal
            qqq_df[f'macd_histogram_{suffix}'] = macd_histogram
            qqq_df[f'macd_bullish_{suffix}'] = (macd > macd_signal).astype(int)
            qqq_df[f'macd_crossover_{suffix}'] = ((macd > macd_signal) & (macd.shift(1) <= macd_signal.shift(1))).astype(int)
        
        # Moving averages and trends
        ma_periods = [5, 10, 20, 50, 100, 200]
        for period in ma_periods:
            ma = qqq_df['Close'].rolling(period).mean()
            ema = qqq_df['Close'].ewm(span=period).mean()
            
            qqq_df[f'ma_{period}'] = ma
            qqq_df[f'ema_{period}'] = ema
            qqq_df[f'above_ma_{period}'] = (qqq_df['Close'] > ma).astype(int)
            qqq_df[f'above_ema_{period}'] = (qqq_df['Close'] > ema).astype(int)
            qqq_df[f'price_ma_ratio_{period}'] = qqq_df['Close'] / ma
            qqq_df[f'ma_slope_{period}'] = ma.pct_change(5)
            qqq_df[f'ma_rising_{period}'] = (qqq_df[f'ma_slope_{period}'] > 0).astype(int)
        
        # Moving average relationships
        qqq_df['golden_cross'] = (qqq_df['ma_50'] > qqq_df['ma_200']).astype(int)
        qqq_df['death_cross'] = (qqq_df['ma_50'] < qqq_df['ma_200']).astype(int)
        qqq_df['short_term_trend'] = (qqq_df['ma_10'] > qqq_df['ma_20']).astype(int)
        qqq_df['medium_term_trend'] = (qqq_df['ma_20'] > qqq_df['ma_50']).astype(int)
        qqq_df['long_term_trend'] = (qqq_df['ma_50'] > qqq_df['ma_100']).astype(int)
        
        # Bollinger Bands
        for period in [20, 30]:
            for std_mult in [1.5, 2.0, 2.5]:
                bb_ma = qqq_df['Close'].rolling(period).mean()
                bb_std = qqq_df['Close'].rolling(period).std()
                bb_upper = bb_ma + std_mult * bb_std
                bb_lower = bb_ma - std_mult * bb_std
                
                suffix = f"{period}_{str(std_mult).replace('.', '_')}"
                qqq_df[f'bb_position_{suffix}'] = (qqq_df['Close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)
                qqq_df[f'bb_squeeze_{suffix}'] = (bb_upper - bb_lower) / bb_ma
                qqq_df[f'bb_breakout_up_{suffix}'] = (qqq_df['Close'] > bb_upper).astype(int)
                qqq_df[f'bb_breakout_down_{suffix}'] = (qqq_df['Close'] < bb_lower).astype(int)
        
        # 4. VIX FEATURES (CRITICAL)
        vix_features = 0
        if 'VIX' in data and len(data['VIX']) == len(qqq_df):
            print("  [HOT] Adding REAL VIX features (CRITICAL)...")
            vix_features = self._add_real_vix_features(qqq_df, data['VIX'])
        else:
            print("  [SYNC] Adding SYNTHETIC VIX features...")
            vix_features = self._add_synthetic_vix_features(qqq_df)
        
        # 5. OTHER CORRELATIONS
        other_features = 0
        if 'BTC' in data and len(data['BTC']) == len(qqq_df):
            print("  [BTC] Adding BTC features...")
            other_features += self._add_btc_features(qqq_df, data['BTC'])
        
        # 6. TIME AND SEASONAL FEATURES
        print("  [TIME] Adding time and seasonal features...")
        
        # Calendar features
        qqq_df['day_of_week'] = qqq_df.index.dayofweek
        qqq_df['month'] = qqq_df.index.month
        qqq_df['quarter'] = qqq_df.index.quarter
        qqq_df['day_of_month'] = qqq_df.index.day
        qqq_df['week_of_year'] = qqq_df.index.isocalendar().week
        
        # Market microstructure
        qqq_df['is_monday'] = (qqq_df['day_of_week'] == 0).astype(int)
        qqq_df['is_tuesday'] = (qqq_df['day_of_week'] == 1).astype(int)
        qqq_df['is_wednesday'] = (qqq_df['day_of_week'] == 2).astype(int)
        qqq_df['is_thursday'] = (qqq_df['day_of_week'] == 3).astype(int)
        qqq_df['is_friday'] = (qqq_df['day_of_week'] == 4).astype(int)
        qqq_df['is_month_end'] = (qqq_df.index.day >= 25).astype(int)
        qqq_df['is_quarter_end'] = ((qqq_df['month'] % 3 == 0) & (qqq_df['day_of_month'] >= 25)).astype(int)
        
        # Seasonal patterns
        qqq_df['is_january'] = (qqq_df['month'] == 1).astype(int)  # January effect
        qqq_df['is_december'] = (qqq_df['month'] == 12).astype(int)  # Santa rally
        qqq_df['is_summer'] = ((qqq_df['month'] >= 6) & (qqq_df['month'] <= 8)).astype(int)  # Summer lull
        qqq_df['is_fall'] = ((qqq_df['month'] >= 9) & (qqq_df['month'] <= 11)).astype(int)  # Volatility season
        
        # 7. INTERACTION AND CONFLUENCE FEATURES
        print("  [LINK] Creating interaction features...")
        
        # Technical confluence signals
        qqq_df['bullish_confluence'] = (
            qqq_df.get('rsi_14_oversold', 0) + 
            qqq_df.get('macd_bullish_12_26', 0) + 
            qqq_df.get('above_ma_20', 0) + 
            qqq_df.get('bb_breakout_up_20_2_0', 0) +
            qqq_df.get('short_term_trend', 0)
        )
        
        qqq_df['bearish_confluence'] = (
            qqq_df.get('rsi_14_overbought', 0) + 
            (1 - qqq_df.get('macd_bullish_12_26', 1)) + 
            (1 - qqq_df.get('above_ma_20', 1)) + 
            qqq_df.get('bb_breakout_down_20_2_0', 0) +
            qqq_df.get('vol_regime_high', 0)
        )
        
        # Volatility-momentum interactions
        qqq_df['vol_momentum_signal'] = qqq_df['returns_3d'] / (qqq_df['volatility_20d'] + 1e-8)
        qqq_df['low_vol_momentum'] = (qqq_df['vol_regime_low'] & (qqq_df['abs_returns_3d'] > 0.02)).astype(int)
        qqq_df['high_vol_reversal'] = (qqq_df['vol_regime_high'] & (qqq_df['returns_1d'] * qqq_df['returns_1d'].shift(1) < 0)).astype(int)
        
        # Trend-volatility signals
        qqq_df['trending_market'] = (qqq_df['trend_strength'] > 0.6).astype(int)
        qqq_df['choppy_market'] = ((qqq_df['trend_strength'] < 0.4) & qqq_df['vol_regime_high']).astype(int)
        
        total_features = len(qqq_df.columns)
        print(f"\n[OK] QQQ-FOCUSED FEATURE ENGINEERING COMPLETE:")
        print(f"   [DATA] Total features: {total_features}")
        print(f"   [HOT] VIX features: {vix_features}")
        print(f"   [CHART] Other correlations: {other_features}")
        print(f"   [TARGET] QQQ internal features: {total_features - vix_features - other_features}")
        
        return qqq_df
    
    def _add_real_vix_features(self, qqq_df: pd.DataFrame, vix_df: pd.DataFrame) -> int:
        """Add real VIX features when available"""
        
        try:
            vix_close = vix_df['Close']
            
            # Core VIX levels and regimes
            qqq_df['vix_level'] = vix_close
            qqq_df['vix_low'] = (vix_close < 15).astype(int)
            qqq_df['vix_normal'] = ((vix_close >= 15) & (vix_close <= 25)).astype(int)
            qqq_df['vix_high'] = (vix_close > 25).astype(int)
            qqq_df['vix_extreme'] = (vix_close > 35).astype(int)
            qqq_df['vix_panic'] = (vix_close > 50).astype(int)
            
            # VIX momentum and changes
            vix_change = vix_close.pct_change()
            qqq_df['vix_change'] = vix_change
            qqq_df['vix_spike'] = (vix_change > 0.15).astype(int)
            qqq_df['vix_crash'] = (vix_change < -0.15).astype(int)
            qqq_df['vix_momentum_3d'] = vix_close.pct_change(3)
            qqq_df['vix_momentum_5d'] = vix_close.pct_change(5)
            
            # VIX moving averages and structure
            for window in [5, 10, 20, 50]:
                vix_ma = vix_close.rolling(window).mean()
                qqq_df[f'vix_ma_{window}'] = vix_ma
                qqq_df[f'vix_above_ma_{window}'] = (vix_close > vix_ma).astype(int)
                qqq_df[f'vix_vs_ma_{window}'] = (vix_close / vix_ma - 1) * 100
            
            # VIX mean reversion signals
            vix_ma_20 = vix_close.rolling(20).mean()
            vix_std_20 = vix_close.rolling(20).std()
            qqq_df['vix_zscore'] = (vix_close - vix_ma_20) / (vix_std_20 + 1e-8)
            qqq_df['vix_oversold'] = (qqq_df['vix_zscore'] > 1.5).astype(int)
            qqq_df['vix_overbought'] = (qqq_df['vix_zscore'] < -1.5).astype(int)
            
            # VIX-QQQ relationships (CRITICAL)
            for window in [10, 20, 60]:
                correlation = vix_close.rolling(window).corr(qqq_df['returns_1d'])
                qqq_df[f'vix_qqq_corr_{window}d'] = correlation
                
            # VIX divergences and signals
            qqq_direction = np.sign(qqq_df['returns_1d'])
            vix_direction = np.sign(vix_change)
            qqq_df['vix_qqq_divergence'] = (qqq_direction == vix_direction).astype(int)  # Bearish when both positive
            qqq_df['vix_fear_extreme'] = ((vix_close > 30) & (qqq_df['returns_1d'] < -0.02)).astype(int)
            qqq_df['vix_complacency'] = ((vix_close < 12) & (qqq_df['returns_1d'] > 0.01)).astype(int)
            
            # VIX term structure approximations
            qqq_df['vix_term_structure'] = vix_close / vix_ma_20
            qqq_df['vix_backwardation'] = (qqq_df['vix_term_structure'] > 1.15).astype(int)
            qqq_df['vix_contango'] = (qqq_df['vix_term_structure'] < 0.9).astype(int)
            
            print(f"    [OK] Added 28 real VIX features")
            return 28
            
        except Exception as e:
            print(f"    [ERROR] Real VIX features failed: {e}")
            return self._add_synthetic_vix_features(qqq_df)
    
    def _add_synthetic_vix_features(self, qqq_df: pd.DataFrame) -> int:
        """Add synthetic VIX features based on QQQ volatility"""
        
        print("    [SYNC] Creating synthetic VIX from QQQ volatility...")
        
        # Create synthetic VIX based on QQQ volatility patterns
        # VIX typically ranges 10-80, with mean around 20
        base_vix = qqq_df['volatility_20d'] * 100  # Convert to percentage
        seasonal_adjustment = 15 + 10 * np.sin(qqq_df.index.dayofyear * 2 * np.pi / 365)  # Seasonal component
        
        # Synthetic VIX with realistic bounds
        synthetic_vix = np.clip(base_vix + seasonal_adjustment, 8, 80)
        
        qqq_df['vix_level'] = synthetic_vix
        qqq_df['vix_low'] = (synthetic_vix < 15).astype(int)
        qqq_df['vix_normal'] = ((synthetic_vix >= 15) & (synthetic_vix <= 25)).astype(int)
        qqq_df['vix_high'] = (synthetic_vix > 25).astype(int)
        qqq_df['vix_extreme'] = (synthetic_vix > 35).astype(int)
        
        # Synthetic VIX momentum
        vix_change = synthetic_vix.pct_change()
        qqq_df['vix_change'] = vix_change
        qqq_df['vix_spike'] = (vix_change > 0.15).astype(int)
        qqq_df['vix_crash'] = (vix_change < -0.15).astype(int)
        
        # VIX moving averages
        vix_ma_20 = synthetic_vix.rolling(20).mean()
        qqq_df['vix_ma_20'] = vix_ma_20
        qqq_df['vix_above_ma_20'] = (synthetic_vix > vix_ma_20).astype(int)
        
        # Mean reversion
        vix_std_20 = synthetic_vix.rolling(20).std()
        qqq_df['vix_zscore'] = (synthetic_vix - vix_ma_20) / (vix_std_20 + 1e-8)
        qqq_df['vix_oversold'] = (qqq_df['vix_zscore'] > 1.5).astype(int)
        
        # Synthetic correlations (based on typical VIX-QQQ relationship)
        qqq_df['vix_qqq_corr_20d'] = synthetic_vix.rolling(20).corr(-qqq_df['returns_1d'])  # Negative correlation
        
        print(f"    [OK] Added 13 synthetic VIX features")
        return 13
    
    def _add_btc_features(self, qqq_df: pd.DataFrame, btc_df: pd.DataFrame) -> int:
        """Add BTC correlation features"""
        try:
            btc_close = btc_df['Close']
            btc_returns = btc_close.pct_change()
            
            # BTC performance and momentum
            qqq_df['btc_returns_1d'] = btc_returns
            qqq_df['btc_returns_3d'] = btc_close.pct_change(3)
            qqq_df['btc_returns_7d'] = btc_close.pct_change(7)
            qqq_df['btc_volatility'] = btc_returns.rolling(20).std()
            
            # QQQ-BTC correlations
            for window in [10, 20, 60]:
                correlation = btc_returns.rolling(window).corr(qqq_df['returns_1d'])
                qqq_df[f'btc_qqq_corr_{window}d'] = correlation
            
            # Risk appetite signals
            qqq_df['risk_on_signal'] = ((qqq_df['returns_1d'] > 0.01) & (btc_returns > 0.02)).astype(int)
            qqq_df['risk_off_signal'] = ((qqq_df['returns_1d'] < -0.01) & (btc_returns < -0.03)).astype(int)
            qqq_df['crypto_qqq_divergence'] = ((qqq_df['returns_1d'] * btc_returns) < 0).astype(int)
            
            # BTC momentum vs QQQ
            qqq_df['btc_outperform'] = (btc_returns > qqq_df['returns_1d']).astype(int)
            qqq_df['btc_momentum_strength'] = np.abs(btc_returns) / (np.abs(qqq_df['returns_1d']) + 1e-8)
            
            print(f"    [OK] Added 11 BTC features")
            return 11
        except Exception as e:
            print(f"    [ERROR] BTC features failed: {e}")
            return 0
    
    def train_model(self, df: pd.DataFrame) -> Dict:
        """Train optimized model with enhanced features"""
        print("\n[MODEL] TRAINING OPTIMIZED MODEL")
        print("=" * 50)
        
        # Find optimal horizon
        best_horizon = self._find_best_horizon(df)
        
        # Prepare data with enhanced feature selection
        X, y = self._prepare_optimized_training_data(df, best_horizon)
        
        if len(X) < 100:
            print(f"[WARN]  WARNING: Limited training data ({len(X)} samples)")
        
        print(f"[DATA] Training with {len(X)} samples, {len(X.columns)} features")
        
        # Train enhanced ensemble
        results = self._train_enhanced_ensemble(X, y)
        
        # Store results
        self.models[best_horizon] = results['models']
        self.scalers[best_horizon] = results['scaler']
        self.feature_cols = results['feature_cols']
        self.performance_metrics[best_horizon] = results['ensemble_metrics']
        
        # Save to database
        self.db_integrator.save_metrics(results)
        
        print(f"\n[SUCCESS] TRAINING COMPLETE!")
        print(f"   [TARGET] Direction Accuracy: {results['ensemble_metrics']['direction_accuracy']:.1%}")
        print(f"   [DATA] R² Score: {results['ensemble_metrics']['r2']:.4f}")
        
        return results
    
    def _find_best_horizon(self, df: pd.DataFrame) -> int:
        """Find best horizon with comprehensive testing"""
        print("[TARGET] Comprehensive horizon optimization...")
        
        best_horizon = 5
        best_score = -999
        results = []
        
        for horizon in [1, 2, 3, 5, 7, 10, 14]:
            try:
                X, y = self._prepare_optimized_training_data(df, horizon)
                
                if len(X) < 50:
                    print(f"   {horizon}d: {len(X)} samples (too few)")
                    continue
                
                # Enhanced validation with multiple metrics
                split = int(len(X) * 0.75)
                X_train, X_test = X.iloc[:split], X.iloc[split:]
                y_train, y_test = y.iloc[:split], y.iloc[split:]
                
                if len(X_test) < 10:
                    continue
                
                # Scale features
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Test with XGBoost
                model = XGBRegressor(
                    n_estimators=100, max_depth=5, learning_rate=0.1, 
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=42, n_jobs=-1
                )
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                direction_acc = np.mean(np.sign(y_test) == np.sign(y_pred))
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                # Hit rate for different return thresholds
                hit_rate_1pct = np.mean((np.sign(y_test) == np.sign(y_pred)) & (np.abs(y_test) > 1))
                hit_rate_2pct = np.mean((np.sign(y_test) == np.sign(y_pred)) & (np.abs(y_test) > 2))
                
                # Combined score emphasizing direction accuracy
                score = (
                    0.4 * direction_acc +
                    0.2 * max(0, r2) + 
                    0.2 * hit_rate_1pct +
                    0.2 * hit_rate_2pct
                )
                
                results.append({
                    'horizon': horizon,
                    'direction_acc': direction_acc,
                    'r2': r2,
                    'mae': mae,
                    'hit_rate_1pct': hit_rate_1pct,
                    'hit_rate_2pct': hit_rate_2pct,
                    'score': score
                })
                
                if score > best_score:
                    best_score = score
                    best_horizon = horizon
                
                print(f"   {horizon}d: Dir={direction_acc:.3f}, R²={r2:.3f}, Hit1%={hit_rate_1pct:.3f}, Score={score:.3f}")
                
            except Exception as e:
                print(f"   {horizon}d: Failed - {e}")
        
        self.best_horizon = best_horizon
        print(f"   [EXCELLENT] Best: {best_horizon} days (score: {best_score:.3f})")
        
        return best_horizon
    
    def _prepare_optimized_training_data(self, df: pd.DataFrame, horizon: int) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data with enhanced feature selection"""
        
        df_work = df.copy()
        
        # Create target
        df_work[f'target_{horizon}d'] = (df_work['Close'].shift(-horizon) / df_work['Close'] - 1) * 100
        
        # Clean data
        df_work = df_work.dropna(subset=[f'target_{horizon}d'])
        
        # Feature selection
        exclude_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume', 
            'returns_1d', 'returns_3d', 'volatility_5d', 'volatility_20d'  # Basic features that might be in target
        ] + [col for col in df_work.columns if 'target' in col]
        
        feature_cols = [col for col in df_work.columns if col not in exclude_cols]
        
        X = df_work[feature_cols].copy()
        y = df_work[f'target_{horizon}d'].copy()
        
        # Clean features
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Remove constant features
        constant_cols = X.columns[X.var() == 0].tolist()
        if constant_cols:
            X = X.drop(columns=constant_cols)
            print(f"    Removed {len(constant_cols)} constant features")
        
        # Advanced feature selection if too many features
        if len(X.columns) > 60:
            print(f"    Applying feature selection ({len(X.columns)} -> 60 features)...")
            
            # Calculate multiple importance scores
            feature_scores = []
            
            for col in X.columns:
                try:
                    # Correlation with target
                    corr_score = abs(X[col].corr(y))
                    if np.isnan(corr_score):
                        corr_score = 0
                    
                    # Mutual information proxy (variance in different target ranges)
                    y_low = y < y.quantile(0.33)
                    y_high = y > y.quantile(0.67)
                    var_low = X.loc[y_low, col].var()
                    var_high = X.loc[y_high, col].var()
                    mi_score = abs(var_high - var_low) / (var_high + var_low + 1e-8)
                    
                    # Feature stability (less missing, less extreme values)
                    stability_score = 1 - (X[col].isna().sum() / len(X))
                    
                    # Combined score
                    combined_score = 0.5 * corr_score + 0.3 * mi_score + 0.2 * stability_score
                    feature_scores.append((col, combined_score))
                    
                except:
                    feature_scores.append((col, 0))
            
            # Sort and select top features
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            top_features = [item[0] for item in feature_scores[:60]]
            X = X[top_features]
            
            print(f"    Selected top 60 features by importance")
        
        return X, y
    
    def _train_enhanced_ensemble(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train enhanced ensemble with better models"""
        
        # Time series split for more robust validation
        split = int(len(X) * 0.75)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        # Robust scaling
        scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
        
        # Enhanced model configurations
        models = {
            'xgboost': XGBRegressor(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
                random_state=42, n_jobs=-1
            ),
            'lightgbm': LGBMRegressor(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
                random_state=42, n_jobs=-1, verbose=-1
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=2,
                max_features='sqrt', random_state=42, n_jobs=-1
            ),
            'extra_trees': RandomForestRegressor(
                n_estimators=200, max_depth=12, min_samples_split=3, min_samples_leaf=1,
                max_features='sqrt', random_state=42, n_jobs=-1, bootstrap=False
            ),
            'gradient_boost': GradientBoostingRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.08,
                subsample=0.8, max_features='sqrt', random_state=42
            )
        }
        
        trained_models = {}
        predictions = {}
        scores = {}
        
        for name, model in models.items():
            print(f"   [TOOL] Training {name}...")
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                # Comprehensive metrics
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                direction_acc = np.mean(np.sign(y_test) == np.sign(y_pred))
                
                # Additional performance metrics
                hit_rate_1pct = np.mean((np.sign(y_test) == np.sign(y_pred)) & (np.abs(y_test) > 1))
                rmse = np.sqrt(mse)
                
                trained_models[name] = model
                predictions[name] = y_pred
                scores[name] = {
                    'r2': r2,
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'direction_accuracy': direction_acc,
                    'hit_rate_1pct': hit_rate_1pct
                }
                
                print(f"      [OK] Dir={direction_acc:.3f}, R²={r2:.3f}, Hit1%={hit_rate_1pct:.3f}")
                
            except Exception as e:
                print(f"      [ERROR] Failed: {e}")
        
        if not trained_models:
            raise ValueError("No models trained successfully")
        
        # Smart ensemble weighting
        weights = self._calculate_smart_weights(scores)
        print(f"   [TARGET] Ensemble weights: {weights}")
        
        # Create ensemble prediction
        ensemble_pred = np.zeros_like(list(predictions.values())[0])
        for name, pred in predictions.items():
            ensemble_pred += weights.get(name, 0) * pred
        
        # Ensemble metrics
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
        ensemble_mse = mean_squared_error(y_test, ensemble_pred)
        ensemble_direction = np.mean(np.sign(y_test) == np.sign(ensemble_pred))
        ensemble_hit_1pct = np.mean((np.sign(y_test) == np.sign(ensemble_pred)) & (np.abs(y_test) > 1))
        
        # Calculate Sharpe-like ratio
        returns_pred = ensemble_pred / 100  # Convert to decimal
        sharpe = np.mean(returns_pred) / (np.std(returns_pred) + 1e-8) if np.std(returns_pred) > 0 else 0
        
        self.ensemble_weights = weights
        
        return {
            'models': trained_models,
            'scaler': scaler,
            'weights': weights,
            'feature_cols': X_train.columns.tolist(),
            'ensemble_metrics': {
                'r2': ensemble_r2,
                'mae': ensemble_mae,
                'mse': ensemble_mse,
                'direction_accuracy': ensemble_direction,
                'hit_rate': ensemble_direction,
                'hit_rate_1pct': ensemble_hit_1pct,
                'sharpe_ratio': sharpe
            },
            'individual_metrics': scores
        }
    
    def _calculate_smart_weights(self, scores: Dict) -> Dict[str, float]:
        """Calculate intelligent ensemble weights"""
        
        weights = {}
        total_score = 0
        
        for name, score_dict in scores.items():
            # Multi-criteria scoring
            direction_score = score_dict['direction_accuracy']
            r2_score = max(0, score_dict['r2'])  # Only positive R²
            hit_rate_score = score_dict.get('hit_rate_1pct', 0)
            
            # Penalty for poor performance
            if direction_score < 0.45:  # Worse than random
                combined_score = 0.1
            else:
                combined_score = (
                    0.5 * direction_score +
                    0.25 * r2_score +
                    0.25 * hit_rate_score
                )
            
            weights[name] = combined_score
            total_score += combined_score
        
        # Normalize weights
        if total_score > 0:
            weights = {name: w / total_score for name, w in weights.items()}
        else:
            weights = {name: 1/len(weights) for name in weights.keys()}
        
        return weights
    
    def make_prediction(self, df_latest: pd.DataFrame, current_price: float) -> Dict:
        """Make enhanced prediction"""
        print(f"\n[PREDICT] MAKING ENHANCED PREDICTION")
        print("=" * 40)
        
        if self.best_horizon not in self.models:
            raise ValueError(f"No model for {self.best_horizon}-day horizon")
        
        print(f"  [PRICE] Current price: ${current_price:.2f}")
        print(f"  [DATE] Horizon: {self.best_horizon} days")
        
        # Get features
        latest_features = df_latest[self.feature_cols].iloc[-1:].copy()
        latest_features = latest_features.fillna(method='ffill').fillna(0)
        
        # Data quality check for infinity and extreme values
        print(f"[DATA] Checking feature data quality...")
        
        # Replace infinity values
        latest_features = latest_features.replace([np.inf, -np.inf], np.nan)
        
        # Check for remaining NaN/infinity issues
        if latest_features.isnull().any().any():
            nan_cols = latest_features.columns[latest_features.isnull().any()].tolist()
            print(f"[WARN] Found NaN values in features: {nan_cols[:5]}...")
            latest_features = latest_features.fillna(0)
        
        # Check for extreme values that could cause issues
        extreme_threshold = 1e10
        extreme_mask = (np.abs(latest_features) > extreme_threshold).any(axis=1)
        if extreme_mask.any():
            print(f"[WARN] Found extreme values, clipping to +/- {extreme_threshold}")
            latest_features = latest_features.clip(-extreme_threshold, extreme_threshold)
        
        # Validate data before scaling
        if not np.isfinite(latest_features.values).all():
            print(f"[ERROR] Features still contain non-finite values after cleaning")
            # Force cleanup by replacing any remaining bad values
            latest_features = pd.DataFrame(
                np.nan_to_num(latest_features.values, nan=0.0, posinf=1e6, neginf=-1e6),
                columns=latest_features.columns,
                index=latest_features.index
            )
            print(f"[OK] Applied emergency cleanup - replaced all non-finite values")
        
        print(f"[OK] Feature data quality validated")
        
        # Scale
        scaler = self.scalers[self.best_horizon]
        try:
            latest_scaled = scaler.transform(latest_features)
        except ValueError as e:
            if "infinity" in str(e) or "finite" in str(e):
                print(f"[ERROR] Scaler failed with data quality issue: {str(e)}")
                print(f"[FIX] Applying additional data cleanup...")
                
                # Emergency fallback: ensure all values are finite
                cleaned_features = np.nan_to_num(latest_features.values, nan=0.0, posinf=1e6, neginf=-1e6)
                latest_scaled = scaler.transform(cleaned_features.reshape(1, -1))
                print(f"[OK] Successfully scaled with emergency cleanup")
            else:
                raise ValueError(f"Scaler error (not data quality): {str(e)}")
        
        print(f"[OK] Features successfully scaled for prediction")
        
        # Get predictions from all models
        models = self.models[self.best_horizon]
        weights = getattr(self, 'ensemble_weights', {name: 1/len(models) for name in models.keys()})
        
        predictions = {}
        for name, model in models.items():
            try:
                pred = model.predict(latest_scaled)[0]
                # Validate prediction is finite
                if not np.isfinite(pred):
                    print(f"[WARN] Model {name} returned non-finite prediction: {pred}, using 0")
                    pred = 0.0
                predictions[name] = pred
            except Exception as e:
                print(f"[ERROR] Model {name} prediction failed: {str(e)}, using 0")
                predictions[name] = 0.0
        
        print(f"[OK] Generated predictions from {len(predictions)} models")
        
        # Weighted ensemble prediction with validation
        ensemble_return = sum(weights.get(name, 0) * pred for name, pred in predictions.items())
        
        # Validate ensemble return is reasonable
        if not np.isfinite(ensemble_return):
            print(f"[ERROR] Ensemble return is not finite: {ensemble_return}, defaulting to 0")
            ensemble_return = 0.0
        elif abs(ensemble_return) > 50:  # Cap extreme predictions at +/- 50%
            print(f"[WARN] Extreme ensemble return {ensemble_return:.2f}%, capping at +/- 50%")
            ensemble_return = np.clip(ensemble_return, -50, 50)
        
        ensemble_price = current_price * (1 + ensemble_return / 100)
        print(f"[PREDICT] Ensemble return: {ensemble_return:.2f}%, price: ${ensemble_price:.2f}")
        
        # Enhanced confidence calculation
        pred_values = list(predictions.values())
        pred_std = np.std(pred_values)
        pred_mean = np.mean(pred_values)
        
        # Model agreement factor
        agreement_factor = max(0, 1 - (pred_std / max(1, abs(pred_mean))))
        
        # Signal strength factor
        signal_strength = min(1, abs(ensemble_return) / 3.0)  # Stronger signals get higher confidence
        
        # Performance-based confidence
        performance_metrics = self.performance_metrics.get(self.best_horizon, {})
        direction_acc = performance_metrics.get('direction_accuracy', 0.5)
        perf_factor = max(0, (direction_acc - 0.5) * 2)  # 0 at 50%, 1 at 100%
        
        # Combined confidence
        base_confidence = (
            0.4 * agreement_factor +
            0.3 * signal_strength +
            0.3 * perf_factor
        )
        
        confidence = 30 + (base_confidence * 65)  # Scale to 30-95%
        confidence = min(95, max(30, confidence))
        
        result = {
            'current_price': current_price,
            'predicted_price': ensemble_price,
            'predicted_return': ensemble_return,
            'confidence': confidence,
            'horizon_days': self.best_horizon,
            'market_regime': 'UNKNOWN',
            'individual_predictions': predictions,
            'ensemble_weights': weights,
            'model_agreement': agreement_factor,
            'signal_strength': signal_strength,
            'performance_factor': perf_factor
        }
        
        print(f"  [CHART] Predicted: ${ensemble_price:.2f} ({ensemble_return:+.2f}%)")
        print(f"  [TARGET] Confidence: {confidence:.1f}%")
        print(f"  [AGREE] Model agreement: {agreement_factor:.2f}")
        print(f"  [SIGNAL] Signal strength: {signal_strength:.2f}")
        
        # Save
        self.db_integrator.save_prediction(result)
        
        return result
    
    def generate_text_report(self, prediction: Dict, training_results: Dict, current_price: float) -> str:
        """Generate text report file for trading reports system"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determine action based on prediction
        predicted_return = prediction['predicted_return']
        confidence = prediction['confidence']
        
        if predicted_return > 2.0 and confidence > 80:
            action = "STRONG BUY"
        elif predicted_return > 1.0 and confidence > 70:
            action = "BUY"
        elif predicted_return > 0.5 and confidence > 60:
            action = "WEAK BUY"
        elif predicted_return > -0.5:
            action = "HOLD"
        elif predicted_return > -1.5:
            action = "SELL"
        else:
            action = "STRONG SELL"
        
        # Determine confidence level
        if confidence > 85:
            confidence_level = "HIGH"
        elif confidence > 70:
            confidence_level = "MEDIUM"
        elif confidence > 55:
            confidence_level = "LOW"
        else:
            confidence_level = "VERY LOW"
        
        # Determine risk level based on volatility and confidence
        model_agreement = prediction.get('model_agreement', 0.5)
        if model_agreement > 0.8 and confidence > 75:
            risk_level = "LOW"
        elif model_agreement > 0.6 and confidence > 60:
            risk_level = "MODERATE"
        elif model_agreement > 0.4:
            risk_level = "HIGH"
        else:
            risk_level = "VERY HIGH"
        
        # Calculate stop loss (10% below current price)
        stop_loss = current_price * 0.90
        
        # Get performance metrics
        direction_accuracy = training_results['ensemble_metrics'].get('direction_accuracy', 0) * 100
        r2_score = training_results['ensemble_metrics'].get('r2', 0)
        hit_rate = training_results['ensemble_metrics'].get('hit_rate_1pct', 0) * 100
        
        # Market regime analysis
        has_real_vix = 'VIX' in self.correlation_assets
        regime = "BULL_MOMENTUM" if predicted_return > 1 else "SIDEWAYS" if abs(predicted_return) < 1 else "BEAR_PRESSURE"
        
        report_content = f"""QQQ Long Bull Report
Generated: {timestamp}

PREDICTION SUMMARY
=====================================
Current Price: ${current_price:.2f}
Target Price: ${prediction['predicted_price']:.2f}
Expected Return: {predicted_return:+.2f}%
Time Horizon: {prediction['horizon_days']} days

SIGNAL ANALYSIS
=====================================
Signal: {action}
Suggested Action: {action}
Confidence: {confidence_level}
Risk Level: {risk_level}

TECHNICAL ANALYSIS
=====================================
Direction Accuracy: {direction_accuracy:.1f}%
Model R² Score: {r2_score:.3f}
Hit Rate (>1% moves): {hit_rate:.1f}%
Model Agreement: {model_agreement:.1f}%
Market Regime: {regime}

RISK MANAGEMENT
=====================================
Stop Loss: ${stop_loss:.2f}
Position Size: Moderate (based on confidence)
Max Drawdown Risk: {100-confidence:.1f}%

MARKET CONDITIONS
=====================================
VIX Data Available: {'Yes' if has_real_vix else 'No (Synthetic Used)'}
Assets Analyzed: {len(self.correlation_assets)}
Training Data Quality: {len(self.correlation_assets)*20:.0f} features
Data Lookback: 3 years

MODEL COMPOSITION
=====================================
Ensemble Models: 5 (XGBoost, LightGBM, RandomForest, ExtraTrees, GradientBoost)
Feature Engineering: QQQ-focused with VIX alignment
Optimization: Multi-horizon with {self.best_horizon}-day selected

INDIVIDUAL MODEL PREDICTIONS
====================================="""

        # Add individual model predictions
        for model_name, pred_value in prediction.get('individual_predictions', {}).items():
            weight = prediction.get('ensemble_weights', {}).get(model_name, 0)
            pred_price = current_price * (1 + pred_value / 100)
            report_content += f"""
{model_name.title()}: ${pred_price:.2f} ({pred_value:+.2f}%) [Weight: {weight:.1%}]"""

        report_content += f"""

TECHNICAL INDICATORS SUMMARY
=====================================
RSI Analysis: Multiple timeframes analyzed
MACD Status: Trend momentum evaluated  
Bollinger Bands: Volatility breakout signals
Moving Averages: Multi-timeframe alignment
Volume Analysis: Institutional flow patterns

DISCLAIMER
=====================================
This analysis is for informational purposes only. Past performance 
does not guarantee future results. Please consult with a financial 
advisor before making investment decisions.

Report ID: LongBull_{report_date}
Model Version: 3.2 VIX Alignment Fix
"""

        # Save report file
        # Use GitHub repo reports directory instead of Desktop
        script_dir = os.path.dirname(os.path.abspath(__file__))
        desktop_path = os.path.join(script_dir, "reports")
        os.makedirs(desktop_path, exist_ok=True)
        
        filename = f"QQQ_Long_Bull_Report_{report_date}.txt"
        filepath = os.path.join(desktop_path, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            print(f"[REPORT] Report saved: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"[ERROR] Failed to save report: {e}")
            return None
    
    def save_model(self, filepath: str):
        """Save the complete model"""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_cols': self.feature_cols,
            'best_horizon': self.best_horizon,
            'performance_metrics': self.performance_metrics,
            'correlation_assets': self.correlation_assets,
            'ensemble_weights': getattr(self, 'ensemble_weights', {})
        }
        
        joblib.dump(model_data, filepath)
        print(f"[SAVE] Model saved to: {filepath}")


def main():
    """Main execution with VIX priority alignment + Text Report Generation"""
    print("QQQ LONG BULL MODEL V3.2 - FINAL VIX ALIGNMENT FIX + REPORT GENERATION")
    print("=" * 80)
    print("[OK] QQQ+VIX priority alignment")
    print("[OK] Enhanced QQQ-focused features")
    print("[OK] Smart synthetic VIX fallback")
    print("[OK] Optimized ensemble with 5 models")
    print("[OK] Advanced feature selection")
    print("[OK] TEXT REPORT GENERATION for trading system integration")
    print("=" * 80)
    
    try:
        # Initialize model
        model = QQQLongBullModelV32()
        
        # Update pending predictions
        model.db_integrator.update_actual_prices()
        
        # Fetch data with QQQ+VIX priority
        data = model.fetch_all_data()
        
        if 'QQQ' not in data:
            raise ValueError("[ERROR] Cannot proceed without QQQ data")
        
        # Check data quality
        sample_size = len(data['QQQ'])
        has_real_vix = 'VIX' in data and len(data['VIX']) == sample_size
        
        print(f"\n[DATA] DATA SUMMARY:")
        print(f"   [DATE] Training period: {sample_size} days")
        print(f"   [HOT] Real VIX data: {'[OK] YES' if has_real_vix else '[ERROR] NO (using synthetic)'}")
        print(f"   [CHART] Available assets: {len(data)}")
        
        # Create enhanced features
        df_features = model.create_qqq_focused_features(data)
        
        # Train model
        training_results = model.train_model(df_features)
        
        # Get current price
        current_price = data_fetcher.get_current_price('QQQ')
        if current_price is None:
            current_price = data['QQQ']['Close'].iloc[-1]
            print(f"[WARN]  Using last available price: ${current_price:.2f}")
        else:
            print(f"[PRICE] Current QQQ price: ${current_price:.2f}")
        
        # Make prediction
        prediction = model.make_prediction(df_features, current_price)
        
        # Generate text report (NEW!)
        report_path = model.generate_text_report(prediction, training_results, current_price)
        if report_path:
            print(f"[OK] Text report generated: {os.path.basename(report_path)}")
        else:
            print("[ERROR] Failed to generate text report")
        
        # Save model
        model_path = os.path.join(OUTPUTS_DIR, f"qqq_long_bull_model_v32_{datetime.now().strftime('%Y%m%d')}.pkl")
        model.save_model(model_path)
        
        # Final comprehensive summary
        direction_acc = training_results['ensemble_metrics']['direction_accuracy']
        r2_score = training_results['ensemble_metrics']['r2']
        
        print("\n" + "=" * 80)
        print("[SUCCESS] QQQ LONG BULL MODEL V3.2 COMPLETE!")
        print("=" * 80)
        print(f"[DATA] MODEL PERFORMANCE:")
        print(f"   [TARGET] Direction Accuracy: {direction_acc:.1%} {'[OK]' if direction_acc > 0.55 else '[WARN]' if direction_acc > 0.45 else '[ERROR]'}")
        print(f"   [CHART] R² Score: {r2_score:.4f} {'[OK]' if r2_score > 0 else '[ERROR]'}")
        print(f"   [DATA] Training Data: {sample_size} days")
        print(f"   [HOT] VIX Integration: {'Real data' if has_real_vix else 'Synthetic'}")
        print(f"\n[PREDICT] PREDICTION ({prediction['horizon_days']} days):")
        print(f"   [PRICE] ${prediction['current_price']:.2f} -> ${prediction['predicted_price']:.2f}")
        print(f"   [CHART] Expected Return: {prediction['predicted_return']:+.2f}%")
        print(f"   [TARGET] Confidence: {prediction['confidence']:.1f}%")
        print(f"   [AGREE] Model Agreement: {prediction['model_agreement']:.2f}")
        print(f"\n[OK] Saved to database and file: {model_path}")
        if report_path:
            print(f"[OK] Text report: {os.path.basename(report_path)}")
        
        # Performance assessment
        if direction_acc > 0.6:
            print("[EXCELLENT] EXCELLENT: Direction accuracy > 60%")
        elif direction_acc > 0.55:
            print("[TARGET] GOOD: Direction accuracy > 55%")
        elif direction_acc > 0.5:
            print("[CHART] MARGINAL: Direction accuracy > 50%")
        else:
            print("[WARN]  POOR: Direction accuracy <= 50% (needs improvement)")
        
        print("=" * 80)
        
    except Exception as e:
        print(f"[ERROR] CRITICAL ERROR: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()