# NVIDIA Bull Momentum Model - AI Stock Prediction
# Optimized for NVIDIA's high-volatility, growth-focused trading patterns

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from datetime import datetime, timedelta
import os
import warnings
import traceback
from sklearn.base import clone
import requests
import time
from model_db_integrator import quick_save_prediction, quick_save_metrics

# Try to import yfinance for fallback
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("yfinance not available. Install with 'pip install yfinance' for better data access.")

warnings.filterwarnings('ignore')

# Configuration
TICKER = "NVDA"
START_DATE = "2020-01-01"  # NVIDIA's AI boom period
ALPHA_VANTAGE_API_KEY = "HMHALLINAHS2FF4Z"
BASE_URL = "https://www.alphavantage.co/query"

# Desktop path detection
def get_desktop_path():
    """Get the user's desktop path, accounting for OneDrive"""
    try:
        user_profile = os.environ.get("USERPROFILE", "")
        if not user_profile:
            desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
            if os.path.exists(desktop_path):
                return desktop_path
            reports_path = os.path.join(os.getcwd(), "nvidia_reports")
            os.makedirs(reports_path, exist_ok=True)
            return reports_path

        onedrive_desktop = os.path.join(user_profile, "OneDrive", "Desktop")
        if os.path.exists(onedrive_desktop):
            return onedrive_desktop
        
        standard_desktop = os.path.join(user_profile, "Desktop")
        if os.path.exists(standard_desktop):
            return standard_desktop
        
        fallback_desktop = os.path.join(os.getcwd(), "Desktop_Fallback")
        os.makedirs(fallback_desktop, exist_ok=True)
        return fallback_desktop

    except Exception as e:
        fallback_desktop = os.path.join(os.getcwd(), "Desktop_Fallback")
        os.makedirs(fallback_desktop, exist_ok=True)
        return fallback_desktop

DESKTOP_PATH = get_desktop_path()
print(f"Using desktop path: {DESKTOP_PATH}")

def fetch_alpha_vantage_data(symbol, api_key, outputsize='full'):
    """Fetch historical data from Alpha Vantage using FREE endpoints"""
    print(f"Fetching {symbol} data from Alpha Vantage (FREE endpoint)...")
    
    # Use the FREE TIME_SERIES_DAILY endpoint instead of premium adjusted
    params = {
        'function': 'TIME_SERIES_DAILY',  # FREE endpoint
        'symbol': symbol,
        'outputsize': outputsize,
        'apikey': api_key
    }
    
    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Enhanced error handling
        if 'Error Message' in data:
            print(f"Alpha Vantage Error: {data['Error Message']}")
            return None
            
        if 'Note' in data:
            print(f"Alpha Vantage Note: {data['Note']}")
            print("API call frequency limit reached. Waiting 60 seconds...")
            time.sleep(60)
            return fetch_alpha_vantage_data(symbol, api_key, outputsize)
        
        if 'Information' in data:
            print(f"Alpha Vantage Information: {data['Information']}")
            if 'call frequency' in data['Information'].lower():
                print("API rate limit hit. Waiting 60 seconds and retrying...")
                time.sleep(60)
                return fetch_alpha_vantage_data(symbol, api_key, outputsize)
            else:
                print("API issue detected. Trying with compact output...")
                if outputsize == 'full':
                    return fetch_alpha_vantage_data(symbol, api_key, 'compact')
                else:
                    return None
        
        if 'Time Series (Daily)' not in data:
            print(f"Unexpected response format: {list(data.keys())}")
            if len(data) > 0:
                print(f"Response content: {data}")
            return None
        
        time_series = data['Time Series (Daily)']
        
        # Convert to DataFrame using FREE endpoint format
        df_data = []
        for date_str, values in time_series.items():
            df_data.append({
                'Date': pd.to_datetime(date_str),
                'Open': float(values['1. open']),
                'High': float(values['2. high']),
                'Low': float(values['3. low']),
                'Close': float(values['4. close']),
                'Volume': int(values['5. volume'])
            })
        
        df = pd.DataFrame(df_data)
        df = df.set_index('Date').sort_index()
        
        # For free endpoint, Close = Adj_Close (no dividend adjustments available for free)
        df['Adj_Close'] = df['Close']
        
        print(f"Successfully fetched {len(df)} days of {symbol} data (FREE API)")
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Network error fetching {symbol}: {e}")
        return None
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def fetch_yahoo_fallback(symbol):
    """Fallback data source using Yahoo Finance API (free alternative)"""
    print(f"Attempting to fetch {symbol} data from Yahoo Finance as fallback...")
    
    if not YFINANCE_AVAILABLE:
        print("yfinance not available. Install with: pip install yfinance")
        return None
    
    try:
        # Fetch data using yfinance
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="5y", interval="1d")  # 5 years of daily data
        
        if data.empty:
            print("No data retrieved from Yahoo Finance")
            return None
        
        # Convert to expected format
        data = data.reset_index()
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index('Date')
        
        # Rename columns to match Alpha Vantage format
        data = data.rename(columns={
            'Open': 'Open',
            'High': 'High', 
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        })
        
        # Create Adj_Close (yfinance Close is already adjusted)
        data['Adj_Close'] = data['Close']
        
        print(f"Successfully fetched {len(data)} days from Yahoo Finance")
        return data
        
    except Exception as e:
        print(f"Yahoo Finance fallback failed: {e}")
        return None

def fetch_manual_fallback():
    """Create sample NVIDIA data for demonstration if no data sources work"""
    print("Creating sample NVIDIA data for demonstration...")
    
    try:
        # Create 2 years of sample data with NVIDIA-like characteristics
        start_date = datetime.now() - timedelta(days=730)
        dates = pd.date_range(start=start_date, periods=500, freq='D')
        
        # Remove weekends
        dates = dates[dates.dayofweek < 5]
        
        # Generate realistic NVIDIA price movement
        np.random.seed(42)
        
        initial_price = 400.0  # Starting price
        returns = np.random.normal(0.001, 0.04, len(dates))  # Daily returns with high volatility
        
        # Add some trend and momentum
        trend = np.linspace(0, 0.5, len(dates))  # Upward trend over time
        momentum_cycles = np.sin(np.arange(len(dates)) * 0.1) * 0.02  # Some cyclical patterns
        
        adjusted_returns = returns + trend * 0.001 + momentum_cycles
        
        # Generate price series
        prices = [initial_price]
        for ret in adjusted_returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create OHLCV data
        data = pd.DataFrame(index=dates)
        data['Close'] = prices
        data['Adj_Close'] = prices
        
        # Generate realistic OHLC
        data['Open'] = data['Close'].shift(1) * (1 + np.random.normal(0, 0.005, len(data)))
        data['High'] = np.maximum(data['Open'], data['Close']) * (1 + np.abs(np.random.normal(0, 0.01, len(data))))
        data['Low'] = np.minimum(data['Open'], data['Close']) * (1 - np.abs(np.random.normal(0, 0.01, len(data))))
        
        # Generate volume (NVIDIA typically has high volume)
        base_volume = 50000000  # 50M shares
        volume_multiplier = 1 + np.abs(adjusted_returns) * 5  # Higher volume on big moves
        data['Volume'] = (base_volume * volume_multiplier * np.random.uniform(0.5, 1.5, len(data))).astype(int)
        
        # Clean up
        data = data.dropna()
        data = data[data.index >= pd.to_datetime(START_DATE)]
        
        print(f"Created {len(data)} days of sample NVIDIA data")
        print(f"Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
        print("Note: This is simulated data for demonstration purposes only!")
        
        return data
        
    except Exception as e:
        print(f"Failed to create sample data: {e}")
        return None

def fetch_current_price(symbol, api_key):
    """Fetch current price from Alpha Vantage using FREE endpoint"""
    print(f"Fetching current {symbol} price using FREE Alpha Vantage endpoint...")
    
    # Try GLOBAL_QUOTE first (free)
    params = {
        'function': 'GLOBAL_QUOTE',
        'symbol': symbol,
        'apikey': api_key
    }
    
    try:
        response = requests.get(BASE_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if 'Global Quote' in data and data['Global Quote']:
            price = float(data['Global Quote']['05. price'])
            print(f"Current {symbol} price (Alpha Vantage FREE): ${price:.2f}")
            return price
        else:
            print(f"Global Quote failed, trying alternative method...")
            
            # Fallback to TIME_SERIES_INTRADAY (also free, but limited)
            params_intraday = {
                'function': 'TIME_SERIES_INTRADAY',
                'symbol': symbol,
                'interval': '1min',
                'apikey': api_key
            }
            
            response = requests.get(BASE_URL, params=params_intraday, timeout=15)
            data = response.json()
            
            if 'Time Series (1min)' in data:
                latest_time = max(data['Time Series (1min)'].keys())
                latest_price = float(data['Time Series (1min)'][latest_time]['4. close'])
                print(f"Current {symbol} price (Intraday): ${latest_price:.2f}")
                return latest_price
            else:
                print(f"Current price fetch failed for {symbol}")
                return None
            
    except Exception as e:
        print(f"Error fetching current price for {symbol}: {e}")
        return None

def calculate_ema(series, window):
    """Calculate Exponential Moving Average"""
    return series.ewm(span=window, adjust=False).mean()

def calculate_rsi(series, window=14):
    """Calculate Relative Strength Index optimized for NVIDIA's volatility"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    """Calculate MACD - crucial for NVIDIA momentum detection"""
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def nvidia_feature_engineering(stock_data):
    """NVIDIA-specific feature engineering focusing on AI/tech momentum patterns"""
    print("Starting NVIDIA-specific feature engineering...")
    df = pd.DataFrame(index=stock_data.index)
    
    # Use adjusted close for splits/dividends
    close = stock_data['Adj_Close'].squeeze()
    high = stock_data['High'].squeeze()
    low = stock_data['Low'].squeeze()
    volume = stock_data['Volume'].squeeze()
    
    # Handle missing data
    close = close.fillna(method='ffill').fillna(method='bfill')
    high = high.fillna(close)
    low = low.fillna(close)
    volume = volume.fillna(volume.median())
    
    # === MOMENTUM FEATURES (Critical for NVIDIA) ===
    print("Calculating momentum features...")
    df['Returns_1d'] = close.pct_change()
    df['Returns_3d'] = close.pct_change(3)
    df['Returns_5d'] = close.pct_change(5)
    df['Returns_10d'] = close.pct_change(10)
    df['Returns_20d'] = close.pct_change(20)  # Monthly momentum
    
    # Momentum acceleration (key for NVIDIA's explosive moves)
    df['Momentum_Accel_3d'] = df['Returns_3d'] - df['Returns_3d'].shift(3)
    df['Momentum_Accel_5d'] = df['Returns_5d'] - df['Returns_5d'].shift(5)
    
    # === TREND STRENGTH FEATURES ===
    print("Calculating trend strength features...")
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma100 = close.rolling(100).mean()
    
    # Price position relative to key moving averages
    df['Price_vs_MA10'] = np.where(ma10 > 0, close / ma10 - 1, 0)
    df['Price_vs_MA20'] = np.where(ma20 > 0, close / ma20 - 1, 0)
    df['Price_vs_MA50'] = np.where(ma50 > 0, close / ma50 - 1, 0)
    df['Price_vs_MA100'] = np.where(ma100 > 0, close / ma100 - 1, 0)
    
    # Moving average trend strength
    df['MA_Trend_10_20'] = np.where(ma20 > 0, ma10 / ma20 - 1, 0)
    df['MA_Trend_20_50'] = np.where(ma50 > 0, ma20 / ma50 - 1, 0)
    
    # === VOLATILITY FEATURES (NVIDIA-specific) ===
    print("Calculating volatility features...")
    returns_1d = df['Returns_1d']
    df['Volatility_5d'] = returns_1d.rolling(5).std()
    df['Volatility_20d'] = returns_1d.rolling(20).std()
    df['Volatility_50d'] = returns_1d.rolling(50).std()
    
    # Volatility regime detection (NVIDIA switches between high/low vol)
    vol_ma = df['Volatility_20d'].rolling(50).mean()
    df['Vol_Regime'] = np.where(vol_ma > 0, df['Volatility_20d'] / vol_ma - 1, 0)
    
    # === TECHNICAL INDICATORS ===
    print("Calculating technical indicators...")
    
    # RSI with NVIDIA-optimized parameters
    rsi_14 = calculate_rsi(close, 14)
    rsi_7 = calculate_rsi(close, 7)  # Faster RSI for quick moves
    df['RSI_14'] = np.where(rsi_14.notna(), (rsi_14 - 50) / 50, 0)
    df['RSI_7'] = np.where(rsi_7.notna(), (rsi_7 - 50) / 50, 0)
    df['RSI_Divergence'] = df['RSI_14'] - df['RSI_7']
    
    # MACD system
    macd_line, signal_line, histogram = calculate_macd(close)
    df['MACD'] = macd_line / close * 100  # Normalized
    df['MACD_Signal'] = signal_line / close * 100
    df['MACD_Histogram'] = histogram / close * 100
    
    # === VOLUME ANALYSIS (Critical for NVIDIA breakouts) ===
    print("Calculating volume features...")
    vol_ma_20 = volume.rolling(20).mean()
    vol_ma_50 = volume.rolling(50).mean()
    
    # Volume ratios
    df['Volume_Ratio_20'] = np.where(vol_ma_20 > 0, volume / vol_ma_20, 1.0)
    df['Volume_Ratio_50'] = np.where(vol_ma_50 > 0, volume / vol_ma_50, 1.0)
    
    # Volume-price relationship (accumulation/distribution)
    df['Volume_Price_Trend'] = df['Returns_1d'] * df['Volume_Ratio_20']
    
    # Volume surge detection (NVIDIA often has volume spikes before big moves)
    volume_surge = (df['Volume_Ratio_20'] > 2.0) & (abs(df['Returns_1d']) > 0.03)
    df['Volume_Surge'] = volume_surge.astype(float)
    
    # === PRICE RANGE ANALYSIS ===
    print("Calculating price range features...")
    
    # Daily range analysis
    daily_range = (high - low) / close
    df['Daily_Range'] = daily_range
    df['Range_MA'] = daily_range.rolling(20).mean()
    df['Range_Expansion'] = np.where(df['Range_MA'] > 0, daily_range / df['Range_MA'] - 1, 0)
    
    # Gap analysis (NVIDIA often gaps on news)
    prev_close = close.shift(1)
    gap = (stock_data['Open'] - prev_close) / prev_close
    df['Gap_Size'] = gap.fillna(0)
    df['Gap_Direction'] = np.sign(df['Gap_Size'])
    
    # === NVIDIA-SPECIFIC PATTERNS ===
    print("Calculating NVIDIA-specific pattern features...")
    
    # Earnings seasonality proxy (quarterly patterns)
    df['Quarter_Phase'] = (df.index.dayofyear % 91) / 91  # Quarterly cycle
    
    # Tech sector momentum proxy using price momentum clusters
    momentum_5d = df['Returns_5d'].rolling(5).mean()
    df['Momentum_Persistence'] = momentum_5d * df['Returns_5d']
    
    # Breakout detection (NVIDIA loves clean breakouts)
    price_high_20 = close.rolling(20).max()
    price_low_20 = close.rolling(20).min()
    df['Near_High_20'] = (close / price_high_20)
    df['Near_Low_20'] = (close / price_low_20)
    
    # === DATA CLEANING AND VALIDATION ===
    print("Cleaning and validating features...")
    
    # Replace infinite values with NaN
    for col in df.columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        # Cap extreme values
        if df[col].notna().sum() > 0:
            q001 = df[col].quantile(0.001)
            q999 = df[col].quantile(0.999)
            df[col] = np.clip(df[col], q001, q999)
        
        # Fill NaN with neutral values
        df[col] = df[col].fillna(0)
    
    # Final validation
    infinite_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()
    if infinite_counts.sum() > 0:
        print("WARNING: Still have infinite values after cleaning:")
        for col, count in infinite_counts.items():
            if count > 0:
                print(f"  {col}: {count} infinite values")
                df[col] = df[col].replace([np.inf, -np.inf], 0)
    
    print(f"Feature engineering complete: {len(df.columns)} features created")
    return df

def nvidia_walk_forward_backtest(data, features, model, train_ratio=0.75, max_predictions=300):
    """Walk-forward backtesting optimized for NVIDIA's volatility patterns"""
    split_idx = int(len(data) * train_ratio)
    total_test_points = len(data) - split_idx
    
    print(f"NVIDIA backtest: {total_test_points} total test points, sampling up to {max_predictions}")
    
    # Sample test points for reasonable runtime
    if total_test_points > max_predictions:
        step_size = max(1, total_test_points // max_predictions)
        test_indices = range(split_idx, len(data), step_size)
        print(f"Sampling every {step_size} points for efficiency")
    else:
        test_indices = range(split_idx, len(data))
    
    predictions = []
    prediction_indices = []
    
    for i, test_idx in enumerate(test_indices):
        try:
            # Use expanding window for NVIDIA (trend-following approach)
            train_start = max(0, split_idx - 500)  # Minimum training window
            train_data = data.iloc[train_start:test_idx]
            test_data = data.iloc[test_idx:test_idx+1]
            
            # Data validation
            if len(train_data) < 200:  # Need substantial history for NVIDIA
                continue
                
            train_clean = train_data.dropna()
            if len(train_clean) < 100:
                continue
            
            if test_data[features].isna().any().any():
                continue
            
            # Clone and fit model
            model_clone = clone(model)
            model_clone.fit(train_clean[features], train_clean['Target_5d'])
            
            # Make prediction
            pred = model_clone.predict(test_data[features])[0]
            predictions.append(pred)
            prediction_indices.append(data.index[test_idx])
            
            if (i + 1) % 25 == 0:
                print(f"NVIDIA backtest progress: {i + 1}/{len(test_indices)} predictions...")
                
        except Exception as e:
            continue
    
    print(f"NVIDIA backtest completed: {len(predictions)} predictions generated")
    return pd.Series(predictions, index=prediction_indices)


def create_standardized_nvidia_report(current_price, predicted_5d_price, predicted_1d_price, 
                                    predicted_5d_return, predicted_1d_return, signal, confidence):
    """Create standardized NVIDIA report for integration"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        desktop_path = get_desktop_path()
        
        report_content = f"""=== NVIDIA Bull Momentum Analysis Report ===
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Symbol: NVDA
Current Price: ${current_price:.2f}

--- TRADING SIGNALS ---
Predicted 5-Day Price: ${predicted_5d_price:.2f}
Predicted 5-Day Return: {predicted_5d_return:+.2%}
Predicted 1-Day Price: ${predicted_1d_price:.2f}
Predicted 1-Day Return: {predicted_1d_return:+.2%}

Signal: {signal}
Confidence: {confidence}

--- SUGGESTED ACTION ---
Suggested Action: {signal}

--- DISCLAIMER ---
This model is for educational and informational purposes only.
Past performance does not guarantee future results.
Always consult with a qualified financial advisor before making investment decisions.

Data Source: Alpha Vantage / Yahoo Finance
Model Version: NVIDIA Bull Momentum v1.0
"""
        
        report_path = os.path.join(desktop_path, f"NVIDIA_Bull_Momentum_Report_{timestamp}.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"Standardized NVIDIA report saved: {report_path}")
        return report_path
        
    except Exception as e:
        print(f"Error creating standardized report: {e}")
        return None


def main():
    try:
        print("=== NVIDIA Bull Momentum Model ===")
        print("AI-focused price prediction optimized for NVIDIA's growth patterns")
        
        # Data Acquisition with multiple fallback options
        print("\n=== DATA ACQUISITION ===")
        stock_data = fetch_alpha_vantage_data(TICKER, ALPHA_VANTAGE_API_KEY)
        
        if stock_data is None or stock_data.empty:
            print("Alpha Vantage failed. Trying Yahoo Finance fallback...")
            stock_data = fetch_yahoo_fallback(TICKER)
            
        if stock_data is None or stock_data.empty:
            print("All external sources failed. Using sample data for demonstration...")
            stock_data = fetch_manual_fallback()
        
        if stock_data is None or stock_data.empty:
            print(f"Failed to fetch {TICKER} data. Exiting.")
            return
        
        # Filter data from start date
        start_date_dt = pd.to_datetime(START_DATE)
        stock_data = stock_data[stock_data.index >= start_date_dt]
        print(f"NVIDIA data shape: {stock_data.shape}")
        print(f"Date range: {stock_data.index.min()} to {stock_data.index.max()}")
        
        # Get current price with fallback
        print("Fetching current NVIDIA price...")
        current_price = fetch_current_price(TICKER, ALPHA_VANTAGE_API_KEY)
        if current_price is None:
            # Try to get from yfinance as fallback
            try:
                if YFINANCE_AVAILABLE:
                    ticker = yf.Ticker(TICKER)
                    current_data = ticker.history(period="1d")
                    if not current_data.empty:
                        current_price = current_data['Close'].iloc[-1]
                        print(f"Current NVIDIA price (Yahoo): ${current_price:.2f}")
                    else:
                        current_price = stock_data['Adj_Close'].iloc[-1]
                        print(f"Using last historical price: ${current_price:.2f}")
                else:
                    current_price = stock_data['Adj_Close'].iloc[-1]
                    print(f"Using last historical price: ${current_price:.2f}")
            except:
                current_price = stock_data['Adj_Close'].iloc[-1]
                print(f"Using last historical price: ${current_price:.2f}")
        else:
            print(f"Current NVIDIA price (Alpha Vantage): ${current_price:.2f}")
        
        # Add small delay to respect API limits
        time.sleep(1)
        
        # Feature Engineering
        print("\n=== FEATURE ENGINEERING ===")
        df = nvidia_feature_engineering(stock_data)
        
        # Target Engineering (5-day forward returns for NVIDIA's momentum patterns)
        print("Creating target variable (5-day forward returns)...")
        close = stock_data['Adj_Close'].squeeze()
        forward_returns = close.pct_change(5).shift(-5)
        
        # Handle end-of-data for recent predictions
        for i in range(5):
            if pd.isna(forward_returns.iloc[-(i+1)]):
                # Use scaled daily return as proxy
                daily_return = close.pct_change().iloc[-(i+1)]
                if not pd.isna(daily_return):
                    forward_returns.iloc[-(i+1)] = daily_return * 5
        
        df['Target_5d'] = forward_returns
        
        # Data Cleaning
        print("\n=== DATA PREPARATION ===")
        df = df.dropna(subset=['Target_5d'])
        
        if len(df) < 200:
            print("Insufficient data after cleaning. Exiting.")
            return
        
        # Feature validation
        features = [col for col in df.columns if col != 'Target_5d']
        print(f"Total features before validation: {len(features)}")
        
        # Remove zero-variance features
        valid_features = []
        for feature in features:
            if df[feature].var() > 1e-10:
                valid_features.append(feature)
            else:
                print(f"Removing {feature} - zero variance")
        
        features = valid_features
        print(f"Final features: {len(features)}")
        
        # Final data validation
        feature_data = df[features]
        target_data = df['Target_5d']
        
        # Remove any remaining problematic rows
        valid_mask = (
            feature_data.notna().all(axis=1) & 
            target_data.notna() &
            np.isfinite(feature_data).all(axis=1) &
            np.isfinite(target_data)
        )
        
        df = df[valid_mask]
        print(f"Clean dataset: {len(df)} rows")
        
        if len(df) < 200:
            print(f"Insufficient clean data ({len(df)} rows). Exiting.")
            return
        
        # Model Configuration (NVIDIA-optimized XGBoost)
        print("\n=== MODEL TRAINING ===")
        model = Pipeline([
            ('scaler', RobustScaler()),
            ('model', XGBRegressor(
                objective='reg:squarederror',
                eval_metric='rmse',
                tree_method='hist',
                random_state=42,
                n_estimators=400,      # Higher for NVIDIA complexity
                max_depth=8,           # Deeper for capturing momentum patterns
                learning_rate=0.03,    # Lower for better generalization
                subsample=0.85,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.5,        # Higher regularization
                n_jobs=-1
            ))
        ])
        
        # Training on full dataset for final model
        print(f"Training on {len(df)} samples...")
        model.fit(df[features], df['Target_5d'])
        print("Training complete!")
        
        # Backtesting
        print("\n=== BACKTESTING ===")
        backtest_preds = nvidia_walk_forward_backtest(df, features, model, max_predictions=250)
        
        if len(backtest_preds) == 0:
            print("No predictions generated during backtesting.")
            return
        
        # Evaluation
        print("\n=== PERFORMANCE EVALUATION ===")
        actual_returns = df['Target_5d']
        
        # Align predictions with actuals
        evaluation_dates = backtest_preds.index
        actual_aligned = actual_returns.reindex(evaluation_dates)
        
        # Clean evaluation data
        valid_eval_mask = actual_aligned.notna() & backtest_preds.notna()
        actual_clean = actual_aligned[valid_eval_mask]
        predicted_clean = backtest_preds[valid_eval_mask]
        
        print(f"Evaluation dataset: {len(actual_clean)} predictions")
        
        if len(actual_clean) < 20:
            print("Insufficient predictions for evaluation.")
            return
        
        # Performance metrics
        mse = mean_squared_error(actual_clean, predicted_clean)
        r2 = r2_score(actual_clean, predicted_clean)
        
        # Trading strategy evaluation
        confidence_threshold = 0.01  # 1% confidence threshold for NVIDIA
        strong_confidence_threshold = 0.025  # 2.5% for strong signals
        
        # Direction accuracy
        direction_correct = ((predicted_clean > 0) & (actual_clean > 0)) | \
                          ((predicted_clean <= 0) & (actual_clean <= 0))
        hit_rate = direction_correct.mean()
        
        # Strategy returns (only trade on confident predictions)
        confident_long = predicted_clean > confidence_threshold
        strong_long = predicted_clean > strong_confidence_threshold
        
        strategy_returns = np.where(confident_long, actual_clean, 0)
        strong_strategy_returns = np.where(strong_long, actual_clean, 0)
        
        # Performance calculations
        total_return = (1 + strategy_returns).prod() - 1
        strong_total_return = (1 + strong_strategy_returns).prod() - 1
        
        returns_std = strategy_returns.std()
        sharpe_ratio = (strategy_returns.mean() / returns_std * np.sqrt(52)) if returns_std > 0 else 0
        
        # Drawdown calculation
        cumulative_returns = (1 + strategy_returns).cumprod()
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        
        # Predictions for next periods
        print("\n=== GENERATING PREDICTIONS ===")
        latest_features = df.iloc[-1:][features]
        predicted_5d_return = model.predict(latest_features)[0]
        predicted_1d_return = predicted_5d_return / 5  # Daily approximation
        
        predicted_1d_price = current_price * (1 + predicted_1d_return)
        predicted_5d_price = current_price * (1 + predicted_5d_return)
        
        # Feature importance analysis
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': model.named_steps['model'].feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Generate signal
        if predicted_5d_return > strong_confidence_threshold:
            signal = "STRONG BUY"
            confidence = "Very High"
        elif predicted_5d_return > confidence_threshold:
            signal = "BUY"
            confidence = "High"
        elif predicted_5d_return < -confidence_threshold:
            signal = "SELL"
            confidence = "High"
        elif predicted_5d_return < -strong_confidence_threshold:
            signal = "STRONG SELL"
            confidence = "Very High"
        else:
            signal = "HOLD"
            confidence = "Low"
        
        # Generate comprehensive report
        report = f"""=== NVIDIA Bull Momentum Analysis Report ===
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Symbol: NVDA
Current Price: ${current_price:.2f}

--- TRADING SIGNALS ---
5-Day Price Target: ${predicted_5d_price:.2f}
5-Day Return Prediction: {predicted_5d_return:+.2%}
1-Day Price Target: ${predicted_1d_price:.2f}
1-Day Return Prediction: {predicted_1d_return:+.2%}

SIGNAL: {signal}
CONFIDENCE: {confidence}

--- MODEL PERFORMANCE ---
Overall Hit Rate: {hit_rate:.1%}
R-squared Score: {r2:.4f}
Mean Squared Error: {mse:.6f}

Strategy Performance:
- Total Return (Confident Trades): {total_return:.1%}
- Total Return (Strong Signals Only): {strong_total_return:.1%}
- Sharpe Ratio: {sharpe_ratio:.2f}
- Maximum Drawdown: {max_drawdown:.1%}
- Backtested Predictions: {len(predicted_clean)}

--- TOP PREDICTIVE FEATURES ---"""

        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            report += f"\n{i+1:2d}. {row['Feature']}: {row['Importance']:.4f}"

        report += f"""

--- MODEL SPECIFICATIONS ---
Algorithm: XGBoost Gradient Boosting
Training Period: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}
Training Samples: {len(df):,}
Features: {len(features)} NVIDIA-optimized technical indicators
Prediction Horizon: 5 trading days
Model Parameters:
  - Estimators: 400
  - Max Depth: 8
  - Learning Rate: 0.03
  - Regularization: L1=0.1, L2=1.5

--- NVIDIA-SPECIFIC OPTIMIZATIONS ---
1. AI/Tech momentum pattern detection
2. Volume surge analysis for breakout prediction
3. Volatility regime classification
4. Extended lookback periods for trend analysis
5. Earnings seasonality proxy features
6. Gap analysis for news-driven movements

--- RISK CONSIDERATIONS ---
• NVIDIA exhibits high volatility - position sizing is critical
• Model optimized for 5-day holding periods
• Strong correlation with tech sector and AI sentiment
• Earnings events can cause significant deviations
• Crypto market movements may impact predictions

--- DISCLAIMER ---
This model is for educational and informational purposes only.
Past performance does not guarantee future results.
NVIDIA stock carries significant volatility and risk.
Always consult with a qualified financial advisor before making investment decisions.
Never invest more than you can afford to lose.

Data Source: Alpha Vantage / Yahoo Finance
Model Version: NVIDIA Bull Momentum v1.0
"""

        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"NVIDIA_Bull_Momentum_Report_{timestamp}.txt"
        report_path = os.path.join(DESKTOP_PATH, report_filename)
        
        try:
            with open(report_path, "w") as f:
                f.write(report)
            print(f"\nReport saved to: {report_path}")
        except Exception as e:
            print(f"Error saving report: {e}")
        
        # Display report
        print("\n" + "="*80)
        print(report)
        print("="*80)
        
        # Display key statistics
        print(f"\n=== QUICK SUMMARY ===")
        print(f"Current NVDA Price: ${current_price:.2f}")
        print(f"5-Day Target: ${predicted_5d_price:.2f} ({predicted_5d_return:+.1%})")
        print(f"Signal: {signal} ({confidence} confidence)")
        print(f"Model Hit Rate: {hit_rate:.1%}")
        print(f"Strategy Return: {total_return:+.1%}")
        # Generate standardized report
        try:
            create_standardized_nvidia_report(
                current_price, predicted_5d_price, predicted_1d_price,
                predicted_5d_return, predicted_1d_return, signal, confidence
            )
        except Exception as e:
            print(f"Error generating report: {e}")
        
        # Save predictions to database
        print("\n[DATABASE] Saving predictions to database...")
        
        # Save 1-day prediction
        db_success_1d = quick_save_prediction(
            model_name="NVIDIA Bull Momentum Model",
            symbol="NVDA",
            current_price=current_price,
            predicted_price=predicted_1d_price,
            confidence=float(confidence.rstrip('%')),
            horizon_days=1,
            suggested_action=signal
        )
        
        # Save 5-day prediction  
        db_success_5d = quick_save_prediction(
            model_name="NVIDIA Bull Momentum Model",
            symbol="NVDA",
            current_price=current_price,
            predicted_price=predicted_5d_price,
            confidence=float(confidence.rstrip('%')),
            horizon_days=5,
            suggested_action=signal
        )
        
        if db_success_1d:
            print("[DATABASE] ✅ 1-day prediction saved to database")
        else:
            print("[DATABASE] ❌ Failed to save 1-day prediction")
            
        if db_success_5d:
            print("[DATABASE] ✅ 5-day prediction saved to database")
        else:
            print("[DATABASE] ❌ Failed to save 5-day prediction")
        
        # Save model metrics to database
        print("[DATABASE] Saving model metrics...")
        nvidia_metrics = {
            "hit_rate": hit_rate,
            "r2_score": r2,
            "mse": mse,
            "total_return": total_return,
            "win_rate": win_rate,
            "avg_return": avg_return
        }
        
        metrics_success = quick_save_metrics("NVIDIA Bull Momentum Model", "NVDA", nvidia_metrics)
        if metrics_success:
            print("[DATABASE] ✅ Model metrics saved to database")
        else:
            print("[DATABASE] ❌ Failed to save model metrics")


    except Exception as e:
        print(f"\nCritical Error in NVIDIA Bull Model: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()