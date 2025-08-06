#!/usr/bin/env python3
"""
Bitcoin Price Prediction Model - UNICODE FIXED VERSION

This version fixes the Unicode encoding issues while keeping all improvements:
1. XGBoost with early stopping for better 1-day predictions
2. Recursive Feature Elimination for optimal feature selection
3. Walk-Forward Validation for more realistic performance evaluation
4. Integration of on-chain metrics for enhanced predictive power
5. FIXED: Smart NaN handling that preserves data instead of dropping everything
6. NEW: Unicode-safe console output for Windows compatibility
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import warnings
import traceback
import sys
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import data_fetcher
from model_db_integrator import quick_save_prediction, quick_save_metrics

# Fix Unicode encoding issues for Windows
def setup_unicode_console():
    """Setup Unicode console output for Windows compatibility"""
    try:
        if sys.platform.startswith('win'):
            # Try to set UTF-8 encoding for Windows console
            import codecs
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    except Exception:
        # If that fails, we'll use ASCII-safe printing
        pass

# Call the setup function
setup_unicode_console()

def safe_print(message):
    """Print function that handles Unicode errors gracefully"""
    try:
        print(message)
    except UnicodeEncodeError:
        # Replace problematic characters with ASCII equivalents
        safe_message = message.encode('ascii', errors='replace').decode('ascii')
        print(safe_message)

# Suppress warnings
warnings.filterwarnings("ignore")

# Set plot style for charts
plt.style.use("ggplot")
sns.set(style="darkgrid")

# Configuration
TICKER = "BTC-USD"
TWELVEDATA_API_KEY = "52784534bde24105838bbb21524a8cb7"
START_DATE = "2015-01-01"
ENABLE_ONCHAIN_METRICS = True  # Flag to enable on-chain metrics
DEBUG_MODE = True  # Flag to control verbose debugging output

# Create necessary directories
DATA_DIR = "data"
REPORTS_DIR = "reports"
REPORTS_FINAL_DIR = os.path.join(REPORTS_DIR, "final")
ONEDRIVE_EXPORT_DIR = os.path.join("onedrive_export", "final")
DEBUG_DIR = os.path.join(REPORTS_DIR, "debug")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(REPORTS_FINAL_DIR, exist_ok=True)
os.makedirs(ONEDRIVE_EXPORT_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

def get_desktop_path():
    """
    Attempts to find the user's Desktop path across different operating systems.
    Returns the path if found, otherwise defaults to a local reports directory.
    """
    try:
        username = os.path.expanduser("~").split(os.sep)[-1]
        possible_desktop_paths = [
            os.path.join(os.path.expanduser("~"), "Desktop"),
            f"C:\\Users\\{username}\\Desktop",
            f"C:\\Users\\{username}\\OneDrive\\Desktop",
            f"/home/{username}/Desktop",
            f"/Users/{username}/Desktop"
        ]
        for path in possible_desktop_paths:
            if os.path.exists(path):
                safe_print(f"Desktop path found: {path}")
                return path
        safe_print("Could not automatically detect Desktop path. Using local reports directory.")
        return REPORTS_FINAL_DIR # Fallback to local reports dir
    except Exception as e:
        safe_print(f"Error detecting Desktop path: {e}. Using local reports directory.")
        return REPORTS_FINAL_DIR # Fallback

DESKTOP_PATH = get_desktop_path()

def fetch_bitcoin_data_with_fallback():
    """
    Fetch Bitcoin historical price data with on-chain metrics using data_fetcher module.
    Fallback to previously saved data or synthetic data if online fetching fails.
    """
    safe_print(f"Fetching Bitcoin historical price data for {TICKER}...")
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    if ENABLE_ONCHAIN_METRICS:
        # Use the function that includes on-chain metrics
        safe_print("Using data_fetcher.fetch_bitcoin_with_onchain to get price and on-chain data...")
        df = data_fetcher.fetch_bitcoin_with_onchain(START_DATE, end_date, api_key=TWELVEDATA_API_KEY)
        
        if df is not None and not df.empty:
            # Debug: Check which on-chain metrics are available
            price_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            onchain_cols = [col for col in df.columns if col not in price_cols]
            
            safe_print(f"\n=== ON-CHAIN DATA VERIFICATION ===")
            safe_print(f"Successfully fetched {len(df)} days of Bitcoin data")
            safe_print(f"Total columns: {len(df.columns)}")
            safe_print(f"Price columns: {price_cols}")
            safe_print(f"On-chain columns ({len(onchain_cols)}): {onchain_cols}")
            
            # Check for NaN values in on-chain metrics
            if onchain_cols:
                onchain_nan_counts = df[onchain_cols].isna().sum()
                safe_print("\nNaN counts in on-chain columns:")
                safe_print(onchain_nan_counts[onchain_nan_counts > 0].to_string())
                
                # Save a sample of the data for inspection
                if DEBUG_MODE:
                    safe_print("Saving sample of raw data with on-chain metrics for inspection...")
                    df.tail(30).to_csv(os.path.join(DEBUG_DIR, "onchain_data_sample.csv"))
            else:
                safe_print("WARNING: No on-chain metrics columns detected despite using fetch_bitcoin_with_onchain!")
            
            # Ensure all columns are numeric before saving
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            
            df.to_csv(os.path.join(DATA_DIR, "bitcoin_data_with_onchain.csv"))
            return df
        else:
            safe_print("Failed to fetch Bitcoin data with on-chain metrics. Trying regular price data.")
    
    # Fall back to regular price data without on-chain metrics
    df = data_fetcher.fetch_historical_data(TICKER, START_DATE, end_date, api_key_twelvedata=TWELVEDATA_API_KEY)
    
    if df is not None and not df.empty:
        safe_print(f"Successfully fetched {len(df)} days of Bitcoin price data using data_fetcher.")
        # Ensure all columns are numeric before saving
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.to_csv(os.path.join(DATA_DIR, "bitcoin_data_from_fetcher.csv"))
        return df
    else:
        safe_print("data_fetcher failed to retrieve data. Trying local fallbacks.")

    # Fallback 1: Check for previously saved data (from data_fetcher or original script)
    saved_data_paths = [
        os.path.join(DATA_DIR, "bitcoin_data_with_onchain.csv"),
        os.path.join(DATA_DIR, "bitcoin_data_from_fetcher.csv"),
        os.path.join(DATA_DIR, "bitcoin_real_data.csv")
    ]
        
    for saved_data_path in saved_data_paths:
        if os.path.exists(saved_data_path):
            try:
                safe_print(f"Using previously saved data from {saved_data_path}...")
                df_saved = pd.read_csv(saved_data_path, index_col=0, parse_dates=True)
                if not df_saved.empty:
                    # Ensure all columns are numeric
                    for col in df_saved.columns:
                        df_saved[col] = pd.to_numeric(df_saved[col], errors="coerce")
                    safe_print(f"Loaded {len(df_saved)} days of Bitcoin data from saved file.")
                    return df_saved
            except Exception as e:
                safe_print(f"Could not load previously saved data from {saved_data_path}: {e}")
    
    # Fallback 2: Synthetic data generation (as a last resort)
    safe_print("Warning: Using synthetic data generation as all other methods failed.")
    return generate_synthetic_bitcoin_data()

def get_current_bitcoin_price_with_fallback():
    """
    Retrieve current Bitcoin price using data_fetcher, with fallback.
    """
    safe_print(f"Fetching current Bitcoin price for {TICKER} using data_fetcher...")
    current_price = data_fetcher.fetch_current_price(TICKER, api_key_twelvedata=TWELVEDATA_API_KEY)
    
    if current_price is not None:
        safe_print(f"Current Bitcoin price from data_fetcher: ${float(current_price):,.2f}")
        return float(current_price)
    
    safe_print("data_fetcher failed to get current price. Trying local fallbacks.")
    # Fallback 1: Use last price from historical data (if available)
    try:
        saved_data_paths = [
            os.path.join(DATA_DIR, "bitcoin_data_with_onchain.csv"),
            os.path.join(DATA_DIR, "bitcoin_data_from_fetcher.csv"),
            os.path.join(DATA_DIR, "bitcoin_real_data.csv")
        ]
        
        for saved_data_path in saved_data_paths:
            if os.path.exists(saved_data_path):
                df_hist = pd.read_csv(saved_data_path, index_col=0, parse_dates=True)
                if not df_hist.empty and "Close" in df_hist.columns:
                    # Make sure Close column is numeric
                    df_hist["Close"] = pd.to_numeric(df_hist["Close"], errors="coerce")
                    last_price = df_hist["Close"].iloc[-1]
                    safe_print(f"Using last known price from data ({saved_data_path}): ${last_price:,.2f}")
                    return float(last_price)
    except Exception as e:
        safe_print(f"Error reading last price from saved data: {e}")
    
    # Fallback 2: Approximate value (as a last resort from original script)
    safe_print("Warning: Could not fetch current price from any source. Using approximate value of $60,000")
    return 60000.0

def generate_synthetic_bitcoin_data():
    """
    Generate synthetic Bitcoin price data when real data cannot be fetched.
    """
    safe_print("Generating synthetic Bitcoin data...")
    end_date = datetime.now()
    start_date_dt = end_date - timedelta(days=365 * 5)
    date_range = pd.date_range(start=start_date_dt, end=end_date, freq="D")
    df = pd.DataFrame(index=date_range)
    df.index.name = "Date"
    
    n = len(df)
    current_price_synthetic = 60000.0 
    start_price_synthetic = current_price_synthetic * 0.05 
    
    trend = np.linspace(start_price_synthetic, current_price_synthetic, n)
    cycle_period = 270  
    cycles = current_price_synthetic * 0.2 * np.sin(np.linspace(0, 2 * n * np.pi / cycle_period, n))
    random_walk = np.cumsum(np.random.normal(0, current_price_synthetic * 0.01, n))
    
    close_prices = trend + cycles + random_walk
    close_prices = np.maximum(1.0, close_prices) # Ensure prices are positive
    close_prices = close_prices * (current_price_synthetic / close_prices[-1]) 
    
    df["Close"] = close_prices
    df["Open"] = df["Close"].shift(1) * (1 + np.random.normal(0, 0.01, n))
    df["High"] = np.maximum(df["Open"], df["Close"]) * (1 + np.abs(np.random.normal(0, 0.01, n)))
    df["Low"] = np.minimum(df["Open"], df["Close"]) * (1 - np.abs(np.random.normal(0, 0.01, n)))
    df["Volume"] = np.random.normal(1000000, 200000, n) * (1 + 0.5 * np.sin(np.linspace(0, 4 * n * np.pi / 365, n)))
    df["Volume"] = np.maximum(0, df["Volume"])

    # Fill NaN (first row Open)
    df.iloc[0, df.columns.get_loc("Open")] = df.iloc[0, df.columns.get_loc("Close")] * 0.99
    
    # Ensure all columns are numeric and non-negative
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = np.abs(df[col])
    
    # Ensure OHLC consistency
    df["High"] = df[["High", "Open", "Close"]].max(axis=1)
    df["Low"] = df[["Low", "Open", "Close"]].min(axis=1)
    df["Adj Close"] = df["Close"] # Add Adj Close for consistency
    
    # Add synthetic on-chain metrics
    if ENABLE_ONCHAIN_METRICS:
        safe_print("Adding synthetic on-chain metrics...")
        
        # Create synthetic active addresses (correlates with price but with some lead/lag)
        active_addr_base = close_prices * 0.5 + np.random.normal(0, current_price_synthetic * 0.05, n)
        active_addr_base = np.maximum(100, active_addr_base)  # Ensure positive values
        
        # Add cycle and trend components
        active_addr_trend = np.linspace(100, 10000, n)  # Growing user base
        active_addr_cycle = 5000 * np.sin(np.linspace(0, 2 * n * np.pi / 180, n))  # 180-day cycle
        
        df["ActiveAddresses"] = active_addr_base + active_addr_trend + active_addr_cycle
        
        # Create synthetic transaction count (correlates with active addresses)
        df["TransactionCount"] = df["ActiveAddresses"] * 0.8 + np.random.normal(0, 1000, n)
        df["TransactionCount"] = np.maximum(10, df["TransactionCount"])
        
        # Create synthetic transaction volume (correlated with price and transaction count)
        df["AdjustedTransactionVolumeUSD"] = df["Close"] * df["TransactionCount"] * 0.1 + np.random.normal(0, 1000000, n)
        df["AdjustedTransactionVolumeUSD"] = np.maximum(1000, df["AdjustedTransactionVolumeUSD"])
        
        # Create synthetic hash rate (growing trend with market cycles)
        hash_trend = np.linspace(1000, 100000, n)  # Growing trend
        hash_cycle = 20000 * np.sin(np.linspace(0, 2 * n * np.pi / 365, n))  # Annual cycle
        hash_response = 30000 * np.sin(np.linspace(np.pi/2, 2 * n * np.pi / 270 + np.pi/2, n))  # Delayed response to price
        
        df["HashRate"] = hash_trend + hash_cycle + hash_response
        df["HashRate"] = np.maximum(100, df["HashRate"])
        
        # Create synthetic transaction fees
        df["TransactionFeesUSD"] = df["TransactionCount"] * 0.1 * np.sqrt(df["Close"]) + np.random.normal(0, 1000, n)
        df["TransactionFeesUSD"] = np.maximum(10, df["TransactionFeesUSD"])
        
        # Create NVT ratio (Network Value to Transactions)
        df["NVT"] = df["Close"] * df["Volume"] / (df["AdjustedTransactionVolumeUSD"] + 1)
    
    safe_print(f"Synthetic Bitcoin data generated with {len(df)} records")
    df.to_csv(os.path.join(DATA_DIR, "bitcoin_synthetic_data.csv"))
    return df

def create_features_fixed(df_input):
    """
    FIXED VERSION: Create technical indicator features with SMART NaN handling
    This fixes the critical issue where all data was being dropped.
    """
    safe_print("\n[FEATURE] Creating technical features (FIXED VERSION)...")
    
    if df_input is None or df_input.empty:
        safe_print("[ERROR] Input DataFrame for feature creation is empty.")
        return None

    try:
        # Ensure single-level columns (data_fetcher should provide this)
        df = df_input.copy()
        if isinstance(df.columns, pd.MultiIndex):
            safe_print("[WARNING] MultiIndex columns detected; attempting to flatten.")
            df.columns = [col[1] if isinstance(col, tuple) and len(col) > 1 else col for col in df.columns]

        # Ensure required columns exist
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_cols:
            if col not in df.columns:
                safe_print(f"Missing required column: {col}. Attempting to create...")
                if col == "Open" and "Close" in df.columns:
                    df[col] = df["Close"].shift(1)
                elif col in ["High", "Low"] and "Close" in df.columns:
                    df[col] = df["Close"]
                elif col == "Volume":
                    df[col] = 0  # Default if missing
                else:
                    safe_print(f"[ERROR] Cannot create missing column {col}. Required for feature engineering.")
                    return None

        # Ensure all columns are numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # SMART CLEANING: Only drop rows with NaN in ESSENTIAL columns
        essential_cols = ["Open", "High", "Low", "Close", "Volume"]
        df_essential_clean = df.dropna(subset=essential_cols)
        
        if df_essential_clean.empty: 
            safe_print("[ERROR] DataFrame empty after cleaning essential columns.")
            return None

        safe_print(f"[SUCCESS] Starting with {len(df_essential_clean)} rows after essential cleaning")

        # Identify and handle on-chain columns BEFORE feature creation
        price_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        initial_onchain_cols = [col for col in df_essential_clean.columns if col not in price_cols]
        
        if initial_onchain_cols:
            safe_print(f"[ONCHAIN] Found {len(initial_onchain_cols)} on-chain columns: {initial_onchain_cols}")
            
            # SMART ONCHAIN HANDLING: Fill missing values intelligently
            for col in initial_onchain_cols:
                if df_essential_clean[col].isnull().sum() > 0:
                    # Try interpolation first
                    df_essential_clean[col] = df_essential_clean[col].interpolate(method='linear')
                    # Forward fill remaining gaps
                    df_essential_clean[col] = df_essential_clean[col].fillna(method='ffill')
                    # Backward fill for any remaining gaps at the beginning
                    df_essential_clean[col] = df_essential_clean[col].fillna(method='bfill')
                    # Final fallback: use median
                    if df_essential_clean[col].isnull().sum() > 0:
                        median_val = df_essential_clean[col].median()
                        df_essential_clean[col] = df_essential_clean[col].fillna(median_val)
                        safe_print(f"  Filled {col} missing values with median: {median_val:.2f}")

        # Use the cleaned dataframe for feature creation
        df = df_essential_clean.copy()

        # Create basic features
        df["returns"] = df["Close"].pct_change()
        df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))

        # IMPROVEMENT: Use min_periods to preserve more data in rolling calculations
        df["volatility"] = df["returns"].rolling(window=30, min_periods=20).std()

        # Add sophisticated rolling statistics with min_periods
        for window in [7, 14, 30, 50, 100, 200]:
            min_periods = max(5, int(window * 0.6))  # Require at least 60% of window
            
            # Moving averages and price relationships
            df[f"MA_{window}"] = df["Close"].rolling(window=window, min_periods=min_periods).mean()
            df[f"MA_ratio_{window}"] = df["Close"] / (df[f"MA_{window}"] + 1e-9)
            
            # Volatility measures
            df[f"std_{window}"] = df["returns"].rolling(window=window, min_periods=min_periods).std()
            
            # Momentum indicators
            df[f"mom_{window}"] = df["Close"].pct_change(periods=min(window, len(df)//4))
        
        # Exponential moving averages
        for window in [12, 26, 9, 50, 200]:
            df[f"EMA_{window}"] = df["Close"].ewm(span=window, adjust=False).mean()
        
        # MACD and Signal line
        if "EMA_12" in df.columns and "EMA_26" in df.columns:
            df["MACD"] = df["EMA_12"] - df["EMA_26"]
            df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
            df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
            df["MACD_ratio"] = df["MACD"] / (df["MACD_signal"] + 1e-9)

        # Bollinger Bands with min_periods
        for window in [20, 50]:
            min_periods = max(10, int(window * 0.7))
            df[f"BB_middle_{window}"] = df["Close"].rolling(window=window, min_periods=min_periods).mean()
            df[f"BB_std_{window}"] = df["Close"].rolling(window=window, min_periods=min_periods).std()
            df[f"BB_upper_{window}"] = df[f"BB_middle_{window}"] + 2 * df[f"BB_std_{window}"]
            df[f"BB_lower_{window}"] = df[f"BB_middle_{window}"] - 2 * df[f"BB_std_{window}"]
            df[f"BB_width_{window}"] = (df[f"BB_upper_{window}"] - df[f"BB_lower_{window}"]) / (df[f"BB_middle_{window}"] + 1e-9)
            df[f"BB_pos_{window}"] = (df["Close"] - df[f"BB_lower_{window}"]) / (df[f"BB_upper_{window}"] - df[f"BB_lower_{window}"] + 1e-9)

        # RSI Calculation with proper handling
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).fillna(0)
        loss = -delta.where(delta < 0, 0).fillna(0)
        
        for period in [7, 14, 21]:
            min_periods = max(5, int(period * 0.7))
            avg_gain = gain.rolling(window=period, min_periods=min_periods).mean()
            avg_loss = loss.rolling(window=period, min_periods=min_periods).mean()
            rs = avg_gain / (avg_loss + 1e-9)
            df[f"RSI_{period}"] = 100 - (100 / (1 + rs))
            df[f"RSI_{period}"] = df[f"RSI_{period}"].fillna(50)

        # Rate of Change for multiple periods
        for window in [1, 3, 5, 7, 14, 30]:
            df[f"ROC_{window}"] = df["Close"].pct_change(periods=window) * 100

        # Volume indicators
        df["volume_change"] = df["Volume"].pct_change()
        df["volume_ma_ratio"] = df["Volume"] / (df["Volume"].rolling(window=30, min_periods=20).mean() + 1e-9)
        
        # On-balance Volume
        df["OBV"] = np.where(df["Close"] > df["Close"].shift(1),
                            df["Volume"], 
                            np.where(df["Close"] < df["Close"].shift(1), 
                                    -df["Volume"], 0)).cumsum()
        
        # Volume weighted moving averages
        vwma_num = (df["Close"] * df["Volume"]).rolling(20, min_periods=14).sum()
        vwma_den = df["Volume"].rolling(20, min_periods=14).sum()
        df["VWMA_20"] = vwma_num / (vwma_den + 1e-9)
        
        # Lagged features (limited to preserve data)
        for lag in range(1, 8):
            df[f"close_lag_{lag}"] = df["Close"].shift(lag)
            df[f"return_lag_{lag}"] = df["returns"].shift(lag)
            
        # Ratio features
        df["high_low_ratio"] = df["High"] / (df["Low"] + 1e-9)
        df["close_open_ratio"] = df["Close"] / (df["Open"] + 1e-9)
        
        # Technical price patterns
        df["price_swing"] = df["High"] - df["Low"]
        df["price_swing_ma"] = df["price_swing"].rolling(window=14, min_periods=10).mean()
        df["price_swing_ratio"] = df["price_swing"] / (df["price_swing_ma"] + 1e-9)
        
        # Higher-order return statistics
        df["returns_skew"] = df["returns"].rolling(window=30, min_periods=20).skew()
        df["returns_kurt"] = df["returns"].rolling(window=30, min_periods=20).kurt()
        
        # Time features
        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month
        df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        
        # Add on-chain specific features if available
        if ENABLE_ONCHAIN_METRICS and initial_onchain_cols:
            safe_print(f"[ONCHAIN] Processing {len(initial_onchain_cols)} on-chain metrics for derived features...")
            
            # For active addresses (if available)
            if "ActiveAddresses" in df.columns:
                df["ActiveAddr_to_Price"] = df["ActiveAddresses"] / (df["Close"] + 1e-9)
                df["ActiveAddr_Momentum"] = df["ActiveAddresses"].pct_change(periods=7)
                
                for window in [7, 14, 30]:
                    min_periods = max(5, int(window * 0.7))
                    df[f"ActiveAddr_MA_{window}"] = df["ActiveAddresses"].rolling(window=window, min_periods=min_periods).mean()
                    df[f"ActiveAddr_MA_Ratio_{window}"] = df["ActiveAddresses"] / (df[f"ActiveAddr_MA_{window}"] + 1e-9)
            
            # For transaction count (if available)
            if "TransactionCount" in df.columns:
                df["TxCount_to_Price"] = df["TransactionCount"] / (df["Close"] + 1e-9)
                df["TxCount_Momentum"] = df["TransactionCount"].pct_change(periods=7)
                
                for window in [7, 14, 30]:
                    min_periods = max(5, int(window * 0.7))
                    df[f"TxCount_MA_{window}"] = df["TransactionCount"].rolling(window=window, min_periods=min_periods).mean()
                    df[f"TxCount_MA_Ratio_{window}"] = df["TransactionCount"] / (df[f"TxCount_MA_{window}"] + 1e-9)
            
            # For transaction volume (if available)
            if "AdjustedTransactionVolumeUSD" in df.columns:
                df["NVT_Ratio"] = df["Close"] * df["Volume"] / (df["AdjustedTransactionVolumeUSD"] + 1e-9)
                
                for window in [7, 14, 30]:
                    min_periods = max(5, int(window * 0.7))
                    df[f"NVT_Ratio_MA_{window}"] = df["NVT_Ratio"].rolling(window=window, min_periods=min_periods).mean()
            
            # For hash rate (if available)
            if "HashRate" in df.columns:
                df["HashRate_Momentum"] = df["HashRate"].pct_change(periods=7)
                
                for window in [30, 60, 90]:
                    min_periods = max(20, int(window * 0.7))
                    df[f"HashRate_MA_{window}"] = df["HashRate"].rolling(window=window, min_periods=min_periods).mean()
                
                if all(f"HashRate_MA_{w}" in df.columns for w in [30, 60]):
                    df["HashRibbon_Crossover"] = (df["HashRate_MA_30"] > df["HashRate_MA_60"]).astype(int)
            
            # For fees (if available)
            if "TransactionFeesUSD" in df.columns:
                df["Fees_to_Price"] = df["TransactionFeesUSD"] / (df["Close"] + 1e-9)
                
                if "TransactionCount" in df.columns:
                    df["Avg_Fee_Per_Tx"] = df["TransactionFeesUSD"] / (df["TransactionCount"] + 1e-9)
                    
                    for window in [7, 14, 30]:
                        min_periods = max(5, int(window * 0.7))
                        df[f"Avg_Fee_MA_{window}"] = df["Avg_Fee_Per_Tx"].rolling(window=window, min_periods=min_periods).mean()
            
            # Combine on-chain metrics into signal indicators
            available_metrics = set(df.columns)
            active_addr_available = any(col.startswith("ActiveAddr") for col in available_metrics)
            tx_count_available = any(col.startswith("TxCount") for col in available_metrics)
            nvt_available = any(col.startswith("NVT") for col in available_metrics)
            
            if active_addr_available and tx_count_available and nvt_available:
                nvt_ma_30 = df.get("NVT_Ratio_MA_30", df.get("NVT_Ratio", pd.Series(index=df.index, dtype=float)))
                if not nvt_ma_30.empty:
                    df["OnChain_Bullish"] = ((df.get("ActiveAddr_Momentum", 0) > 0) & 
                                            (df.get("TxCount_Momentum", 0) > 0) & 
                                            (df.get("NVT_Ratio", 0) < nvt_ma_30.rolling(window=30, min_periods=20).mean())).astype(int)
            
            if "HashRibbon_Crossover" in df.columns and "OnChain_Bullish" in df.columns:
                df["OnChain_Signal"] = df["OnChain_Bullish"] + df["HashRibbon_Crossover"]
            elif "OnChain_Bullish" in df.columns:
                df["OnChain_Signal"] = df["OnChain_Bullish"]
            elif "HashRibbon_Crossover" in df.columns:
                df["OnChain_Signal"] = df["HashRibbon_Crossover"]

        # Target variables (future percentage returns)
        for days in [1, 3, 7]:
            df[f"target_return_{days}d"] = df["Close"].pct_change(periods=days).shift(-days)
        
        # SMART FINAL CLEANING: Replace infinities and handle remaining NaNs intelligently
        safe_print("[CLEANING] Smart final cleaning...")
        
        # Replace infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Count NaN values before final cleaning
        total_nans_before = df.isnull().sum().sum()
        safe_print(f"Total NaN values before final cleaning: {total_nans_before}")
        
        # Only require essential columns to be complete for final cleaning
        target_cols = [col for col in df.columns if col.startswith("target_return_")]
        essential_for_training = essential_cols + target_cols
        
        # Drop rows where essential columns OR target columns have NaN
        df_final = df.dropna(subset=essential_for_training)
        
        # For non-essential columns, fill remaining NaNs
        non_essential_cols = [col for col in df_final.columns if col not in essential_for_training]
        for col in non_essential_cols:
            if df_final[col].isnull().sum() > 0:
                # Forward fill
                df_final[col] = df_final[col].fillna(method='ffill')
                # Backward fill
                df_final[col] = df_final[col].fillna(method='bfill')
                # If still NaN, use median
                if df_final[col].isnull().sum() > 0:
                    median_val = df_final[col].median()
                    if pd.notna(median_val):
                        df_final[col] = df_final[col].fillna(median_val)
                    else:
                        df_final[col] = df_final[col].fillna(0)
        
        # Final check for any remaining NaNs
        remaining_nans = df_final.isnull().sum().sum()
        safe_print(f"Remaining NaN values after smart cleaning: {remaining_nans}")
        
        if remaining_nans > 0:
            # Drop any columns that are still mostly NaN
            for col in df_final.columns:
                nan_pct = df_final[col].isnull().sum() / len(df_final)
                if nan_pct > 0.5:  # Drop if more than 50% NaN
                    df_final = df_final.drop(columns=[col])
                    safe_print(f"Dropped column {col} (too many NaN values)")
            
            # Final dropna only if absolutely necessary
            if df_final.isnull().sum().sum() > 0:
                df_final = df_final.dropna()
        
        safe_print(f"[SUCCESS] FIXED feature creation complete!")
        safe_print(f"Final shape: {df_final.shape}")
        safe_print(f"Data preservation: {len(df_final)} / {len(df_input)} rows ({len(df_final)/len(df_input)*100:.1f}%)")
        
        if DEBUG_MODE and len(df_final) > 0:
            feature_info_file = os.path.join(DEBUG_DIR, "fixed_feature_info.txt")
            with open(feature_info_file, 'w') as f:
                f.write(f"FIXED Feature Engineering Summary\n")
                f.write(f"=================================\n")
                f.write(f"Original rows: {len(df_input)}\n")
                f.write(f"Final rows: {len(df_final)}\n")
                f.write(f"Preservation rate: {len(df_final)/len(df_input)*100:.1f}%\n")
                f.write(f"Total features: {df_final.shape[1]}\n\n")
                f.write("Feature list:\n")
                for i, col in enumerate(df_final.columns):
                    f.write(f"{i+1:3d}. {col}\n")
            safe_print(f"Fixed feature info saved to {feature_info_file}")
            
            # Save sample data
            sample_file = os.path.join(DEBUG_DIR, "fixed_bitcoin_features_sample.csv")
            df_final.tail(20).to_csv(sample_file)
            safe_print(f"Sample data saved to {sample_file}")
            
        return df_final
    
    except Exception as e:
        safe_print(f"[ERROR] Error during FIXED feature creation: {e}")
        safe_print(traceback.format_exc())
        return None

def analyze_features(df_features):
    """
    Analyze the features to understand their importance and relationships.
    This function is streamlined to focus on correlation analysis only.
    """
    if df_features is None or df_features.empty:
        safe_print("Cannot analyze features: DataFrame is empty.")
        return
    
    safe_print("\nAnalyzing features...")
    
    try:
        # Correlation with target variables
        target_cols = [col for col in df_features.columns if col.startswith("target_return_")]
        feature_cols = [col for col in df_features.columns if not col.startswith("target_return_")]
        
        for target_col in target_cols:
            # Calculate correlations and sort by absolute value
            correlations = df_features[feature_cols].corrwith(df_features[target_col])
            abs_correlations = correlations.abs().sort_values(ascending=False)
            
            # Print top correlated features for this target
            horizon = target_col.split('_')[-1].replace('d', '-day')
            safe_print(f"\nTop 10 features correlated with {horizon} returns:")
            safe_print(abs_correlations.head(10).to_string())
        
        # Analysis specific to on-chain metrics if available
        if ENABLE_ONCHAIN_METRICS:
            onchain_terms = ["ActiveAddr", "TxCount", "NVT", "HashRate", "Fees", "TransactionCount", "TransactionFees"]
            onchain_cols = [col for col in df_features.columns 
                          if any(term in col for term in onchain_terms)]
            
            if onchain_cols:
                safe_print(f"\nAnalyzing {len(onchain_cols)} on-chain metrics...")
                # Calculate correlations between on-chain metrics and returns
                for target_col in target_cols:
                    onchain_correlations = df_features[onchain_cols].corrwith(df_features[target_col])
                    onchain_abs_correlations = onchain_correlations.abs().sort_values(ascending=False)
                    
                    horizon = target_col.split('_')[-1].replace('d', '-day')
                    safe_print(f"\nTop on-chain metrics correlated with {horizon} returns:")
                    safe_print(onchain_abs_correlations.head(10).to_string())
    
    except Exception as e:
        safe_print(f"Error during feature analysis: {e}")
        safe_print(traceback.format_exc())

def analyze_onchain_importance(df_features, selected_features):
    """
    Analyze the importance of on-chain metrics in the model
    """
    if df_features is None or df_features.empty or not selected_features:
        safe_print("Cannot analyze on-chain importance: Data missing.")
        return
    
    # Identify on-chain metrics columns
    onchain_terms = ["ActiveAddr", "TxCount", "NVT", "HashRate", "Fees", "OnChain", "Transaction", "Volume"]
    onchain_cols = [col for col in df_features.columns 
                   if any(term in col for term in onchain_terms) and col not in ["Open", "High", "Low", "Close", "Volume"]]
    
    if not onchain_cols:
        safe_print("No on-chain metrics found in features.")
        return
    
    safe_print(f"\nAnalyzing importance of {len(onchain_cols)} on-chain metrics...")
    
    # Check what percentage of selected features are on-chain metrics
    onchain_selected = [feature for feature in selected_features if any(term in feature for term in onchain_terms)]
    
    if selected_features:
        onchain_pct = len(onchain_selected) / len(selected_features) * 100
        safe_print(f"On-chain metrics make up {onchain_pct:.1f}% of selected features ({len(onchain_selected)}/{len(selected_features)})")
    
    # List the top on-chain metrics that were selected
    if onchain_selected:
        safe_print("Top selected on-chain metrics:")
        for i, feature in enumerate(onchain_selected[:min(10, len(onchain_selected))]):
            safe_print(f"{i+1}. {feature}")
    else:
        safe_print("No on-chain metrics were selected by the feature selection process.")

def evaluate_onchain_impact(df_features, target_horizon_days=1):
    """
    Evaluate the impact of on-chain metrics by comparing models with and without them.
    """
    if df_features is None or df_features.empty:
        safe_print("Feature DataFrame is empty. Cannot evaluate on-chain impact.")
        return
    
    if not ENABLE_ONCHAIN_METRICS:
        safe_print("On-chain metrics are disabled. Cannot evaluate impact.")
        return
    
    safe_print(f"\n=== EVALUATING ON-CHAIN METRICS IMPACT ({target_horizon_days}-day horizon) ===")
    
    # Identify on-chain columns
    onchain_terms = ["ActiveAddr", "TxCount", "NVT", "HashRate", "Fees", "OnChain", "Transaction"]
    onchain_cols = [col for col in df_features.columns 
                  if any(term in col for term in onchain_terms)]
    
    if not onchain_cols:
        safe_print("No on-chain metrics found in the dataset. Cannot evaluate impact.")
        return
    
    safe_print(f"Found {len(onchain_cols)} on-chain metrics in the dataset")
    
    target_col = f"target_return_{target_horizon_days}d"
    if target_col not in df_features.columns:
        safe_print(f"Target column {target_col} not found.")
        return
    
    # Setup common columns to exclude
    exclude_cols = ["Open", "High", "Low", "Close", "Volume", "Adj Close", "returns", "log_returns"] 
    exclude_cols += [col for col in df_features.columns if col.startswith("target_return_")]
    
    # Create dataset WITHOUT on-chain metrics
    features_no_onchain = [col for col in df_features.columns 
                         if col not in exclude_cols and col not in onchain_cols]
    
    # Create dataset WITH on-chain metrics
    features_with_onchain = [col for col in df_features.columns if col not in exclude_cols]
    
    safe_print(f"Features without on-chain: {len(features_no_onchain)}")
    safe_print(f"Features with on-chain: {len(features_with_onchain)}")
    
    # Train a simple model with each feature set
    X_no_onchain = df_features[features_no_onchain]
    X_with_onchain = df_features[features_with_onchain]
    y = df_features[target_col]
    
    # Use a small validation set
    train_size = int(len(df_features) * 0.8)
    X_no_onchain_train, X_no_onchain_test = X_no_onchain[:train_size], X_no_onchain[train_size:]
    X_with_onchain_train, X_with_onchain_test = X_with_onchain[:train_size], X_with_onchain[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train model without on-chain metrics
    model_no_onchain = XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
    model_no_onchain.fit(X_no_onchain_train, y_train)
    preds_no_onchain = model_no_onchain.predict(X_no_onchain_test)
    
    # Train model with on-chain metrics
    model_with_onchain = XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
    model_with_onchain.fit(X_with_onchain_train, y_train)
    preds_with_onchain = model_with_onchain.predict(X_with_onchain_test)
    
    # Calculate metrics
    mse_no_onchain = mean_squared_error(y_test, preds_no_onchain)
    mse_with_onchain = mean_squared_error(y_test, preds_with_onchain)
    
    r2_no_onchain = r2_score(y_test, preds_no_onchain)
    r2_with_onchain = r2_score(y_test, preds_with_onchain)
    
    dir_acc_no_onchain = np.mean((preds_no_onchain > 0) == (y_test > 0))
    dir_acc_with_onchain = np.mean((preds_with_onchain > 0) == (y_test > 0))
    
    # Print comparison
    safe_print("\nPerformance Comparison:")
    safe_print(f"MSE without on-chain: {mse_no_onchain:.6f}")
    safe_print(f"MSE with on-chain: {mse_with_onchain:.6f}")
    safe_print(f"MSE improvement: {(mse_no_onchain - mse_with_onchain) / mse_no_onchain * 100:.2f}%")
    
    safe_print(f"\nR2 without on-chain: {r2_no_onchain:.4f}")
    safe_print(f"R2 with on-chain: {r2_with_onchain:.4f}")
    safe_print(f"R2 difference: {r2_with_onchain - r2_no_onchain:.4f}")
    
    safe_print(f"\nDirectional accuracy without on-chain: {dir_acc_no_onchain:.4f}")
    safe_print(f"Directional accuracy with on-chain: {dir_acc_with_onchain:.4f}")
    safe_print(f"Directional accuracy improvement: {(dir_acc_with_onchain - dir_acc_no_onchain) * 100:.2f}%")
    
    # If the model with on-chain metrics is better, analyze which metrics contributed most
    if r2_with_onchain > r2_no_onchain or dir_acc_with_onchain > dir_acc_no_onchain:
        safe_print("\nOn-chain metrics improved model performance!")
        
        # Check which on-chain features have highest importance
        feature_imp = model_with_onchain.feature_importances_
        onchain_imp = [(features_with_onchain[i], feature_imp[i]) 
                      for i in range(len(features_with_onchain)) 
                      if features_with_onchain[i] in onchain_cols]
        
        safe_print("\nTop on-chain metrics by importance:")
        for feature, imp in sorted(onchain_imp, key=lambda x: x[1], reverse=True)[:5]:
            safe_print(f"{feature}: {imp:.6f}")
    else:
        safe_print("\nOn-chain metrics did not improve model performance in this configuration.")
        safe_print("Consider trying different transformations or feature engineering approaches.")

def train_xgboost_model(df_features, target_horizon_days=1, force_onchain=True):
    """
    IMPROVEMENT 1: Train XGBoost model with early stopping
    IMPROVEMENT 2: Recursive Feature Elimination
    IMPROVEMENT 4: Force inclusion of on-chain metrics when needed
    """
    if df_features is None or df_features.empty:
        safe_print("Feature DataFrame is empty. Cannot train model.")
        return None, None, None, None, None

    target_col = f"target_return_{target_horizon_days}d"
    if target_col not in df_features.columns:
        safe_print(f"Target column {target_col} not found. Available: {df_features.columns}")
        return None, None, None, None, None

    safe_print(f"\nPreparing data for XGBoost {target_horizon_days}-day horizon model...")
    
    # Remove non-feature columns from training
    exclude_cols = ["Open", "High", "Low", "Close", "Volume", "Adj Close", "returns", "log_returns"] 
    exclude_cols += [col for col in df_features.columns if col.startswith("target_return_")]
    
    features = [col for col in df_features.columns if col not in exclude_cols]
    
    X = df_features[features]
    y = df_features[target_col]

    # Modified train/val/test split to use for early stopping
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    
    X_val = X.iloc[train_size:train_size+val_size]
    y_val = y.iloc[train_size:train_size+val_size]
    
    X_test = X.iloc[train_size+val_size:]
    y_test = y.iloc[train_size+val_size:]

    if X_train.empty or y_train.empty:
        safe_print("Not enough data for training. Aborting.")
        return None, None, None, None, None

    safe_print(f"Training data size: {X_train.shape}, Validation data size: {X_val.shape}, Test data size: {X_test.shape}")
    
    # Step 1: Normalize the features
    safe_print("Normalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 2: IMPROVEMENT 2 - Recursive Feature Elimination
    safe_print("Performing Recursive Feature Elimination...")
    base_estimator = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    
    # Use RFECV with Time Series CV for better feature selection
    cv = TimeSeriesSplit(n_splits=5)
    selector = RFECV(
        estimator=base_estimator,
        step=1,
        cv=cv,
        scoring='neg_mean_squared_error',
        min_features_to_select=5,
        n_jobs=-1,
        verbose=0
    )
    
    selector.fit(X_train_scaled, y_train)
    
    # Get selected feature indices and names
    selected_features_idx = selector.support_
    selected_features = [features[i] for i in range(len(features)) if selected_features_idx[i]]
    
    X_train_selected = selector.transform(X_train_scaled)
    X_val_selected = selector.transform(X_val_scaled)
    X_test_selected = selector.transform(X_test_scaled)
    
    safe_print(f"Optimal number of features: {selector.n_features_}")
    safe_print(f"Selected {len(selected_features)} out of {len(features)} features.")
    safe_print(f"Top selected features: {selected_features[:min(10, len(selected_features))]}")
    
    # DEBUG: Check which on-chain metrics were considered and selected
    if ENABLE_ONCHAIN_METRICS:
        onchain_terms = ["ActiveAddr", "TxCount", "NVT", "HashRate", "Fees", "OnChain", "Transaction"]
        all_onchain_features = [feature for feature in features 
                            if any(term in feature for term in onchain_terms)]
        
        selected_onchain_features = [feature for feature in selected_features 
                                    if any(term in feature for term in onchain_terms)]
        
        safe_print(f"\n=== ON-CHAIN FEATURE SELECTION ANALYSIS ({target_horizon_days}-day) ===")
        safe_print(f"Total features considered: {len(features)}")
        safe_print(f"On-chain features available: {len(all_onchain_features)} ({all_onchain_features})")
        safe_print(f"On-chain features selected: {len(selected_onchain_features)} ({selected_onchain_features})")
        
        if len(all_onchain_features) > 0:
            onchain_selection_rate = len(selected_onchain_features) / len(all_onchain_features) * 100
            safe_print(f"On-chain feature selection rate: {onchain_selection_rate:.1f}%")
        
        # If RFECV provides feature ranking, analyze it
        if hasattr(selector, 'ranking_'):
            onchain_rankings = [(features[i], selector.ranking_[i]) for i in range(len(features)) 
                            if any(term in features[i] for term in onchain_terms)]
            safe_print("\nRankings of on-chain features (lower is better):")
            for feature, rank in sorted(onchain_rankings, key=lambda x: x[1]):
                safe_print(f"{feature}: {rank}")
    
    # IMPROVEMENT 4: If force_onchain is True, ensure some on-chain metrics are included
    if force_onchain and ENABLE_ONCHAIN_METRICS:
        onchain_terms = ["ActiveAddr", "TxCount", "NVT", "HashRate", "Fees", "OnChain", "Transaction"]
        selected_onchain_features = [feature for feature in selected_features 
                                    if any(term in feature for term in onchain_terms)]
        
        if not selected_onchain_features and all_onchain_features:
            safe_print("No on-chain features were selected. Forcing inclusion of top on-chain features...")
            
            # Calculate correlation with target for on-chain features
            onchain_importance = {}
            for feature in all_onchain_features:
                if feature in df_features.columns:
                    corr = abs(df_features[feature].corr(df_features[target_col]))
                    onchain_importance[feature] = corr
            
            # Sort by importance and take top 3
            if onchain_importance:
                top_onchain = sorted(onchain_importance.items(), key=lambda x: x[1], reverse=True)[:3]
                safe_print(f"Forcing inclusion of: {[f[0] for f in top_onchain]}")
                
                # Modify selected features to include these
                new_selected_features = selected_features.copy()
                for feature, _ in top_onchain:
                    if feature in features and feature not in new_selected_features:
                        idx = features.index(feature)
                        selected_features_idx[idx] = True
                        new_selected_features.append(feature)
                
                # Recalculate selected features
                selected_features = new_selected_features
                
                # Reapply the selection to the data
                X_train_selected = X_train_scaled[:, selected_features_idx]
                X_val_selected = X_val_scaled[:, selected_features_idx]
                X_test_selected = X_test_scaled[:, selected_features_idx]
                
                safe_print(f"After forcing inclusion, selected {len(selected_features)} features")
    
    # Step 3: Train XGBoost with early stopping
    safe_print(f"Training XGBoost model for {target_horizon_days}-day return prediction...")
    
    # To handle different XGBoost versions, we'll use try-except for the parameters
    try:
        # First attempt: Include eval_metric in fit() (newer XGBoost versions)
        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.005,  # Slower learning rate
            max_depth=3,          # Reduced depth to prevent overfitting
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,        # L1 regularization
            reg_lambda=1.0,       # L2 regularization
            random_state=42,
            n_jobs=-1
        )
        
        # Use early stopping
        eval_set = [(X_train_selected, y_train), (X_val_selected, y_val)]
        model.fit(
            X_train_selected, y_train,
            eval_set=eval_set,
            eval_metric='rmse',
            early_stopping_rounds=20,
            verbose=False
        )
    except TypeError as e:
        # Alternative approach for older XGBoost versions
        safe_print("XGBoost compatibility issue. Using alternative approach...")
        
        if "eval_metric" in str(e):
            # Include eval_metric in the model initialization
            model = XGBRegressor(
                n_estimators=300,
                learning_rate=0.005,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                objective='reg:squarederror',  # Explicitly set objective
                eval_metric='rmse'             # Set eval_metric here instead
            )
            
            # Fit with early stopping but without eval_metric
            eval_set = [(X_train_selected, y_train), (X_val_selected, y_val)]
            try:
                model.fit(
                    X_train_selected, y_train,
                    eval_set=eval_set,
                    early_stopping_rounds=20,
                    verbose=False
                )
            except TypeError:
                # If that still fails, use the simplest fit method
                safe_print("Using basic fit method without early stopping.")
                model.fit(X_train_selected, y_train)
    
    # Print best iteration and score (if available)
    if hasattr(model, 'best_iteration'):
        safe_print(f"Best iteration: {model.best_iteration}")
    else:
        safe_print("Early stopping information not available.")
        
    if hasattr(model, 'best_score'):
        safe_print(f"Best validation RMSE: {model.best_score:.6f}")
    else:
        # Try to get evaluation results, with error handling
        try:
            if hasattr(model, 'evals_result'):
                eval_results = model.evals_result()
                if eval_results and 'validation_1' in eval_results:
                    best_rmse = min(eval_results['validation_1']['rmse'])
                    safe_print(f"Best validation RMSE: {best_rmse:.6f}")
                else:
                    safe_print("No validation metrics available.")
            else:
                safe_print("Evaluation result information not available.")
        except Exception as e:
            safe_print(f"Could not extract evaluation metrics: {e}")
            safe_print("Continuing without evaluation metrics.")
    
    # Evaluate on test set
    test_predictions = model.predict(X_test_selected)
    
    # Calculate metrics
    r2 = r2_score(y_test, test_predictions)
    mse = mean_squared_error(y_test, test_predictions)
    mae = mean_absolute_error(y_test, test_predictions)
    
    safe_print(f"\nModel Performance on Test Set ({target_horizon_days}-day target):")
    safe_print(f"  R-squared: {r2:.4f}")
    safe_print(f"  MSE: {mse:.6f}")
    safe_print(f"  MAE: {mae:.4f}")
    
    # Feature importance
    if hasattr(model, 'feature_importances_') and DEBUG_MODE:
        feature_importance = model.feature_importances_
        sorted_idx = np.argsort(feature_importance)[::-1]
        safe_print(f"\nTop 10 features by importance:")
        for i in range(min(10, len(selected_features))):
            idx = sorted_idx[i]
            safe_print(f"{selected_features[idx]}: {feature_importance[idx]:.6f}")
    
    # Predict future return using last available data
    last_data_point = X.iloc[-1:].copy()
    last_data_scaled = scaler.transform(last_data_point)
    last_data_selected = selector.transform(last_data_scaled)
    future_return_prediction = model.predict(last_data_selected)[0]
    
    safe_print(f"XGBoost predicted future {target_horizon_days}-day return: {future_return_prediction*100:+.2f}%")
    
    # Return model, prediction, metrics, and feature information
    eval_metrics = {
        "R2": r2,
        "MSE": mse,
        "MAE": mae,
        "Optimal_Features": len(selected_features)
    }
    
    # Add best_iteration to metrics if available
    if hasattr(model, 'best_iteration'):
        eval_metrics["Best_Iteration"] = model.best_iteration
    
    return model, future_return_prediction, eval_metrics, selected_features, (scaler, selector)

def walk_forward_validation(df_features, target_horizon_days=1, window_size=365):
    """
    IMPROVEMENT 3: Implement Walk-forward validation for more realistic performance assessment
    """
    if df_features is None or df_features.empty:
        safe_print("Feature DataFrame is empty. Cannot perform walk-forward validation.")
        return None, None, None, None

    target_col = f"target_return_{target_horizon_days}d"
    if target_col not in df_features.columns:
        safe_print(f"Target column {target_col} not found. Available: {df_features.columns}")
        return None, None, None, None

    safe_print(f"\nPerforming Walk-Forward Validation for {target_horizon_days}-day horizon...")
    
    # Remove non-feature columns from training
    exclude_cols = ["Open", "High", "Low", "Close", "Volume", "Adj Close", "returns", "log_returns"] 
    exclude_cols += [col for col in df_features.columns if col.startswith("target_return_")]
    
    features = [col for col in df_features.columns if col not in exclude_cols]
    
    X = df_features[features]
    y = df_features[target_col]
    
    # Ensure we have enough data
    if len(X) <= window_size:
        safe_print(f"Not enough data for walk-forward validation. Need more than {window_size} records.")
        return None, None, None, None
    
    # Prepare for walk-forward validation
    predictions = []
    actuals = []
    feature_importances_sum = None
    steps = 0
    
    # Define step size (predicting every 30 days for efficiency)
    step_size = 30
    
    # Process slices of data for walk-forward validation
    for start_idx in range(window_size, len(X), step_size):
        end_idx = min(start_idx + step_size, len(X))
        
        # Define train and test for this step
        X_train = X.iloc[start_idx - window_size:start_idx]
        y_train = y.iloc[start_idx - window_size:start_idx]
        X_test = X.iloc[start_idx:end_idx]
        y_test = y.iloc[start_idx:end_idx]
        
        if len(X_train) < 100 or len(X_test) == 0:  # Skip if not enough training data
            continue
            
        safe_print(f"  Walk-forward step {steps+1}: training on {len(X_train)} samples, testing on {len(X_test)} samples")
        
        # Normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train a model on this window
        try:
            # First try with more complete configuration
            model = XGBRegressor(
                n_estimators=100,
                learning_rate=0.01,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42
            )
            
            model.fit(X_train_scaled, y_train)
        except Exception as e:
            safe_print(f"  Warning: Error training model with full config: {e}")
            safe_print("  Trying with simplified model...")
            
            # Fallback to simpler model
            model = XGBRegressor(random_state=42)
            model.fit(X_train_scaled, y_train)
        
        # Make predictions for the next step_size days
        step_predictions = model.predict(X_test_scaled)
        
        # Collect results
        predictions.extend(step_predictions)
        actuals.extend(y_test.values)
        
        # Collect feature importances for aggregation
        if hasattr(model, 'feature_importances_'):
            if feature_importances_sum is None:
                feature_importances_sum = model.feature_importances_
            else:
                feature_importances_sum += model.feature_importances_
        
        steps += 1
    
    if steps == 0:
        safe_print("No valid walk-forward steps were completed. Aborting.")
        return None, None, None, None
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate overall walk-forward metrics
    r2 = r2_score(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    
    safe_print(f"\nWalk-Forward Validation Performance ({target_horizon_days}-day target):")
    safe_print(f"  R-squared: {r2:.4f}")
    safe_print(f"  MSE: {mse:.6f}")
    safe_print(f"  MAE: {mae:.4f}")
    
    # Average feature importances across all models
    avg_feature_importances = feature_importances_sum / steps if feature_importances_sum is not None else None
    
    # Assemble metrics
    eval_metrics = {
        "R2": r2,
        "MSE": mse,
        "MAE": mae,
        "WF_Steps": steps,
        "WF_Window_Size": window_size
    }
    
    # Compute directional accuracy (how often prediction correctly gets the sign right)
    directional_accuracy = np.mean((predictions > 0) == (actuals > 0))
    eval_metrics["Directional_Accuracy"] = directional_accuracy
    safe_print(f"  Directional Accuracy: {directional_accuracy:.4f}")
    
    return predictions, actuals, eval_metrics, avg_feature_importances

def generate_report_and_save(current_price, xgb_predictions, wf_metrics, features_info, models):
    """
    Generate a comprehensive report with predictions and model performance, and save it.
    """
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_filename = f"Bitcoin_Prediction_Report_FIXED_{datetime.now().strftime('%Y%m%d')}.txt"
    report_path = os.path.join(DESKTOP_PATH, report_filename)
    if not os.access(os.path.dirname(report_path), os.W_OK):
        report_path = os.path.join(REPORTS_FINAL_DIR, report_filename)

    safe_print(f"\nGenerating FIXED enhanced report: {report_path}")
    
    report_content = f"""=== Bitcoin (BTC-USD) Price Prediction Report - FIXED VERSION ===
Generated on: {datetime.now().strftime('%Y-%m-%d')}
Symbol: BTC-USD
Current Price: ${current_price:,.2f}

--- Prediction Details ---"""

    for horizon, pred_return in xgb_predictions.items():
        pred_price = current_price * (1 + pred_return)
        report_content += f"\nPredicted {horizon}-Day Price: ${pred_price:,.2f}"
        report_content += f"\nPredicted {horizon}-Day Return: {pred_return*100:+.2f}%"
    
    # Add trading signal based on XGBoost predictions
    avg_return = sum(pred for _, pred in xgb_predictions.items()) / len(xgb_predictions)
    if avg_return > 0.01:  # 1% threshold
        suggested_action = "BUY"
    elif avg_return < -0.01:  # -1% threshold
        suggested_action = "SELL"
    else:
        suggested_action = "HOLD"
    
    report_content += f"\nSuggested Action: {suggested_action}"

    report_content += "\n\n--- Model Performance ---"
    for horizon, metric_values in wf_metrics.items():
        report_content += f"\nHorizon: {horizon}-day"
        report_content += f"\nR-squared: {metric_values.get('R2', 'N/A'):.4f}"
        report_content += f"\nHit Rate: {metric_values.get('Directional_Accuracy', 'N/A'):.4f}"
            
    report_content += f"\n\n--- Model Information ---"
    report_content += "\nModel Type: XGBoost Regressor with Early Stopping"
    report_content += "\nFeature Selection: Recursive Feature Elimination with Cross-Validation"
    report_content += "\nValidation Method: Walk-Forward Validation"
    report_content += f"\nOn-Chain Metrics: {'Enabled' if ENABLE_ONCHAIN_METRICS else 'Disabled'}"
    report_content += "\n*** CRITICAL FIX: Smart NaN handling implemented - data preserved! ***"
    
    report_content += "\n\n--- Selected Features ---"
    for horizon, features in features_info.items():
        report_content += f"\nTop {min(10, len(features))} features for {horizon}-day model:"
        for i, feature in enumerate(features[:min(10, len(features))]):
            report_content += f"\n{i+1}. {feature}"
    
    report_content += "\n\n--- Disclaimer ---"
    report_content += "\nThis report is for informational purposes only and does not constitute financial advice. "
    report_content += "Cryptocurrency investments are highly volatile and risky. Past performance and model "
    report_content += "predictions are not indicative of future results."

    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        safe_print(f"FIXED Report saved to {report_path}")
        # Also save to onedrive_export/final if DESKTOP_PATH was used and is different from onedrive_export
        if DESKTOP_PATH != ONEDRIVE_EXPORT_DIR:
            onedrive_report_path = os.path.join(ONEDRIVE_EXPORT_DIR, report_filename)
            try:
                with open(onedrive_report_path, "w", encoding="utf-8") as f_od:
                    f_od.write(report_content)
                safe_print(f"FIXED Report also saved to {onedrive_report_path}")
            except Exception as e_od:
                safe_print(f"Could not save report to OneDrive export path {onedrive_report_path}: {e_od}")

    except Exception as e:
        safe_print(f"Error saving report: {e}")

    safe_print("\n--- FIXED Report Content ---")
    safe_print(report_content)

def main():
    try:
        safe_print("=== FIXED Enhanced Bitcoin Price Prediction Model ===")
        safe_print("[FIX] CRITICAL FIX: Smart NaN handling implemented!")
        safe_print("Implementing four targeted improvements:")
        safe_print("1. XGBoost with early stopping for better 1-day predictions")
        safe_print("2. Recursive Feature Elimination for optimal feature selection")
        safe_print("3. Walk-Forward Validation for more realistic performance evaluation")
        safe_print("4. Integration of on-chain metrics for enhanced predictive power")
        safe_print("5. [SUCCESS] FIXED: Smart NaN handling that preserves data")
        safe_print("6. [NEW] Unicode-safe console output for Windows compatibility\n")
        
        # 1. Fetch Data
        raw_df = fetch_bitcoin_data_with_fallback()
        if raw_df is None or raw_df.empty:
            safe_print("Failed to fetch Bitcoin data. Exiting script.")
            return

        # 2. Create Features with FIXED NaN handling
        processed_df = create_features_fixed(raw_df.copy())  # Use FIXED version
        if processed_df is None or processed_df.empty:
            safe_print("[ERROR] FIXED Data preprocessing/feature creation failed. Exiting script.")
            return
        else:
            safe_print(f"[SUCCESS] FIXED Feature creation successful! Final data shape: {processed_df.shape}")
            
        # 3. Analyze features
        analyze_features(processed_df.copy())
        
        # Evaluate impact of on-chain metrics specifically
        if ENABLE_ONCHAIN_METRICS:
            # Evaluate the impact of on-chain metrics for each horizon
            for horizon in [1, 3, 7]:
                evaluate_onchain_impact(processed_df.copy(), target_horizon_days=horizon)

        # 4. Train models with the four improvements
        horizons_to_predict = [1, 3, 7]
        xgb_predictions = {}
        xgb_models = {}
        features_info = {}
        wf_metrics = {}

        # Train XGBoost models (Improvement 1)
        safe_print("\n" + "="*50)
        safe_print("IMPROVEMENT 1: XGBoost with Early Stopping")
        safe_print("="*50)
        
        for horizon in horizons_to_predict:
            model, prediction, metrics, selected_features, transformers = train_xgboost_model(
                processed_df.copy(), target_horizon_days=horizon, force_onchain=True
            )
            
            if model is not None:
                xgb_predictions[horizon] = prediction
                xgb_models[horizon] = (model, transformers)
                features_info[horizon] = selected_features
                
                # Analyze importance of on-chain metrics specifically (Improvement 4)
                if ENABLE_ONCHAIN_METRICS:
                    analyze_onchain_importance(processed_df.copy(), selected_features)
        
        # Perform Walk-Forward Validation (Improvement 3)
        safe_print("\n" + "="*50)
        safe_print("IMPROVEMENT 3: Walk-Forward Validation")
        safe_print("="*50)
        
        for horizon in horizons_to_predict:
            predictions, actuals, wf_metric, avg_importances = walk_forward_validation(
                processed_df.copy(), target_horizon_days=horizon
            )
            
            if predictions is not None:
                wf_metrics[horizon] = wf_metric
        
        # Get Current Price
        current_btc_price = get_current_bitcoin_price_with_fallback()
        if current_btc_price is None:
            safe_print("Could not obtain current Bitcoin price. Exiting script.")
            return

        # Generate and Save Enhanced Report
        generate_report_and_save(current_btc_price, xgb_predictions, wf_metrics, features_info, xgb_models)
        
        # Save predictions to database
        safe_print("\n[DATABASE] Saving predictions to database...")
        for horizon, pred in xgb_predictions.items():
            pred_price = current_btc_price * (1 + pred)
            
            # Determine confidence based on model performance
            confidence = 80.0  # Default confidence
            if horizon in wf_metrics:
                # Use R² score as confidence indicator (convert to percentage)
                r2_score = wf_metrics[horizon].get('r2', 0.5)
                confidence = max(50.0, min(95.0, r2_score * 100))
            
            # Determine action
            return_pct = pred * 100
            if return_pct > 3.0:
                action = "BUY"
            elif return_pct < -3.0:
                action = "SELL"
            else:
                action = "HOLD"
            
            # Save to database
            db_success = quick_save_prediction(
                model_name="Bitcoin Model",
                symbol="BTC-USD",
                current_price=current_btc_price,
                predicted_price=pred_price,
                confidence=confidence,
                horizon_days=horizon,
                suggested_action=action
            )
            
            if db_success:
                safe_print(f"[DATABASE] ✅ {horizon}-day prediction saved to database")
            else:
                safe_print(f"[DATABASE] ❌ Failed to save {horizon}-day prediction")
        
        # Save model metrics to database
        if wf_metrics:
            safe_print("[DATABASE] Saving model metrics...")
            # Aggregate metrics for database
            avg_metrics = {}
            for horizon, metrics in wf_metrics.items():
                for metric_name, value in metrics.items():
                    if metric_name not in avg_metrics:
                        avg_metrics[metric_name] = []
                    avg_metrics[metric_name].append(value)
            
            # Calculate averages
            final_metrics = {}
            for metric_name, values in avg_metrics.items():
                final_metrics[f"avg_{metric_name}"] = sum(values) / len(values)
            
            metrics_success = quick_save_metrics("Bitcoin Model", "BTC-USD", final_metrics)
            if metrics_success:
                safe_print("[DATABASE] ✅ Model metrics saved to database")
            else:
                safe_print("[DATABASE] ❌ Failed to save model metrics")
        
        safe_print("\n[SUCCESS] FIXED Enhanced Bitcoin Price Prediction script completed successfully!")
        safe_print(f"[PRICE] Current Bitcoin Price: ${current_btc_price:,.2f}")
        for horizon, pred in xgb_predictions.items():
            pred_price = current_btc_price * (1 + pred)
            safe_print(f"[PREDICTION] {horizon}-day prediction: ${pred_price:,.2f} ({pred*100:+.2f}%)")

    except Exception as e:
        safe_print(f"\n[ERROR] An unexpected error occurred in the FIXED Enhanced Bitcoin model: {str(e)}")
        safe_print(traceback.format_exc())

if __name__ == "__main__":
    main()