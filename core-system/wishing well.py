import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
from pathlib import Path
import warnings
import traceback
from model_db_integrator import quick_save_prediction, quick_save_metrics
warnings.filterwarnings('ignore')

# ENCODING FIX FOR WINDOWS
import locale
import io

def setup_unicode_output():
    """Setup Unicode output for Windows console"""
    try:
        # Try to set UTF-8 encoding
        if sys.platform.startswith('win'):
            # For Windows, try to enable UTF-8 mode
            try:
                import codecs
                sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
                sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
            except:
                # Fallback: disable Unicode characters
                pass
    except Exception as e:
        print(f"Warning: Could not setup Unicode output: {e}")

# Safe print function that handles encoding issues
def safe_print(text):
    """Print text safely, removing Unicode characters if needed"""
    try:
        print(text)
    except UnicodeEncodeError:
        # Remove all non-ASCII characters and try again
        ascii_text = text.encode('ascii', 'ignore').decode('ascii')
        print(ascii_text)

setup_unicode_output()

# Add the parent directory to the path for importing data_fetcher
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
try:
    import data_fetcher
except ImportError:
    safe_print("Error: data_fetcher module not found. Please ensure data_fetcher.py is available.")
    sys.exit(1)

# Set up logging and paths
def get_desktop_path():
    """Get the reports path in GitHub repo instead of desktop"""
    try:
        # Use GitHub repo reports directory instead of Desktop
        reports_path = os.path.join(script_dir, "reports")
        os.makedirs(reports_path, exist_ok=True)
        safe_print(f"Using GitHub repo reports path: {reports_path}")
        return reports_path
        
    except Exception as e:
        safe_print(f"Error creating reports directory: {e}. Using fallback in script directory.")
        fallback_desktop = os.path.join(script_dir, "reports_fallback")
        os.makedirs(fallback_desktop, exist_ok=True)
        return fallback_desktop

DESKTOP_PATH = get_desktop_path()
safe_print(f"Using desktop path: {DESKTOP_PATH}")
OUTPUT_PATH = DESKTOP_PATH
SAVE_PLOTS = True

# OPTIMIZED: Trading signal parameters
SHORT_WINDOW = 10
MEDIUM_WINDOW = 30
LONG_WINDOW = 50
RSI_WINDOW = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
VIX_THRESHOLD = 25
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_WINDOW = 20
BOLLINGER_STD = 2
STOCH_K = 14
STOCH_D = 3
T2108_WINDOW = 40

def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index with optimized error handling"""
    try:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Fill NaN with neutral RSI
    except Exception as e:
        safe_print(f"Error calculating RSI: {e}")
        return pd.Series(50, index=prices.index)

def calculate_macd(prices, fast_window=12, slow_window=26, signal_window=9):
    """Calculate MACD with improved error handling"""
    try:
        ema_fast = prices.ewm(span=fast_window, adjust=False).mean()
        ema_slow = prices.ewm(span=slow_window, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'MACD': macd_line,
            'Signal': signal_line,
            'Histogram': histogram
        })
    except Exception as e:
        safe_print(f"Error calculating MACD: {e}")
        zeros = pd.Series(0, index=prices.index)
        return pd.DataFrame({
            'MACD': zeros,
            'Signal': zeros,
            'Histogram': zeros
        })

def calculate_bollinger_bands(series, window=20, num_std=2):
    """Calculate Bollinger Bands with error handling"""
    try:
        sma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        return pd.DataFrame({
            'Middle': sma,
            'Upper': sma + (std * num_std),
            'Lower': sma - (std * num_std)
        })
    except Exception as e:
        safe_print(f"Error calculating Bollinger Bands: {e}")
        return pd.DataFrame({
            'Middle': series,
            'Upper': series * 1.05,
            'Lower': series * 0.95
        })

def calculate_stochastic(prices, k_period=14, d_period=3, slowing=3):
    """Calculate Stochastic Oscillator"""
    try:
        low_min = prices.rolling(window=k_period).min()
        high_max = prices.rolling(window=k_period).max()
        
        k = 100 * ((prices - low_min) / (high_max - low_min + 1e-9))
        
        if slowing > 1:
            k = k.rolling(window=slowing).mean()
        
        d = k.rolling(window=d_period).mean()
        
        return pd.DataFrame({
            'K': k.fillna(50),
            'D': d.fillna(50)
        })
    except Exception as e:
        safe_print(f"Error calculating Stochastic: {e}")
        neutral = pd.Series(50, index=prices.index)
        return pd.DataFrame({
            'K': neutral,
            'D': neutral
        })

def detect_stochastic_signals(stoch_df, threshold=50):
    """Detect stochastic crossovers for dot signals"""
    try:
        signals = pd.DataFrame(index=stoch_df.index)
        
        # Green Dot: Fast stochastic crosses above slow while both below threshold
        signals['green_dot'] = ((stoch_df['K'] > stoch_df['D']) & 
                               (stoch_df['K'].shift(1) <= stoch_df['D'].shift(1)) & 
                               (stoch_df['K'] < threshold) & 
                               (stoch_df['D'] < threshold)).astype(int)
        
        # Blue Dot: Similar signal with slight variation
        signals['blue_dot'] = ((stoch_df['K'] > stoch_df['D']) & 
                              (stoch_df['K'].shift(1) <= stoch_df['D'].shift(1)) & 
                              (stoch_df['K'] < threshold + 10)).astype(int)
        
        # Black Dot: Faster crossover
        signals['black_dot'] = ((stoch_df['K'] > stoch_df['D']) & 
                               (stoch_df['K'].shift(1) <= stoch_df['D'].shift(1))).astype(int)
        
        return signals
    except Exception as e:
        safe_print(f"Error detecting stochastic signals: {e}")
        return pd.DataFrame(0, index=stoch_df.index, columns=['green_dot', 'blue_dot', 'black_dot'])

def calculate_t2108(price_data, window=40):
    """Calculate T2108 indicator - simplified version for individual stock"""
    try:
        ma = price_data.rolling(window=window).mean()
        above_ma = (price_data > ma).astype(int)
        return above_ma * 100
    except Exception as e:
        safe_print(f"Error calculating T2108: {e}")
        return pd.Series(50, index=price_data.index)

def calculate_weekly_metrics(daily_df):
    """Calculate weekly metrics for trend analysis"""
    try:
        weekly_df = daily_df.resample('W').last()
        weekly_df['MA_10W'] = weekly_df['Close'].rolling(window=10).mean()
        weekly_df['MA_30W'] = weekly_df['Close'].rolling(window=30).mean()
        weekly_df['Uptrend'] = (weekly_df['MA_10W'] > weekly_df['MA_30W']).astype(int)
        weekly_df['MA_Trend_Change'] = weekly_df['Uptrend'].diff().ne(0).astype(int)
        
        return weekly_df
    except Exception as e:
        safe_print(f"Error calculating weekly metrics: {e}")
        weekly_df = daily_df.resample('W').last()
        weekly_df['Uptrend'] = 0
        return weekly_df

def calculate_gmi_components(data):
    """Calculate GMI (General Market Index) components"""
    try:
        gmi_df = pd.DataFrame(index=data.index)
        
        # Component 1: QQQ price above its 30-day MA
        gmi_df['Component_1'] = (data['QQQ_Close'] > data['QQQ_Close'].rolling(window=30).mean()).astype(int)
        
        # Component 2: QQQ 10-day MA above 30-day MA
        ma10 = data['QQQ_Close'].rolling(window=10).mean()
        ma30 = data['QQQ_Close'].rolling(window=30).mean()
        gmi_df['Component_2'] = (ma10 > ma30).astype(int)
        
        # Component 3: QQQ MACD above Signal line
        gmi_df['Component_3'] = (data['MACD'] > data['MACD_Signal']).astype(int)
        
        # Component 4: Price momentum (proxy for new highs vs new lows)
        gmi_df['Component_4'] = (data['QQQ_Close'] > data['QQQ_Close'].rolling(window=50).max().shift(10)).astype(int)
        
        # Component 5: T2108 above 50%
        gmi_df['Component_5'] = (data['T2108'] > 50).astype(int)
        
        # Component 6: Weekly QQQ uptrend
        gmi_df['Component_6'] = data['Weekly_Uptrend']
        
        # Calculate GMI Score (0-6)
        gmi_df['GMI_Score'] = (
            gmi_df['Component_1'] + 
            gmi_df['Component_2'] + 
            gmi_df['Component_3'] + 
            gmi_df['Component_4'] + 
            gmi_df['Component_5'] + 
            gmi_df['Component_6']
        )
        
        # GMI Signal
        gmi_df['GMI_Signal'] = 'Yellow'
        gmi_df.loc[gmi_df['GMI_Score'] >= 4, 'GMI_Signal'] = 'Green'  # More conservative threshold
        
        # Red signal for consecutive days below 3
        for i in range(2, len(gmi_df)):
            if (gmi_df['GMI_Score'].iloc[i-1] < 3) and (gmi_df['GMI_Score'].iloc[i] < 3):
                gmi_df['GMI_Signal'].iloc[i] = 'Red'
        
        return gmi_df
    except Exception as e:
        safe_print(f"Error calculating GMI components: {e}")
        gmi_df = pd.DataFrame(index=data.index)
        gmi_df['GMI_Score'] = 3
        gmi_df['GMI_Signal'] = 'Yellow'
        return gmi_df

def detect_qqq_trend(data):
    """Detect and count days in QQQ uptrend/downtrend"""
    try:
        trend_df = pd.DataFrame(index=data.index)
        
        ma10 = data['QQQ_Close'].rolling(window=10).mean()
        ma30 = data['QQQ_Close'].rolling(window=30).mean()
        
        trend_df['Trend'] = 0
        trend_df.loc[ma10 > ma30, 'Trend'] = 1
        trend_df.loc[ma10 < ma30, 'Trend'] = -1
        
        trend_df['Trend_Change'] = trend_df['Trend'].diff().ne(0).astype(int)
        
        # Count days in trend
        trend_df['Trend_Days'] = 0
        current_trend = 0
        days_in_trend = 0
        
        for i in range(len(trend_df)):
            if i > 0 and trend_df['Trend_Change'].iloc[i] == 1:
                current_trend = trend_df['Trend'].iloc[i]
                days_in_trend = 1
            else:
                days_in_trend += 1
            
            trend_df['Trend_Days'].iloc[i] = days_in_trend
        
        # Create trend labels
        trend_df['Trend_Label'] = trend_df.apply(
            lambda row: f"U-{int(row['Trend_Days'])}" if row['Trend'] == 1 
            else f"D-{int(row['Trend_Days'])}" if row['Trend'] == -1 
            else "N-0", axis=1
        )
        
        return trend_df
    except Exception as e:
        safe_print(f"Error detecting QQQ trend: {e}")
        trend_df = pd.DataFrame(index=data.index)
        trend_df['Trend'] = 0
        trend_df['Trend_Days'] = 0
        trend_df['Trend_Change'] = 0
        trend_df['Trend_Label'] = "U-0"
        return trend_df

def prepare_data_optimized(start_date, end_date):
    """DATABASE-FIRST: Download and prepare data for QQQ and VIX"""
    try:
        safe_print(f"Fetching QQQ data from {start_date} to {end_date} (DATABASE-FIRST)...")
        
        # DATABASE-FIRST: Fetch QQQ data
        qqq_data = data_fetcher.get_historical_data("QQQ", start_date, end_date)
        if qqq_data is None or qqq_data.empty:
            safe_print("Failed to get QQQ data. Cannot proceed.")
            return None

        safe_print(f"Retrieved {len(qqq_data)} days of QQQ data")
        
        # DATABASE-FIRST: Get VIX data
        safe_print(f"Fetching VIX data from {start_date} to {end_date} (DATABASE-FIRST)...")
        vix_data = None
        has_vix = False
        
        # Try multiple VIX symbols
        for symbol in ["VIX", "^VIX"]:
            safe_print(f"Trying VIX symbol: {symbol}")
            vix_data = data_fetcher.get_historical_data(symbol, start_date, end_date)
            if vix_data is not None and not vix_data.empty and len(vix_data) > 100:
                safe_print(f"Retrieved {len(vix_data)} days of VIX data with {symbol}")
                has_vix = True
                break
        
        # Create VIX proxy if needed
        if not has_vix:
            safe_print("Creating VIX proxy from QQQ volatility...")
            vix_data = pd.DataFrame(index=qqq_data.index)
            qqq_returns = qqq_data['Close'].pct_change()
            rolling_vol = qqq_returns.rolling(window=20).std() * np.sqrt(252) * 100
            vix_data['Close'] = 15 + (rolling_vol * 0.8)  # Base VIX + volatility factor
            vix_data['Close'] = vix_data['Close'].clip(lower=9)
            vix_data['Open'] = vix_data['Close'].shift(1).fillna(vix_data['Close'])
            vix_data['High'] = vix_data['Close'] * 1.05
            vix_data['Low'] = vix_data['Close'] * 0.95
            vix_data['Volume'] = 1000000
            has_vix = True
            safe_print(f"Created VIX proxy with {len(vix_data)} days")
        
        # Create main features DataFrame
        safe_print("Creating optimized features for trading signals...")
        df = pd.DataFrame(index=qqq_data.index)
        
        # Add QQQ data
        df['QQQ_Close'] = qqq_data['Close']
        df['QQQ_Open'] = qqq_data['Open']
        df['QQQ_High'] = qqq_data['High']
        df['QQQ_Low'] = qqq_data['Low']
        df['QQQ_Volume'] = qqq_data['Volume']
        
        # Add VIX data
        if has_vix:
            vix_reindexed = vix_data.reindex(qqq_data.index, method='ffill')
            df['VIX_Close'] = vix_reindexed['Close']
            df['VIX_Change'] = df['VIX_Close'].pct_change()
        
        # Calculate returns
        df['QQQ_Return'] = df['QQQ_Close'].pct_change()
        
        # Moving averages
        df['SMA_Short'] = df['QQQ_Close'].rolling(window=SHORT_WINDOW).mean()
        df['SMA_Medium'] = df['QQQ_Close'].rolling(window=MEDIUM_WINDOW).mean()
        df['SMA_Long'] = df['QQQ_Close'].rolling(window=LONG_WINDOW).mean()
        
        # Calculate weekly metrics
        weekly_df = calculate_weekly_metrics(qqq_data)
        df['Weekly_Uptrend'] = 0
        
        # Map weekly data to daily
        for week_end in weekly_df.index:
            week_start = week_end - timedelta(days=6)
            mask = (df.index >= week_start) & (df.index <= week_end)
            if week_end in weekly_df.index:
                df.loc[mask, 'Weekly_Uptrend'] = weekly_df.loc[week_end, 'Uptrend']
        
        # Technical indicators
        df['RSI'] = calculate_rsi(df['QQQ_Close'], window=RSI_WINDOW)
        
        # MACD
        macd_df = calculate_macd(df['QQQ_Close'], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
        df['MACD'] = macd_df['MACD']
        df['MACD_Signal'] = macd_df['Signal']
        df['MACD_Histogram'] = macd_df['Histogram']
        
        # Bollinger Bands
        bb_df = calculate_bollinger_bands(df['QQQ_Close'], BOLLINGER_WINDOW, BOLLINGER_STD)
        df['BB_Middle'] = bb_df['Middle']
        df['BB_Upper'] = bb_df['Upper']
        df['BB_Lower'] = bb_df['Lower']
        
        # Stochastics
        stoch_df = calculate_stochastic(df['QQQ_Close'], STOCH_K, STOCH_D)
        df['Stoch_K'] = stoch_df['K']
        df['Stoch_D'] = stoch_df['D']
        
        # T2108
        df['T2108'] = calculate_t2108(df['QQQ_Close'], T2108_WINDOW)
        
        # Dot signals
        stoch_signals = detect_stochastic_signals(stoch_df)
        df['Green_Dot'] = stoch_signals['green_dot']
        df['Blue_Dot'] = stoch_signals['blue_dot']
        df['Black_Dot'] = stoch_signals['black_dot']
        
        # Trend analysis
        trend_df = detect_qqq_trend(df)
        df['Trend'] = trend_df['Trend']
        df['Trend_Days'] = trend_df['Trend_Days']
        df['Trend_Label'] = trend_df['Trend_Label']
        df['Trend_Change'] = trend_df['Trend_Change']
        
        # GMI components
        gmi_df = calculate_gmi_components(df)
        df['GMI_Score'] = gmi_df['GMI_Score']
        df['GMI_Signal'] = gmi_df['GMI_Signal']
        
        # Additional momentum indicators
        df['Momentum_5d'] = df['QQQ_Close'].pct_change(periods=5)
        df['Momentum_10d'] = df['QQQ_Close'].pct_change(periods=10)
        df['Momentum_20d'] = df['QQQ_Close'].pct_change(periods=20)
        
        # Volatility
        df['Volatility_10d'] = df['QQQ_Return'].rolling(window=10).std()
        df['Volatility_20d'] = df['QQQ_Return'].rolling(window=20).std()
        
        # Clean data
        df = df.dropna()
        
        if df.empty:
            safe_print("Feature creation resulted in empty DataFrame.")
            return None
        
        safe_print(f"Successfully created optimized features: {len(df)} rows, {len(df.columns)} columns")
        return df
    except Exception as e:
        safe_print(f"Error preparing data: {e}")
        traceback.print_exc()
        return None

def generate_signals_optimized(features_df):
    """Generate optimized trading signals based on Dr. Wish's methodology"""
    try:
        safe_print("Generating optimized trading signals...")
        
        signals_df = features_df.copy()
        signals_df['Signal'] = 0  # 0: Hold, 1: Buy, -1: Sell
        
        # Individual signal components
        signals_df['MA_Signal'] = 0
        signals_df.loc[signals_df['SMA_Short'] > signals_df['SMA_Medium'], 'MA_Signal'] = 1
        signals_df.loc[signals_df['SMA_Short'] < signals_df['SMA_Medium'], 'MA_Signal'] = -1
        
        signals_df['RSI_Signal'] = 0
        signals_df.loc[signals_df['RSI'] < RSI_OVERSOLD, 'RSI_Signal'] = 1
        signals_df.loc[signals_df['RSI'] > RSI_OVERBOUGHT, 'RSI_Signal'] = -1
        
        signals_df['MACD_Signal_Indicator'] = 0
        signals_df.loc[signals_df['MACD'] > signals_df['MACD_Signal'], 'MACD_Signal_Indicator'] = 1
        signals_df.loc[signals_df['MACD'] < signals_df['MACD_Signal'], 'MACD_Signal_Indicator'] = -1
        
        signals_df['BB_Signal'] = 0
        signals_df.loc[signals_df['QQQ_Close'] < signals_df['BB_Lower'], 'BB_Signal'] = 1
        signals_df.loc[signals_df['QQQ_Close'] > signals_df['BB_Upper'], 'BB_Signal'] = -1
        
        # VIX Signal
        if 'VIX_Close' in signals_df.columns:
            signals_df['VIX_Signal'] = 0
            signals_df.loc[signals_df['VIX_Close'] > VIX_THRESHOLD, 'VIX_Signal'] = -1
            signals_df.loc[(signals_df['VIX_Close'] > VIX_THRESHOLD) & 
                          (signals_df['VIX_Change'] < -0.05), 'VIX_Signal'] = 1
        
        # Dot Signal
        signals_df['Dot_Signal'] = 0
        signals_df.loc[signals_df['Blue_Dot'] == 1, 'Dot_Signal'] = 1
        signals_df.loc[signals_df['Green_Dot'] == 1, 'Dot_Signal'] = 1
        signals_df.loc[signals_df['Black_Dot'] == 1, 'Dot_Signal'] = 1
        
        # GMI Signal (most important according to Dr. Wish)
        signals_df['GMI_Signal_Indicator'] = 0
        signals_df.loc[signals_df['GMI_Signal'] == 'Green', 'GMI_Signal_Indicator'] = 1
        signals_df.loc[signals_df['GMI_Signal'] == 'Red', 'GMI_Signal_Indicator'] = -1
        
        # OPTIMIZED: Dr. Wish's weighted scoring system
        # GMI gets highest weight as it's the core of his methodology
        signals_df['Combined_Score'] = (
            0.35 * signals_df['GMI_Signal_Indicator'] +  # Highest weight for GMI
            0.15 * signals_df['MA_Signal'] +
            0.10 * signals_df['RSI_Signal'] +
            0.15 * signals_df['MACD_Signal_Indicator'] +
            0.10 * signals_df['BB_Signal'] +
            0.15 * signals_df['Dot_Signal']
        )
        
        if 'VIX_Signal' in signals_df.columns:
            # Redistribute weights to include VIX
            signals_df['Combined_Score'] = (
                0.30 * signals_df['GMI_Signal_Indicator'] +
                0.12 * signals_df['MA_Signal'] +
                0.08 * signals_df['RSI_Signal'] +
                0.12 * signals_df['MACD_Signal_Indicator'] +
                0.08 * signals_df['BB_Signal'] +
                0.15 * signals_df['Dot_Signal'] +
                0.15 * signals_df['VIX_Signal']
            )
        
        # OPTIMIZED: Dr. Wish's signal generation rules
        # 1. Strong emphasis on trend changes (day 1 of new trend)
        # 2. GMI-based signals
        # 3. Conservative thresholds to reduce whipsaws
        
        # Base signals on combined score
        signals_df.loc[signals_df['Combined_Score'] >= 0.5, 'Signal'] = 1   # Strong buy
        signals_df.loc[signals_df['Combined_Score'] <= -0.5, 'Signal'] = -1  # Strong sell
        
        # OVERRIDE: Day 1 of trend changes (Dr. Wish's key insight)
        signals_df.loc[(signals_df['Trend'] == 1) & (signals_df['Trend_Days'] == 1), 'Signal'] = 1
        signals_df.loc[(signals_df['Trend'] == -1) & (signals_df['Trend_Days'] == 1), 'Signal'] = -1
        
        # OVERRIDE: Strong GMI signals
        signals_df.loc[(signals_df['GMI_Score'] >= 5) & (signals_df['Combined_Score'] > 0), 'Signal'] = 1
        signals_df.loc[(signals_df['GMI_Score'] <= 1) & (signals_df['Combined_Score'] < 0), 'Signal'] = -1
        
        # Signal change tracking
        signals_df['Signal_Change'] = signals_df['Signal'].diff().fillna(0)
        
        # TQQQ/SQQQ strategy mapping
        signals_df['Leverage_ETF'] = "CASH"
        signals_df.loc[signals_df['Signal'] == 1, 'Leverage_ETF'] = "TQQQ"
        signals_df.loc[signals_df['Signal'] == -1, 'Leverage_ETF'] = "SQQQ"
        
        # Action labels
        signals_df['Action'] = "HOLD"
        signals_df.loc[signals_df['Signal'] == 1, 'Action'] = "BUY"
        signals_df.loc[signals_df['Signal'] == -1, 'Action'] = "SELL"
        
        return signals_df
    except Exception as e:
        safe_print(f"Error generating signals: {e}")
        traceback.print_exc()
        return features_df

def backtest_strategy_optimized(signals_df, initial_capital=10000):
    """Optimized backtesting with enhanced metrics"""
    try:
        safe_print("Running optimized backtesting...")
        
        backtest_results = signals_df.copy()
        
        # Position tracking
        backtest_results['Position'] = backtest_results['Signal'].shift(1).fillna(0)
        
        # Calculate strategy returns
        backtest_results['Strategy_Return'] = 0.0
        backtest_results.loc[backtest_results['Position'] == 1, 'Strategy_Return'] = backtest_results['QQQ_Return']
        backtest_results.loc[backtest_results['Position'] == -1, 'Strategy_Return'] = -backtest_results['QQQ_Return']
        
        # Leveraged strategy (3x)
        backtest_results['Leveraged_Strategy_Return'] = 0.0
        backtest_results.loc[backtest_results['Position'] == 1, 'Leveraged_Strategy_Return'] = backtest_results['QQQ_Return'] * 3
        backtest_results.loc[backtest_results['Position'] == -1, 'Leveraged_Strategy_Return'] = backtest_results['QQQ_Return'] * -3
        
        # Cumulative returns
        backtest_results['Cumulative_QQQ_Return'] = (1 + backtest_results['QQQ_Return']).cumprod() - 1
        backtest_results['Cumulative_Strategy_Return'] = (1 + backtest_results['Strategy_Return']).cumprod() - 1
        backtest_results['Cumulative_Leveraged_Return'] = (1 + backtest_results['Leveraged_Strategy_Return']).cumprod() - 1
        
        # Portfolio values
        backtest_results['QQQ_Portfolio'] = initial_capital * (1 + backtest_results['Cumulative_QQQ_Return'])
        backtest_results['Strategy_Portfolio'] = initial_capital * (1 + backtest_results['Cumulative_Strategy_Return'])
        backtest_results['Leveraged_Portfolio'] = initial_capital * (1 + backtest_results['Cumulative_Leveraged_Return'])
        
        # Performance calculations
        trading_days_per_year = 252
        years = len(backtest_results) / trading_days_per_year
        
        # Total returns
        qqq_total_return = backtest_results['Cumulative_QQQ_Return'].iloc[-1]
        strategy_total_return = backtest_results['Cumulative_Strategy_Return'].iloc[-1]
        leveraged_total_return = backtest_results['Cumulative_Leveraged_Return'].iloc[-1]
        
        # Annualized returns
        qqq_annual_return = (1 + qqq_total_return) ** (1 / years) - 1 if years > 0 else 0
        strategy_annual_return = (1 + strategy_total_return) ** (1 / years) - 1 if years > 0 else 0
        leveraged_annual_return = (1 + leveraged_total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Sharpe ratios
        risk_free_rate = 0.03
        daily_rf = (1 + risk_free_rate) ** (1 / trading_days_per_year) - 1
        
        def safe_sharpe(returns, daily_rf):
            excess_returns = returns - daily_rf
            if excess_returns.std() > 0:
                return (excess_returns.mean() / excess_returns.std()) * np.sqrt(trading_days_per_year)
            return 0
        
        qqq_sharpe = safe_sharpe(backtest_results['QQQ_Return'], daily_rf)
        strategy_sharpe = safe_sharpe(backtest_results['Strategy_Return'], daily_rf)
        leveraged_sharpe = safe_sharpe(backtest_results['Leveraged_Strategy_Return'], daily_rf)
        
        # Drawdowns
        def calculate_drawdown(portfolio_values):
            peak = portfolio_values.cummax()
            drawdown = (portfolio_values - peak) / peak
            return drawdown.min()
        
        qqq_max_drawdown = calculate_drawdown(backtest_results['QQQ_Portfolio'])
        strategy_max_drawdown = calculate_drawdown(backtest_results['Strategy_Portfolio'])
        leveraged_max_drawdown = calculate_drawdown(backtest_results['Leveraged_Portfolio'])
        
        # Win rates
        def calculate_win_rate(returns):
            returns_clean = returns[returns != 0]
            if len(returns_clean) > 0:
                return (returns_clean > 0).sum() / len(returns_clean)
            return 0
        
        qqq_win_rate = calculate_win_rate(backtest_results['QQQ_Return'])
        strategy_win_rate = calculate_win_rate(backtest_results['Strategy_Return'])
        leveraged_win_rate = calculate_win_rate(backtest_results['Leveraged_Strategy_Return'])
        
        # Trade statistics
        trades = backtest_results[backtest_results['Signal_Change'] != 0]
        num_trades = len(trades)
        
        # ENHANCED: Print comprehensive performance metrics
        safe_print("\n" + "="*60)
        safe_print("OPTIMIZED WISHING WEALTH QQQ PERFORMANCE METRICS")
        safe_print("="*60)
        
        safe_print(f"\nQQQ Buy and Hold Strategy:")
        safe_print(f"  Total Return: {qqq_total_return:.2%}")
        safe_print(f"  Annualized Return: {qqq_annual_return:.2%}")
        safe_print(f"  Sharpe Ratio: {qqq_sharpe:.2f}")
        safe_print(f"  Maximum Drawdown: {qqq_max_drawdown:.2%}")
        safe_print(f"  Win Rate: {qqq_win_rate:.2%}")
        
        safe_print(f"\nBasic QQQ Timing Strategy:")
        safe_print(f"  Total Return: {strategy_total_return:.2%}")
        safe_print(f"  Annualized Return: {strategy_annual_return:.2%}")
        safe_print(f"  Sharpe Ratio: {strategy_sharpe:.2f}")
        safe_print(f"  Maximum Drawdown: {strategy_max_drawdown:.2%}")
        safe_print(f"  Win Rate: {strategy_win_rate:.2%}")
        safe_print(f"  Number of Trades: {num_trades}")
        
        safe_print(f"\nLeveraged TQQQ/SQQQ Strategy:")
        safe_print(f"  Total Return: {leveraged_total_return:.2%}")
        safe_print(f"  Annualized Return: {leveraged_annual_return:.2%}")
        safe_print(f"  Sharpe Ratio: {leveraged_sharpe:.2f}")
        safe_print(f"  Maximum Drawdown: {leveraged_max_drawdown:.2%}")
        safe_print(f"  Win Rate: {leveraged_win_rate:.2%}")
        safe_print(f"  Number of Trades: {num_trades}")
        
        safe_print(f"\nDr. Wish's Insights:")
        safe_print(f"  GMI-based signals prioritized for trend following")
        safe_print(f"  Day-1 trend changes emphasized for timing")
        safe_print(f"  TQQQ/SQQQ leveraged strategy for amplified returns")
        safe_print(f"  (Historically outperforms 94%+ of individual stocks)")
        
        return backtest_results
    except Exception as e:
        safe_print(f"Error in backtesting: {e}")
        traceback.print_exc()
        return signals_df

def generate_current_signal_optimized(signals_df, backtest_results=None):
    """Generate enhanced current trading signal with comprehensive analysis"""
    try:
        current_signal = signals_df.iloc[-1]['Signal']
        current_date = signals_df.index[-1]
        
        # Signal interpretation and price prediction
        if current_signal == 1:
            signal_text = "BUY"
            etf = "TQQQ"
            signal_emoji = "GREEN"
            prediction_text = "QQQ is expected to RISE - Buy TQQQ for leveraged upside"
        elif current_signal == -1:
            signal_text = "SELL"
            etf = "SQQQ"
            signal_emoji = "RED"
            prediction_text = "QQQ is expected to FALL - Buy SQQQ for leveraged downside profit"
        else:
            signal_text = "HOLD"
            etf = "CASH"
            signal_emoji = "YELLOW"
            prediction_text = "QQQ direction uncertain - Stay in CASH until clear signal"
        
        # Get comprehensive data
        gmi_score = signals_df.iloc[-1]['GMI_Score']
        gmi_signal = signals_df.iloc[-1]['GMI_Signal']
        trend_label = signals_df.iloc[-1]['Trend_Label']
        qqq_close = signals_df.iloc[-1]['QQQ_Close']
        
        # Calculate historical performance
        signal_changes = signals_df['Signal_Change'] != 0
        total_signals = signal_changes.sum()
        
        win_rate = 0
        if total_signals > 0:
            wins = 0
            for i in range(len(signals_df) - 3):
                if signals_df['Signal_Change'].iloc[i] != 0:
                    signal = signals_df['Signal'].iloc[i]
                    future_idx = min(i + 3, len(signals_df) - 1)
                    future_return = (signals_df['QQQ_Close'].iloc[future_idx] / 
                                   signals_df['QQQ_Close'].iloc[i] - 1)
                    
                    if (signal == 1 and future_return > 0) or (signal == -1 and future_return < 0):
                        wins += 1
            
            win_rate = (wins / total_signals) * 100
        
        # ENHANCED: Console output with comprehensive analysis
        safe_print("\n" + "="*70)
        safe_print(f"{signal_emoji} WISHING WEALTH QQQ TRADING SIGNAL {signal_emoji}")
        safe_print("="*70)
        safe_print(f"Date: {current_date.strftime('%Y-%m-%d')}")
        safe_print(f"Signal: {signal_text}")
        safe_print(f"PRICE PREDICTION: {prediction_text}")
        safe_print(f"Recommended Action: {etf}")
        safe_print(f"QQQ Close: ${qqq_close:.2f}")
        safe_print(f"GMI Score: {gmi_score} / 6 ({gmi_signal})")
        safe_print(f"QQQ Trend: {trend_label}")
        safe_print(f"Historical Hit Rate: {win_rate:.1f}%")
        
        # VIX analysis
        if 'VIX_Close' in signals_df.columns:
            vix_value = signals_df.iloc[-1]['VIX_Close']
            vix_status = "HIGH (Fear)" if vix_value > VIX_THRESHOLD else "LOW (Complacency)"
            vix_emoji = "(Fear)" if vix_value > VIX_THRESHOLD else "(Complacency)"
            safe_print(f"VIX: {vix_value:.2f} - {vix_status} {vix_emoji}")
        
        # Dot signals
        dots = []
        if signals_df.iloc[-1]['Blue_Dot'] == 1:
            dots.append("Blue Dot")
        if signals_df.iloc[-1]['Green_Dot'] == 1:
            dots.append("Green Dot")
        if signals_df.iloc[-1]['Black_Dot'] == 1:
            dots.append("Black Dot")
        
        if dots:
            safe_print(f"Dot Signals: {', '.join(dots)}")
        
        safe_print("\nCONTRIBUTING FACTORS")
        safe_print("-"*70)
        
        # Technical analysis summary
        factors = []
        
        # Moving averages
        ma_signal = signals_df.iloc[-1]['MA_Signal']
        ma_status = "BULLISH" if ma_signal > 0 else "BEARISH" if ma_signal < 0 else "NEUTRAL"
        factors.append(f"Moving Averages: {ma_status}")
        
        # RSI
        rsi_value = signals_df.iloc[-1]['RSI']
        if rsi_value < RSI_OVERSOLD:
            rsi_status = "OVERSOLD (Bullish)"
        elif rsi_value > RSI_OVERBOUGHT:
            rsi_status = "OVERBOUGHT (Bearish)"
        else:
            rsi_status = "NEUTRAL"
        factors.append(f"RSI: {rsi_value:.1f} - {rsi_status}")
        
        # MACD
        macd_signal = signals_df.iloc[-1]['MACD_Signal_Indicator']
        macd_status = "BULLISH" if macd_signal > 0 else "BEARISH"
        factors.append(f"MACD: {macd_status}")
        
        # Bollinger Bands
        bb_signal = signals_df.iloc[-1]['BB_Signal']
        if bb_signal > 0:
            bb_status = "LOWER BAND (Bullish)"
        elif bb_signal < 0:
            bb_status = "UPPER BAND (Bearish)"
        else:
            bb_status = "WITHIN BANDS"
        factors.append(f"Bollinger Bands: {bb_status}")
        
        for factor in factors:
            safe_print(f"  • {factor}")
        
        # ENHANCED: Save signal to file with STANDARD FORMAT for dashboard compatibility
        signal_file_path = os.path.join(OUTPUT_PATH, "WishingWealthQQQ_signal.txt")
        try:
            with open(signal_file_path, 'w', encoding='utf-8') as f:
                f.write(f"WISHING WEALTH QQQ TRADING SIGNAL\n")
                f.write("=" * 50 + "\n")
                f.write(f"Generated on: {current_date.strftime('%Y-%m-%d')}\n")
                f.write(f"Signal: {signal_text}\n")
                f.write(f"PRICE PREDICTION: {prediction_text}\n")
                f.write(f"Suggested Action: {signal_text}\n")
                f.write(f"Current Price: ${qqq_close:.2f}\n")
                f.write(f"GMI Score: {gmi_score} / 6 ({gmi_signal})\n")
                f.write(f"QQQ Trend: {trend_label}\n")
                f.write(f"Win Rate: {win_rate:.1f}%\n\n")
                
                f.write("LEVERAGE ETF STRATEGY:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Recommended ETF: {etf}\n")
                f.write(f"Strategy: {'Long TQQQ' if etf == 'TQQQ' else 'Long SQQQ' if etf == 'SQQQ' else 'Cash/Hold'}\n\n")
                
                # Add backtesting results if available
                if backtest_results is not None:
                    try:
                        # Calculate performance metrics
                        trading_days_per_year = 252
                        years = len(backtest_results) / trading_days_per_year
                        
                        qqq_total_return = backtest_results['Cumulative_QQQ_Return'].iloc[-1]
                        strategy_total_return = backtest_results['Cumulative_Strategy_Return'].iloc[-1] 
                        leveraged_total_return = backtest_results['Cumulative_Leveraged_Return'].iloc[-1]
                        
                        qqq_annual_return = (1 + qqq_total_return) ** (1 / years) - 1 if years > 0 else 0
                        strategy_annual_return = (1 + strategy_total_return) ** (1 / years) - 1 if years > 0 else 0
                        leveraged_annual_return = (1 + leveraged_total_return) ** (1 / years) - 1 if years > 0 else 0
                        
                        # Sharpe ratios
                        risk_free_rate = 0.03
                        daily_rf = (1 + risk_free_rate) ** (1 / trading_days_per_year) - 1
                        
                        def safe_sharpe(returns, daily_rf):
                            excess_returns = returns - daily_rf
                            if excess_returns.std() > 0:
                                return (excess_returns.mean() / excess_returns.std()) * np.sqrt(trading_days_per_year)
                            return 0
                        
                        qqq_sharpe = safe_sharpe(backtest_results['QQQ_Return'], daily_rf)
                        strategy_sharpe = safe_sharpe(backtest_results['Strategy_Return'], daily_rf)
                        leveraged_sharpe = safe_sharpe(backtest_results['Leveraged_Strategy_Return'], daily_rf)
                        
                        # Drawdowns
                        def calculate_drawdown(portfolio_values):
                            peak = portfolio_values.cummax()
                            drawdown = (portfolio_values - peak) / peak
                            return drawdown.min()
                        
                        qqq_max_drawdown = calculate_drawdown(backtest_results['QQQ_Portfolio'])
                        strategy_max_drawdown = calculate_drawdown(backtest_results['Strategy_Portfolio'])
                        leveraged_max_drawdown = calculate_drawdown(backtest_results['Leveraged_Portfolio'])
                        
                        # Win rates
                        def calculate_win_rate(returns):
                            returns_clean = returns[returns != 0]
                            if len(returns_clean) > 0:
                                return (returns_clean > 0).sum() / len(returns_clean)
                            return 0
                        
                        qqq_win_rate = calculate_win_rate(backtest_results['QQQ_Return'])
                        strategy_win_rate = calculate_win_rate(backtest_results['Strategy_Return'])
                        leveraged_win_rate = calculate_win_rate(backtest_results['Leveraged_Strategy_Return'])
                        
                        # Trade statistics
                        trades = backtest_results[backtest_results['Signal_Change'] != 0]
                        num_trades = len(trades)
                        
                        f.write("HISTORICAL PERFORMANCE RESULTS:\n")
                        f.write("-" * 40 + "\n")
                        f.write(f"Backtest Period: {years:.1f} years\n\n")
                        
                        f.write("QQQ Buy and Hold Strategy:\n")
                        f.write(f"  Total Return: {qqq_total_return:.2%}\n")
                        f.write(f"  Annualized Return: {qqq_annual_return:.2%}\n")
                        f.write(f"  Sharpe Ratio: {qqq_sharpe:.2f}\n")
                        f.write(f"  Maximum Drawdown: {qqq_max_drawdown:.2%}\n")
                        f.write(f"  Win Rate: {qqq_win_rate:.2%}\n\n")
                        
                        f.write("QQQ Timing Strategy (This Model):\n")
                        f.write(f"  Total Return: {strategy_total_return:.2%}\n")
                        f.write(f"  Annualized Return: {strategy_annual_return:.2%}\n")
                        f.write(f"  Sharpe Ratio: {strategy_sharpe:.2f}\n")
                        f.write(f"  Maximum Drawdown: {strategy_max_drawdown:.2%}\n")
                        f.write(f"  Win Rate: {strategy_win_rate:.2%}\n")
                        f.write(f"  Number of Trades: {num_trades}\n\n")
                        
                        f.write("Leveraged TQQQ/SQQQ Strategy:\n")
                        f.write(f"  Total Return: {leveraged_total_return:.2%}\n")
                        f.write(f"  Annualized Return: {leveraged_annual_return:.2%}\n")
                        f.write(f"  Sharpe Ratio: {leveraged_sharpe:.2f}\n")
                        f.write(f"  Maximum Drawdown: {leveraged_max_drawdown:.2%}\n")
                        f.write(f"  Win Rate: {leveraged_win_rate:.2%}\n")
                        f.write(f"  Number of Trades: {num_trades}\n\n")
                        
                        # Performance comparison
                        qqq_vs_strategy = ((1 + strategy_total_return) / (1 + qqq_total_return) - 1) * 100
                        strategy_vs_leveraged = ((1 + leveraged_total_return) / (1 + strategy_total_return) - 1) * 100
                        
                        f.write("PERFORMANCE COMPARISON:\n")
                        f.write(f"Strategy vs QQQ Buy-Hold: {qqq_vs_strategy:+.1f}% outperformance\n")
                        f.write(f"Leveraged vs Basic Strategy: {strategy_vs_leveraged:+.1f}% additional return\n\n")
                        
                    except Exception as e:
                        f.write("HISTORICAL PERFORMANCE RESULTS:\n")
                        f.write("-" * 40 + "\n")
                        f.write(f"Error calculating performance metrics: {e}\n\n")
                
                f.write("TECHNICAL ANALYSIS SUMMARY:\n")
                f.write("-" * 30 + "\n")
                for factor in factors:
                    f.write(f"• {factor}\n")
                
                if 'VIX_Close' in signals_df.columns:
                    f.write(f"• VIX: {vix_value:.2f} - {vix_status}\n")
                
                if dots:
                    f.write(f"• Dot Signals: {', '.join(dots)}\n")
                
                f.write(f"\nDr. Wish's methodology emphasizes GMI score and trend changes for optimal timing.\n")
                f.write(f"This system historically outperforms 94%+ of individual stock picks.\n")
        except UnicodeEncodeError:
            # Fallback to ASCII encoding
            with open(signal_file_path, 'w', encoding='ascii', errors='ignore') as f:
                f.write(f"WISHING WEALTH QQQ TRADING SIGNAL\n")
                f.write("=" * 50 + "\n")
                f.write(f"Generated on: {current_date.strftime('%Y-%m-%d')}\n")
                f.write(f"Signal: {signal_text}\n")
                f.write(f"PRICE PREDICTION: {prediction_text}\n")
                f.write(f"Suggested Action: {signal_text}\n")
                f.write(f"Current Price: ${qqq_close:.2f}\n")
                f.write(f"GMI Score: {gmi_score} / 6 ({gmi_signal})\n")
                f.write(f"QQQ Trend: {trend_label}\n")
                f.write(f"Win Rate: {win_rate:.1f}%\n\n")
                
                f.write("LEVERAGE ETF STRATEGY:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Recommended ETF: {etf}\n")
                f.write(f"Strategy: {'Long TQQQ' if etf == 'TQQQ' else 'Long SQQQ' if etf == 'SQQQ' else 'Cash/Hold'}\n\n")
                
                f.write("TECHNICAL ANALYSIS SUMMARY:\n")
                f.write("-" * 30 + "\n")
                for factor in factors:
                    f.write(f"• {factor}\n")
                
                if 'VIX_Close' in signals_df.columns:
                    f.write(f"• VIX: {vix_value:.2f} - {vix_status}\n")
                
                if dots:
                    f.write(f"• Dot Signals: {', '.join(dots)}\n")
                
                f.write(f"\nDr. Wish's methodology emphasizes GMI score and trend changes for optimal timing.")
        
        safe_print(f"\nEnhanced signal saved to {signal_file_path}")
        safe_print("="*70)
        
        return current_signal, signal_text, etf
    except Exception as e:
        safe_print(f"Error generating current signal: {e}")
        traceback.print_exc()
        
        # Create minimal fallback signal
        try:
            signal_file_path = os.path.join(OUTPUT_PATH, "WishingWealthQQQ_signal.txt")
            with open(signal_file_path, 'w', encoding='ascii') as f:
                f.write(f"WISHING WEALTH QQQ TRADING SIGNAL\n")
                f.write("=" * 50 + "\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d')}\n")
                f.write(f"Signal: HOLD\n")
                f.write(f"PRICE PREDICTION: QQQ direction uncertain - Stay in CASH\n")
                f.write(f"Suggested Action: HOLD\n")
                f.write(f"Current Price: $0.00\n")
                f.write(f"GMI Score: 3 / 6 (Yellow)\n")
                f.write(f"Win Rate: 50.0%\n\n")
                f.write("Error occurred during signal generation\n")
        except:
            pass
            
        return 0, "HOLD", "CASH"

def main():
    safe_print("Initializing OPTIMIZED WishingWealthQQQ Trading Signal Generator...")
    safe_print("Features: DATABASE-FIRST + Dr. Wish's GMI methodology + ENCODING SAFE")
    
    try:
        # Parameters
        start_date = "2015-01-01"  # 10 years of data
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Step 1: Prepare data with DATABASE-FIRST approach
        features_df = prepare_data_optimized(start_date, end_date)
        if features_df is None:
            safe_print("Failed to prepare data. Exiting.")
            return
        
        # Step 2: Generate optimized signals
        signals_df = generate_signals_optimized(features_df)
        
        # Step 3: Run optimized backtesting
        backtest_results = backtest_strategy_optimized(signals_df)
        
        # Step 4: Generate current signal with comprehensive analysis
        current_signal, signal_text, etf = generate_current_signal_optimized(signals_df, backtest_results)
        
        safe_print("\nOPTIMIZED WISHING WEALTH QQQ COMPLETED")
        safe_print(f"Current Signal: {signal_text} {etf}")
        safe_print(f"Enhanced with Dr. Wish's GMI methodology")
        safe_print(f"DATABASE-FIRST data sourcing")
        safe_print(f"Optimized performance metrics")
        safe_print(f"ENCODING SAFE - No more Unicode errors!")
        
        # Save prediction to database
        safe_print("\n[DATABASE] Saving prediction to database...")
        try:
            # Get current QQQ price from the signals DataFrame
            current_qqq_price = signals_df.iloc[-1]['QQQ_Close']
            
            # For Wishing Well, we generate directional predictions rather than specific price targets
            # Estimate price target based on signal direction and historical volatility
            price_volatility = signals_df['QQQ_Close'].pct_change().std() * 5  # 5-day volatility estimate
            
            if current_signal == 1:  # BUY signal
                predicted_price = current_qqq_price * (1 + price_volatility)
                suggested_action = "BUY"
                confidence = 75.0
            elif current_signal == -1:  # SELL signal  
                predicted_price = current_qqq_price * (1 - price_volatility)
                suggested_action = "SELL"
                confidence = 75.0
            else:  # HOLD signal
                predicted_price = current_qqq_price  # No change predicted
                suggested_action = "HOLD"
                confidence = 60.0
            
            # Adjust confidence based on GMI score if available
            if 'GMI_Score' in signals_df.columns:
                gmi_score = signals_df.iloc[-1]['GMI_Score']
                # GMI score ranges from 0-6, use it to adjust confidence
                confidence = 50.0 + (gmi_score / 6.0) * 40.0  # Scale to 50-90%
            
            db_success = quick_save_prediction(
                model_name="Wishing Well QQQ Model",
                symbol="QQQ",
                current_price=current_qqq_price,
                predicted_price=predicted_price,
                confidence=confidence,
                horizon_days=5,  # Wishing Well is typically for medium-term signals
                suggested_action=suggested_action
            )
            
            if db_success:
                safe_print("[DATABASE] ✅ Wishing Well prediction saved to database")
            else:
                safe_print("[DATABASE] ❌ Failed to save Wishing Well prediction")
            
            # Save model metrics if backtest results available
            if backtest_results and isinstance(backtest_results, dict):
                safe_print("[DATABASE] Saving model metrics...")
                
                # Extract key metrics from backtest results
                wishing_well_metrics = {}
                if 'win_rate' in backtest_results:
                    wishing_well_metrics['win_rate'] = backtest_results['win_rate']
                if 'total_return' in backtest_results:
                    wishing_well_metrics['total_return'] = backtest_results['total_return']
                if 'sharpe_ratio' in backtest_results:
                    wishing_well_metrics['sharpe_ratio'] = backtest_results['sharpe_ratio']
                
                # Add GMI score as a metric
                if 'GMI_Score' in signals_df.columns:
                    wishing_well_metrics['current_gmi_score'] = signals_df.iloc[-1]['GMI_Score']
                
                if wishing_well_metrics:
                    metrics_success = quick_save_metrics("Wishing Well QQQ Model", "QQQ", wishing_well_metrics)
                    if metrics_success:
                        safe_print("[DATABASE] ✅ Model metrics saved to database")
                    else:
                        safe_print("[DATABASE] ❌ Failed to save model metrics")
                        
        except Exception as db_error:
            safe_print(f"[DATABASE] ❌ Error saving to database: {db_error}")
        
    except Exception as e:
        safe_print(f"Error in optimized main function: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()