"""
QQQ Trading Signal Generator

This model combines the best elements of the Long Horn Bull Model and simplified approach
to generate clear trading signals with confidence levels and price targets.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import joblib
import warnings
import traceback
import data_fetcher # Import the new data_fetcher module
import sys
sys.path.append('..')
from model_db_integrator import quick_save_prediction, quick_save_metrics

warnings.filterwarnings("ignore")

# Configuration
TICKER = "QQQ"
START_DATE = "2014-08-13"  # 10+ years of historical data for robust training
ALPHA_VANTAGE_API_KEY = "HMHALLINAHS2FF4Z" # User provided API Key

# Use GitHub repo reports directory instead of Desktop
script_dir = os.path.dirname(os.path.abspath(__file__))
DESKTOP_PATH = os.path.join(script_dir, "reports")
os.makedirs(DESKTOP_PATH, exist_ok=True)

class QQQTradingSignalGenerator:
    def __init__(self, start_date=START_DATE, end_date=None, prediction_days=3):
        """
        Initialize the QQQ trading signal generator.
        
        Parameters:
        -----------
        start_date : str
            Start date for historical data in 'YYYY-MM-DD' format
        end_date : str or None
            End date for historical data in 'YYYY-MM-DD' format, defaults to today
        prediction_days : int
            Number of days to look ahead for prediction
        """
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.today().strftime("%Y-%m-%d")
        self.prediction_days = prediction_days
        self.model = None
        self.features = None
        self.df = None
        
    def calculate_ema(self, series, window):
        """Calculate Exponential Moving Average."""
        return series.ewm(span=window, adjust=False).mean()

    def calculate_bollinger_bands(self, series, window=20, num_std=2):
        """Calculate Bollinger Bands."""
        sma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        return sma + (std * num_std), sma, sma - (std * num_std)

    def calculate_rsi(self, series, window=14):
        """Calculate Relative Strength Index."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_atr(self, high, low, close, window=14):
        """Calculate Average True Range."""
        tr = pd.DataFrame({
            "hl": high - low,
            "hc": (high - close.shift()).abs(),
            "lc": (low - close.shift()).abs()
        }).max(axis=1)
        return tr.rolling(window=window).mean()

    def create_features(self, stock_data, vix_data=None):
        """Create features for the trading signal model."""
        print("Creating features for trading signal generation...")
        
        df = pd.DataFrame(index=stock_data.index)
        
        # Extract price data
        close = stock_data["Close"].squeeze()
        high = stock_data["High"].squeeze()
        low = stock_data["Low"].squeeze()
        
        # Basic price data
        df["Close"] = close
        df["Open"] = stock_data["Open"].squeeze()
        df["High"] = high
        df["Low"] = low
        df["Volume"] = stock_data["Volume"].squeeze()
        
        # Price and returns features
        df["Returns_1d"] = close.pct_change()
        df["Returns_5d"] = close.pct_change(5)
        df["Returns_20d"] = close.pct_change(20)
        
        # Moving averages
        df["SMA_5"] = close.rolling(5).mean()
        df["SMA_20"] = close.rolling(20).mean()
        df["SMA_50"] = close.rolling(50).mean()
        df["SMA_200"] = close.rolling(200).mean()
        
        # Moving average crossovers
        df["SMA_5_20_Ratio"] = df["SMA_5"] / df["SMA_20"]
        df["SMA_20_50_Ratio"] = df["SMA_20"] / df["SMA_50"]
        df["SMA_50_200_Ratio"] = df["SMA_50"] / df["SMA_200"]
        
        # Exponential moving averages
        df["EMA_12"] = self.calculate_ema(close, 12)
        df["EMA_26"] = self.calculate_ema(close, 26)
        
        # MACD
        df["MACD"] = df["EMA_12"] - df["EMA_26"]
        df["MACD_Signal"] = self.calculate_ema(df["MACD"], 9)
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
        
        # Bollinger Bands
        df["BB_Upper"], df["BB_Middle"], df["BB_Lower"] = self.calculate_bollinger_bands(close)
        df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
        df["BB_Position"] = (close - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])
        
        # RSI
        df["RSI"] = self.calculate_rsi(close)
        
        # ATR
        df["ATR"] = self.calculate_atr(high, low, close)
        df["ATR_Ratio"] = df["ATR"] / close
        
        # Volume features
        df["Volume_SMA_20"] = df["Volume"].rolling(20).mean()
        df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA_20"]
        
        # Add VIX features (now with restored historical data)
        if vix_data is not None and not vix_data.empty and "Close" in vix_data.columns:
            print(f"Adding real VIX features from {len(vix_data)} VIX data points")
            try:
                # Resample VIX to match stock data index
                vix_resampled = vix_data["Close"].reindex(df.index, method="ffill")
                df["VIX"] = vix_resampled
                df["VIX_SMA_20"] = df["VIX"].rolling(20).mean()
                df["VIX_Ratio"] = df["VIX"] / df["VIX_SMA_20"]
                print("Successfully added real VIX features")
            except Exception as e:
                print(f"Failed to add real VIX features: {e}")
        else:
            print("WARNING: VIX data still unavailable - check data fetching")
        
        # Target variable - future returns
        df[f"Target_{self.prediction_days}d"] = close.pct_change(self.prediction_days).shift(-self.prediction_days)
        
        # More intelligent NaN handling - don't drop all rows
        print(f"Before NaN handling: {len(df)} rows, {df.isna().sum().sum()} total NaN values")
        
        # Fill forward then backward, then drop only rows where target is NaN
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Only drop rows where the target variable is NaN (essential for training)
        target_col = f"Target_{self.prediction_days}d"
        if target_col in df.columns:
            df = df.dropna(subset=[target_col])
        
        print(f"After NaN handling: {len(df)} rows remaining")
        
        return df

    def select_features(self):
        """Select the most relevant features for the model."""
        # Basic feature set - can be expanded based on feature importance analysis
        features = [
            "Returns_1d", "Returns_5d", "Returns_20d",
            "SMA_5_20_Ratio", "SMA_20_50_Ratio", "SMA_50_200_Ratio",
            "MACD", "MACD_Signal", "MACD_Hist",
            "BB_Width", "BB_Position",
            "RSI", "ATR_Ratio", "Volume_Ratio"
        ]
        
        # Add VIX features if available
        if "VIX_Ratio" in self.df.columns:
            features.append("VIX_Ratio")
        
        return features

    def collect_data(self):
        """Download historical data for QQQ and VIX using data_fetcher."""
        print(f"Downloading data for {TICKER} from {self.start_date} to {self.end_date}...")
        
        stock_data = data_fetcher.fetch_historical_data(TICKER, self.start_date, self.end_date, api_key=ALPHA_VANTAGE_API_KEY)
        
        vix_data = data_fetcher.fetch_historical_data("^VIX", self.start_date, self.end_date, api_key=ALPHA_VANTAGE_API_KEY)
        
        if stock_data is None or stock_data.empty:
            print(f"Critical error: Could not download data for {TICKER}. Exiting.")
            # Optionally, raise an exception or handle more gracefully
            return None, None 

        print(f"Downloaded {len(stock_data)} days of data for {TICKER}")
        if vix_data is not None:
            print(f"Downloaded {len(vix_data)} days of data for VIX")
        else:
            print("Warning: Could not download VIX data. Proceeding without VIX features.")
            
        return stock_data, vix_data

    def prepare_data(self, stock_data=None, vix_data=None):
        """Prepare data for model training and prediction."""
        if stock_data is None:
            stock_data, vix_data = self.collect_data()
            if stock_data is None: # Check if collect_data failed
                return None, None
        
        # Create features
        self.df = self.create_features(stock_data, vix_data)
        if self.df is None or self.df.empty:
            print("Feature creation resulted in empty DataFrame. Cannot proceed.")
            return None, None
            
        # Select features
        self.features = self.select_features()
        
        print(f"Data prepared with {len(self.df)} rows and {len(self.features)} features")
        return self.df, self.features

    def train_model(self, train_size=0.8, model_type="xgboost"):
        """Train the trading signal model."""
        print(f"Training {model_type} model...")
        
        # Prepare data if not already done
        if self.df is None or self.features is None:
            prepared_df, prepared_features = self.prepare_data()
            if prepared_df is None or prepared_features is None:
                 print("Model training aborted due to data preparation failure.")
                 return None # Indicate failure
        
        # Train-test split
        split_idx = int(len(self.df) * train_size)
        train_data = self.df.iloc[:split_idx]
        test_data = self.df.iloc[split_idx:]
        
        # Define target
        target = f"Target_{self.prediction_days}d"
        
        if target not in self.df.columns:
            print(f"Target column 	{target}	 not found in DataFrame. Available columns: {self.df.columns}")
            return None

        # Choose model type
        if model_type == "ridge":
            # Simple Ridge regression model with standardization
            self.model = Pipeline([
                ("scaler", RobustScaler()),
                ("model", Ridge(alpha=1.0))
            ])
        elif model_type == "xgboost":
            # XGBoost model
            self.model = Pipeline([
                ("scaler", RobustScaler()),
                ("model", XGBRegressor(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                ))
            ])
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        self.model.fit(train_data[self.features], train_data[target])
        
        # Generate predictions
        train_preds = self.model.predict(train_data[self.features])
        test_preds = self.model.predict(test_data[self.features])
        
        # Calculate performance metrics
        train_metrics = self.calculate_performance_metrics(train_data[target], train_preds)
        test_metrics = self.calculate_performance_metrics(test_data[target], test_preds)
        
        # Calculate feature importance
        if model_type == "ridge":
            importance = pd.DataFrame({
                "Feature": self.features,
                "Importance": np.abs(self.model.named_steps["model"].coef_)
            }).sort_values("Importance", ascending=False)
        else:  # xgboost
            importance = pd.DataFrame({
                "Feature": self.features,
                "Importance": self.model.named_steps["model"].feature_importances_
            }).sort_values("Importance", ascending=False)
        
        print(f"Model trained. Test Hit Rate: {test_metrics['hit_rate']:.2%}")
        
        return {
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "feature_importance": importance,
            "train_data": train_data,
            "test_data": test_data,
            "train_preds": train_preds,
            "test_preds": test_preds
        }

    def calculate_performance_metrics(self, actual, predicted):
        """Calculate comprehensive performance metrics."""
        # Basic regression metrics
        mse = np.mean((actual - predicted) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actual - predicted))
        
        # R-squared
        ss_total = np.sum((actual - np.mean(actual)) ** 2)
        ss_residual = np.sum((actual - predicted) ** 2)
        r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
        
        # Directional accuracy
        actual_direction = np.sign(actual)
        predicted_direction = np.sign(predicted)
        hit_rate = np.mean(actual_direction == predicted_direction)
        
        # Trading strategy metrics
        # Only take positions when prediction exceeds threshold
        threshold = np.std(predicted) * 0.5
        signal = np.where(predicted > threshold, 1, np.where(predicted < -threshold, -1, 0))
        
        # Calculate returns based on signal
        strategy_returns = actual * signal
        
        # Sharpe ratio
        sharpe_ratio = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() != 0 else 0
        
        # Win rate
        wins = np.sum((strategy_returns > 0) & (signal != 0))
        trades = np.sum(signal != 0)
        win_rate = wins / trades if trades > 0 else 0
        
        # Profit factor
        gross_profit = np.sum(strategy_returns[strategy_returns > 0])
        gross_loss = np.abs(np.sum(strategy_returns[strategy_returns < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        
        # Maximum drawdown
        cumulative_returns = (1 + strategy_returns).cumprod()
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r_squared": r_squared,
            "hit_rate": hit_rate,
            "sharpe_ratio": sharpe_ratio,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "trades": trades
        }

    def generate_trading_signal(self, confidence_threshold=0.6):
        """Generate explicit trading signal with confidence level and price targets."""
        if self.model is None:
            print("Model has not been trained. Training model first...")
            train_results = self.train_model()
            if train_results is None:
                print("Failed to train model. Cannot generate signal.")
                return None # Indicate failure
        
        # Get the most recent data
        if self.df is None or self.df.empty or self.features is None:
            print("Data not prepared. Preparing data first...")
            prepared_df, prepared_features = self.prepare_data()
            if prepared_df is None or prepared_features is None:
                print("Failed to prepare data. Cannot generate signal.")
                return None # Indicate failure

        recent_data = self.df.iloc[-1:]
        
        # Make prediction
        predicted_return = float(self.model.predict(recent_data[self.features])[0])
        
        # Get current price using data_fetcher
        current_price = data_fetcher.fetch_current_price(TICKER, api_key_twelvedata=ALPHA_VANTAGE_API_KEY)
        if current_price is None:
            print(f"Could not fetch current price for {TICKER}. Cannot generate signal.")
            return None # Indicate failure
        current_price = float(current_price)
        
        # Calculate predicted price
        predicted_price = current_price * (1 + predicted_return)
        
        # Calculate confidence based on historical volatility
        if "ATR_Ratio" not in recent_data.columns or recent_data["ATR_Ratio"].empty:
            print("ATR_Ratio not available in recent data. Cannot calculate confidence.")
            return None
        daily_volatility = float(recent_data["ATR_Ratio"].iloc[0])
        prediction_volatility = daily_volatility * np.sqrt(self.prediction_days)
        
        # Normalize prediction by volatility to get z-score
        if prediction_volatility == 0: # Avoid division by zero
            z_score = float("inf") if predicted_return != 0 else 0
        else:
            z_score = predicted_return / prediction_volatility
        
        # Convert z-score to confidence (0-100%)
        confidence = min(abs(z_score) * 50, 100)  # Scale and cap at 100%
        
        # Determine signal based on prediction and confidence
        if predicted_return > prediction_volatility and confidence >= confidence_threshold:
            signal = "BUY"
            action = "OPEN LONG"
        elif predicted_return < -prediction_volatility and confidence >= confidence_threshold:
            signal = "SELL"
            action = "OPEN SHORT"
        else:
            signal = "NEUTRAL"
            action = "HOLD"
        
        # Calculate price targets
        take_profit = None
        stop_loss = None
        if signal == "BUY":
            # Set take profit at 2x the predicted move
            take_profit = current_price * (1 + predicted_return * 2)
            # Set stop loss at 1x the predicted volatility below current price
            stop_loss = current_price * (1 - prediction_volatility)
        elif signal == "SELL":
            # Set take profit at 2x the predicted move
            take_profit = current_price * (1 + predicted_return * 2)
            # Set stop loss at 1x the predicted volatility above current price
            stop_loss = current_price * (1 + prediction_volatility)
        
        # Calculate risk-reward ratio
        risk_reward_ratio = None
        if take_profit is not None and stop_loss is not None and current_price != 0 and stop_loss != current_price:
            if signal == "BUY":
                risk = (current_price - stop_loss) / current_price
                reward = (take_profit - current_price) / current_price
            else:  # SELL
                risk = (stop_loss - current_price) / current_price
                reward = (current_price - take_profit) / current_price
            
            risk_reward_ratio = reward / risk if risk != 0 else float("inf")
        
        # Calculate confidence interval
        z_score_95 = 1.96  # 95% confidence interval
        margin = current_price * prediction_volatility * z_score_95
        lower_bound = predicted_price - margin
        upper_bound = predicted_price + margin
        
        signal_output = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ticker": TICKER,
            "current_price": current_price,
            "predicted_return_horizon_days": self.prediction_days,
            "predicted_return_pct": predicted_return * 100,
            "predicted_price": predicted_price,
            "confidence_pct": confidence,
            "signal": signal,
            "action": action,
            "take_profit_price": take_profit,
            "stop_loss_price": stop_loss,
            "risk_reward_ratio": risk_reward_ratio,
            "predicted_price_lower_bound_95ci": lower_bound,
            "predicted_price_upper_bound_95ci": upper_bound,
            "daily_volatility_atr_ratio": daily_volatility,
            "prediction_volatility": prediction_volatility
        }
        
        print("\n--- QQQ Trading Signal ---")
        for key, value in signal_output.items():
            if isinstance(value, float):
                print(f"{key.replace('_', ' ').title()}: {value:.2f}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")
        
        # Save report
        report_path = os.path.join(DESKTOP_PATH, f"{TICKER}_Trading_Signal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(report_path, "w") as f:
            f.write(f"=== {TICKER} Trading Signal Report ===\n")
            for key, value in signal_output.items():
                if isinstance(value, float):
                    f.write(f"{key.replace('_', ' ').title()}: {value:.4f}\n") # More precision in file
                else:
                    f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        print(f"\nReport saved to: {report_path}")
        
        return signal_output

if __name__ == "__main__":
    try:
        # Ensure ALPHA_VANTAGE_API_KEY is available to data_fetcher
        # data_fetcher.py will use os.getenv("AV_API_KEY", "HMHALLINAHS2FF4Z")
        # If you want to force this key, you can set it as an environment variable before running data_fetcher, 
        # or modify data_fetcher.py to accept it as a parameter in its functions if not already.
        # For this script, we pass it directly to the data_fetcher functions.
        os.environ["AV_API_KEY"] = ALPHA_VANTAGE_API_KEY

        generator = QQQTradingSignalGenerator()
        
        # Train model (this will also prepare data if not already done)
        training_results = generator.train_model(model_type="xgboost")
        
        if training_results:
            # Generate and display trading signal
            signal_info = generator.generate_trading_signal()
            if signal_info:
                print("\nTrading signal generation successful.")
                
                # Save prediction to database
                print("\n[DATABASE] Saving prediction to database...")
                db_success = quick_save_prediction(
                    model_name="QQQ Trading Signal",
                    symbol="QQQ",
                    current_price=signal_info['current_price'],
                    predicted_price=signal_info['predicted_price'],
                    confidence=signal_info['confidence_pct'],
                    horizon_days=signal_info['predicted_return_horizon_days'],
                    suggested_action=signal_info['action']
                )
                
                if db_success:
                    print("[DATABASE] [OK] QQQ Trading Signal prediction saved to database")
                else:
                    print("[DATABASE] [ERROR] Failed to save QQQ Trading Signal prediction")
                
                # Save model metrics to database
                if training_results.get('test_metrics'):
                    print("[DATABASE] Saving model metrics...")
                    test_metrics = training_results['test_metrics']
                    metrics_success = quick_save_metrics("QQQ Trading Signal", "QQQ", test_metrics)
                    if metrics_success:
                        print("[DATABASE] [OK] Model metrics saved to database")
                    else:
                        print("[DATABASE] [ERROR] Failed to save model metrics")
                
            else:
                print("\nFailed to generate trading signal.")
        else:
            print("\nModel training failed. Cannot generate trading signal.")
            
    except Exception as e:
        print(f"\nCritical Error in QQQ Trading Signal script: {str(e)}")
        traceback.print_exc()

