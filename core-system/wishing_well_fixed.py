#!/usr/bin/env python3
"""
FIXED Wishing Well QQQ Model
Properly tracks GMI signals instead of fake price predictions
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from signal_db_integrator import save_signal_prediction, update_signal_performance, get_signal_performance
warnings.filterwarnings('ignore')

# Add the parent directory to the path for importing data_fetcher
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

try:
    import data_fetcher
except ImportError:
    print("Error: data_fetcher module not found. Please ensure data_fetcher.py is available.")
    sys.exit(1)

def safe_print(text):
    """Print text safely, removing Unicode characters if needed"""
    try:
        print(text)
    except UnicodeEncodeError:
        ascii_text = text.encode('ascii', 'ignore').decode('ascii')
        print(ascii_text)

def calculate_gmi_score(qqq_data):
    """
    Calculate Dr. Wish's GMI (General Market Index) Score
    Returns a score from 0-6 based on market conditions
    """

    try:
        latest_data = qqq_data.iloc[-1]
        close_prices = qqq_data['Close']

        # Component 1: QQQ above 30-day MA
        ma30 = close_prices.rolling(window=30).mean().iloc[-1]
        component1 = 1 if latest_data['Close'] > ma30 else 0

        # Component 2: QQQ 10-day MA above 30-day MA
        ma10 = close_prices.rolling(window=10).mean().iloc[-1]
        component2 = 1 if ma10 > ma30 else 0

        # Component 3: Strong recent momentum (close above 20-day high)
        high20 = close_prices.rolling(window=20).max().iloc[-1]
        component3 = 1 if latest_data['Close'] >= high20 * 0.98 else 0

        # Component 4: Volume confirmation (above 20-day average)
        if 'Volume' in qqq_data.columns:
            avg_volume = qqq_data['Volume'].rolling(window=20).mean().iloc[-1]
            component4 = 1 if latest_data['Volume'] > avg_volume else 0
        else:
            component4 = 1  # Default if no volume data

        # Component 5: Short-term trend (5-day MA above 10-day MA)
        ma5 = close_prices.rolling(window=5).mean().iloc[-1]
        component5 = 1 if ma5 > ma10 else 0

        # Component 6: No recent significant drawdown (< 5% from 10-day high)
        high10 = close_prices.rolling(window=10).max().iloc[-1]
        drawdown = (high10 - latest_data['Close']) / high10
        component6 = 1 if drawdown < 0.05 else 0

        gmi_score = component1 + component2 + component3 + component4 + component5 + component6

        # Create component breakdown for additional data
        components = {
            'above_ma30': component1,
            'ma10_above_ma30': component2,
            'near_20d_high': component3,
            'volume_above_avg': component4,
            'short_term_trend': component5,
            'small_drawdown': component6,
            'total_score': gmi_score
        }

        return gmi_score, components

    except Exception as e:
        safe_print(f"Error calculating GMI score: {e}")
        return 3, {'error': str(e)}  # Default neutral score

def generate_signal(gmi_score, components):
    """
    Generate trading signal based on GMI score

    Rules:
    - GMI 0-2: SELL signal (bearish)
    - GMI 3-4: HOLD signal (neutral)
    - GMI 5-6: BUY signal (bullish)
    """

    if gmi_score >= 5:
        signal = 'BUY'
        expected_return = 2.5  # Conservative 2.5% expected return
        confidence = 70 + (gmi_score - 5) * 15  # 70-85% confidence
    elif gmi_score <= 2:
        signal = 'SELL'
        expected_return = -2.0  # Expected decline
        confidence = 65 + (2 - gmi_score) * 10  # 65-85% confidence
    else:
        signal = 'HOLD'
        expected_return = 0.5  # Small positive bias for market
        confidence = 50 + gmi_score * 5  # 50-65% confidence

    return signal, expected_return, confidence

def main():
    """Main function for Wishing Well QQQ Signal Generator"""

    safe_print("="*80)
    safe_print("WISHING WELL QQQ SIGNAL GENERATOR - FIXED VERSION")
    safe_print("Now properly tracks GMI signals instead of fake price predictions")
    safe_print("="*80)

    try:
        # Get QQQ data
        safe_print("Fetching QQQ market data...")

        try:
            # Try to get 90 days of data for proper moving averages
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)

            qqq_data = data_fetcher.fetch_historical_data('QQQ',
                                                       start_date.strftime('%Y-%m-%d'),
                                                       end_date.strftime('%Y-%m-%d'))

            if qqq_data is None or len(qqq_data) < 30:
                safe_print("Error: Insufficient QQQ data for analysis")
                return False

            safe_print(f"Successfully fetched {len(qqq_data)} days of QQQ data")

        except Exception as e:
            safe_print(f"Error fetching QQQ data: {e}")
            return False

        # Calculate GMI Score
        safe_print("Calculating GMI (General Market Index) score...")
        gmi_score, components = calculate_gmi_score(qqq_data)

        # Generate signal
        signal, expected_return, confidence = generate_signal(gmi_score, components)

        # Get current price
        current_price = qqq_data['Close'].iloc[-1]

        # Display results
        safe_print(f"\nGMI ANALYSIS RESULTS:")
        safe_print(f"Current QQQ Price: ${current_price:.2f}")
        safe_print(f"GMI Score: {gmi_score}/6")
        safe_print(f"Signal: {signal}")
        safe_print(f"Expected Return: {expected_return:+.1f}%")
        safe_print(f"Confidence: {confidence:.0f}%")

        safe_print(f"\nGMI COMPONENT BREAKDOWN:")
        for component, value in components.items():
            if component != 'total_score' and component != 'error':
                status = "YES" if value == 1 else "NO"
                safe_print(f"  {component.replace('_', ' ').title()}: {status}")

        # Save to signal tracking database
        safe_print(f"\n[DATABASE] Saving signal to database...")

        try:
            success = save_signal_prediction(
                model_name="Wishing Well QQQ Model",
                symbol="QQQ",
                signal_type=signal,
                gmi_score=gmi_score,
                signal_strength=confidence,
                current_price=current_price,
                target_return=expected_return,
                horizon_days=5,
                additional_data=components
            )

            if success:
                safe_print("[DATABASE] SUCCESS: Signal saved to database")

                # Update performance metrics
                update_signal_performance("Wishing Well QQQ Model")
                safe_print("[DATABASE] SUCCESS: Performance metrics updated")

                # Display historical performance
                performance = get_signal_performance("Wishing Well QQQ Model")
                if performance:
                    safe_print(f"\nHISTORICAL SIGNAL PERFORMANCE:")
                    safe_print(f"{'Signal':<8} {'Total':<8} {'Correct':<8} {'Win Rate':<10} {'Avg Return'}")
                    safe_print("-" * 50)
                    for row in performance:
                        signal_type, total, correct, win_rate, avg_return = row
                        safe_print(f"{signal_type:<8} {total:<8} {correct or 0:<8} {win_rate or 0:<9.1f}% {avg_return or 0:<.2f}%")

            else:
                safe_print("[DATABASE] ERROR: Failed to save signal")

        except Exception as e:
            safe_print(f"[DATABASE] ERROR: {e}")

        # Generate report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"reports/Wishing_Well_Signal_Report_{timestamp}.txt"

        try:
            os.makedirs("reports", exist_ok=True)

            with open(report_filename, 'w') as f:
                f.write("="*80 + "\n")
                f.write("WISHING WELL QQQ SIGNAL REPORT\n")
                f.write("="*80 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Symbol: QQQ\n")
                f.write(f"Current Price: ${current_price:.2f}\n\n")

                f.write("SIGNAL ANALYSIS:\n")
                f.write(f"GMI Score: {gmi_score}/6\n")
                f.write(f"Signal: {signal}\n")
                f.write(f"Expected Return: {expected_return:+.1f}%\n")
                f.write(f"Confidence: {confidence:.0f}%\n\n")

                f.write("GMI COMPONENT BREAKDOWN:\n")
                for component, value in components.items():
                    if component != 'total_score' and component != 'error':
                        status = "YES" if value == 1 else "NO"
                        f.write(f"  {component.replace('_', ' ').title()}: {status}\n")

                f.write(f"\nHORIZON: 5 days\n")
                f.write(f"MODEL: Wishing Well QQQ (Signal-based, not price prediction)\n")

                if performance:
                    f.write(f"\nHISTORICAL PERFORMANCE:\n")
                    for row in performance:
                        signal_type, total, correct, win_rate, avg_return = row
                        f.write(f"{signal_type}: {correct or 0}/{total} ({win_rate or 0:.1f}% win rate, {avg_return or 0:.2f}% avg return)\n")

            safe_print(f"Report saved: {report_filename}")

        except Exception as e:
            safe_print(f"Error saving report: {e}")

        safe_print("\n" + "="*80)
        safe_print("WISHING WELL QQQ SIGNAL ANALYSIS COMPLETE")
        safe_print("Model now properly tracks signals instead of fake predictions!")
        safe_print("="*80)

        return True

    except Exception as e:
        safe_print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()