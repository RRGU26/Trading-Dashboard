#!/usr/bin/env python3
"""
Fix NVIDIA stock split issue in the model
NVIDIA had a 10-for-1 stock split on June 7, 2024
"""

import pandas as pd
import numpy as np
from datetime import datetime

def adjust_for_stock_split(df, split_date='2024-06-07', split_ratio=10):
    """
    Adjust historical prices for NVIDIA's 10-for-1 stock split
    
    Parameters:
    - df: DataFrame with price data
    - split_date: Date of the stock split
    - split_ratio: Split ratio (10 for 10-for-1 split)
    """
    print(f"Adjusting for {split_ratio}-for-1 stock split on {split_date}")
    
    # Convert split_date to datetime
    split_date = pd.to_datetime(split_date)
    
    # Create a copy to avoid modifying original
    adjusted_df = df.copy()
    
    # Ensure Date column is datetime
    if 'Date' in adjusted_df.columns:
        adjusted_df['Date'] = pd.to_datetime(adjusted_df['Date'])
        date_col = 'Date'
    else:
        # Assume index is the date
        adjusted_df.index = pd.to_datetime(adjusted_df.index)
        date_col = None
    
    # Get the date column for comparison
    if date_col:
        dates = adjusted_df[date_col]
    else:
        dates = adjusted_df.index
    
    # Find rows before the split date
    pre_split_mask = dates < split_date
    
    # Adjust price columns for pre-split data
    price_columns = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Adjusted Close']
    
    for col in price_columns:
        if col in adjusted_df.columns:
            # Divide pre-split prices by the split ratio
            adjusted_df.loc[pre_split_mask, col] = adjusted_df.loc[pre_split_mask, col] / split_ratio
            print(f"  Adjusted {col} column")
    
    # Adjust volume for pre-split data (multiply by split ratio)
    if 'Volume' in adjusted_df.columns:
        adjusted_df.loc[pre_split_mask, 'Volume'] = adjusted_df.loc[pre_split_mask, 'Volume'] * split_ratio
        print(f"  Adjusted Volume column")
    
    # Show before/after comparison
    if not adjusted_df.empty:
        # Find a date just before the split for comparison
        pre_split_data = adjusted_df[pre_split_mask]
        if not pre_split_data.empty:
            last_pre_split = pre_split_data.iloc[-1]
            print(f"\nExample adjustment (last pre-split day):")
            if 'Close' in adjusted_df.columns:
                original_close = df[pre_split_mask].iloc[-1]['Close'] if not df[pre_split_mask].empty else 0
                adjusted_close = last_pre_split['Close']
                print(f"  Original Close: ${original_close:.2f}")
                print(f"  Adjusted Close: ${adjusted_close:.2f}")
    
    return adjusted_df


def check_if_split_adjusted(df, current_price=None):
    """
    Check if data needs split adjustment
    Returns True if adjustment is needed
    """
    if df.empty:
        return False
    
    # Get recent prices
    recent_closes = df['Close'].tail(10) if 'Close' in df.columns else df['Adj_Close'].tail(10)
    avg_recent = recent_closes.mean()
    
    print(f"Recent average price: ${avg_recent:.2f}")
    
    # If recent prices are above $500, likely not adjusted for split
    if avg_recent > 500:
        print("WARNING: Prices appear to be pre-split (>$500)")
        return True
    
    # If we have current price, check consistency
    if current_price:
        print(f"Current market price: ${current_price:.2f}")
        if avg_recent > current_price * 2:
            print("WARNING: Historical prices much higher than current price")
            return True
    
    print("Prices appear to be already adjusted for split")
    return False


# Test the adjustment
if __name__ == "__main__":
    print("NVIDIA Stock Split Adjustment Tool")
    print("=" * 50)
    
    # Example usage
    import yfinance as yf
    
    # Get current price
    nvda = yf.Ticker("NVDA")
    current = nvda.info.get('regularMarketPrice', 180)
    print(f"Current NVDA price: ${current:.2f}")
    
    # Create sample data showing the issue
    print("\nExample of the problem:")
    print("If seeing prices like $633.99, they're likely pre-split")
    print("Should be around $63.40 post-split (divided by 10)")
    
    # Adjustment example
    pre_split_price = 633.99
    post_split_price = pre_split_price / 10
    print(f"\nAdjustment: ${pre_split_price:.2f} -> ${post_split_price:.2f}")
    print("This would bring NVDA price to a realistic ~$63 range")