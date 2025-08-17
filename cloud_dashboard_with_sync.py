"""
Cloud Dashboard with GitHub Data Sync
Reads data from JSON files updated by local system
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import requests
import os

# Configuration
st.set_page_config(
    page_title="Trading Models Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# GitHub raw file URLs
GITHUB_REPO = "RRGU26/Trading-Dashboard"
BASE_URL = f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/data/"

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 100%;
    }
    .main-header {
        font-size: 1.8rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .signal-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .buy-signal { border-left-color: #28a745; }
    .sell-signal { border-left-color: #dc3545; }
    .hold-signal { border-left-color: #ffc107; }
    
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.4rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Data loading functions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_predictions_from_github():
    """Load latest predictions from GitHub"""
    try:
        url = BASE_URL + "latest_predictions.json"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            st.error(f"Failed to load data from GitHub: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data(ttl=300)
def load_summary_from_github():
    """Load summary data from GitHub"""
    try:
        url = BASE_URL + "summary.json"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
            
    except Exception as e:
        return None

# Fallback sample data
def get_sample_data():
    """Sample data for demo when GitHub data unavailable"""
    return {
        'last_updated': datetime.now().isoformat(),
        'export_date': datetime.now().strftime('%Y-%m-%d'),
        'total_predictions': 5,
        'system_status': 'demo_mode',
        'predictions': [
            {
                'model': 'QQQ Long Bull Model',
                'symbol': 'QQQ',
                'current_price': 577.34,
                'predicted_price': 572.13,
                'suggested_action': 'SELL',
                'confidence': 95,
                'expected_return': -0.9,
                'prediction_date': datetime.now().strftime('%Y-%m-%d')
            },
            {
                'model': 'QQQ Master Model',
                'symbol': 'QQQ',
                'suggested_action': 'SELL',
                'confidence': 95,
                'expected_return': -1.13,
                'prediction_date': datetime.now().strftime('%Y-%m-%d')
            },
            {
                'model': 'NVIDIA Bull Momentum',
                'symbol': 'NVDA',
                'current_price': 180.45,
                'predicted_price': 171.86,
                'suggested_action': 'SELL',
                'confidence': 85,
                'expected_return': -4.76,
                'prediction_date': datetime.now().strftime('%Y-%m-%d')
            },
            {
                'model': 'Wishing Well QQQ',
                'symbol': 'QQQ',
                'suggested_action': 'BUY',
                'confidence': 85,
                'expected_return': 2.5,
                'prediction_date': datetime.now().strftime('%Y-%m-%d')
            },
            {
                'model': 'Bitcoin Model',
                'symbol': 'BTC-USD',
                'current_price': 117776,
                'suggested_action': 'HOLD',
                'confidence': 75,
                'expected_return': 0.08,
                'prediction_date': datetime.now().strftime('%Y-%m-%d')
            }
        ]
    }

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Trading Models Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    data = load_predictions_from_github()
    summary = load_summary_from_github()
    
    if data is None:
        st.warning("Using demo data - live data unavailable")
        data = get_sample_data()
    
    # Status bar
    status_color = "🟢" if data.get('system_status') == 'operational' else "🟡"
    last_update = data.get('last_updated', 'Unknown')
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_text = "Live Data" if data.get('system_status') == 'operational' else "Demo Mode"
        st.metric("Status", f"{status_color} {status_text}")
    
    with col2:
        if last_update != 'Unknown':
            update_time = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
            st.metric("Last Update", update_time.strftime("%I:%M %p"))
        else:
            st.metric("Last Update", "Unknown")
    
    with col3:
        st.metric("Active Models", str(data.get('total_predictions', 0)))
    
    with col4:
        # Calculate next run time
        now = datetime.now()
        next_run = now.replace(hour=15, minute=40, second=0, microsecond=0)
        if now > next_run:
            next_run += timedelta(days=1)
        hours_until = (next_run - now).total_seconds() / 3600
        st.metric("Next Run", f"{hours_until:.1f}h")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📈 Current Signals", "📊 Analytics", "⚙️ Settings"])
    
    with tab1:
        st.header("Today's Trading Signals")
        
        predictions = data.get('predictions', [])
        
        if predictions:
            # Convert to DataFrame for easier handling
            df = pd.DataFrame(predictions)
            
            # Signal summary
            if 'suggested_action' in df.columns:
                signal_counts = df['suggested_action'].value_counts()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🟢 BUY Signals", signal_counts.get('BUY', 0))
                with col2:
                    st.metric("🔴 SELL Signals", signal_counts.get('SELL', 0))
                with col3:
                    st.metric("🟡 HOLD Signals", signal_counts.get('HOLD', 0))
            
            st.divider()
            
            # Individual signal cards
            for prediction in predictions:
                signal = prediction.get('suggested_action', 'UNKNOWN')
                signal_class = f"{signal.lower()}-signal" if signal in ['BUY', 'SELL', 'HOLD'] else ""
                
                with st.container():
                    st.markdown(f'<div class="signal-card {signal_class}">', unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
                    
                    with col1:
                        st.subheader(prediction.get('model', 'Unknown Model'))
                        if 'symbol' in prediction:
                            st.caption(f"Symbol: {prediction['symbol']}")
                    
                    with col2:
                        signal_emoji = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '🟡'}.get(signal, '⚪')
                        st.metric("Signal", f"{signal_emoji} {signal}")
                    
                    with col3:
                        confidence = prediction.get('confidence', 0)
                        st.metric("Confidence", f"{confidence:.0f}%")
                    
                    with col4:
                        expected_return = prediction.get('expected_return', 0)
                        st.metric("Expected Return", f"{expected_return:+.2f}%")
                    
                    # Additional details
                    if 'current_price' in prediction:
                        st.caption(f"Current Price: ${prediction['current_price']:.2f}")
                    if 'predicted_price' in prediction:
                        st.caption(f"Target Price: ${prediction['predicted_price']:.2f}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.divider()
        else:
            st.info("No predictions available. Models run daily at 3:40 PM ET.")
    
    with tab2:
        st.header("Performance Analytics")
        
        if predictions:
            df = pd.DataFrame(predictions)
            
            # Expected returns chart
            if 'expected_return' in df.columns and 'model' in df.columns:
                fig = px.bar(
                    df,
                    x='model',
                    y='expected_return',
                    color='suggested_action',
                    title='Expected Returns by Model',
                    color_discrete_map={
                        'BUY': '#28a745',
                        'SELL': '#dc3545',
                        'HOLD': '#ffc107'
                    }
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            # Confidence levels
            if 'confidence' in df.columns:
                fig2 = px.bar(
                    df,
                    x='model',
                    y='confidence',
                    title='Model Confidence Levels',
                    color='confidence',
                    color_continuous_scale='RdYlGn'
                )
                fig2.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig2, use_container_width=True)
        
        # Data freshness indicator
        if last_update != 'Unknown':
            try:
                update_time = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                minutes_ago = (datetime.now() - update_time.replace(tzinfo=None)).total_seconds() / 60
                
                if minutes_ago < 60:
                    st.success(f"Data is fresh ({minutes_ago:.0f} minutes old)")
                elif minutes_ago < 1440:  # 24 hours
                    st.warning(f"Data is {minutes_ago/60:.1f} hours old")
                else:
                    st.error(f"Data is {minutes_ago/1440:.1f} days old")
            except:
                st.info("Unable to determine data freshness")
    
    with tab3:
        st.header("Dashboard Configuration")
        
        # Data source info
        st.subheader("Data Source")
        st.info(f"📊 Reading from: {BASE_URL}")
        st.info("🔄 Auto-refreshes every 5 minutes")
        
        # Sync status
        st.subheader("Sync Status")
        if data.get('system_status') == 'operational':
            st.success("✅ Connected to live trading system")
        else:
            st.warning("⚠️ Running in demo mode")
        
        # Manual refresh
        if st.button("🔄 Refresh Data Now"):
            st.cache_data.clear()
            st.experimental_rerun()
        
        # Download data
        st.subheader("Export Data")
        if st.button("📥 Download Current Predictions"):
            if predictions:
                df = pd.DataFrame(predictions)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
    
    # Sidebar
    with st.sidebar:
        st.header("Quick Stats")
        
        if summary:
            st.subheader("Today's Summary")
            signal_counts = summary.get('signal_counts', {})
            
            for signal, count in signal_counts.items():
                emoji = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '🟡'}.get(signal, '⚪')
                st.metric(f"{emoji} {signal}", count)
            
            avg_conf = summary.get('average_confidence', 0)
            st.metric("Avg Confidence", f"{avg_conf:.1f}%")
        
        st.divider()
        
        # System info
        st.subheader("System Info")
        st.caption(f"Dashboard v3.0")
        st.caption(f"Models: 7 active")
        st.caption(f"Update: Daily at 3:40 PM ET")
        
        # Quick links
        st.divider()
        st.subheader("Links")
        st.markdown("[📧 Get Email Reports](mailto:RRGU26@gmail.com)")
        st.markdown("[📊 Raw Data](https://github.com/yourusername/trading-models)")
    
    # Footer
    st.divider()
    st.caption("🔄 Auto-syncs with local trading system via GitHub • Updates every 5 minutes")

if __name__ == "__main__":
    main()