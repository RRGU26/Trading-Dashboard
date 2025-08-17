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

# Enhanced Custom CSS with improved visual hierarchy
st.markdown("""
<style>
    /* Global app styling */
    .stApp {
        max-width: 100%;
        font-size: 14px;
    }
    
    /* Reduce default streamlit text sizes */
    .stMarkdown p, .stText {
        font-size: 13px;
        line-height: 1.4;
    }
    
    .stMetric > div > div {
        font-size: 12px;
    }
    
    .stCaption {
        font-size: 11px;
        opacity: 0.7;
    }
    
    /* Header styling */
    .main-header {
        font-size: 1.6rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1.5rem;
    }
    
    /* Enhanced signal cards with modern design */
    .signal-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.25rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .signal-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: #64748b;
    }
    
    .signal-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    /* Signal-specific card styling */
    .buy-signal::before { background: linear-gradient(90deg, #10b981, #34d399); }
    .sell-signal::before { background: linear-gradient(90deg, #ef4444, #f87171); }
    .hold-signal::before { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
    
    /* Prominent price displays */
    .price-display {
        font-size: 1.8rem;
        font-weight: 800;
        color: #1e293b;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
    }
    
    .predicted-price {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0.25rem 0;
    }
    
    .price-current { color: #3b82f6; }
    .price-target-up { color: #10b981; }
    .price-target-down { color: #ef4444; }
    
    /* Enhanced signal display */
    .signal-badge {
        display: inline-flex;
        align-items: center;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin: 0.5rem 0;
    }
    
    .signal-buy {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    
    .signal-sell {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
    }
    
    .signal-hold {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.3);
    }
    
    /* Model name styling */
    .model-name {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    
    /* Confidence and return styling */
    .confidence-high { color: #10b981; font-weight: 600; }
    .confidence-medium { color: #f59e0b; font-weight: 600; }
    .confidence-low { color: #ef4444; font-weight: 600; }
    
    .return-positive { color: #10b981; font-weight: 700; }
    .return-negative { color: #ef4444; font-weight: 700; }
    
    /* Status indicators */
    .status-live {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    
    .status-demo {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.3rem;
            margin-bottom: 1rem;
        }
        
        .signal-card {
            padding: 1rem;
            margin-bottom: 0.75rem;
        }
        
        .price-display {
            font-size: 1.4rem;
        }
        
        .predicted-price {
            font-size: 1.2rem;
        }
        
        .signal-badge {
            font-size: 12px;
            padding: 6px 12px;
        }
        
        .model-name {
            font-size: 1rem;
        }
        
        .stMetric > div > div {
            font-size: 11px;
        }
    }
    
    @media (max-width: 480px) {
        .main-header {
            font-size: 1.1rem;
        }
        
        .price-display {
            font-size: 1.2rem;
        }
        
        .predicted-price {
            font-size: 1rem;
        }
        
        .signal-card {
            padding: 0.75rem;
        }
    }
    
    /* Summary cards styling */
    .summary-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    
    /* Tab styling improvements */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
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
        status_class = "status-live" if data.get('system_status') == 'operational' else "status-demo"
        st.markdown(f'<div class="{status_class}">{status_color} {status_text}</div>', unsafe_allow_html=True)
    
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
                    buy_count = signal_counts.get('BUY', 0)
                    st.markdown(f'''
                    <div class="summary-card" style="border-left: 4px solid #10b981;">
                        <div style="font-size: 1.5rem; font-weight: 700; color: #10b981;">{buy_count}</div>
                        <div style="font-size: 12px; color: #64748b;">🟢 BUY SIGNALS</div>
                    </div>
                    ''', unsafe_allow_html=True)
                with col2:
                    sell_count = signal_counts.get('SELL', 0)
                    st.markdown(f'''
                    <div class="summary-card" style="border-left: 4px solid #ef4444;">
                        <div style="font-size: 1.5rem; font-weight: 700; color: #ef4444;">{sell_count}</div>
                        <div style="font-size: 12px; color: #64748b;">🔴 SELL SIGNALS</div>
                    </div>
                    ''', unsafe_allow_html=True)
                with col3:
                    hold_count = signal_counts.get('HOLD', 0)
                    st.markdown(f'''
                    <div class="summary-card" style="border-left: 4px solid #f59e0b;">
                        <div style="font-size: 1.5rem; font-weight: 700; color: #f59e0b;">{hold_count}</div>
                        <div style="font-size: 12px; color: #64748b;">🟡 HOLD SIGNALS</div>
                    </div>
                    ''', unsafe_allow_html=True)
            
            st.divider()
            
            # Individual signal cards
            for prediction in predictions:
                signal = prediction.get('suggested_action', 'UNKNOWN')
                signal_class = f"{signal.lower()}-signal" if signal in ['BUY', 'SELL', 'HOLD'] else ""
                
                with st.container():
                    st.markdown(f'<div class="signal-card {signal_class}">', unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
                    
                    with col1:
                        model_name = prediction.get("model", "Unknown Model")
                        
                        # Add time horizon for Bitcoin and NVIDIA models
                        if 'target_date' in prediction and 'prediction_date' in prediction:
                            try:
                                pred_date = datetime.strptime(prediction['prediction_date'], '%Y-%m-%d')
                                target_date = datetime.strptime(prediction['target_date'], '%Y-%m-%d')
                                days_ahead = (target_date - pred_date).days
                                
                                if days_ahead > 0:
                                    model_name += f" ({days_ahead}d)"
                            except:
                                pass
                        
                        st.markdown(f'<div class="model-name">{model_name}</div>', unsafe_allow_html=True)
                        if 'symbol' in prediction:
                            st.markdown(f'<span style="font-size: 12px; color: #64748b; font-weight: 500;">{prediction["symbol"]}</span>', unsafe_allow_html=True)
                        
                        # Prominent price displays
                        if 'current_price' in prediction and prediction['current_price'] is not None:
                            try:
                                price = float(prediction['current_price'])
                                st.markdown(f'<div class="price-display price-current">${price:,.2f}</div>', unsafe_allow_html=True)
                                st.markdown('<span style="font-size: 11px; color: #64748b;">Current Price</span>', unsafe_allow_html=True)
                            except (ValueError, TypeError):
                                pass
                    
                    with col2:
                        # Enhanced signal badge
                        signal_class = f"signal-{signal.lower()}" if signal in ['BUY', 'SELL', 'HOLD'] else ""
                        signal_emoji = {'BUY': '⬆', 'SELL': '⬇', 'HOLD': '⏸'}.get(signal, '⚪')
                        st.markdown(f'<div class="signal-badge {signal_class}">{signal_emoji} {signal}</div>', unsafe_allow_html=True)
                        
                        # Target price if available
                        if 'predicted_price' in prediction and prediction['predicted_price'] is not None:
                            try:
                                target_price = float(prediction['predicted_price'])
                                current_price = float(prediction.get('current_price', 0))
                                price_class = "price-target-up" if target_price > current_price else "price-target-down"
                                st.markdown(f'<div class="predicted-price {price_class}">${target_price:,.2f}</div>', unsafe_allow_html=True)
                                st.markdown('<span style="font-size: 11px; color: #64748b;">Target Price</span>', unsafe_allow_html=True)
                            except (ValueError, TypeError):
                                pass
                    
                    with col3:
                        confidence = prediction.get('confidence', 0)
                        if confidence is not None and isinstance(confidence, (int, float)):
                            conf_class = "confidence-high" if confidence >= 80 else "confidence-medium" if confidence >= 60 else "confidence-low"
                            st.markdown(f'<div style="font-size: 1.2rem; font-weight: 700;" class="{conf_class}">{confidence:.0f}%</div>', unsafe_allow_html=True)
                            st.markdown('<span style="font-size: 11px; color: #64748b;">Confidence</span>', unsafe_allow_html=True)
                        else:
                            st.markdown('<span style="color: #64748b;">N/A</span>', unsafe_allow_html=True)
                    
                    with col4:
                        expected_return = prediction.get('expected_return', 0)
                        if expected_return is not None and isinstance(expected_return, (int, float)):
                            return_class = "return-positive" if expected_return > 0 else "return-negative"
                            st.markdown(f'<div style="font-size: 1.2rem; font-weight: 700;" class="{return_class}">{expected_return:+.2f}%</div>', unsafe_allow_html=True)
                            st.markdown('<span style="font-size: 11px; color: #64748b;">Expected Return</span>', unsafe_allow_html=True)
                        else:
                            st.markdown('<span style="color: #64748b;">N/A</span>', unsafe_allow_html=True)
                    
                    # Additional context information
                    if 'prediction_date' in prediction:
                        st.markdown(f'<span style="font-size: 10px; color: #94a3b8;">📅 {prediction["prediction_date"]}</span>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.divider()
        else:
            st.info("No predictions available. Models run daily at 3:40 PM ET.")
    
    with tab2:
        st.header("Performance Analytics")
        
        if predictions:
            df = pd.DataFrame(predictions)
            
            # Clean and validate data for charts
            # Convert expected_return to numeric, handling None/invalid values
            if 'expected_return' in df.columns:
                df['expected_return'] = pd.to_numeric(df['expected_return'], errors='coerce')
                df = df.dropna(subset=['expected_return'])  # Remove rows with invalid returns
            
            # Convert confidence to numeric
            if 'confidence' in df.columns:
                df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
            
            # Expected returns chart (only if we have valid data)
            if 'expected_return' in df.columns and 'model' in df.columns and len(df) > 0:
                try:
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
                except Exception as e:
                    st.warning("Unable to display expected returns chart - data formatting issue")
            
            # Confidence levels chart
            if 'confidence' in df.columns and len(df.dropna(subset=['confidence'])) > 0:
                try:
                    # Filter to only rows with valid confidence
                    df_conf = df.dropna(subset=['confidence'])
                    fig2 = px.bar(
                        df_conf,
                        x='model',
                        y='confidence',
                        title='Model Confidence Levels',
                        color='confidence',
                        color_continuous_scale='RdYlGn'
                    )
                    fig2.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig2, use_container_width=True)
                except Exception as e:
                    st.warning("Unable to display confidence chart - data formatting issue")
        
        # Model Performance vs Actual Results
        st.subheader("📊 Predictions vs Actual Results")
        
        # This would typically come from your database with actual prices
        # For now, showing placeholder structure for when you add this data
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("**Coming Soon**: Track how well predictions performed vs actual market movements")
            st.write("Features will include:")
            st.write("• Prediction accuracy by model")
            st.write("• Actual vs predicted price movements") 
            st.write("• Win/loss ratios for signals")
            st.write("• Historical performance trends")
        
        with col2:
            # Real accuracy data from your database
            st.subheader("Model Accuracy (Last 30 Days)")
            
            if summary and 'performance_data' in summary and summary['performance_data']:
                performance_df = pd.DataFrame(summary['performance_data'])
                
                try:
                    fig3 = px.bar(
                        performance_df, 
                        x='model', 
                        y='accuracy',
                        title='Direction Prediction Accuracy %',
                        color='accuracy',
                        color_continuous_scale='RdYlGn',
                        text='accuracy'
                    )
                    fig3.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    fig3.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig3, use_container_width=True)
                    
                    # Show detailed performance table
                    st.subheader("📊 Detailed Performance Metrics")
                    st.dataframe(
                        performance_df.round(2),
                        column_config={
                            "model": "Model",
                            "total_predictions": "Total Predictions",
                            "accuracy": st.column_config.NumberColumn("Accuracy %", format="%.1f%%"),
                            "avg_error": st.column_config.NumberColumn("Avg Error %", format="%.2f%%")
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.warning("Unable to display performance data")
            else:
                # Fallback to placeholder when no real data available
                st.info("Performance tracking will show here once models have historical data with actual price comparisons.")
                
                # Show sample structure
                sample_df = pd.DataFrame({
                    'Model': ['QQQ Long Bull', 'QQQ Master', 'NVIDIA'],
                    'Accuracy': [62.3, 58.5, 55.2],
                    'Predictions': [25, 25, 20]
                })
                
                try:
                    fig3 = px.bar(
                        sample_df, 
                        x='Model', 
                        y='Accuracy',
                        title='Sample: Direction Prediction Accuracy %',
                        color='Accuracy',
                        color_continuous_scale='RdYlGn'
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                except Exception as e:
                    st.warning("Sample chart unavailable")
        
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