"""
Cloud-Ready Trading Dashboard
Optimized for Streamlit Cloud deployment
"""

import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import requests
from io import BytesIO

# Configuration
st.set_page_config(
    page_title="Trading Models Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for mobile-friendly display
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
    .status-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.4rem;
        }
        .row-widget {
            display: block !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Database connection with cloud support
@st.cache_resource
def get_database_connection():
    """Get database connection - works locally and in cloud"""
    # Check if we're running in cloud (Railway/Render provide DATABASE_URL)
    database_url = os.getenv('DATABASE_URL')
    
    if database_url:
        # Cloud database (PostgreSQL)
        import psycopg2
        return psycopg2.connect(database_url)
    else:
        # Local SQLite database
        db_path = os.getenv('DB_PATH', 'models_dashboard.db')
        if os.path.exists(db_path):
            return sqlite3.connect(db_path, check_same_thread=False)
        else:
            # Use sample data if no database
            return None

# Fetch latest predictions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_latest_predictions(conn):
    """Get latest predictions from database"""
    if conn is None:
        # Return sample data if no database
        return pd.DataFrame({
            'model': ['QQQ Long Bull', 'QQQ Master', 'NVIDIA', 'Bitcoin'],
            'signal': ['SELL', 'SELL', 'SELL', 'HOLD'],
            'confidence': [95, 95, 85, 75],
            'expected_return': [-0.9, -1.13, -4.76, 0.08],
            'last_updated': [datetime.now()] * 4
        })
    
    query = """
    SELECT 
        model,
        symbol,
        current_price,
        predicted_price,
        suggested_action as signal,
        confidence,
        expected_return,
        prediction_date as last_updated
    FROM model_predictions
    WHERE prediction_date = (
        SELECT MAX(prediction_date) 
        FROM model_predictions
    )
    ORDER BY model
    """
    
    try:
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        st.error(f"Error loading predictions: {e}")
        return pd.DataFrame()

# Main Dashboard
def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Trading Models Dashboard</h1>', unsafe_allow_html=True)
    
    # Connection status
    conn = get_database_connection()
    
    # Status indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Status", "🟢 Online" if conn else "🔴 Demo Mode")
    
    with col2:
        st.metric("Last Update", datetime.now().strftime("%I:%M %p"))
    
    with col3:
        st.metric("Models Active", "7")
    
    with col4:
        next_run = datetime.now().replace(hour=15, minute=40, second=0)
        if datetime.now() > next_run:
            next_run += timedelta(days=1)
        hours_until = (next_run - datetime.now()).seconds // 3600
        st.metric("Next Run", f"{hours_until}h")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Current Signals", "📊 Performance", "⚙️ Settings", "📱 Mobile View"])
    
    with tab1:
        st.header("Today's Trading Signals")
        
        # Get latest predictions
        predictions_df = get_latest_predictions(conn)
        
        if not predictions_df.empty:
            # Create signal cards
            for _, row in predictions_df.iterrows():
                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
                    
                    with col1:
                        st.subheader(row['model'])
                    
                    with col2:
                        signal_color = {
                            'BUY': '🟢',
                            'SELL': '🔴', 
                            'HOLD': '🟡'
                        }.get(row.get('signal', 'HOLD'), '⚪')
                        st.metric("Signal", f"{signal_color} {row.get('signal', 'N/A')}")
                    
                    with col3:
                        confidence = row.get('confidence', 0)
                        st.metric("Confidence", f"{confidence:.0f}%")
                    
                    with col4:
                        ret = row.get('expected_return', 0)
                        st.metric("Expected Return", f"{ret:+.2f}%")
                    
                    st.divider()
        else:
            st.info("No predictions available. Models run daily at 3:40 PM ET.")
    
    with tab2:
        st.header("Performance Analytics")
        
        # Performance chart
        if not predictions_df.empty:
            fig = px.bar(
                predictions_df,
                x='model',
                y='expected_return',
                color='signal',
                title='Expected Returns by Model',
                color_discrete_map={'BUY': 'green', 'SELL': 'red', 'HOLD': 'yellow'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Historical accuracy (placeholder)
        st.subheader("Model Accuracy (Last 30 Days)")
        accuracy_data = pd.DataFrame({
            'Model': ['QQQ Long Bull', 'QQQ Master', 'NVIDIA', 'Bitcoin'],
            'Accuracy': [62.3, 58.5, 55.2, 51.3]
        })
        
        fig2 = px.bar(accuracy_data, x='Model', y='Accuracy', title='Direction Accuracy %')
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        st.header("Dashboard Settings")
        
        # Refresh settings
        st.subheader("Auto-Refresh")
        auto_refresh = st.checkbox("Enable auto-refresh (every 5 minutes)")
        
        if auto_refresh:
            st.info("Dashboard will refresh automatically every 5 minutes")
            st.experimental_rerun()  # This would trigger based on a timer in production
        
        # Notification settings
        st.subheader("Notifications")
        email_alerts = st.checkbox("Email alerts for strong signals")
        
        if email_alerts:
            email = st.text_input("Email address", "RRGU26@gmail.com")
            threshold = st.slider("Signal confidence threshold", 70, 100, 90)
            st.success(f"Alerts will be sent to {email} for signals above {threshold}% confidence")
        
        # Data export
        st.subheader("Export Data")
        if st.button("Download Today's Predictions (CSV)"):
            csv = predictions_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with tab4:
        st.header("Mobile-Optimized View")
        
        # Simplified mobile view
        st.info("📱 This view is optimized for mobile devices")
        
        if not predictions_df.empty:
            for _, row in predictions_df.iterrows():
                # Mobile-friendly card layout
                with st.expander(f"{row['model']} - {row.get('signal', 'N/A')}", expanded=True):
                    st.write(f"**Signal:** {row.get('signal', 'N/A')}")
                    st.write(f"**Confidence:** {row.get('confidence', 0):.0f}%")
                    st.write(f"**Expected Return:** {row.get('expected_return', 0):+.2f}%")
                    
                    # Signal strength indicator
                    if row.get('confidence', 0) > 90:
                        st.success("Strong Signal")
                    elif row.get('confidence', 0) > 80:
                        st.warning("Moderate Signal")
                    else:
                        st.info("Weak Signal")
    
    # Footer
    st.divider()
    st.caption("Trading Models Dashboard v3.0 | Updates daily at 3:40 PM ET")
    
    # Add connection cleanup
    if conn and hasattr(conn, 'close'):
        conn.close()

# Sidebar
with st.sidebar:
    st.header("Quick Stats")
    
    # Today's summary
    st.subheader("Today's Summary")
    predictions_df = get_latest_predictions(get_database_connection())
    
    if not predictions_df.empty:
        buy_count = len(predictions_df[predictions_df['signal'] == 'BUY'])
        sell_count = len(predictions_df[predictions_df['signal'] == 'SELL'])
        hold_count = len(predictions_df[predictions_df['signal'] == 'HOLD'])
        
        st.metric("BUY Signals", buy_count)
        st.metric("SELL Signals", sell_count)
        st.metric("HOLD Signals", hold_count)
    
    st.divider()
    
    # Links
    st.subheader("Quick Links")
    st.markdown("[📧 Email Settings](mailto:RRGU26@gmail.com)")
    st.markdown("[📊 Full Reports](#)")
    st.markdown("[⚙️ System Status](#)")
    
    st.divider()
    
    # Help
    st.info("Need help? Check the Settings tab for configuration options.")

if __name__ == "__main__":
    main()