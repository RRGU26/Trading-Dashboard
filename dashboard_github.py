import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from pathlib import Path

# GitHub-ready configuration - dynamically find database
def find_database():
    """Find the models_dashboard.db file dynamically"""
    current_dir = Path(__file__).parent
    
    # Try multiple possible locations
    possible_paths = [
        current_dir / "databases" / "models_dashboard.db",
        current_dir / "core-system" / "models_dashboard.db", 
        current_dir / "models_dashboard.db",
        current_dir.parent / "databases" / "models_dashboard.db"
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    return None

DB_PATH = find_database()

# Dashboard configuration
st.set_page_config(
    page_title="Trading Models Accuracy Dashboard",
    page_icon="📈",
    layout="wide",
)

# Header
st.markdown("# 📈 Trading Models Accuracy Dashboard")
st.markdown("**Focus: Prediction Accuracy Analysis**")
st.markdown("---")

def load_prediction_data():
    """Load prediction accuracy data from database"""
    if not DB_PATH or not os.path.exists(DB_PATH):
        st.error(f"❌ Database not found. Checked locations: {DB_PATH}")
        st.stop()
        return None
    
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Get predictions with accuracy data
        query = """
        SELECT 
            model,
            symbol,
            prediction_date,
            target_date,
            current_price,
            predicted_price,
            actual_price,
            expected_return,
            actual_return,
            direction_correct,
            error_pct,
            horizon_days,
            created_timestamp,
            confidence,
            signal
        FROM model_predictions
        WHERE actual_price IS NOT NULL
        ORDER BY created_timestamp DESC
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            st.warning("⚠️ No prediction accuracy data found in database")
            return None
            
        return df
        
    except Exception as e:
        st.error(f"❌ Database error: {str(e)}")
        return None

def calculate_accuracy_metrics(df):
    """Calculate accuracy metrics by model"""
    if df is None or df.empty:
        return None
    
    # Group by model and calculate metrics
    model_metrics = []
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model].copy()
        
        if len(model_data) == 0:
            continue
            
        # Calculate metrics
        total_predictions = len(model_data)
        
        # Direction accuracy
        direction_correct = model_data['direction_correct'].sum() if 'direction_correct' in model_data.columns else 0
        direction_accuracy = (direction_correct / total_predictions * 100) if total_predictions > 0 else 0
        
        # Price accuracy (average absolute error)
        price_errors = model_data['error_pct'].abs().dropna()
        avg_price_error = price_errors.mean() if len(price_errors) > 0 else 0
        
        # Return accuracy
        return_errors = (model_data['actual_return'] - model_data['expected_return']).abs().dropna()
        avg_return_error = return_errors.mean() if len(return_errors) > 0 else 0
        
        # Recent performance (last 10 predictions)
        recent_data = model_data.head(10)
        recent_direction_accuracy = 0
        if len(recent_data) > 0 and 'direction_correct' in recent_data.columns:
            recent_direction_correct = recent_data['direction_correct'].sum()
            recent_direction_accuracy = (recent_direction_correct / len(recent_data) * 100)
        
        # Latest prediction date
        latest_prediction = model_data['prediction_date'].max()
        
        model_metrics.append({
            'Model': model,
            'Total Predictions': total_predictions,
            'Direction Accuracy (%)': round(direction_accuracy, 1),
            'Avg Price Error (%)': round(avg_price_error, 2),
            'Avg Return Error (%)': round(avg_return_error, 2),
            'Recent Direction Accuracy (%)': round(recent_direction_accuracy, 1),
            'Latest Prediction': latest_prediction,
            'Status': '🟢' if direction_accuracy >= 60 else '🟡' if direction_accuracy >= 50 else '🔴'
        })
    
    return pd.DataFrame(model_metrics)

def show_model_performance_chart(df):
    """Show model performance over time"""
    if df is None or df.empty:
        return
    
    st.subheader("📊 Model Performance Over Time")
    
    # Direction accuracy over time
    df['prediction_date'] = pd.to_datetime(df['prediction_date'])
    
    # Calculate rolling accuracy for each model
    chart_data = []
    for model in df['model'].unique():
        model_data = df[df['model'] == model].sort_values('prediction_date')
        
        if len(model_data) >= 5:  # Need at least 5 predictions for rolling average
            model_data['rolling_accuracy'] = model_data['direction_correct'].rolling(window=5, min_periods=1).mean() * 100
            
            for _, row in model_data.iterrows():
                chart_data.append({
                    'Model': model,
                    'Date': row['prediction_date'],
                    'Rolling Accuracy (%)': row['rolling_accuracy']
                })
    
    if chart_data:
        chart_df = pd.DataFrame(chart_data)
        fig = px.line(chart_df, x='Date', y='Rolling Accuracy (%)', 
                     color='Model', title='5-Day Rolling Direction Accuracy')
        fig.add_hline(y=60, line_dash="dash", line_color="green", 
                     annotation_text="Good (60%)")
        fig.add_hline(y=50, line_dash="dash", line_color="orange", 
                     annotation_text="Break Even (50%)")
        st.plotly_chart(fig, use_container_width=True)

def show_prediction_details(df):
    """Show detailed prediction table"""
    st.subheader("🔍 Recent Prediction Details")
    
    if df is None or df.empty:
        st.info("No prediction data available")
        return
    
    # Select relevant columns for display
    display_cols = ['model', 'symbol', 'prediction_date', 'predicted_price', 
                   'actual_price', 'error_pct', 'direction_correct', 'confidence']
    
    available_cols = [col for col in display_cols if col in df.columns]
    display_df = df[available_cols].head(20)
    
    # Format for better display
    if 'error_pct' in display_df.columns:
        display_df['error_pct'] = display_df['error_pct'].round(2)
    if 'predicted_price' in display_df.columns:
        display_df['predicted_price'] = display_df['predicted_price'].round(2)
    if 'actual_price' in display_df.columns:
        display_df['actual_price'] = display_df['actual_price'].round(2)
    
    st.dataframe(display_df, use_container_width=True)

def main():
    """Main dashboard function"""
    
    # Database status
    col1, col2 = st.columns([3, 1])
    with col1:
        if DB_PATH and os.path.exists(DB_PATH):
            st.success(f"✅ Database connected: {os.path.basename(DB_PATH)}")
        else:
            st.error("❌ Database not found")
            st.stop()
    
    with col2:
        if st.button("🔄 Refresh Data"):
            st.rerun()
    
    # Load data
    df = load_prediction_data()
    
    if df is None or df.empty:
        st.warning("No prediction accuracy data available yet.")
        st.info("💡 Predictions will appear here after models run and actual prices are recorded.")
        return
    
    # Calculate metrics
    metrics_df = calculate_accuracy_metrics(df)
    
    if metrics_df is not None and not metrics_df.empty:
        # Display metrics table
        st.subheader("🎯 Model Accuracy Summary")
        
        # Color code the metrics
        def highlight_accuracy(val):
            if isinstance(val, (int, float)):
                if val >= 60:
                    return 'background-color: lightgreen'
                elif val >= 50:
                    return 'background-color: lightyellow'
                else:
                    return 'background-color: lightcoral'
            return ''
        
        # Apply styling to direction accuracy columns
        styled_df = metrics_df.style.applymap(
            highlight_accuracy, 
            subset=['Direction Accuracy (%)', 'Recent Direction Accuracy (%)']
        )
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Performance chart
        show_model_performance_chart(df)
        
        # Detailed predictions
        show_prediction_details(df)
    
    # Footer
    st.markdown("---")
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %I:%M %p ET')}")
    st.markdown("**Repository:** GitHub-ready dashboard for trading model accuracy tracking")

if __name__ == "__main__":
    main()