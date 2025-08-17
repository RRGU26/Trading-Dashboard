import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from pathlib import Path

# Database configuration
def find_database():
    current_dir = Path(__file__).parent
    possible_paths = [
        current_dir / "core-system" / "models_dashboard.db",  # PRIORITIZE this one
        current_dir / "databases" / "models_dashboard.db",
        current_dir / "models_dashboard.db",
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    return None

DB_PATH = find_database()

# Page config
st.set_page_config(
    page_title="Trading Models Accuracy Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Trading Models Accuracy Dashboard")
st.markdown("**Real-time prediction accuracy tracking by model**")
st.markdown("---")

def load_predictions_with_accuracy():
    """Load predictions and calculate accuracy against actual prices"""
    if not DB_PATH or not os.path.exists(DB_PATH):
        st.error(f"❌ Database not found at: {DB_PATH}")
        return None
    
    conn = sqlite3.connect(DB_PATH)
    
    # Get predictions with actual prices for comparison
    query = """
    SELECT 
        mp.model,
        mp.symbol,
        mp.prediction_date,
        mp.target_date,
        mp.predicted_price,
        mp.current_price as price_when_predicted,
        mp.actual_price,
        mp.id as created_timestamp,
        mp.horizon,
        mp.confidence,
        mp.direction_correct,
        mp.error_pct
    FROM model_predictions mp
    WHERE mp.prediction_date >= '2025-08-01'
    ORDER BY mp.id DESC
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        return None
    
    # Convert columns to proper numeric types
    numeric_cols = ['predicted_price', 'price_when_predicted', 'actual_price', 'confidence']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Filter out rows with invalid data
    df = df.dropna(subset=['predicted_price', 'price_when_predicted'])
    
    if df.empty:
        return None
    
    # Calculate accuracy metrics only where we have actual prices
    mask = df['actual_price'].notna()
    df.loc[mask, 'prediction_error'] = df.loc[mask, 'predicted_price'] - df.loc[mask, 'actual_price']
    df.loc[mask, 'prediction_error_pct'] = (df.loc[mask, 'prediction_error'] / df.loc[mask, 'actual_price'] * 100).round(2)
    df['abs_error_pct'] = abs(df['prediction_error_pct']).round(2)
    
    # Direction accuracy (can calculate even without actual prices)
    df['predicted_direction'] = (df['predicted_price'] > df['price_when_predicted']).map({True: 'UP', False: 'DOWN'})
    
    # Only calculate actual direction where we have actual prices
    actual_mask = df['actual_price'].notna()
    df.loc[actual_mask, 'actual_direction'] = (df.loc[actual_mask, 'actual_price'] > df.loc[actual_mask, 'price_when_predicted']).map({True: 'UP', False: 'DOWN'})
    df.loc[actual_mask, 'direction_correct'] = df.loc[actual_mask, 'predicted_direction'] == df.loc[actual_mask, 'actual_direction']
    
    # Format dates
    df['prediction_date'] = pd.to_datetime(df['prediction_date']).dt.strftime('%Y-%m-%d')
    df['target_date'] = pd.to_datetime(df['target_date']).dt.strftime('%Y-%m-%d')
    
    return df

def show_model_summary(df):
    """Show summary stats by model"""
    st.subheader("📊 Model Performance Summary")
    
    # Calculate summary by model
    summary = df.groupby('model').agg({
        'direction_correct': ['count', 'sum'],
        'abs_error_pct': 'mean',
        'prediction_error_pct': 'mean',
        'actual_price': 'count'
    }).round(2)
    
    summary.columns = ['Total_Predictions', 'Correct_Direction', 'Avg_Abs_Error_%', 'Avg_Error_%', 'Count']
    summary['Direction_Accuracy_%'] = (summary['Correct_Direction'] / summary['Total_Predictions'] * 100).round(1)
    
    # Display with color coding
    def color_accuracy(val):
        if val >= 70:
            return 'background-color: lightgreen'
        elif val >= 50:
            return 'background-color: lightyellow'
        else:
            return 'background-color: lightcoral'
    
    styled_summary = summary[['Total_Predictions', 'Direction_Accuracy_%', 'Avg_Abs_Error_%', 'Avg_Error_%']].style.applymap(
        color_accuracy, subset=['Direction_Accuracy_%']
    )
    
    st.dataframe(styled_summary, use_container_width=True)

def show_detailed_predictions(df):
    """Show detailed predictions by model"""
    st.subheader("🔍 Detailed Predictions by Model")
    
    # Model selector
    models = ['All Models'] + sorted(df['model'].unique().tolist())
    selected_model = st.selectbox("Select Model:", models)
    
    if selected_model != 'All Models':
        model_df = df[df['model'] == selected_model].copy()
    else:
        model_df = df.copy()
    
    if model_df.empty:
        st.warning("No predictions found for selected model")
        return
    
    # Sort by most recent
    model_df = model_df.sort_values('created_timestamp', ascending=False)
    
    # Display detailed table
    display_cols = [
        'model', 'symbol', 'prediction_date', 'target_date',
        'price_when_predicted', 'predicted_price', 'actual_price',
        'prediction_error_pct', 'predicted_direction', 'actual_direction', 'direction_correct'
    ]
    
    # Filter to available columns
    available_cols = [col for col in display_cols if col in model_df.columns]
    display_df = model_df[available_cols].copy()
    
    # Rename for better display
    column_names = {
        'model': 'Model',
        'symbol': 'Symbol',
        'prediction_date': 'Prediction Date',
        'target_date': 'Target Date',
        'price_when_predicted': 'Price When Predicted',
        'predicted_price': 'Predicted Price',
        'actual_price': 'Actual Price',
        'prediction_error_pct': 'Error %',
        'predicted_direction': 'Predicted Dir',
        'actual_direction': 'Actual Dir',
        'direction_correct': 'Direction Correct'
    }
    
    display_df = display_df.rename(columns=column_names)
    
    # Format numbers
    if 'Price When Predicted' in display_df.columns:
        display_df['Price When Predicted'] = display_df['Price When Predicted'].round(2)
    if 'Predicted Price' in display_df.columns:
        display_df['Predicted Price'] = display_df['Predicted Price'].round(2)
    if 'Actual Price' in display_df.columns:
        display_df['Actual Price'] = display_df['Actual Price'].round(2)
    
    # Color code direction accuracy
    def highlight_direction(row):
        if 'Direction Correct' in row.index:
            if row['Direction Correct']:
                return ['background-color: lightgreen'] * len(row)
            else:
                return ['background-color: lightcoral'] * len(row)
        return [''] * len(row)
    
    styled_df = display_df.style.apply(highlight_direction, axis=1)
    st.dataframe(styled_df, use_container_width=True)

def show_accuracy_charts(df):
    """Show accuracy visualization charts"""
    st.subheader("📈 Accuracy Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Direction accuracy by model
        direction_accuracy = df.groupby('model')['direction_correct'].agg(['count', 'sum']).reset_index()
        direction_accuracy['accuracy_pct'] = (direction_accuracy['sum'] / direction_accuracy['count'] * 100).round(1)
        
        fig1 = px.bar(
            direction_accuracy, 
            x='model', 
            y='accuracy_pct',
            title='Direction Accuracy by Model (%)',
            color='accuracy_pct',
            color_continuous_scale='RdYlGn'
        )
        fig1.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="Break Even (50%)")
        fig1.add_hline(y=70, line_dash="dash", line_color="green", annotation_text="Good (70%)")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Prediction error distribution
        fig2 = px.box(
            df, 
            x='model', 
            y='abs_error_pct',
            title='Prediction Error Distribution (%)',
            labels={'abs_error_pct': 'Absolute Error (%)'}
        )
        st.plotly_chart(fig2, use_container_width=True)

def show_recent_activity(df):
    """Show recent prediction activity"""
    st.subheader("🕐 Recent Activity")
    
    # Get last 10 predictions
    recent = df.head(10)[['model', 'symbol', 'prediction_date', 'predicted_price', 'actual_price', 'direction_correct']].copy()
    
    if recent.empty:
        st.warning("No recent predictions found")
        return
    
    # Format for display
    recent['Status'] = recent['direction_correct'].map({True: '✅ Correct', False: '❌ Wrong'})
    recent['Predicted Price'] = recent['predicted_price'].round(2)
    recent['Actual Price'] = recent['actual_price'].round(2)
    
    display_recent = recent[['model', 'symbol', 'prediction_date', 'Predicted Price', 'Actual Price', 'Status']]
    display_recent.columns = ['Model', 'Symbol', 'Date', 'Predicted', 'Actual', 'Result']
    
    st.dataframe(display_recent, use_container_width=True)

def show_pending_predictions(df):
    """Show pending predictions waiting for results"""
    # Model selector for pending
    models = ['All Models'] + sorted(df['model'].unique().tolist())
    selected_model = st.selectbox("Filter Pending by Model:", models, key="pending_model")
    
    if selected_model != 'All Models':
        pending_df = df[df['model'] == selected_model].copy()
    else:
        pending_df = df.copy()
    
    if pending_df.empty:
        st.info("No pending predictions for selected model")
        return
    
    # Sort by target date (soonest first)
    pending_df = pending_df.sort_values('target_date')
    
    # Calculate days until target
    from datetime import date
    today = date.today()
    pending_df['target_date_dt'] = pd.to_datetime(pending_df['target_date']).dt.date
    pending_df['days_until_target'] = pending_df['target_date_dt'].apply(lambda x: (x - today).days)
    pending_df['status'] = pending_df['days_until_target'].apply(
        lambda x: "🔴 Overdue" if x < 0 else f"⏳ {x} days" if x > 0 else "📅 Today"
    )
    
    # Display columns
    display_cols = ['model', 'symbol', 'prediction_date', 'target_date', 'price_when_predicted', 'predicted_price', 'predicted_direction', 'status']
    available_cols = [col for col in display_cols if col in pending_df.columns]
    
    display_pending = pending_df[available_cols].copy()
    
    # Rename columns
    column_names = {
        'model': 'Model',
        'symbol': 'Symbol', 
        'prediction_date': 'Prediction Date',
        'target_date': 'Target Date',
        'price_when_predicted': 'Price When Predicted',
        'predicted_price': 'Predicted Price',
        'predicted_direction': 'Direction',
        'status': 'Status'
    }
    
    display_pending = display_pending.rename(columns=column_names)
    
    # Format prices
    if 'Price When Predicted' in display_pending.columns:
        display_pending['Price When Predicted'] = display_pending['Price When Predicted'].round(2)
    if 'Predicted Price' in display_pending.columns:
        display_pending['Predicted Price'] = display_pending['Predicted Price'].round(2)
    
    st.dataframe(display_pending, use_container_width=True)
    
    # Show summary stats
    overdue = sum(pending_df['days_until_target'] < 0)
    today_count = sum(pending_df['days_until_target'] == 0) 
    future_count = sum(pending_df['days_until_target'] > 0)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔴 Overdue", overdue)
    with col2:
        st.metric("📅 Due Today", today_count)
    with col3:
        st.metric("⏳ Future", future_count)

def main():
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
    
    # Load and process data
    df = load_predictions_with_accuracy()
    
    if df is None or df.empty:
        st.warning("📊 No prediction data available with actual price comparisons")
        st.info("Data will appear here after:")
        st.markdown("- Models make predictions")
        st.markdown("- Target dates are reached")
        st.markdown("- Actual prices are recorded")
        return
    
    # Show all predictions, but highlight which have actual results
    df_with_actuals = df.dropna(subset=['actual_price'])
    df_pending = df[df['actual_price'].isna()]
    
    # Show prediction overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Predictions", len(df))
    with col2:
        st.metric("With Results", len(df_with_actuals))
    with col3:
        st.metric("Pending", len(df_pending))
    
    # Show results if we have any
    if not df_with_actuals.empty:
        st.subheader("📊 Results Available")
        show_model_summary(df_with_actuals)
        show_accuracy_charts(df_with_actuals)
        show_detailed_predictions(df_with_actuals)
        show_recent_activity(df_with_actuals)
    else:
        st.info("⏳ No actual prices available yet for accuracy comparison")
    
    # Always show pending predictions
    if not df_pending.empty:
        st.subheader("⏰ Pending Predictions")
        show_pending_predictions(df_pending)
    
    # Footer
    st.markdown("---")
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %I:%M %p ET')}")
    st.markdown(f"**Predictions Analyzed:** {len(df_with_actuals)} with actual price comparisons")

if __name__ == "__main__":
    main()