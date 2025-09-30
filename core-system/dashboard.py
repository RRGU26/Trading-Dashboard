import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "reports_tracking.db")

# Dashboard title and configuration
st.set_page_config(
    page_title="Trading Models Performance Dashboard",
    page_icon="📈",
    layout="wide",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-align: center;
    }
    .metrics-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
        padding-top: 1rem;
        border-top: 1px solid #e6e6e6;
    }
    .debug-info {
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
        margin-bottom: 1rem;
    }
    .high-accuracy {
        color: #28a745;
        font-weight: bold;
    }
    .medium-accuracy {
        color: #007bff;
        font-weight: bold;
    }
    .low-accuracy {
        color: #fd7e14;
        font-weight: bold;
    }
    .poor-accuracy {
        color: #dc3545;
        font-weight: bold;
    }
    .table-container {
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 1rem;
        background-color: #f8f9fa;
    }
    .today-highlight {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load data function with better error handling and no caching during development
def load_data():
    """Load prediction data from SQLite database with improved date handling"""
    try:
        if not os.path.exists(DB_PATH):
            st.error(f"Database not found at: {DB_PATH}")
            return pd.DataFrame()
        
        conn = sqlite3.connect(DB_PATH)
        
        # Enhanced query with better date handling
        query = """
        SELECT 
            model,
            symbol,
            prediction_date,
            target_date,
            horizon,
            current_price,
            predicted_price,
            actual_price,
            confidence,
            suggested_action,
            error_pct,
            direction_correct
        FROM 
            model_predictions
        ORDER BY 
            prediction_date DESC, model, horizon
        """
        df = pd.read_sql(query, conn)
        
        if df.empty:
            st.warning("No predictions found in database")
            conn.close()
            return df
        
        # Convert dates to datetime with explicit format
        df['prediction_date'] = pd.to_datetime(df['prediction_date'])
        df['target_date'] = pd.to_datetime(df['target_date'])
        
        # Try to get model metrics (R² values and hit rates)
        try:
            metrics_query = """
            SELECT model, date, metric_type, metric_value 
            FROM model_metrics 
            ORDER BY date DESC
            """
            metrics_df = pd.read_sql(metrics_query, conn)
            
            if not metrics_df.empty:
                # Pivot metrics for easier joining
                r2_df = metrics_df[metrics_df['metric_type'] == 'r2'].copy()
                r2_df['date'] = pd.to_datetime(r2_df['date'])
                r2_df = r2_df.rename(columns={'metric_value': 'r2_value'})
                
                hit_rate_df = metrics_df[metrics_df['metric_type'] == 'hit_rate'].copy()
                hit_rate_df['date'] = pd.to_datetime(hit_rate_df['date'])
                hit_rate_df = hit_rate_df.rename(columns={'metric_value': 'hit_rate'})
                
                # Join with main dataframe
                df = df.merge(r2_df[['model', 'date', 'r2_value']], 
                             left_on=['model', 'prediction_date'], 
                             right_on=['model', 'date'], 
                             how='left', suffixes=('', '_r2'))
                
                df = df.merge(hit_rate_df[['model', 'date', 'hit_rate']], 
                             left_on=['model', 'prediction_date'], 
                             right_on=['model', 'date'], 
                             how='left', suffixes=('', '_hit'))
                
                # Clean up duplicate date columns
                df = df.drop(columns=[col for col in df.columns if col.endswith('_r2') or col.endswith('_hit')])
            else:
                df['r2_value'] = None
                df['hit_rate'] = None
                
        except Exception as e:
            st.warning(f"Could not load model metrics: {e}")
            df['r2_value'] = None
            df['hit_rate'] = None
        
        conn.close()
        
        # Calculate accuracy if actual prices exist
        df['accuracy'] = None
        mask = df['actual_price'].notna() & df['current_price'].notna()
        if mask.any():
            df.loc[mask, 'accuracy'] = 100 * (1 - abs((df.loc[mask, 'actual_price'] - df.loc[mask, 'predicted_price']) / df.loc[mask, 'current_price']))
        
        # Add days to target for active predictions
        today = pd.Timestamp(datetime.now().date())
        df['days_to_target'] = (df['target_date'] - today).dt.days
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Dashboard title
st.markdown("<div class='main-header'>Trading Models Performance Dashboard</div>", unsafe_allow_html=True)
st.write("Monitoring and analyzing the performance of predictive trading models")

# Add refresh button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("🔄 Refresh Data", type="primary"):
        st.cache_data.clear()
        st.rerun()

# Load data
data = load_data()

if data.empty:
    st.error("No data available. Please check:")
    st.write("1. Database exists at:", DB_PATH)
    st.write("2. Models have run today at 3:45 PM")
    st.write("3. Dashboard data collection completed successfully")
    if st.button("🔧 Run Quick Database Check"):
        st.code("python quick_database_check.py", language="bash")
    st.stop()

# Calculate date variables right after loading data
min_date = data['prediction_date'].min().date()
max_date = data['prediction_date'].max().date()
today = datetime.now().date()

# Calculate today's predictions for use throughout the dashboard
today_predictions = data[data['prediction_date'].dt.date == today]

# Debug information
with st.expander("🔍 Debug Information", expanded=False):
    st.markdown('<div class="debug-info">', unsafe_allow_html=True)
    st.write(f"**Database Path:** {DB_PATH}")
    st.write(f"**Database Exists:** {os.path.exists(DB_PATH)}")
    st.write(f"**Total Records:** {len(data)}")
    st.write(f"**Date Range:** {data['prediction_date'].min()} to {data['prediction_date'].max()}")
    st.write(f"**Models Found:** {', '.join(data['model'].unique())}")
    st.write(f"**Today's Date:** {today}")
    
    # Show today's data specifically
    today_data = data[data['prediction_date'].dt.date == today]
    st.write(f"**Predictions for Today ({today}):** {len(today_data)}")
    
    if len(today_data) > 0:
        st.success("✅ Today's data found!")
        for model in today_data['model'].unique():
            model_count = len(today_data[today_data['model'] == model])
            st.write(f"  - {model}: {model_count} predictions")
    else:
        if max_date < today:
            st.warning("⚠️ Today's data not yet available")
            st.write(f"Most recent data: {max_date}")
            st.write("Possible reasons:")
            st.write("- Models haven't run today at 3:45 PM yet")
            st.write("- Dashboard data collection failed")
            st.write("- Check wrapper_log.txt for errors")
        else:
            st.error("❌ No predictions found for today")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Sidebar filters
st.sidebar.header("Filter Options")

# Date range selector with better defaults
# Smart default logic - handle case where data doesn't include today
if max_date >= today:
    # Data includes today or future dates
    default_start = max(min_date, today - timedelta(days=7))
    default_end = today
    date_max_value = max(max_date, today)
else:
    # Data only has past dates (today's data not yet collected)
    default_start = max(min_date, max_date - timedelta(days=7))
    default_end = max_date
    date_max_value = max_date

start_date = st.sidebar.date_input(
    "Start Date", 
    value=default_start, 
    min_value=min_date, 
    max_value=date_max_value
)

end_date = st.sidebar.date_input(
    "End Date", 
    value=default_end, 
    min_value=min_date, 
    max_value=date_max_value
)

# Quick date filter buttons
st.sidebar.write("**Quick Filters:**")

# Show data status
if max_date < today:
    st.sidebar.warning("⚠️ No data for today")
    st.sidebar.write(f"Latest data: {max_date}")
    st.sidebar.write("Models may not have run yet")
else:
    st.sidebar.success(f"✅ Data current through {max_date}")

col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("Today Only"):
        if max_date >= today:
            start_date = today
            end_date = today
        else:
            # If no today data, show most recent date
            start_date = max_date
            end_date = max_date
        st.rerun()

with col2:
    if st.button("Last 7 Days"):
        if max_date >= today:
            start_date = today - timedelta(days=7)
            end_date = today
        else:
            # If no today data, show last 7 days from most recent date
            start_date = max_date - timedelta(days=7)
            end_date = max_date
        st.rerun()

# Model selector
all_models = sorted(data['model'].unique().tolist())
selected_models = st.sidebar.multiselect(
    "Select Models", 
    all_models,
    default=all_models
)

if not selected_models:
    st.warning("Please select at least one model.")
    st.stop()

# Horizon selector
show_all_horizons = st.sidebar.checkbox("Show All Horizons", value=True)

# Filter data by date range
filtered_data = data[
    (data['prediction_date'].dt.date >= start_date) & 
    (data['prediction_date'].dt.date <= end_date) &
    (data['model'].isin(selected_models))
].copy()

# Show filter results
if filtered_data.empty:
    st.warning(f"No data found for the selected date range ({start_date} to {end_date}) and models.")
    st.info("Try:")
    st.write("- Expanding the date range")
    st.write("- Selecting different models")
    st.write("- Checking if reports ran today at 3:45 PM")
    st.stop()

# Horizon filtering
if not show_all_horizons and not filtered_data.empty:
    horizons = sorted(filtered_data['horizon'].unique().tolist())
    if horizons:
        selected_horizon = st.sidebar.selectbox("Prediction Horizon (days)", horizons, index=0)
        filtered_data = filtered_data[filtered_data['horizon'] == selected_horizon]

# Show current filter summary
st.sidebar.markdown("---")
st.sidebar.write("**Current Filter Results:**")
st.sidebar.write(f"📅 Date Range: {start_date} to {end_date}")
st.sidebar.write(f"📊 Total Predictions: {len(filtered_data)}")

# Highlight today's predictions if any
if len(today_predictions) > 0:
    st.sidebar.markdown(f"**🎯 Today's Predictions: {len(today_predictions)}**")

# MODEL ACCURACY SUMMARY SECTION
st.markdown("<div class='metrics-header'>📊 Model Performance Summary</div>", unsafe_allow_html=True)

# Calculate accuracy summary for each model and horizon
accuracy_summary = []

for model in selected_models:
    model_data = filtered_data[filtered_data['model'] == model]
    horizons = sorted(model_data['horizon'].unique())
    
    for horizon in horizons:
        horizon_data = model_data[model_data['horizon'] == horizon]
        
        # Get completed predictions (with actual prices)
        completed = horizon_data[horizon_data['actual_price'].notna()]
        
        # Calculate metrics
        total_predictions = len(horizon_data)
        completed_predictions = len(completed)
        
        if completed_predictions > 0:
            # Calculate average accuracy
            avg_accuracy = completed['accuracy'].mean() if 'accuracy' in completed.columns and completed['accuracy'].notna().any() else None
            
            # Calculate direction accuracy
            correct_directions = completed['direction_correct'].sum() if 'direction_correct' in completed.columns else 0
            direction_accuracy = (correct_directions / completed_predictions * 100) if completed_predictions > 0 else 0
        else:
            # Use hit_rate from latest prediction if available
            latest = horizon_data.sort_values('prediction_date', ascending=False).iloc[0] if not horizon_data.empty else None
            avg_accuracy = latest['hit_rate'] if latest is not None and pd.notna(latest.get('hit_rate')) else None
            direction_accuracy = None
        
        accuracy_summary.append({
            'Model': model,
            'Horizon (days)': horizon,
            'Accuracy (%)': round(avg_accuracy, 1) if avg_accuracy is not None else "N/A",
            'Direction Accuracy (%)': round(direction_accuracy, 1) if direction_accuracy is not None else "N/A",
            'Total Predictions': total_predictions,
            'Completed': completed_predictions
        })

if accuracy_summary:
    accuracy_df = pd.DataFrame(accuracy_summary)
    st.markdown('<div class="table-container">', unsafe_allow_html=True)
    st.dataframe(accuracy_df, use_container_width=True, height=300)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("No accuracy data available for the selected filters.")

# TODAY'S PREDICTIONS HIGHLIGHT
# Format prices based on symbol - shared function
def format_price_by_symbol(price, symbol):
    if pd.isna(price):
        return "N/A"
    if symbol in ['ALGO-USD']:
        return f"${price:.4f}"
    elif symbol in ['BTC-USD']:
        return f"${price:,.2f}"
    else:
        return f"${price:.2f}"

if len(today_predictions) > 0:
    st.markdown("<div class='metrics-header'>🎯 Today's Predictions</div>", unsafe_allow_html=True)
    st.markdown('<div class="today-highlight">', unsafe_allow_html=True)
    
    # Format today's predictions for display
    today_display = today_predictions[['model', 'symbol', 'horizon', 'current_price', 'predicted_price', 'suggested_action', 'target_date']].copy()
    
    today_display['Current Price'] = [
        format_price_by_symbol(price, symbol) 
        for price, symbol in zip(today_display['current_price'], today_display['symbol'])
    ]
    
    today_display['Predicted Price'] = [
        format_price_by_symbol(price, symbol) 
        for price, symbol in zip(today_display['predicted_price'], today_display['symbol'])
    ]
    
    today_display['Target Date'] = today_display['target_date'].dt.strftime('%Y-%m-%d')
    
    # Calculate expected return
    today_display['Expected Return'] = (
        (today_display['predicted_price'] / today_display['current_price'] - 1) * 100
    ).round(2).astype(str) + '%'
    
    # Select and rename columns for display
    display_cols = ['model', 'symbol', 'horizon', 'Current Price', 'Predicted Price', 'Expected Return', 'suggested_action', 'Target Date']
    today_formatted = today_display[display_cols].copy()
    today_formatted.columns = ['Model', 'Symbol', 'Horizon (days)', 'Current Price', 'Predicted Price', 'Expected Return', 'Action', 'Target Date']
    
    st.dataframe(today_formatted, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif max_date < today:
    # No today's data available yet
    st.markdown("<div class='metrics-header'>📅 Latest Available Predictions</div>", unsafe_allow_html=True)
    st.info(f"Today's predictions not yet available. Showing latest data from {max_date}")
    
    # Show most recent predictions
    latest_predictions = data[data['prediction_date'].dt.date == max_date]
    if len(latest_predictions) > 0:
        st.markdown('<div class="today-highlight">', unsafe_allow_html=True)
        
        # Use same formatting as today's predictions
        latest_display = latest_predictions[['model', 'symbol', 'horizon', 'current_price', 'predicted_price', 'suggested_action', 'target_date']].copy()
        
        latest_display['Current Price'] = [
            format_price_by_symbol(price, symbol) 
            for price, symbol in zip(latest_display['current_price'], latest_display['symbol'])
        ]
        
        latest_display['Predicted Price'] = [
            format_price_by_symbol(price, symbol) 
            for price, symbol in zip(latest_display['predicted_price'], latest_display['symbol'])
        ]
        
        latest_display['Target Date'] = latest_display['target_date'].dt.strftime('%Y-%m-%d')
        
        latest_display['Expected Return'] = (
            (latest_display['predicted_price'] / latest_display['current_price'] - 1) * 100
        ).round(2).astype(str) + '%'
        
        display_cols = ['model', 'symbol', 'horizon', 'Current Price', 'Predicted Price', 'Expected Return', 'suggested_action', 'Target Date']
        latest_formatted = latest_display[display_cols].copy()
        latest_formatted.columns = ['Model', 'Symbol', 'Horizon (days)', 'Current Price', 'Predicted Price', 'Expected Return', 'Action', 'Target Date']
        
        st.dataframe(latest_formatted, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# DETAILED PREDICTION RESULTS SECTION
st.markdown("<div class='metrics-header'>📋 Detailed Prediction Results</div>", unsafe_allow_html=True)

if not filtered_data.empty:
    # Sort by prediction date (newest first), then by model
    detail_df = filtered_data.sort_values(['prediction_date', 'model', 'horizon'], ascending=[False, True, True])
    
    # Format for display
    display_df = detail_df.copy()
    
    # Format prices
    display_df['formatted_current'] = [
        format_price_by_symbol(price, symbol) 
        for price, symbol in zip(display_df['current_price'], display_df['symbol'])
    ]
    
    display_df['formatted_predicted'] = [
        format_price_by_symbol(price, symbol) 
        for price, symbol in zip(display_df['predicted_price'], display_df['symbol'])
    ]
    
    display_df['formatted_actual'] = [
        format_price_by_symbol(price, symbol) if pd.notna(price) else "N/A"
        for price, symbol in zip(display_df['actual_price'], display_df['symbol'])
    ]
    
    # Format dates
    display_df['formatted_pred_date'] = display_df['prediction_date'].dt.strftime('%Y-%m-%d')
    display_df['formatted_target_date'] = display_df['target_date'].dt.strftime('%Y-%m-%d')
    
    # Format error percentage
    display_df['formatted_error'] = display_df['error_pct'].apply(
        lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
    )
    
    # Days to target
    display_df['formatted_days'] = display_df['days_to_target'].apply(
        lambda x: f"{x} days" if pd.notna(x) and x > 0 else "Past Due" if pd.notna(x) and x <= 0 else "N/A"
    )
    
    # Select columns for final display
    final_display = display_df[[
        'model', 'symbol', 'formatted_pred_date', 'formatted_target_date', 'horizon',
        'formatted_current', 'formatted_predicted', 'formatted_actual', 
        'formatted_error', 'suggested_action', 'formatted_days'
    ]].copy()
    
    final_display.columns = [
        'Model', 'Symbol', 'Prediction Date', 'Target Date', 'Horizon (days)',
        'Price at Prediction', 'Predicted Price', 'Actual Price', 
        'Error %', 'Action', 'Days to Target'
    ]
    
    st.markdown('<div class="table-container">', unsafe_allow_html=True)
    st.dataframe(final_display, use_container_width=True, height=500)
    st.markdown('</div>', unsafe_allow_html=True)

# MODEL PERFORMANCE BY PREDICTION
st.markdown("<div class='metrics-header'>🎯 Model Performance by Prediction</div>", unsafe_allow_html=True)

# Group detailed predictions by model and horizon
if not filtered_data.empty:
    model_tabs = st.tabs(selected_models)
    
    for i, model in enumerate(selected_models):
        with model_tabs[i]:
            model_data = filtered_data[filtered_data['model'] == model]
            
            if not model_data.empty:
                # Get unique horizons for this model
                horizons = sorted(model_data['horizon'].unique())
                
                for horizon in horizons:
                    horizon_data = model_data[model_data['horizon'] == horizon].sort_values('prediction_date', ascending=False)
                    
                    if not horizon_data.empty:
                        with st.expander(f"📊 {horizon}-Day Predictions", expanded=(len(horizons) == 1)):
                            # Format data for display
                            perf_data = []
                            for _, row in horizon_data.head(10).iterrows():  # Show last 10 predictions
                                # Format prices
                                if row['symbol'] in ['ALGO-USD']:
                                    pred_price = f"${row['predicted_price']:.4f}"
                                    actual_price = f"${row['actual_price']:.4f}" if pd.notna(row['actual_price']) else "Pending"
                                elif row['symbol'] in ['BTC-USD']:
                                    pred_price = f"${row['predicted_price']:,.2f}"
                                    actual_price = f"${row['actual_price']:,.2f}" if pd.notna(row['actual_price']) else "Pending"
                                else:
                                    pred_price = f"${row['predicted_price']:.2f}"
                                    actual_price = f"${row['actual_price']:.2f}" if pd.notna(row['actual_price']) else "Pending"
                                
                                # Calculate accuracy
                                if pd.notna(row['actual_price']):
                                    accuracy = 100 * (1 - abs((row['actual_price'] - row['predicted_price']) / row['current_price']))
                                    accuracy_str = f"{accuracy:.1f}%"
                                else:
                                    accuracy_str = "Pending"
                                
                                perf_data.append({
                                    'Prediction Date': row['prediction_date'].strftime('%Y-%m-%d'),
                                    'Target Date': row['target_date'].strftime('%Y-%m-%d'),
                                    'Predicted Price': pred_price,
                                    'Actual Price': actual_price,
                                    'Accuracy': accuracy_str,
                                    'Action': row['suggested_action']
                                })
                            
                            if perf_data:
                                perf_df = pd.DataFrame(perf_data)
                                st.dataframe(perf_df, use_container_width=True)
                            else:
                                st.info(f"No {horizon}-day predictions available")
            else:
                st.info(f"No data available for {model}")

# PREDICTION ERROR STATISTICS
st.markdown("<div class='metrics-header'>📈 Prediction Error Statistics</div>", unsafe_allow_html=True)

completed_predictions = filtered_data[filtered_data['actual_price'].notna()].copy()

if not completed_predictions.empty:
    # Group by model and horizon to calculate stats
    error_stats = []
    
    for model in selected_models:
        model_data = completed_predictions[completed_predictions['model'] == model]
        
        if not model_data.empty:
            horizons = sorted(model_data['horizon'].unique())
            
            for horizon in horizons:
                horizon_data = model_data[model_data['horizon'] == horizon]
                
                if len(horizon_data) > 0:
                    errors = horizon_data['error_pct']
                    
                    error_stats.append({
                        'Model': model,
                        'Horizon (days)': horizon,
                        'Count': len(horizon_data),
                        'Mean Error (%)': f"{errors.mean():.2f}",
                        'Std Dev (%)': f"{errors.std():.2f}",
                        'Min Error (%)': f"{errors.min():.2f}",
                        'Max Error (%)': f"{errors.max():.2f}",
                        'Abs Mean Error (%)': f"{abs(errors).mean():.2f}"
                    })
    
    if error_stats:
        error_df = pd.DataFrame(error_stats)
        st.markdown('<div class="table-container">', unsafe_allow_html=True)
        st.dataframe(error_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No completed predictions available for error statistics.")
else:
    st.info("No completed predictions available for error statistics.")

# MODEL COMPARISON CHART
if len(filtered_data) > 0:
    st.markdown("<div class='metrics-header'>📊 Model Comparison Chart</div>", unsafe_allow_html=True)
    
    # Get latest predictions for each model (3-day horizon preferred)
    comparison_data = []
    for model in selected_models:
        model_data = filtered_data[filtered_data['model'] == model]
        
        # Prefer 3-day horizon, but take any if 3-day not available
        horizon_3d = model_data[model_data['horizon'] == 3]
        if not horizon_3d.empty:
            latest = horizon_3d.sort_values('prediction_date', ascending=False).iloc[0]
        else:
            latest = model_data.sort_values('prediction_date', ascending=False).iloc[0]
        
        expected_return = ((latest['predicted_price'] / latest['current_price']) - 1) * 100
        
        comparison_data.append({
            'Model': model,
            'Expected Return (%)': expected_return,
            'Action': latest['suggested_action'],
            'Horizon': f"{latest['horizon']} days",
            'Symbol': latest['symbol']
        })
    
    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        
        # Create bar chart
        fig = px.bar(
            comp_df,
            x='Model',
            y='Expected Return (%)',
            color='Action',
            color_discrete_map={'BUY': '#28a745', 'SELL': '#dc3545', 'HOLD': '#ffc107'},
            title='Latest Predicted Returns by Model',
            hover_data=['Symbol', 'Horizon']
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# ACCURACY TRENDS OVER TIME (NEW)
if not filtered_data.empty:
    st.markdown("<div class='metrics-header'>📈 Accuracy Trends Over Time</div>", unsafe_allow_html=True)
    
    # Calculate daily accuracy for models with completed predictions
    accuracy_data = []
    for model in selected_models:
        model_completed = completed_predictions[completed_predictions['model'] == model]
        
        if len(model_completed) > 1:
            # Group by prediction date and calculate average accuracy
            daily_accuracy = model_completed.groupby(model_completed['prediction_date'].dt.date).agg({
                'accuracy': 'mean',
                'error_pct': lambda x: abs(x).mean()
            }).reset_index()
            
            daily_accuracy['model'] = model
            accuracy_data.append(daily_accuracy)
    
    if accuracy_data:
        all_accuracy = pd.concat(accuracy_data, ignore_index=True)
        
        # Create line chart
        fig = px.line(
            all_accuracy,
            x='prediction_date',
            y='accuracy',
            color='model',
            title='Model Accuracy Trends Over Time',
            labels={'prediction_date': 'Prediction Date', 'accuracy': 'Accuracy (%)'}
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough completed predictions to show accuracy trends.")

# PERFORMANCE INSIGHTS
st.markdown("<div class='metrics-header'>🚀 Performance Insights</div>", unsafe_allow_html=True)

# Check for recent performance monitoring reports
desktop_path = os.path.join(os.environ.get("USERPROFILE", ""), "OneDrive", "Desktop")
perf_reports = [f for f in os.listdir(desktop_path) if f.startswith("performance_monitoring_report_") and f.endswith(".json")]

if perf_reports:
    # Get the most recent report
    latest_report = max(perf_reports, key=lambda x: x.split("_")[-1].split(".")[0])
    report_path = os.path.join(desktop_path, latest_report)
    
    try:
        with open(report_path, 'r') as f:
            perf_data = json.load(f)
        
        st.success(f"✅ Performance monitoring active - Last analysis: {latest_report.split('_')[-1].split('.')[0]}")
        
        if 'summary' in perf_data:
            summary = perf_data['summary']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Active Models", summary.get('active_models', 0))
            with col2:
                st.metric("Need Attention", summary.get('models_need_attention', 0))
            with col3:
                st.metric("Critical Issues", summary.get('critical_recommendations', 0))
            with col4:
                st.metric("High Priority", summary.get('high_priority_recommendations', 0))
        
        # Show top recommendations if any
        if 'optimization_recommendations' in perf_data and perf_data['optimization_recommendations']:
            with st.expander("🎯 Top Performance Recommendations", expanded=False):
                for i, rec in enumerate(perf_data['optimization_recommendations'][:3], 1):
                    priority_color = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}.get(rec['priority'], '⚪')
                    st.write(f"{priority_color} **{rec['priority'].title()}**: {rec['message']}")
                    if 'action' in rec:
                        st.write(f"   → Action: {rec['action']}")
    
    except Exception as e:
        st.warning(f"Could not load performance report: {e}")
else:
    st.info("🔧 Performance monitoring available - run `python performance_monitoring_system.py` to generate insights")

# FOOTER
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.write(f"**Data Source:** {os.path.basename(DB_PATH)}")

with col2:
    st.write(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with col3:
    if st.button("💾 Export Data"):
        # Create downloadable CSV
        export_data = filtered_data[['model', 'symbol', 'prediction_date', 'target_date', 'horizon', 'current_price', 'predicted_price', 'actual_price', 'suggested_action']]
        csv = export_data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"trading_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )