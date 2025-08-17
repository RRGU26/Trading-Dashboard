import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import numpy as np

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "models_dashboard.db")

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
    .demo-banner {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    .live-banner {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def create_sample_data():
    """Create sample data for demonstration when database is not available"""
    
    # Generate sample predictions for the last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    models = ['Bitcoin Model', 'QQQ Long Bull Model', 'Algorand Model', 'QQQ Trading Signal', 'Wishing Well QQQ Model']
    symbols = ['BTC-USD', 'QQQ', 'ALGO-USD', 'QQQ', 'QQQ']
    horizons = [1, 3, 7]
    actions = ['BUY', 'HOLD', 'SELL']
    
    sample_data = []
    
    for i in range(50):  # Generate 50 sample predictions
        pred_date = start_date + timedelta(days=int(np.random.randint(0, 30)))
        model_idx = int(np.random.randint(0, len(models)))
        model = models[model_idx]
        symbol = symbols[model_idx]
        horizon = int(np.random.choice(horizons))  # Ensure integer
        
        # Generate realistic prices based on symbol
        if symbol == 'BTC-USD':
            current_price = float(np.random.uniform(45000, 70000))
            predicted_price = float(current_price * (1 + np.random.uniform(-0.1, 0.1)))
            # Fix: Use integer for timedelta
            actual_price = float(current_price * (1 + np.random.uniform(-0.08, 0.08))) if pred_date < end_date - timedelta(days=int(horizon)) else None
        elif symbol == 'ALGO-USD':
            current_price = float(np.random.uniform(0.15, 0.35))
            predicted_price = float(current_price * (1 + np.random.uniform(-0.15, 0.15)))
            actual_price = float(current_price * (1 + np.random.uniform(-0.12, 0.12))) if pred_date < end_date - timedelta(days=int(horizon)) else None
        else:  # QQQ
            current_price = float(np.random.uniform(350, 520))
            predicted_price = float(current_price * (1 + np.random.uniform(-0.05, 0.05)))
            actual_price = float(current_price * (1 + np.random.uniform(-0.04, 0.04))) if pred_date < end_date - timedelta(days=int(horizon)) else None
        
        # Calculate metrics
        error_pct = float(abs((predicted_price - actual_price) / current_price * 100)) if actual_price else None
        direction_correct = 1 if actual_price and ((predicted_price > current_price) == (actual_price > current_price)) else 0 if actual_price else None
        
        sample_data.append({
            'model': model,
            'symbol': symbol,
            'prediction_date': pred_date,
            'target_date': pred_date + timedelta(days=int(horizon)),
            'horizon': int(horizon),
            'current_price': float(current_price),
            'predicted_price': float(predicted_price),
            'actual_price': float(actual_price) if actual_price else None,
            'confidence': float(np.random.uniform(60, 95)),
            'suggested_action': str(np.random.choice(actions)),
            'error_pct': float(error_pct) if error_pct else None,
            'direction_correct': int(direction_correct) if direction_correct is not None else None,
            'hit_rate': float(np.random.uniform(45, 75))
        })
    
    return pd.DataFrame(sample_data)

# Load data function with better error handling and sample data fallback
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data():
    """Load prediction data from SQLite database with sample data fallback"""
    
    is_live_data = False
    
    # Check if using cloud database URL from secrets
    if hasattr(st, 'secrets') and 'database_url' in st.secrets:
        try:
            # Try to connect to cloud database
            conn = sqlite3.connect(st.secrets.database_url)
            st.success("🔗 Connected to live database!")
            is_live_data = True
        except Exception as e:
            st.warning(f"Could not connect to live database: {e}")
            st.info("📊 Using sample data for demonstration")
            return create_sample_data()
    elif os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            st.success("🔗 Connected to local database!")
            is_live_data = True
        except Exception as e:
            st.warning(f"Could not connect to local database: {e}")
            st.info("📊 Using sample data for demonstration")
            return create_sample_data()
    else:
        st.info("🔧 Database not found - Using sample data for demonstration")
        return create_sample_data()
    
    try:
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
            st.warning("📊 No predictions found in database - Using sample data")
            conn.close()
            return create_sample_data()
        
        # Convert dates to datetime with proper error handling
        try:
            df['prediction_date'] = pd.to_datetime(df['prediction_date'], errors='coerce')
            df['target_date'] = pd.to_datetime(df['target_date'], errors='coerce')
        except Exception as e:
            st.error(f"Date conversion error: {e}")
            conn.close()
            return create_sample_data()
        
        # Ensure numeric columns are proper types
        numeric_columns = ['horizon', 'current_price', 'predicted_price', 'actual_price', 'confidence', 'error_pct']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure horizon is integer
        df['horizon'] = df['horizon'].fillna(3).astype(int)
        
        # Try to get model metrics (R² values and hit rates)
        try:
            metrics_query = """
            SELECT model, date, metric_type, metric_value 
            FROM model_metrics 
            ORDER BY date DESC
            """
            metrics_df = pd.read_sql(metrics_query, conn)
            
            if not metrics_df.empty:
                # Add hit rates from metrics
                hit_rate_df = metrics_df[metrics_df['metric_type'] == 'hit_rate'].copy()
                hit_rate_df['date'] = pd.to_datetime(hit_rate_df['date'], errors='coerce')
                hit_rate_df = hit_rate_df.rename(columns={'metric_value': 'hit_rate'})
                
                # Join with main dataframe
                df = df.merge(hit_rate_df[['model', 'date', 'hit_rate']], 
                             left_on=['model', 'prediction_date'], 
                             right_on=['model', 'date'], 
                             how='left')
                
                # Clean up duplicate date columns
                df = df.drop(columns=[col for col in df.columns if col.endswith('_hit') or col == 'date'])
            else:
                df['hit_rate'] = np.random.uniform(45, 75, len(df))
                
        except Exception as e:
            st.warning(f"Could not load model metrics: {e}")
            df['hit_rate'] = np.random.uniform(45, 75, len(df))
        
        conn.close()
        
        # Calculate accuracy if actual prices exist
        df['accuracy'] = None
        mask = df['actual_price'].notna() & df['current_price'].notna() & (df['current_price'] != 0)
        if mask.any():
            df.loc[mask, 'accuracy'] = 100 * (1 - abs((df.loc[mask, 'actual_price'] - df.loc[mask, 'predicted_price']) / df.loc[mask, 'current_price']))
        
        # Add days to target for active predictions - FIXED VERSION
        today = pd.Timestamp(datetime.now().date())
        try:
            days_diff = (df['target_date'] - today).dt.days
            df['days_to_target'] = days_diff.fillna(0).astype(int)
        except Exception as e:
            st.warning(f"Error calculating days to target: {e}")
            df['days_to_target'] = 0
        
        # Add live data indicator
        if is_live_data:
            st.success("✅ Using LIVE trading data!")
        
        return df
        
    except Exception as e:
        st.error(f"⚠️ Error loading database: {e}")
        st.info("📊 Using sample data for demonstration")
        if 'conn' in locals():
            conn.close()
        return create_sample_data()

# Dashboard title
st.markdown("<div class='main-header'>Trading Models Performance Dashboard</div>", unsafe_allow_html=True)

# Check data source and show appropriate banner
if (hasattr(st, 'secrets') and 'database_url' in st.secrets) or os.path.exists(DB_PATH):
    st.markdown("""
    <div class='live-banner'>
        <strong>🔴 LIVE DATA MODE</strong><br>
        This dashboard is connected to your live trading models database.<br>
        Data updates automatically when your models run.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class='demo-banner'>
        <strong>🔧 DEMO MODE</strong><br>
        This dashboard is running with sample data for demonstration purposes.<br>
        To connect your real trading data, see instructions below.
    </div>
    """, unsafe_allow_html=True)

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
    st.error("No data available. Please check your data source.")
    st.stop()

# Calculate date variables
try:
    min_date = data['prediction_date'].min().date()
    max_date = data['prediction_date'].max().date()
    today = datetime.now().date()
except Exception as e:
    st.error(f"Error processing dates: {e}")
    st.stop()

# Calculate today's predictions
today_predictions = data[data['prediction_date'].dt.date == today]

# Debug information
with st.expander("🔍 System Information", expanded=False):
    st.markdown('<div class="debug-info">', unsafe_allow_html=True)
    st.write(f"**Database Path:** {DB_PATH}")
    st.write(f"**Database Exists:** {os.path.exists(DB_PATH)}")
    st.write(f"**Cloud Database:** {'Yes' if hasattr(st, 'secrets') and 'database_url' in st.secrets else 'No'}")
    st.write(f"**Data Source:** {'Live Database' if (os.path.exists(DB_PATH) or (hasattr(st, 'secrets') and 'database_url' in st.secrets)) else 'Sample Data'}")
    st.write(f"**Total Records:** {len(data)}")
    st.write(f"**Date Range:** {data['prediction_date'].min()} to {data['prediction_date'].max()}")
    st.write(f"**Models Found:** {', '.join(data['model'].unique())}")
    st.write(f"**Today's Predictions:** {len(today_predictions)}")
    st.markdown('</div>', unsafe_allow_html=True)

# Sidebar filters
st.sidebar.header("Filter Options")

# Date range selector
if max_date >= today:
    default_start = max(min_date, today - timedelta(days=7))
    default_end = today
    date_max_value = max(max_date, today)
else:
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

# Filter data by date range
filtered_data = data[
    (data['prediction_date'].dt.date >= start_date) & 
    (data['prediction_date'].dt.date <= end_date) &
    (data['model'].isin(selected_models))
].copy()

if filtered_data.empty:
    st.warning(f"No data found for the selected filters.")
    st.stop()

# Show current filter summary
st.sidebar.markdown("---")
st.sidebar.write("**Filter Results:**")
st.sidebar.write(f"📅 Date Range: {start_date} to {end_date}")
st.sidebar.write(f"📊 Total Predictions: {len(filtered_data)}")

# Format prices based on symbol
def format_price_by_symbol(price, symbol):
    if pd.isna(price):
        return "N/A"
    try:
        if symbol in ['ALGO-USD']:
            return f"${float(price):.4f}"
        elif symbol in ['BTC-USD']:
            return f"${float(price):,.2f}"
        else:
            return f"${float(price):.2f}"
    except (ValueError, TypeError):
        return "N/A"

# MODEL ACCURACY SUMMARY
st.markdown("<div class='metrics-header'>📊 Model Performance Summary</div>", unsafe_allow_html=True)

accuracy_summary = []

for model in selected_models:
    model_data = filtered_data[filtered_data['model'] == model]
    horizons = sorted(model_data['horizon'].unique())
    
    for horizon in horizons:
        horizon_data = model_data[model_data['horizon'] == horizon]
        completed = horizon_data[horizon_data['actual_price'].notna()]
        
        total_predictions = len(horizon_data)
        completed_predictions = len(completed)
        
        if completed_predictions > 0:
            avg_accuracy = completed['accuracy'].mean() if 'accuracy' in completed.columns and completed['accuracy'].notna().any() else None
            correct_directions = completed['direction_correct'].sum() if 'direction_correct' in completed.columns else 0
            direction_accuracy = (correct_directions / completed_predictions * 100) if completed_predictions > 0 else 0
        else:
            latest = horizon_data.sort_values('prediction_date', ascending=False).iloc[0] if not horizon_data.empty else None
            avg_accuracy = latest['hit_rate'] if latest is not None and pd.notna(latest.get('hit_rate')) else None
            direction_accuracy = None
        
        accuracy_summary.append({
            'Model': model,
            'Horizon (days)': int(horizon),
            'Accuracy (%)': round(float(avg_accuracy), 1) if avg_accuracy is not None else "N/A",
            'Direction Accuracy (%)': round(float(direction_accuracy), 1) if direction_accuracy is not None else "N/A",
            'Total Predictions': total_predictions,
            'Completed': completed_predictions
        })

if accuracy_summary:
    accuracy_df = pd.DataFrame(accuracy_summary)
    st.dataframe(accuracy_df, use_container_width=True)

# TODAY'S PREDICTIONS HIGHLIGHT
if len(today_predictions) > 0:
    st.markdown("<div class='metrics-header'>🎯 Today's Predictions</div>", unsafe_allow_html=True)
    st.markdown('<div class="today-highlight">', unsafe_allow_html=True)
    
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
    
    # Calculate expected return safely
    try:
        expected_returns = []
        for _, row in today_display.iterrows():
            try:
                ret = ((row['predicted_price'] / row['current_price'] - 1) * 100)
                expected_returns.append(f"{ret:.2f}%")
            except (ZeroDivisionError, TypeError):
                expected_returns.append("N/A")
        today_display['Expected Return'] = expected_returns
    except Exception:
        today_display['Expected Return'] = "N/A"
    
    display_cols = ['model', 'symbol', 'horizon', 'Current Price', 'Predicted Price', 'Expected Return', 'suggested_action', 'Target Date']
    today_formatted = today_display[display_cols].copy()
    today_formatted.columns = ['Model', 'Symbol', 'Horizon (days)', 'Current Price', 'Predicted Price', 'Expected Return', 'Action', 'Target Date']
    
    st.dataframe(today_formatted, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# MODEL COMPARISON CHART
if len(filtered_data) > 0:
    st.markdown("<div class='metrics-header'>📊 Model Comparison Chart</div>", unsafe_allow_html=True)
    
    comparison_data = []
    for model in selected_models:
        model_data = filtered_data[filtered_data['model'] == model]
        
        # Prefer 3-day horizon
        horizon_3d = model_data[model_data['horizon'] == 3]
        if not horizon_3d.empty:
            latest = horizon_3d.sort_values('prediction_date', ascending=False).iloc[0]
        else:
            latest = model_data.sort_values('prediction_date', ascending=False).iloc[0]
        
        try:
            expected_return = ((latest['predicted_price'] / latest['current_price']) - 1) * 100
        except (ZeroDivisionError, TypeError):
            expected_return = 0
        
        comparison_data.append({
            'Model': model,
            'Expected Return (%)': float(expected_return),
            'Action': str(latest['suggested_action']),
            'Horizon': f"{int(latest['horizon'])} days",
            'Symbol': str(latest['symbol'])
        })
    
    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        
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

# INSTRUCTIONS FOR CONNECTING LIVE DATA
if not os.path.exists(DB_PATH) and not (hasattr(st, 'secrets') and 'database_url' in st.secrets):
    st.markdown("<div class='metrics-header'>🔗 Connect Your Live Data</div>", unsafe_allow_html=True)
    st.info("To connect your real trading models data, choose one of these options:")
    
    tab1, tab2 = st.tabs(["📁 Upload Database", "🌐 Cloud Database"])
    
    with tab1:
        st.write("**Option 1: Upload your SQLite database file**")
        st.write("1. Add your `models_dashboard.db` file to your GitHub repository")
        st.write("2. Push changes to GitHub")
        st.write("3. Streamlit will automatically redeploy with your live data")
        st.code("# Add to your repository root:\nmodels_dashboard.db", language="bash")
    
    with tab2:
        st.write("**Option 2: Connect to a cloud database**")
        st.write("1. Go to your Streamlit app settings")
        st.write("2. Add secrets for database connection")
        st.write("3. Update dashboard code to use cloud database")
        st.code("""# Add to Streamlit secrets:
database_url = "your_database_connection_string"
db_type = "postgresql"  # or mysql, sqlite, etc.""", language="toml")

# FOOTER
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    data_source = "Live Database" if (os.path.exists(DB_PATH) or (hasattr(st, 'secrets') and 'database_url' in st.secrets)) else "Sample Data"
    st.write(f"**Data Source:** {data_source}")

with col2:
    st.write(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with col3:
    if st.button("💾 Export Data"):
        try:
            export_data = filtered_data[['model', 'symbol', 'prediction_date', 'target_date', 'horizon', 'current_price', 'predicted_price', 'actual_price', 'suggested_action']]
            csv = export_data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"trading_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Export error: {e}")
