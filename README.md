# Trading Dashboard 📊

**Live cloud dashboard for trading models with real-time predictions and mobile-friendly interface**

## 🚀 Live Dashboard

🌐 **Access your live dashboard:** https://rrgu26-trading-dashboard.streamlit.app

## 📊 Features

- **📱 Mobile-Optimized** - Responsive design for all devices
- **🔄 Auto-Updates** - Syncs with local trading system every 5 minutes  
- **📈 Real-Time Signals** - Live predictions from all 7 trading models
- **📊 Performance Charts** - Interactive analytics and visualizations
- **📥 Data Export** - Download predictions as CSV

## 🎯 Trading Models Included

1. **QQQ Long Bull Model** - Primary QQQ predictions with VIX alignment
2. **QQQ Trading Signal** - Confidence-based trading signals  
3. **Algorand Price Prediction** - Crypto volatility-filtered predictions
4. **Bitcoin Model** - Enhanced with on-chain metrics
5. **NVIDIA Bull Momentum** - AI stock growth-focused patterns
6. **Wishing Well QQQ** - Alternative QQQ strategy
7. **QQQ Master Model** - Optimized ensemble approach

## 🏗️ Repository Structure

```
trading-models-system/
├── core-system/           # 22 core Python files
│   ├── wrapper.py         # Master orchestration
│   ├── data_fetcher.py    # Critical: API data access
│   ├── model_db_integrator.py # Critical: Database integration  
│   ├── [7 model files]    # All trading models
│   ├── [dashboard files]  # Health monitoring
│   └── [email system]     # Automated reports
├── databases/             # SQLite databases
│   ├── models_dashboard.db # Main predictions database
│   └── *.db               # Additional databases
├── cache/                 # API data caching
├── data/                  # Processed data storage  
├── qqq_data_cache/        # Historical market data
├── outputs/               # Model outputs & trained models
├── config/                # Configuration files
└── reports/               # Historical reports archive
    ├── daily_reports/     # Trading reports
    ├── performance_reports/ # Model performance
    └── analysis_reports/  # Dashboard analysis
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Required packages: `pandas`, `numpy`, `sklearn`, `xgboost`, `yfinance`
- API keys for Alpha Vantage and market data

### Running the System

1. **Run All Models:**
   ```bash
   cd core-system
   python wrapper.py
   ```

2. **Launch Dashboard:**
   ```bash
   python dashboard_launcher.py
   ```

3. **Check System Health:**
   ```bash
   python enhanced_dashboard_automation.py
   ```

## 📈 Model Performance

- **24 active predictions** as of latest run
- **QQQ Master Model**: 24 predictions with 53% confidence
- **System Status**: Recently fixed database corruption, all models operational
- **Email Reports**: Automated daily reports to 7 recipients

## 🔧 System Features

### Automated Workflow
1. **wrapper.py** executes all 7 models sequentially
2. Models save predictions to `models_dashboard.db`
3. **send_report.py** emails trading reports  
4. **dashboard.data.py** collects performance data
5. **enhanced_dashboard_automation.py** monitors system health

### Data Pipeline
- **data_fetcher.py**: Handles all API calls (Alpha Vantage, Yahoo Finance)
- **model_db_integrator.py**: Universal database integration
- Automatic caching for improved performance
- Error handling and Unicode compatibility

### Dashboard & Monitoring
- Real-time model health status
- Performance trend analysis
- Prediction accuracy tracking
- Automated alert system

## 📊 Recent Performance Data

**Latest Dashboard Analysis (Aug 6, 2025):**
- Active Models: 7
- Recent Predictions: 39 total
- System Status: All models operational after database fixes
- Direction Accuracy: Varies by model (Bitcoin: 97.66% price accuracy)

## ⚙️ Configuration

Key configuration files:
- `trading_config.py` - System settings
- `trading_reports_config.py` - Email configuration
- API keys stored in environment variables

## 🔄 Migration Notes

**This repository represents a complete migration from desktop to GitHub:**
- All 335+ files migrated successfully
- Preserved exact file structure for compatibility
- All dependencies included (data_fetcher.py, model_db_integrator.py)
- Historical data and reports archived

## 📝 Recent Updates

- **Aug 6, 2025**: Fixed corrupted database, restored full functionality
- **Aug 6, 2025**: Complete GitHub migration with all dependencies
- **Jul 31, 2025**: Enhanced dashboard automation system
- **Jul 30, 2025**: Model performance audit and optimization

## 🚨 Important Notes

- Models currently expect OneDrive/Desktop paths - GitHub execution requires path updates
- Database paths are hardcoded to Desktop location
- API keys required for data fetching
- Email system configured for 7 recipients

## 📞 Support

System maintained by trading team. All models operational as of latest health check.

**Generated**: August 6, 2025  
**Total Files**: 335+ (22 core Python files, databases, cache, reports)
**Status**: ✅ Fully Migrated and Operational
=======
# Trading Models Performance Dashboard

A comprehensive Streamlit dashboard for monitoring and analyzing the performance of predictive trading models including Bitcoin, QQQ, Algorand, and VIX predictions.

## 🚀 Live Demo

[View Live Dashboard](https://your-app-name.streamlit.app) *(Will be available after deployment)*

## 📊 Features

- **Real-time Model Performance Tracking**: Monitor accuracy and predictions across multiple trading models
- **Interactive Data Visualization**: Charts and graphs showing model performance over time
- **Multi-timeframe Analysis**: View predictions across different horizons (1-day, 3-day, 7-day)
- **Export Functionality**: Download prediction data as CSV files
- **Responsive Design**: Works on desktop and mobile devices

## 🏗️ Models Included

- **Bitcoin Model**: Cryptocurrency price predictions with on-chain metrics
- **QQQ Long Bull Model**: NASDAQ-100 ETF predictions
- **Algorand Model**: ALGO cryptocurrency predictions  
- **QQQ Trading Signal**: Technical analysis-based signals
- **Wishing Well QQQ Model**: Advanced QQQ predictions

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Database**: SQLite
- **Deployment**: Streamlit Community Cloud

## 📁 Project Structure
>>>>>>> 32045aa7e458d93913248c6e2d7079ba073acdf7
