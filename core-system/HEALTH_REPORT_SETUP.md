# Trading System Daily Health Report

## Overview
The daily health report provides comprehensive monitoring of your trading system, checking all critical components every morning at 9 AM.

## What Gets Checked

### 📊 Database Health
- ✅ Database file accessibility  
- ✅ Core tables integrity (model_predictions, price_history, model_metrics)
- ✅ Recent prediction activity (last 7 days)
- ⚠️ Overdue predictions needing actual prices

### 💰 Price Data Currency  
- ✅ Latest price data dates for QQQ, NVDA, BTC-USD
- ✅ API connectivity (yfinance)
- ⚠️ Data freshness (alerts if > 5 days old)

### 🤖 Model Performance
- ✅ Recent predictions by model (30 days)
- ✅ Accuracy rates and error percentages
- ✅ Predictions with actual results vs pending

### 🌐 Dashboard Status
- ✅ Dashboard file exists and accessible
- ✅ Streamlit server running on http://localhost:8502

### 💾 System Resources
- ✅ Available disk space (alerts if < 1GB)
- ✅ Reports directory writable

### 🔄 Recent Model Execution
- ✅ Latest model runs and log files
- ✅ Most recent predictions by each model

## Current System Status

**Overall Health Score:** 80/100 - EXCELLENT

### Key Metrics (as of last report):
- **Total Predictions:** 27
- **With Results:** 10  
- **Overall Accuracy:** 70.0%
- **Overdue Predictions:** 1

### Issues to Address:
1. **NVDA Price Data:** 13 days old (needs update)
2. **Model Accuracy:** Some models below 60% accuracy threshold
3. **Overdue Predictions:** 1 prediction needs actual price

## How to Use

### Manual Health Check
```bash
cd "C:\Users\rrose\trading-models-system\core-system"
python daily_health_report.py
```

### Automated Daily Reports (9 AM)

#### Option 1: PowerShell Script
```powershell
# Run once to test
PowerShell -ExecutionPolicy Bypass -File "C:\Users\rrose\trading-models-system\core-system\run_daily_health_check.ps1"
```

#### Option 2: Windows Task Scheduler
```cmd
# Run as Administrator to setup
C:\Users\rrose\trading-models-system\core-system\schedule_health_report.bat
```

### View Historical Reports
Reports are saved to: `C:\Users\rrose\trading-models-system\core-system\reports\health_report_YYYY-MM-DD_HH-mm-ss.log`

## Health Score Interpretation

- **90-100:** EXCELLENT - System operating optimally
- **80-89:** EXCELLENT - Minor issues present  
- **60-79:** GOOD - Some attention needed
- **40-59:** FAIR - Multiple issues requiring action
- **0-39:** NEEDS ATTENTION - Critical issues present

## Automated Alerts

The system will flag issues when:
- Overdue predictions > 5
- Model accuracy < 50%
- Price data > 5 days old
- Disk space < 1GB
- No recent model executions (24+ hours)

## Recommendations Based on Current Status

1. **Update NVDA Price Data:** Import recent NVDA prices to price_history table
2. **Review Model Performance:** Investigate models with < 60% accuracy
3. **Clear Overdue Predictions:** Run update_overdue_predictions.py
4. **Monitor API Limits:** yfinance rate limiting detected

## Files Created

- `daily_health_report.py` - Main health check script
- `run_daily_health_check.ps1` - PowerShell runner with logging  
- `schedule_health_report.bat` - Windows Task Scheduler setup
- `HEALTH_REPORT_SETUP.md` - This documentation

## Next Steps

1. ✅ Health report system created and tested
2. ⏳ Set up automated 9 AM scheduling
3. ⏳ Address current NVDA data gap
4. ⏳ Monitor and improve model accuracy
5. ⏳ Integrate with email/SMS alerts (future enhancement)