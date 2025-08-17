@echo off
echo Creating scheduled task for Trading Models Wrapper...
schtasks /create /tn "TradingModelsWrapper" /tr "C:\Users\rrose\trading-models-system\core-system\run_wrapper.bat" /sc daily /st 03:40 /f
echo.
echo Task created successfully!
echo The wrapper will run daily at 3:40 AM
echo.
echo To run the task immediately for testing, use:
echo schtasks /run /tn "TradingModelsWrapper"
pause