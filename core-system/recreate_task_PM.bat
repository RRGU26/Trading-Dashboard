@echo off
echo Deleting old task...
schtasks /delete /tn "TradingModelsWrapper" /f 2>nul

echo Creating new scheduled task for 3:40 PM...
schtasks /create /tn "TradingModelsWrapper" /tr "C:\Users\rrose\trading-models-system\core-system\run_wrapper.bat" /sc daily /st 15:40 /f

echo.
echo Task created successfully!
echo The wrapper will run daily at 3:40 PM
echo.
echo Now testing the system...
schtasks /run /tn "TradingModelsWrapper"
echo.
echo System test started! Check wrapper_log.txt for progress.
pause