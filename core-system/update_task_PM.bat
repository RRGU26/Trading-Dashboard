@echo off
echo Updating scheduled task for Trading Models Wrapper to 3:40 PM...
schtasks /change /tn "TradingModelsWrapper" /st 15:40 /f
echo.
echo Task updated successfully!
echo The wrapper will now run daily at 3:40 PM
echo.
echo To run the task immediately for testing, use:
echo schtasks /run /tn "TradingModelsWrapper"
pause