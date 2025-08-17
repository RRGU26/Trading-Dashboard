@echo off
REM Daily Health Report Scheduler for Trading System
REM Run this script to set up automated 9am daily health reports

echo Setting up Daily Trading System Health Report...
echo.

REM Create the scheduled task
schtasks /create /tn "Trading System Health Report" /tr "python \"C:\Users\rrose\trading-models-system\core-system\daily_health_report.py\" > \"C:\Users\rrose\trading-models-system\core-system\reports\health_report_%date:~-4,4%_%date:~-10,2%_%date:~-7,2%.log\" 2>&1" /sc daily /st 09:00 /f

if %errorlevel% == 0 (
    echo SUCCESS: Daily health report scheduled for 9:00 AM
    echo.
    echo The health report will run automatically every day at 9 AM
    echo Reports will be saved to: C:\Users\rrose\trading-models-system\core-system\reports\
    echo.
    echo To view scheduled tasks: schtasks /query /tn "Trading System Health Report"
    echo To delete the task: schtasks /delete /tn "Trading System Health Report" /f
) else (
    echo ERROR: Failed to create scheduled task
    echo Please run this script as Administrator
)

echo.
pause