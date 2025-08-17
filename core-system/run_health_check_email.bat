@echo off
echo Starting Daily Health Check Email at %date% %time%
cd /d "C:\Users\rrose\trading-models-system\core-system"
"C:\Users\rrose\AppData\Local\Programs\Python\Python311\python.exe" daily_health_check_email.py
echo Health check email completed at %date% %time%
exit