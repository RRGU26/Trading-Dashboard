@echo off
echo Starting Trading Models Wrapper at %date% %time%
cd /d "C:\Users\rrose\trading-models-system\core-system"
"C:\Users\rrose\AppData\Local\Programs\Python\Python311\python.exe" wrapper.py
echo Wrapper execution completed at %date% %time%
exit