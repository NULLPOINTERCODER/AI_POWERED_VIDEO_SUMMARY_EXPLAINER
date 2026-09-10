@echo off
echo ================================================
echo   AI Video Assistant - Starting Streamlit App
echo ================================================
echo.
cd /d "%~dp0"
.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
pause
