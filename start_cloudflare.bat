@echo off
cls
echo ============================================================
echo AI Interview Coach - Cloudflare Tunnel Deployment
echo ============================================================
echo.

echo [Step 1/2] Cleaning up existing processes...
taskkill /f /im streamlit.exe >nul 2>&1
taskkill /f /im cloudflared.exe >nul 2>&1
timeout /t 2 /nobreak >nul

echo [Step 2/2] Starting Streamlit app...
start /B streamlit run hack.py --server.port=8501
echo Waiting for Streamlit to start...
timeout /t 10 /nobreak >nul

echo [Step 3/3] Creating Cloudflare Tunnel...
echo.
echo Your public URL will appear below:
echo ============================================================
echo.

"%USERPROFILE%\cloudflared.exe" tunnel --url http://localhost:8501

echo.
echo ============================================================
echo Tunnel stopped. 
pause
