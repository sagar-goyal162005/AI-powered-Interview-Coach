@echo off
echo ============================================================
echo AI Interview Coach - Cloudflare Tunnel Deployment
echo ============================================================
echo.

echo [Step 1/2] Starting Streamlit app...
start /B streamlit run hack.py --server.port=8501
timeout /t 10 /nobreak > nul

echo [Step 2/2] Creating Cloudflare Tunnel...
echo.
echo Your public URL will appear below:
echo ============================================================
echo.

"%USERPROFILE%\cloudflared.exe" tunnel --url http://localhost:8501

echo.
echo ============================================================
echo Tunnel stopped. Press any key to exit...
pause > nul
