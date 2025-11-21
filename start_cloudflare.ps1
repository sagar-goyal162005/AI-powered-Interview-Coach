# AI Interview Coach - Cloudflare Tunnel Deployment
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "AI Interview Coach - Cloudflare Tunnel Deployment" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[Step 1/2] Starting Streamlit app..." -ForegroundColor Yellow
Start-Process -NoNewWindow -FilePath "streamlit" -ArgumentList "run", "hack.py", "--server.port=8501"
Start-Sleep -Seconds 10

Write-Host "[Step 2/2] Creating Cloudflare Tunnel..." -ForegroundColor Yellow
Write-Host ""
Write-Host "Your public URL will appear below:" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

& "$env:USERPROFILE\cloudflared.exe" tunnel --url http://localhost:8501

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Tunnel stopped." -ForegroundColor Red
Read-Host "Press Enter to exit"
