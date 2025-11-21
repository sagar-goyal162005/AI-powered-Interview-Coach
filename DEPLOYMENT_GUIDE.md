# ============================================================
# AI Interview Coach - Deployment Instructions
# ============================================================

## How to Deploy Your App Publicly

### IMPORTANT: Use CMD (Command Prompt), NOT PowerShell!

1. Open **Command Prompt** (CMD):
   - Press `Windows + R`
   - Type `cmd`
   - Press Enter

2. Navigate to your project folder:
   ```
   cd /d d:\AIInterviewCoach-main
   ```

3. Run the deployment script:
   ```
   start_cloudflare.bat
   ```

4. Wait 10-15 seconds. You'll see a URL like:
   ```
   https://something-random-words.trycloudflare.com
   ```

5. **Copy that URL and share it with your teammates!**

6. **Keep the CMD window open!** Closing it will stop the tunnel.

---

## Alternative: Manual 2-Terminal Method

### Terminal 1 (Start Streamlit):
```
cd /d d:\AIInterviewCoach-main
streamlit run hack.py --server.port=8501
```

### Terminal 2 (Start Cloudflare Tunnel):
```
cd /d d:\AIInterviewCoach-main
%USERPROFILE%\cloudflared.exe tunnel --url http://localhost:8501
```

The public URL will appear in Terminal 2!

---

## Troubleshooting

**If you see "& was unexpected":**
- You're using PowerShell by mistake
- Close PowerShell and open CMD instead

**If cloudflared.exe is not found:**
- Run this in CMD:
  ```
  powershell -Command "Invoke-WebRequest -Uri 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe' -OutFile '$env:USERPROFILE\cloudflared.exe'"
  ```

---

## Quick Reference

**Check if Streamlit is running:**
```
netstat -ano | findstr ":8501"
```

**Stop all processes:**
```
taskkill /f /im streamlit.exe
taskkill /f /im cloudflared.exe
```
