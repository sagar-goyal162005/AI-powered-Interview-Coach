# How to Access the AI Interview Coach App

## 🚀 Quick Start (For Host/Developer)

### Option 1: Using Cloudflare Tunnel (Recommended - FREE & NO WARNING PAGE!)

Simply run this command:
```powershell
cd d:\AIInterviewCoach-main
.\start_cloudflare.ps1
```

**You'll get a public URL like:** `https://something-random-words.trycloudflare.com`

### Option 2: Manual Setup

**Terminal 1 - Start Streamlit:**
```powershell
cd d:\AIInterviewCoach-main
streamlit run hack.py --server.port=8501
```

**Terminal 2 - Start Cloudflare Tunnel:**
```powershell
& "$env:USERPROFILE\cloudflared.exe" tunnel --url http://localhost:8501
```

The public URL will appear in the output!

---

## 📋 Instructions for Teammates

### Step 1: Get the URL from Host
Ask the person running the app for the Cloudflare tunnel URL
- It will look like: `https://xxxxx-xxxxx-xxxxx.trycloudflare.com`

### Step 2: Open the URL
- Simply click/paste the URL in your browser
- **NO warning page!** (Unlike ngrok)
- **NO account needed!**
- Works directly!

### Step 3: Use the App
- The app loads immediately
- No additional clicks required
- Works from anywhere in the world

---

## ✅ Why Cloudflare Tunnel is Better

| Feature | Cloudflare Tunnel | ngrok (free) |
|---------|------------------|--------------|
| **Warning Page** | ❌ None | ✅ Yes (annoying) |
| **Account Required** | ❌ No | ✅ Yes |
| **Speed** | ⚡ Fast | 🐌 Slower |
| **Reliability** | 🎯 Excellent | ⚠️ Limited |
| **Blank Screen Issues** | ❌ Rare | ✅ Common |

---

## ⚠️ Troubleshooting

If you see a blank screen (rare with Cloudflare):

1. **Hard Refresh**: Press `Ctrl + Shift + R` (Windows) or `Cmd + Shift + R` (Mac)
2. **Wait**: Sometimes takes 10-20 seconds on first load
3. **Try Different Browser**: Use Chrome, Firefox, or Edge
4. **Check Console**: Press F12 for any error messages

---

## 🔧 For Developer (Keep Running)

**Keep the PowerShell terminal open!**
- Closing it will stop the public access
- Your teammates won't be able to access the app
- Streamlit must also keep running

---

## ⏰ Availability
- App is available **only when host keeps the tunnel running**
- If teammates can't access, check if both terminals are running
- URL changes each time you restart the tunnel

## 💡 Pro Tip
Share the URL via WhatsApp/Slack/Email to your teammates!
