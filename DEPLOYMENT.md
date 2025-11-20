# Deployment Guide: AI-Powered Interview Coach

## ✅ Recommended Deployment Options

### Option 1: Render (Recommended - Easy & Reliable) ⭐

**Render offers free hosting with excellent Streamlit support!**

**🎉 This repository is now fully configured for one-click Render deployment!**

#### Quick Deploy with Render (Automatic Configuration):

This repository includes a `render.yaml` file for automatic deployment!

1. **One-Click Deploy**
   - Click: [![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/sagar-goyal162005/AI-powered-Interview-Coach)
   - Or visit: https://render.com and click "New +" → "Blueprint"
   - Select this repository

2. **Add Environment Variables**
   - Render will prompt you for:
     - `OPENAI_API_KEY`: Your OpenAI API key
     - `FREEPIK_API_KEY`: Your Freepik API key
   - Get your OpenAI key from: https://platform.openai.com/api-keys
   - Get your Freepik key from: https://www.freepik.com/api

3. **Deploy!**
   - Click "Apply" or "Create Web Service"
   - Render will automatically:
     - Install system dependencies (ffmpeg, audio libraries)
     - Install Python dependencies
     - Configure the service with correct settings
     - Deploy your app with SSL/HTTPS
   - Your app will be live at: `https://ai-interview-coach.onrender.com`

#### Manual Setup (Alternative):

If you prefer manual configuration:

1. **Push your code to GitHub** (Already done! ✅)

2. **Go to Render Dashboard**
   - Visit: https://render.com
   - Sign in with your GitHub account

3. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository: `sagar-goyal162005/AI-powered-Interview-Coach`

4. **Configure the Service**
   - **Name**: `ai-interview-coach` (or your preferred name)
   - **Environment**: `Python 3`
   - **Build Command**: `chmod +x build.sh && ./build.sh`
   - **Start Command**: `streamlit run hack.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`
   - **Instance Type**: Select `Free` or `Starter`

5. **Add Environment Variables**
   - Click "Environment" tab
   - Add the following:
     - `OPENAI_API_KEY`: Your OpenAI API key
     - `FREEPIK_API_KEY`: Your Freepik API key
     - `PYTHON_VERSION`: `3.11.0`

6. **Deploy!**
   - Click "Create Web Service"
   - Render will automatically build and deploy your app
   - Your app will be live at: `https://ai-interview-coach.onrender.com`

#### Benefits of Render:
- ✅ Free tier available (with 750 hours/month)
- ✅ Automatic deployments from GitHub
- ✅ Built-in SSL/HTTPS
- ✅ Easy environment variable management
- ✅ Supports WebSocket connections
- ✅ Docker support (optional)
- ✅ Auto-scaling available

#### Notes:
- Free tier may spin down after inactivity (takes ~30 seconds to wake up)
- For production with zero downtime, upgrade to paid tier ($7/month)

---

### Option 2: Streamlit Community Cloud (FREE & BEST)

**Perfect for Streamlit apps!**

1. **Push your code to GitHub** (Already done! ✅)

2. **Go to Streamlit Community Cloud**
   - Visit: https://streamlit.io/cloud
   - Sign in with your GitHub account

3. **Deploy your app**
   - Click "New app"
   - Select your repository: `sagar-goyal162005/AI-powered-Interview-Coach`
   - Main file path: `hack.py`
   - Click "Deploy"

4. **Add Environment Variables**
   - In the app settings, add:
     - `OPENAI_API_KEY`: Your OpenAI API key
     - `FREEPIK_API_KEY`: Your Freepik API key

5. **Your app will be live at**: `https://[your-app-name].streamlit.app`

---

### Option 3: Heroku (Recommended for Full Features)

1. **Create a Procfile**
   ```
   web: streamlit run hack.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Create runtime.txt**
   ```
   python-3.11.0
   ```

3. **Deploy to Heroku**
   ```bash
   heroku login
   heroku create your-app-name
   git push heroku main
   heroku config:set OPENAI_API_KEY=your_key
   heroku config:set FREEPIK_API_KEY=your_key
   ```

### Option 3: AWS EC2 or Azure VM

For full control and all features working:

1. **Launch an EC2 instance or Azure VM**
2. **SSH into the server**
3. **Clone your repository**
   ```bash
   git clone https://github.com/sagar-goyal162005/AI-powered-Interview-Coach.git
   cd AI-powered-Interview-Coach
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set environment variables**
   ```bash
   export OPENAI_API_KEY=your_key
   export FREEPIK_API_KEY=your_key
   ```

6. **Run the app**
   ```bash
   streamlit run hack.py --server.port=80 --server.address=0.0.0.0
   ```

### Option 4: Railway.app (Easy & Modern)

1. **Go to Railway.app**
   - Visit: https://railway.app
   - Sign in with GitHub

2. **Create New Project**
   - Select "Deploy from GitHub repo"
   - Choose your repository

3. **Add Environment Variables**
   - `OPENAI_API_KEY`
   - `FREEPIK_API_KEY`

4. **Railway will auto-detect and deploy your Streamlit app**

### Option 5: Google Cloud Run

1. **Create a Dockerfile**
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 8080
   CMD streamlit run hack.py --server.port=8080 --server.address=0.0.0.0
   ```

2. **Deploy to Cloud Run**
   ```bash
   gcloud run deploy ai-interview-coach --source . --platform managed
   ```

## 🎯 Recommended: Streamlit Community Cloud

For your project, I **strongly recommend Streamlit Community Cloud** because:
- ✅ FREE hosting
- ✅ Built specifically for Streamlit apps
- ✅ Direct GitHub integration
- ✅ Easy environment variable management
- ✅ Automatic SSL/HTTPS
- ✅ All Streamlit features work perfectly
- ✅ WebRTC support for video features
- ✅ Simple one-click deployment

## 🔒 Security Notes

Before deploying:
1. ✅ Never commit `.env` file (already in .gitignore)
2. ✅ Add API keys via platform's environment variables
3. ✅ Use strong passwords for user accounts
4. ✅ Consider adding rate limiting for API calls

## 📊 Performance Considerations

- Video generation may take time (30-60 seconds)
- Consider adding loading indicators (already implemented)
- For production, consider implementing queue system for video generation
- Monitor API usage to stay within limits

## 🐛 Troubleshooting

**If deployment fails:**
1. Check Python version compatibility
2. Verify all dependencies in requirements.txt
3. Ensure environment variables are set correctly
4. Check logs for specific error messages

**For Streamlit Community Cloud:**
- Check the app logs in the dashboard
- Ensure your repository is public or grant access
- Verify requirements.txt has all dependencies

## 📞 Need Help?

- Streamlit Community: https://discuss.streamlit.io/
- Streamlit Docs: https://docs.streamlit.io/
- Your GitHub Issues: https://github.com/sagar-goyal162005/AI-powered-Interview-Coach/issues

---

**Start with Streamlit Community Cloud - it's the easiest and best option for this project!** 🚀
