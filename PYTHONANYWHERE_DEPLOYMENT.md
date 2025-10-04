# 🚀 PythonAnywhere Deployment Guide

## Quick Deploy to PythonAnywhere

Your OS Study Bot is fully compatible with PythonAnywhere! Follow these steps to deploy it.

## 📋 Prerequisites

- PythonAnywhere account (free account works)
- Google Gemini API key
- OpenAI API key (optional)

## 🛠️ Deployment Steps

### 1. Sign Up for PythonAnywhere
- Go to [pythonanywhere.com](https://www.pythonanywhere.com/)
- Create a free account
- Verify your email

### 2. Open Bash Console
- Log into your PythonAnywhere dashboard
- Click on "Consoles" tab
- Click "Bash" to open a new console

### 3. Clone Your Repository
```bash
git clone https://github.com/IzAnHaMzA/StudyBot.git
cd StudyBot
```

### 4. Set Up Virtual Environment
```bash
python3.10 -m venv venv
source venv/bin/activate
```

### 5. Install Dependencies
```bash
pip install -r requirements.txt
```

### 6. Configure API Keys
```bash
nano api_keys_config.json
```
Add your API keys:
```json
{
  "gemini_keys": [
    {
      "name": "Primary",
      "api_key": "your_actual_gemini_api_key_here"
    }
  ],
  "openai_keys": [
    {
      "name": "Primary", 
      "api_key": "your_actual_openai_api_key_here"
    }
  ]
}
```

### 7. Update WSGI Configuration
```bash
nano wsgi.py
```
Replace `yourusername` with your actual PythonAnywhere username:
```python
project_home = '/home/YOUR_USERNAME/StudyBot'
```

### 8. Configure Web App
1. Go to "Web" tab in PythonAnywhere dashboard
2. Click "Add a new web app"
3. Choose "Manual configuration"
4. Select Python 3.10
5. Configure paths:
   - **Source code**: `/home/YOUR_USERNAME/StudyBot`
   - **Working directory**: `/home/YOUR_USERNAME/StudyBot`
   - **WSGI file**: `/home/YOUR_USERNAME/StudyBot/wsgi.py`

### 9. Deploy
Click "Reload" button to deploy your application.

## 🌐 Access Your App

Your OS Study Bot will be available at:
```
https://YOUR_USERNAME.pythonanywhere.com
```

## 🔧 Configuration Options

### Environment Variables (Recommended)
Instead of using the JSON file, you can use environment variables:

```bash
export GEMINI_API_KEY="your_gemini_key"
export OPENAI_API_KEY="your_openai_key"
```

### Custom Domain (Paid Plans)
- Hacker plan ($5/month): Custom domains
- Web Developer plan ($12/month): More resources

## 📊 Features That Work on PythonAnywhere

✅ **AI-Powered Q&A** - Full Gemini integration  
✅ **Interactive Test Center** - Complete grading system  
✅ **Q&A Caching** - Save and organize content  
✅ **Modern UI** - Responsive neural interface  
✅ **API Key Rotation** - Multiple provider support  
✅ **Page Tracking** - Advanced content mapping  

## 🚨 Important Notes

### Free Account Limitations
- Apps sleep after 3 months of inactivity
- Limited CPU time per day
- No custom domains
- 512MB disk space

### Production Recommendations
- Upgrade to Hacker plan for better performance
- Use environment variables for API keys
- Set up regular backups
- Monitor usage and performance

## 🔍 Troubleshooting

### Common Issues

**Import Error:**
```bash
# Make sure you're in the right directory
cd /home/YOUR_USERNAME/StudyBot
# Check if virtual environment is activated
source venv/bin/activate
```

**API Key Issues:**
```bash
# Check if API keys are properly configured
cat api_keys_config.json
```

**WSGI Errors:**
- Check the error log in Web tab
- Ensure wsgi.py has correct paths
- Verify all dependencies are installed

### Getting Help
- Check PythonAnywhere documentation
- Visit the community forum
- Contact support for paid plans

## 🎉 Success!

Once deployed, your OS Study Bot will be live on the web with:
- Full AI-powered functionality
- Interactive learning features
- Professional web interface
- Accessible from anywhere

**Your AI-powered OS Study Bot is now live on the internet! 🌟**
