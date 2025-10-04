@echo off
echo ============================================================
echo 🚀 OS Study Bot - Quick Setup (Windows)
echo ============================================================
echo AI-Powered Operating Systems Learning Assistant
echo ============================================================

echo.
echo 📦 Creating virtual environment...
python -m venv env
if errorlevel 1 (
    echo ❌ Failed to create virtual environment!
    pause
    exit /b 1
)

echo ✅ Virtual environment created!

echo.
echo 📚 Installing dependencies...
env\Scripts\pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Failed to install dependencies!
    pause
    exit /b 1
)

echo ✅ Dependencies installed!

echo.
echo 🔑 Setting up API keys...
if not exist api_keys_config.json (
    copy api_keys_config.json.example api_keys_config.json
    echo ✅ API keys configuration file created!
    echo ⚠️  Please edit 'api_keys_config.json' and add your API keys
) else (
    echo ✅ API keys configuration file already exists!
)

echo.
echo 🎉 Setup Complete!
echo ============================================================
echo Next steps:
echo 1. Activate the virtual environment:
echo    env\Scripts\activate
echo.
echo 2. Add your API keys to 'api_keys_config.json'
echo.
echo 3. Run the application:
echo    python app2.py
echo.
echo 4. Open your browser and go to:
echo    http://localhost:5001
echo ============================================================
echo 📖 For more information, visit:
echo    https://github.com/IzAnHaMzA/StudyBot
echo ============================================================
pause
