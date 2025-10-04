#!/usr/bin/env python3
"""
PythonAnywhere Setup Script for OS Study Bot
============================================

This script helps you deploy the OS Study Bot to PythonAnywhere.
"""

import os
import sys

def print_banner():
    """Print the setup banner."""
    print("=" * 60)
    print("🚀 OS Study Bot - PythonAnywhere Deployment")
    print("=" * 60)
    print("AI-Powered Operating Systems Learning Assistant")
    print("=" * 60)

def check_files():
    """Check if required files exist."""
    required_files = [
        'app2.py',
        'requirements.txt',
        'wsgi.py',
        'api_keys_config.json.example'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ All required files found!")
    return True

def create_api_config():
    """Create API configuration file."""
    if not os.path.exists('api_keys_config.json'):
        if os.path.exists('api_keys_config.json.example'):
            import shutil
            shutil.copy('api_keys_config.json.example', 'api_keys_config.json')
            print("✅ API configuration file created from example!")
            print("⚠️  Please edit 'api_keys_config.json' and add your API keys")
            return True
        else:
            print("❌ API configuration example file not found!")
            return False
    else:
        print("✅ API configuration file already exists!")
        return True

def print_deployment_steps():
    """Print the deployment steps."""
    print("\n🎯 PythonAnywhere Deployment Steps:")
    print("=" * 60)
    print("1. Sign up for PythonAnywhere (free account available)")
    print("   https://www.pythonanywhere.com/")
    print()
    print("2. Open a Bash console in PythonAnywhere")
    print()
    print("3. Clone your repository:")
    print("   git clone https://github.com/IzAnHaMzA/StudyBot.git")
    print("   cd StudyBot")
    print()
    print("4. Create a virtual environment:")
    print("   python3.10 -m venv venv")
    print("   source venv/bin/activate")
    print()
    print("5. Install dependencies:")
    print("   pip install -r requirements.txt")
    print()
    print("6. Configure API keys:")
    print("   nano api_keys_config.json")
    print("   # Add your Google Gemini and OpenAI API keys")
    print()
    print("7. Update wsgi.py:")
    print("   nano wsgi.py")
    print("   # Replace 'yourusername' with your PythonAnywhere username")
    print()
    print("8. Go to Web tab in PythonAnywhere dashboard")
    print("9. Click 'Add a new web app'")
    print("10. Choose 'Manual configuration'")
    print("11. Select Python 3.10")
    print("12. Set Source code: /home/yourusername/StudyBot")
    print("13. Set Working directory: /home/yourusername/StudyBot")
    print("14. Set WSGI file: /home/yourusername/StudyBot/wsgi.py")
    print("15. Click 'Reload' to deploy")
    print("=" * 60)
    print("🌐 Your app will be available at:")
    print("   https://yourusername.pythonanywhere.com")
    print("=" * 60)

def print_important_notes():
    """Print important deployment notes."""
    print("\n📝 Important Notes:")
    print("=" * 60)
    print("• Free accounts have some limitations:")
    print("  - Apps sleep after 3 months of inactivity")
    print("  - Limited CPU time per day")
    print("  - No custom domains")
    print()
    print("• For production use, consider upgrading to:")
    print("  - Hacker plan ($5/month)")
    print("  - Web Developer plan ($12/month)")
    print()
    print("• API Keys:")
    print("  - Keep your API keys secure")
    print("  - Don't commit them to git")
    print("  - Use environment variables in production")
    print()
    print("• File Paths:")
    print("  - Update wsgi.py with your actual username")
    print("  - Ensure all file paths are correct")
    print("=" * 60)

def main():
    """Main setup function."""
    print_banner()
    
    # Check required files
    if not check_files():
        sys.exit(1)
    
    # Create API configuration
    if not create_api_config():
        sys.exit(1)
    
    # Print deployment steps
    print_deployment_steps()
    
    # Print important notes
    print_important_notes()

if __name__ == "__main__":
    main()
