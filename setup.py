#!/usr/bin/env python3
"""
OS Study Bot Setup Script
=========================

This script helps you set up the OS Study Bot quickly and easily.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def print_banner():
    """Print the setup banner."""
    print("=" * 60)
    print("🚀 OS Study Bot - Setup Script")
    print("=" * 60)
    print("AI-Powered Operating Systems Learning Assistant")
    print("=" * 60)

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required!")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def create_virtual_environment():
    """Create a virtual environment."""
    print("\n📦 Creating virtual environment...")
    try:
        subprocess.run([sys.executable, "-m", "venv", "env"], check=True)
        print("✅ Virtual environment created successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to create virtual environment!")
        return False

def get_activation_command():
    """Get the appropriate activation command for the OS."""
    if os.name == 'nt':  # Windows
        return "env\\Scripts\\activate"
    else:  # Unix/Linux/macOS
        return "source env/bin/activate"

def install_dependencies():
    """Install required dependencies."""
    print("\n📚 Installing dependencies...")
    activation_cmd = get_activation_command()
    
    if os.name == 'nt':  # Windows
        pip_cmd = "env\\Scripts\\pip"
    else:  # Unix/Linux/macOS
        pip_cmd = "env/bin/pip"
    
    try:
        subprocess.run([pip_cmd, "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies!")
        return False

def setup_api_keys():
    """Set up API keys configuration."""
    print("\n🔑 Setting up API keys...")
    
    if not os.path.exists("api_keys_config.json.example"):
        print("❌ API keys example file not found!")
        return False
    
    if not os.path.exists("api_keys_config.json"):
        shutil.copy("api_keys_config.json.example", "api_keys_config.json")
        print("✅ API keys configuration file created!")
        print("⚠️  Please edit 'api_keys_config.json' and add your API keys:")
        print("   - Google Gemini API key")
        print("   - OpenAI API key (optional)")
        return True
    else:
        print("✅ API keys configuration file already exists!")
        return True

def print_next_steps():
    """Print the next steps for the user."""
    activation_cmd = get_activation_command()
    
    print("\n🎉 Setup Complete!")
    print("=" * 60)
    print("Next steps:")
    print("1. Activate the virtual environment:")
    print(f"   {activation_cmd}")
    print("\n2. Add your API keys to 'api_keys_config.json'")
    print("\n3. Run the application:")
    print("   python app2.py")
    print("\n4. Open your browser and go to:")
    print("   http://localhost:5001")
    print("=" * 60)
    print("📖 For more information, visit:")
    print("   https://github.com/IzAnHaMzA/StudyBot")
    print("=" * 60)

def main():
    """Main setup function."""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Setup API keys
    if not setup_api_keys():
        sys.exit(1)
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()
