#!/usr/bin/env python3
"""
PythonAnywhere Disk Space Optimization Script
=============================================

This script helps optimize disk space usage on PythonAnywhere free accounts.
"""

import os
import shutil
import subprocess
import sys

def print_banner():
    """Print the optimization banner."""
    print("=" * 60)
    print("💾 PythonAnywhere Disk Space Optimizer")
    print("=" * 60)
    print("Optimizing your OS Study Bot for free account limits")
    print("=" * 60)

def check_disk_usage():
    """Check current disk usage."""
    try:
        result = subprocess.run(['df', '-h'], capture_output=True, text=True)
        print("📊 Current disk usage:")
        print(result.stdout)
        return True
    except Exception as e:
        print(f"⚠️  Could not check disk usage: {e}")
        return False

def clean_python_cache():
    """Clean Python cache files."""
    print("\n🧹 Cleaning Python cache files...")
    
    cache_dirs = [
        '__pycache__',
        '.pytest_cache',
        '.mypy_cache',
        '.coverage',
        'htmlcov'
    ]
    
    cleaned = 0
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                shutil.rmtree(cache_dir)
                print(f"✅ Removed {cache_dir}/")
                cleaned += 1
            except Exception as e:
                print(f"⚠️  Could not remove {cache_dir}/: {e}")
    
    # Clean .pyc files
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.pyc'):
                try:
                    os.remove(os.path.join(root, file))
                    cleaned += 1
                except Exception as e:
                    pass
    
    print(f"✅ Cleaned {cleaned} cache files/directories")
    return cleaned

def remove_unnecessary_files():
    """Remove unnecessary files for deployment."""
    print("\n🗑️  Removing unnecessary files...")
    
    files_to_remove = [
        'os.zip',
        'ml_training.log',
        'test_*.py',
        '*.log',
        '.DS_Store',
        'Thumbs.db'
    ]
    
    removed = 0
    for pattern in files_to_remove:
        if '*' in pattern:
            # Handle wildcard patterns
            import glob
            for file in glob.glob(pattern):
                try:
                    os.remove(file)
                    print(f"✅ Removed {file}")
                    removed += 1
                except Exception as e:
                    pass
        else:
            if os.path.exists(pattern):
                try:
                    os.remove(pattern)
                    print(f"✅ Removed {pattern}")
                    removed += 1
                except Exception as e:
                    pass
    
    print(f"✅ Removed {removed} unnecessary files")
    return removed

def optimize_virtual_environment():
    """Optimize virtual environment."""
    print("\n🔧 Optimizing virtual environment...")
    
    if not os.path.exists('venv'):
        print("❌ Virtual environment not found!")
        return False
    
    # Remove unnecessary packages from venv
    venv_packages_to_remove = [
        'pip',
        'setuptools',
        'wheel'
    ]
    
    try:
        for package in venv_packages_to_remove:
            subprocess.run(['venv/bin/pip', 'uninstall', '-y', package], 
                         capture_output=True)
        print("✅ Optimized virtual environment")
        return True
    except Exception as e:
        print(f"⚠️  Could not optimize venv: {e}")
        return False

def create_lightweight_app():
    """Create a lightweight version of the app."""
    print("\n⚡ Creating lightweight app version...")
    
    # Create a minimal app.py for deployment
    lightweight_app = '''#!/usr/bin/env python3
"""
Lightweight OS Study Bot for PythonAnywhere
"""

from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import os
import json

app = Flask(__name__)

# Load API keys
try:
    with open('api_keys_config.json', 'r') as f:
        config = json.load(f)
        api_key = config['gemini_keys'][0]['api_key']
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
except Exception as e:
    print(f"API key error: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    if not model:
        return jsonify({"ok": False, "error": "AI model not available"})
    
    try:
        data = request.get_json()
        question = data.get('query', '')
        
        response = model.generate_content(question)
        return jsonify({
            "ok": True,
            "answers": [{
                "type": "normal",
                "answer": response.text,
                "source": "gemini"
            }]
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

if __name__ == '__main__':
    app.run(debug=False)
'''
    
    with open('app_lightweight.py', 'w') as f:
        f.write(lightweight_app)
    
    print("✅ Created lightweight app version")
    return True

def print_optimization_tips():
    """Print additional optimization tips."""
    print("\n💡 Additional Optimization Tips:")
    print("=" * 60)
    print("1. Use minimal requirements:")
    print("   pip install -r requirements_minimal.txt")
    print()
    print("2. Remove large files:")
    print("   rm -rf data/ models/ uploads/ history/")
    print()
    print("3. Use lightweight app:")
    print("   # Update wsgi.py to use app_lightweight.py")
    print()
    print("4. Consider upgrading to Hacker plan ($5/month)")
    print("   - 3GB disk space")
    print("   - Custom domains")
    print("   - More CPU time")
    print()
    print("5. Monitor disk usage:")
    print("   du -sh *")
    print("=" * 60)

def main():
    """Main optimization function."""
    print_banner()
    
    # Check disk usage
    check_disk_usage()
    
    # Clean cache files
    clean_python_cache()
    
    # Remove unnecessary files
    remove_unnecessary_files()
    
    # Optimize virtual environment
    optimize_virtual_environment()
    
    # Create lightweight app
    create_lightweight_app()
    
    # Print tips
    print_optimization_tips()
    
    print("\n🎉 Optimization complete!")
    print("Try installing dependencies again with:")
    print("pip install -r requirements_minimal.txt")

if __name__ == "__main__":
    main()
