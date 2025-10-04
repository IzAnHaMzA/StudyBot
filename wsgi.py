#!/usr/bin/env python3
"""
WSGI entry point for PythonAnywhere deployment
"""

import sys
import os

# Add the project directory to the Python path
project_home = '/home/yourusername/StudyBot'  # Replace 'yourusername' with your PythonAnywhere username
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Change to the project directory
os.chdir(project_home)

# Import the Flask app
from app2 import app as application

# Configure the application for production
if __name__ == "__main__":
    application.run()
