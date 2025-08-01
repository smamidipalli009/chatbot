import sys
import os
import logging
import warnings

# Suppress scikit-learn version warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Import Flask app from web_app directory
from web_app.app import app

def initialize_app():
    """Initialize the application for production use"""
    print("="*50)
    print("Initializing Medical Chatbot App")
    print("="*50)

    # App initialization is handled automatically by app.py
    print("[INFO] App initialized successfully")
    print("[INFO] Ready for Gunicorn")

    return app

# Initialize the app when module is imported
initialize_app()

# This is what gunicorn will use: run_chatbot:app
if __name__ == '__main__':
    # Only runs when called directly (not via gunicorn)
    print("[INFO] Running in development mode")
    print("[INFO] For production, use: gunicorn run_chatbot:app")
    app.run(debug=True, host='0.0.0.0', port=9090)
