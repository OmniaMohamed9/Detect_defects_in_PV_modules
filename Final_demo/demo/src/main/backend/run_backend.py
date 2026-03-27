#!/usr/bin/env python3
"""
Backend server runner for Camera PDF Generator
"""
import os
import sys

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

from app import app

if __name__ == "__main__":
    print("Starting Camera PDF Generator Backend...")
    print("Backend will be available at: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server")
    
    # Run the Flask app
    app.run(host="127.0.0.1", port=5000, debug=True)

