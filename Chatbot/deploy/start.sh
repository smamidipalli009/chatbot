#!/bin/bash

# Simple start script - Gunicorn + Nginx only

echo "Starting Medical App..."
echo "======================"

# Create logs directory
mkdir -p ../logs

# Stop any existing processes
echo "Cleaning up existing processes..."
sudo pkill nginx 2>/dev/null || true
pkill -f gunicorn 2>/dev/null || true

# Start Gunicorn
echo "Starting Gunicorn (Flask app server)..."
cd ../web_app
gunicorn --config ../deploy/gunicorn.conf.py app:app &
cd ../deploy

# Wait for Gunicorn to start
sleep 3

# Start Nginx
echo "Starting Nginx (web server)..."
sudo nginx -c $(pwd)/nginx.conf

echo ""
echo "🚀 App started successfully!"
echo "📱 Access at: http://localhost"
echo "💾 Logs in: ../logs/"
echo ""
echo "To stop: ./stop.sh"