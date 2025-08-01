#!/bin/bash

# Fixed start script for your directory structure
# Run this from root directory with: bash deploy/start.sh

echo "Starting Medical App..."
echo "======================"

# Create logs directory
mkdir -p logs

# Stop any existing processes
echo "Cleaning up existing processes..."
sudo pkill nginx 2>/dev/null || true
pkill -f gunicorn 2>/dev/null || true

# Remove stale PID files
if [ -f gunicorn.pid ]; then
    echo "Removing stale gunicorn.pid"
    rm gunicorn.pid
fi

if [ -f nginx.pid ]; then
    echo "Removing stale nginx.pid"
    rm nginx.pid
fi

# Wait a moment for processes to fully stop
sleep 2

# Start Gunicorn (from web_app directory)
echo "Starting Gunicorn (Flask app server)..."
cd web_app
../deploy_env/bin/gunicorn --config ../deploy/gunicorn_config.py app:app &
cd ..

# Wait for Gunicorn to start
sleep 3

# Start Nginx
echo "Starting Nginx (web server)..."
NGINX_CONFIG=$(pwd)/deploy/nginx_config
echo "Using config: $NGINX_CONFIG"
if [ -f "$NGINX_CONFIG" ]; then
    sudo nginx -c $NGINX_CONFIG
else
    echo "ERROR: Nginx config file not found: $NGINX_CONFIG"
    exit 1
fi

echo ""
echo "App started successfully!"
echo "Access at: http://localhost"
echo "Logs in: logs/"
echo ""
echo "To stop: bash deploy/stop.sh"
