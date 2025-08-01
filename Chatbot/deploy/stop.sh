#!/bin/bash

# Fixed stop script for your directory structure
# Run this from root directory with: bash deploy/stop.sh

echo "Stopping Medical App..."
echo "======================"

# Stop Nginx
echo "Stopping Nginx..."
sudo pkill nginx 2>/dev/null || true

# Stop Gunicorn
echo "Stopping Gunicorn..."
pkill -f gunicorn 2>/dev/null || true

# Clean up PID files
if [ -f nginx.pid ]; then
    rm nginx.pid
fi

if [ -f gunicorn.pid ]; then
    rm gunicorn.pid
fi

echo ""
echo "App stopped successfully!"
