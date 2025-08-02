#!/bin/bash

# Simple stop script for gunicorn
# Run from web_app directory

echo "Stopping Medical App..."

# Kill gunicorn using PID file
if [ -f gunicorn.pid ]; then
    PID=$(cat gunicorn.pid)
    echo "Killing Gunicorn PID: $PID"
    kill $PID 2>/dev/null || true
    rm gunicorn.pid
    echo "Gunicorn stopped"
else
    echo "No PID file found, killing all gunicorn processes..."
    pkill -f gunicorn 2>/dev/null || true
    echo "All gunicorn processes killed"
fi

echo "App stopped successfully!"
