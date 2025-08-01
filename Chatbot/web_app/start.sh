#!/bin/bash

echo "Starting Medical App..."

# Kill existing gunicorn processes
echo "Killing existing processes..."
pkill -f gunicorn 2>/dev/null || true
sleep 2

# Start gunicorn and save PID
echo "Starting Gunicorn..."
nohup ../deploy_env/bin/gunicorn --config config/gunicorn_config.py app:app &
GUNICORN_PID=$!

# Save PID to file
echo $GUNICORN_PID > gunicorn.pid
echo "Gunicorn started with PID: $GUNICORN_PID"

echo "App started successfully!"
