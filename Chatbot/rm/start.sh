#!/bin/bash

echo "Starting Medical App..."

# Kill existing processes
pkill -f gunicorn 2>/dev/null || true
sleep 2

# Create logs directory
mkdir -p ../logs

# Start gunicorn as daemon (best for automation)
echo "Starting Gunicorn daemon..."
#../deploy_env/bin/gunicorn --config config/gunicorn_config.py --daemon app:app
nohup ../deploy_env/bin/gunicorn --config config/gunicorn_config.py app:app > ../logs/gunicorn.log 2>&1 &

# Wait and get PID
sleep 2
GUNICORN_PID=$(pgrep -f "gunicorn.*app:app")

if [ -n "$GUNICORN_PID" ]; then
    echo $GUNICORN_PID > gunicorn.pid
    echo "Gunicorn started successfully with PID: $GUNICORN_PID"
    echo "App accessible at: http://localhost:9090"
    exit 0
else
    echo "ERROR: Gunicorn failed to start"
    exit 1
fi
