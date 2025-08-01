#!/bin/bash

# Fixed installation and startup script

echo "Installing and starting Medical App..."

# Install nginx (fixed typo + added sudo)
sudo apt-get update
sudo apt-get install -y nginx

# Create virtual environment
python3 -m venv deploy_env
source deploy_env/Scripts/activate

# Upgrade pip tools
pip3 install --upgrade pip setuptools wheel  --break-system-packages

# Install requirements (fixed filename)
pip3 install -r deploy/requirements_deploy.txt  --break-system-packages

deactivate
# Make scripts executable
chmod +x deploy/start.sh deploy/stop.sh

# Start the application
bash deploy/start.sh
