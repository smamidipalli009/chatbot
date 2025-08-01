#!/bin/bash

# Fixed installation and startup script

echo "Installing and starting Medical App..."

# Install nginx (fixed typo + added sudo)
sudo apt-get update
sudo apt-get install -y nginx

# Create virtual environment
python3 -m venv deploy_env
source deploy_env/bin/activate

# Upgrade pip tools
pip3 install --upgrade pip setuptools wheel  --break-system-packages

# Install requirements (fixed filename)
pip3 install -r requirements_deploy.txt  --break-system-packages

#deactivate
# Make scripts executable
chmod +x web_app/start.sh web_app/stop.sh

# Start the application
cd web_app
bash start.sh

cd ..
pwd 
ls -lthra 
chmod +x nginx_setup.sh
ls -lthra nginx_setup.sh
bash nginx_setup.sh
