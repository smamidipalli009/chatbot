
#!/bin/bash

# Fixed installation and startup script

echo "Installing and starting Medical App..."

# Install nginx (fixed typo + added sudo)
#sudo apt-get update
#sudo apt-get install -y nginx

sudo yum update -y
dnf install python3.13

# Create virtual environment
python3 -m venv deploy_env
source deploy_env/bin/activate

# Upgrade pip tools
pip3.13 install --upgrade pip setuptools wheel  --break-system-packages

# Install requirements (fixed filename)
pip3.13 install -r requirements_deploy.txt  --break-system-packages

nohup deploy_env/bin/gunicorn --bind 0.0.0.0:9090 --workers 2 run_chatbot:app > gunicorn.log 2>&1 &

#deactivate
# Make scripts executable
#chmod +x web_app/start.sh web_app/stop.sh

# Start the application
#cd web_app

#setsid nohup ../deploy_env/bin/gunicorn --bind 0.0.0.0:9090 --workers 2 app:app > gunicorn.log 2>&1 &
#echo $! > gunicorn.pid

#chmod +x nginx_setup.sh
#bash nginx_setup.sh
