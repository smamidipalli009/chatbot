#!/bin/bash

set -e  # Exit on any error

echo "[INFO] Updating system packages..."
sudo yum update -y

echo "[INFO] Installing Python and required packages..."
dnf install python3.13
#sudo yum install -y python3 python3-pip git

sudo dnf install python3.13 python3.13-pip python3.13-pip-wheel nginx
python3.13 -m venv venv_py313
source venv_py313/bin/activate
pip3.13 install --upgrade pip setuptools wheel

pip3.13 install -r requirements.txt

print_status "Getting EC2 public IP..."
EC2_PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "localhost")
print_status "EC2 Public IP: $EC2_PUBLIC_IP"


echo "[INFO] Starting Flask app with Waitress on port 9000..."
nohup venv_py313/bin/waitress-serve --host=0.0.0.0 --port=9000 run_chatbot:app > chatbot.log 2>&1 &

echo "[SUCCESS] Chatbot is running in the background. Logs: chatbot.log"


sudo tee /etc/nginx/conf.d/medical-chatbot.conf << EOF
server {
    listen 80;
    server_name $EC2_PUBLIC_IP;
    
    location / {
        proxy_pass http://127.0.0.1:9000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 120s;
        proxy_connect_timeout 120s;
    }
}
EOF

systemctl start nginx
