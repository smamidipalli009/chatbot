#!/bin/bash

# Simple script to setup nginx config

echo "Setting up nginx configuration..."

# Copy config to sites-available
sudo cp config/nginx.conf /etc/nginx/sites-available/medical-app

# Enable the site
sudo ln -sf /etc/nginx/sites-available/medical-app /etc/nginx/sites-enabled/

# Remove default site
sudo rm -f /etc/nginx/sites-enabled/default

# Test config
sudo nginx -t

if [ $? -eq 0 ]; then
    echo "Config is valid, restarting nginx..."
    sudo systemctl restart nginx
    echo "Nginx setup complete!"
    echo "Medical app will be available on port 80"
else
    echo "Nginx config has errors!"
    exit 1
fi
