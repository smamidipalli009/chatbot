#!/bin/bash

echo "Setting up nginx with custom configuration..."

# Backup original nginx.conf
sudo cp /etc/nginx/nginx.conf /etc/nginx/nginx.conf.backup

# Copy our custom nginx.conf
sudo cp nginx.conf /etc/nginx/nginx.conf

# Test configuration
sudo nginx -t

if [ $? -eq 0 ]; then
    echo "✓ Configuration is valid"

    # Restart and enable nginx
    sudo systemctl restart nginx
    sudo systemctl enable nginx

    echo "✓ Nginx setup complete!"
    echo "✓ Medical app available on port 80 -> forwarding to port 9090"

    # Show status
    sudo systemctl status nginx --no-pager -l

else
    echo "✗ Configuration has errors!"
    echo "Restoring backup..."
    sudo cp /etc/nginx/nginx.conf.backup /etc/nginx/nginx.conf
    exit 1
fi
