#!/bin/bash

# Enhanced Nginx Setup Script with Comprehensive Logging
# Configures nginx for Medical App deployment

LOG_FILE="nginx_setup.log"
DATE_FORMAT="+%Y-%m-%d %H:%M:%S"

# Logging functions
log_info() {
    echo "$(date "$DATE_FORMAT") [INFO] $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo "$(date "$DATE_FORMAT") [ERROR] $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo "$(date "$DATE_FORMAT") [SUCCESS] $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo "$(date "$DATE_FORMAT") [WARNING] $1" | tee -a "$LOG_FILE"
}

# Error handling function
check_command() {
    if [ $? -eq 0 ]; then
        log_success "$1"
        return 0
    else
        log_error "$1 failed"
        return 1
    fi
}

# Start nginx setup
log_info "=== Starting Nginx Configuration Setup ==="
log_info "Setup started by user: $(whoami)"
log_info "Current working directory: $(pwd)"
log_info "System: $(uname -a)"

echo "Setting up nginx with custom configuration..."

# Check if nginx is installed
log_info "Verifying nginx installation..."
if command -v nginx >/dev/null 2>&1; then
    NGINX_VERSION=$(nginx -v 2>&1)
    log_success "Nginx is installed: $NGINX_VERSION"
else
    log_error "Nginx is not installed. Please install nginx first."
    exit 1
fi

# Check if custom nginx.conf exists
log_info "Checking for custom nginx.conf file..."
if [ -f "nginx.conf" ]; then
    log_success "Custom nginx.conf found in current directory"
    log_info "File size: $(du -h nginx.conf | cut -f1)"
    log_info "File permissions: $(ls -la nginx.conf | awk '{print $1, $3, $4}')"
else
    log_error "Custom nginx.conf not found in current directory"
    log_info "Expected file: $(pwd)/nginx.conf"
    exit 1
fi

# Check current nginx status
log_info "Checking current nginx status..."
if systemctl is-active --quiet nginx; then
    log_info "Nginx is currently running"
    NGINX_PID=$(systemctl show nginx --property=MainPID --value)
    log_info "Current nginx PID: $NGINX_PID"
else
    log_info "Nginx is not currently running"
fi

# Backup original nginx.conf
log_info "Creating backup of original nginx configuration..."
if [ -f "/etc/nginx/nginx.conf" ]; then
    sudo cp /etc/nginx/nginx.conf /etc/nginx/nginx.conf.backup >> "$LOG_FILE" 2>&1
    if check_command "Original nginx.conf backed up to nginx.conf.backup"; then
        BACKUP_SIZE=$(sudo du -h /etc/nginx/nginx.conf.backup | cut -f1)
        log_info "Backup file size: $BACKUP_SIZE"
    fi
else
    log_warning "Original /etc/nginx/nginx.conf not found"
fi

# Copy custom nginx.conf
log_info "Installing custom nginx configuration..."
sudo cp nginx.conf /etc/nginx/nginx.conf >> "$LOG_FILE" 2>&1
check_command "Custom nginx.conf copied to /etc/nginx/"

# Verify file was copied correctly
log_info "Verifying configuration file copy..."
if [ -f "/etc/nginx/nginx.conf" ]; then
    NEW_CONFIG_SIZE=$(sudo du -h /etc/nginx/nginx.conf | cut -f1)
    log_success "New configuration file size: $NEW_CONFIG_SIZE"
else
    log_error "Failed to copy configuration file"
    exit 1
fi

# Test nginx configuration
log_info "Testing nginx configuration syntax..."
sudo nginx -t >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    log_success "✓ Configuration syntax is valid"
    
    # Restart nginx
    log_info "Restarting nginx service..."
    sudo systemctl restart nginx >> "$LOG_FILE" 2>&1
    if check_command "Nginx service restarted"; then
        sleep 2
        
        # Verify nginx is running
        if systemctl is-active --quiet nginx; then
            NEW_NGINX_PID=$(systemctl show nginx --property=MainPID --value)
            log_success "Nginx is running with PID: $NEW_NGINX_PID"
        else
            log_error "Nginx failed to start after restart"
            exit 1
        fi
    else
        log_error "Failed to restart nginx"
        exit 1
    fi
    
    # Enable nginx to start on boot
    log_info "Enabling nginx to start on boot..."
    sudo systemctl enable nginx >> "$LOG_FILE" 2>&1
    check_command "Nginx enabled for auto-start"
    
    log_success "✓ Nginx setup complete!"
    log_success "✓ Medical app available on port 80 -> forwarding to port 9090"
    
    # Show nginx status
    log_info "Current nginx service status:"
    sudo systemctl status nginx --no-pager -l >> "$LOG_FILE" 2>&1
    
    # Test port accessibility
    log_info "Testing port accessibility..."
    if ss -tuln | grep -q ":80 "; then
        log_success "Nginx is listening on port 80"
    elif lsof -i :80 >/dev/null 2>&1; then
        log_success "Port 80 is active (verified with lsof)"
    else
        log_warning "Cannot verify port 80 is listening"
    fi
    
    # Test configuration with curl if available
    if command -v curl >/dev/null 2>&1; then
        log_info "Testing nginx response..."
        if curl -s -o /dev/null -w "%{http_code}" http://localhost/ | grep -q "200\|301\|302"; then
            log_success "Nginx is responding to HTTP requests"
        else
            log_warning "Nginx may not be responding correctly to HTTP requests"
        fi
    fi
    
else
    log_error "✗ Configuration has syntax errors!"
    echo "Configuration validation failed. Check the logs for details:"
    sudo nginx -t
    
    log_info "Restoring backup configuration..."
    if [ -f "/etc/nginx/nginx.conf.backup" ]; then
        sudo cp /etc/nginx/nginx.conf.backup /etc/nginx/nginx.conf >> "$LOG_FILE" 2>&1
        check_command "Backup configuration restored"
        
        # Test restored configuration
        log_info "Testing restored configuration..."
        sudo nginx -t >> "$LOG_FILE" 2>&1
        if [ $? -eq 0 ]; then
            log_success "Restored configuration is valid"
            sudo systemctl restart nginx >> "$LOG_FILE" 2>&1
            check_command "Nginx restarted with restored configuration"
        else
            log_error "Even the backup configuration has errors!"
        fi
    else
        log_error "No backup file found to restore"
    fi
    
    exit 1
fi

# Final summary
log_info "=== Nginx Setup Summary ==="
log_info "Configuration file: /etc/nginx/nginx.conf"
log_info "Backup file: /etc/nginx/nginx.conf.backup"
log_info "Service status: $(systemctl is-active nginx)"
log_info "Auto-start enabled: $(systemctl is-enabled nginx)"
log_info "Listening ports: $(ss -tuln | grep nginx || echo 'Use: ss -tuln | grep :80')"
log_success "=== Nginx configuration completed successfully ==="
