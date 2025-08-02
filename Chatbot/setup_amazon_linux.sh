#!/bin/bash

# Enhanced Medical App Installation and Startup Script
# Added comprehensive logging and error handling

LOG_FILE="deployment.log"
DATE_FORMAT="+%Y-%m-%d %H:%M:%S"

# Logging function
log_info() {
    echo "$(date "$DATE_FORMAT") [INFO] $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo "$(date "$DATE_FORMAT") [ERROR] $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo "$(date "$DATE_FORMAT") [SUCCESS] $1" | tee -a "$LOG_FILE"
}

# Error handling function
check_command() {
    if [ $? -eq 0 ]; then
        log_success "$1"
    else
        log_error "$1 failed"
        exit 1
    fi
}

# Start deployment
log_info "=== Starting Medical App Deployment ==="
log_info "Deployment started by user: $(whoami)"
log_info "System: $(uname -a)"

# Update system packages
log_info "Updating system packages..."
sudo yum update -y >> "$LOG_FILE" 2>&1
check_command "System update completed"

# Install required packages
log_info "Installing Python 3.13, pip, and nginx..."
sudo dnf install python3.13 python3.13-pip python3.13-pip-wheel nginx -y >> "$LOG_FILE" 2>&1
check_command "Package installation completed"

# Verify installations
log_info "Verifying installations..."
python3.13 --version >> "$LOG_FILE" 2>&1
check_command "Python 3.13 verification"

nginx -v >> "$LOG_FILE" 2>&1
check_command "Nginx verification"

# Create virtual environment
log_info "Creating Python virtual environment..."
python3.13 -m venv deploy_env >> "$LOG_FILE" 2>&1
check_command "Virtual environment created"

# Activate virtual environment
log_info "Activating virtual environment..."
source deploy_env/bin/activate
check_command "Virtual environment activated"

# Upgrade pip tools
log_info "Upgrading pip, setuptools, and wheel..."
pip3.13 install --upgrade pip setuptools wheel --break-system-packages >> "$LOG_FILE" 2>&1
check_command "Pip tools upgraded"

# Install application requirements
log_info "Installing application requirements from requirements_deploy.txt..."
if [ -f "requirements_deploy.txt" ]; then
    pip3.13 install -r requirements_deploy.txt --break-system-packages >> "$LOG_FILE" 2>&1
    check_command "Application requirements installed"
else
    log_error "requirements_deploy.txt not found"
    exit 1
fi

# Start Gunicorn server
log_info "Starting Gunicorn server on port 9090..."
nohup deploy_env/bin/gunicorn --bind 0.0.0.0:9090 --workers 2 run_chatbot:app > gunicorn.log 2>&1 &
GUNICORN_PID=$!

# Verify Gunicorn started successfully
sleep 3
if ps -p $GUNICORN_PID > /dev/null; then
    log_success "Gunicorn server started successfully (PID: $GUNICORN_PID)"
    echo $GUNICORN_PID > gunicorn.pid
    log_info "Process ID saved to gunicorn.pid"
else
    log_error "Failed to start Gunicorn server"
    exit 1
fi

# Check if port is listening (using ss instead of netstat)
log_info "Verifying server is listening on port 9090..."
sleep 2
if ss -tuln | grep -q ":9090 "; then
    log_success "Server is listening on port 9090"
elif lsof -i :9090 >/dev/null 2>&1; then
    log_success "Server is listening on port 9090 (verified with lsof)"
elif curl -s http://localhost:9090 >/dev/null 2>&1; then
    log_success "Server is responding on port 9090 (verified with curl)"
else
    log_error "Cannot verify server is listening on port 9090"
    log_info "You can manually check with: ss -tuln | grep 9090"
fi

# Final status
log_info "=== Deployment Summary ==="
log_info "Application: Medical App"
log_info "Server: Gunicorn"
log_info "Port: 9090"
log_info "Workers: 2"
log_info "Process ID: $GUNICORN_PID"
log_info "Log files: deployment.log, gunicorn.log"
log_success "=== Medical App deployment completed successfully ==="
