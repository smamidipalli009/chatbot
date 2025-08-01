# Gunicorn configuration for your directory structure
# This file should be named gunicorn_config.py (not gunicorn.conf.py)

# Server socket
bind = "0.0.0.0:9090"
backlog = 2048

# Worker processes
workers = 2
worker_class = "sync"
worker_connections = 1000
timeout = 30
keepalive = 2

# Logging (paths relative to where gunicorn runs - from web_app/ directory)
loglevel = "info"
accesslog = "../logs/gunicorn_access.log"
errorlog = "../logs/gunicorn_error.log"

# Process naming
proc_name = 'medical_app'

# Server mechanics
daemon = False
pidfile = '../gunicorn.pid'
user = None
group = None
tmp_upload_dir = None

# SSL (disabled for basic lab setup)
keyfile = None
certfile = None
