# Optimized Gunicorn settings for better performance
workers = int(os.environ.get('GUNICORN_PROCESSES', '4'))  # Increased from 3
threads = int(os.environ.get('GUNICORN_THREADS', '2'))    # Increased from 1
bind = os.environ.get('GUNICORN_BIND', '0.0.0.0:8080')
static_folder='static'
forwarded_allow_ips = '*'
secure_scheme_headers = { 'X-Forwarded-Proto': 'https' }
timeout = 120  # Reduced from 240 for faster timeouts
keepalive = 2  # Keep connections alive for better performance
max_requests = 1000  # Restart worker after 1000 requests to prevent memory leaks
max_requests_jitter = 50  # Add jitter to prevent thundering herd
preload_app = True  # Preload application for better startup performance

ROBOT_API = os.environ.get('ROBOT_API', 'http://hub-controller-live.hub-controller.svc.cluster.local:8080/robot')
ROBOT_NAME = os.environ.get('ROBOT_NAME', '<REPLACE_WITH_ROBOT_NAME>')
INFERENCING_API = os.environ.get('INFERENCING_API', '<REPLACE_WITH_INFERENCING_API>')
INFERENCING_API_TOKEN  = os.environ.get('INFERENCING_API_TOKEN', '<REPLACE_WITH_INFERENCING_API_TOKEN>')
