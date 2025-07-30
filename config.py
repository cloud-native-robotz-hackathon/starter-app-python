import os

# modify these values
ROBOT_NAME = os.environ.get('ROBOT_NAME', '<REPLACE_WITH_ROBOT_NAME>')
INFERENCING_API = os.environ.get('INFERENCING_API', '<REPLACE_WITH_INFERENCING_API>')
INFERENCING_API_TOKEN  = os.environ.get('INFERENCING_API_TOKEN', '<REPLACE_WITH_INFERENCING_API_TOKEN>')

# Model parameters
CONFIDENCE_THRESHOLD = 0.6
CLASS_LABELS = ['Fedora']


# Keep these settings as is
workers = int(os.environ.get('GUNICORN_PROCESSES', '3'))
threads = int(os.environ.get('GUNICORN_THREADS', '1'))
bind = os.environ.get('GUNICORN_BIND', '0.0.0.0:8080')
static_folder='static'
forwarded_allow_ips = '*'
secure_scheme_headers = { 'X-Forwarded-Proto': 'https' }
timeout = 240

if os.environ.get('ROBOT_API') is not None:
    ROBOT_API = os.environ.get('ROBOT_API')
elif os.environ.get('ROBOT_API_ENDPOINT') is not None:
    ROBOT_API_ENDPOINT = os.environ.get('ROBOT_API_ENDPOINT')
    ROBOT_API = f"http://{ROBOT_API_ENDPOINT}:5000/"
else:
    ROBOT_API = 'http://api.hub-controller.svc.cluster.local/robot'

