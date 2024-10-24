import os

workers = int(os.environ.get('GUNICORN_PROCESSES', '3'))
threads = int(os.environ.get('GUNICORN_THREADS', '1'))
bind = os.environ.get('GUNICORN_BIND', '0.0.0.0:8080')

forwarded_allow_ips = '*'
secure_scheme_headers = { 'X-Forwarded-Proto': 'https' }
timeout = 240

ROBOT_API = os.environ.get('ROBOT_API', 'http://hub-controller-live.red-hat-service-interconnect-data-center.svc.cluster.local:8080/robot')
ROBOT_NAME = os.environ.get('ROBOT_NAME', '<REPLACE_WITH_ROBOT_NAME>')
INFERENCING_API = os.environ.get('INFERENCING_API', '<REPLACE_WITH_INFERENCING_API>')
INFERENCING_API_TOKEN  = os.environ.get('INFERENCING_API_TOKEN', '<REPLACE_WITH_INFERENCING_API_TOKEN>')
