import os

workers = int(os.environ.get('GUNICORN_PROCESSES', '3'))
threads = int(os.environ.get('GUNICORN_THREADS', '1'))
bind = os.environ.get('GUNICORN_BIND', '0.0.0.0:8080')
static_folder='static'
forwarded_allow_ips = '*'
secure_scheme_headers = { 'X-Forwarded-Proto': 'https' }
timeout = 240

