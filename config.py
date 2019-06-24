import os

workers = int(os.environ.get('GUNICORN_PROCESSES', '3'))
threads = int(os.environ.get('GUNICORN_THREADS', '1'))

forwarded_allow_ips = '*'
secure_scheme_headers = { 'X-Forwarded-Proto': 'https' }
timeout = 240

URI = 'https://api-2445582274375.production.gw.apicast.io/api/robot'
APITOKEN = ''
