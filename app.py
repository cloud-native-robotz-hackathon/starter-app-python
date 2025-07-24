from flask import Flask, render_template, jsonify, send_from_directory
from jinja2.exceptions import TemplateNotFound
import requests
import time
import threading
import os
from lib.robot_utils import (
    log_with_timestamp,
    bypass_obstacle,
    search_for_hat_step,
    get_stream_data
)

# Define variables for Flask proxy/web/application server
application = Flask(__name__)
application.config.from_object('config')
thread_event = threading.Event()

# Define model parameters
class_labels = ['Fedora',]
image_resolution_x = 640
confidence_threshold = 0.6
delta_threshold = 280
hat_found_and_intercepted = False
min_distance_to_obstacle = 300
angle_delta = 90

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/run', methods=['POST'])
def run():
    try:
        log_with_timestamp("/run endpoint called.")
        thread_event.set()
        log_with_timestamp("Creating and starting the startRobot thread.")
        thread = threading.Thread(target=startRobot)
        thread.start()
        log_with_timestamp("/run endpoint finished and returned 'Robot started'.")
        return "Robot started"
    except Exception as error:
        return str(error)

@application.route('/stop', methods=['POST'])
def stop():
    try:
        log_with_timestamp("/stop endpoint called.")
        thread_event.clear()
        return "Robot stopped"
    except Exception as error:
        return str(error)

@application.route('/status', methods=['POST'])
def status():
    response = requests.get(application.config['ROBOT_API'] + '/remote_status?user_key=' + application.config['ROBOT_NAME'], verify=False)
    return response.text

@application.route('/get_stream', methods=['GET'])
def stream():
    """
    Web route to get the processed image stream.
    Calls the utility function to handle the logic.
    """
    data, status_code = get_stream_data(application.config, class_labels)
    return jsonify(data), status_code

def startRobot():
    log_with_timestamp("startRobot thread has started.")
    while thread_event.is_set():
        log_with_timestamp("Entering main control loop.")

        # add your code here

    log_with_timestamp("Exited main control loop.")

@application.route('/<string:page_name>')
def serve_page(page_name):
    try:
        return render_template(f'{page_name}.html')
    except TemplateNotFound:
        return "<h1>Page not found</h1><p>The requested page does not exist.</p>", 404

@application.route('/templates/<path:filename>')
def serve_template_asset(filename):
    return send_from_directory(os.path.join(application.root_path, 'templates'), filename)

if __name__ == '__main__':
   application.run(host="0.0.0.0", port=8080)
