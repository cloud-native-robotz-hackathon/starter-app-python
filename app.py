from flask import Flask, render_template, jsonify, send_from_directory
from jinja2.exceptions import TemplateNotFound
import requests
import time
import threading
import os
import config
import math
from lib.robot_utils import (
    log_with_timestamp,
    bypass_obstacle,
    search_for_hat_step,
    get_stream_data,
    move_forward,
    turn_left,
    turn_right,
    move_backward,
    distance,
    distance_int,
    take_picture_and_detect_objects,
    find_highest_score,
    take_picture
)

# Defined variables for Flask proxy/web/application server
application = Flask(__name__)
application.config.from_object('config')
thread_event = threading.Event()

# Defined model parameters
image_resolution_x = 640
delta_threshold = 280
hat_found_and_intercepted = False
min_distance_to_obstacle = 300
angle_delta = 90

# Function you will be working on
def startRobot():

    # Drop your code here
    print('Done')


# API and helper functions
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
    Web route to get the plain camera image stream.
    Calls the utility function to handle the logic.
    """
    data, status_code = get_stream_data()
    return jsonify(data), status_code

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
