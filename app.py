from flask import Flask, render_template, jsonify, send_from_directory
from jinja2.exceptions import TemplateNotFound
import requests
import time
import threading
import os
import asyncio
import aiohttp
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
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

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=4)

# Cache for expensive operations
@lru_cache(maxsize=128)
def cached_robot_status(robot_api, robot_name, timestamp):
    """Cache robot status for a short period"""
    try:
        response = requests.get(f"{robot_api}/remote_status?user_key={robot_name}", 
                              verify=False, timeout=5)
        return response.text
    except requests.RequestException as e:
        return f"Error: {str(e)}"

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
        # Use executor for better thread management
        future = executor.submit(startRobot)
        log_with_timestamp("/run endpoint finished and returned 'Robot started'.")
        return jsonify({"status": "Robot started", "message": "Robot control thread initiated"})
    except Exception as error:
        return jsonify({"error": str(error)}), 500

@application.route('/stop', methods=['POST'])
def stop():
    try:
        log_with_timestamp("/stop endpoint called.")
        thread_event.clear()
        return jsonify({"status": "Robot stopped"})
    except Exception as error:
        return jsonify({"error": str(error)}), 500

@application.route('/status', methods=['GET'])
def status():
    try:
        # Use caching with 1-second granularity
        current_time = int(time.time())
        cached_status = cached_robot_status(
            application.config['ROBOT_API'], 
            application.config['ROBOT_NAME'], 
            current_time
        )
        return jsonify({"status": cached_status})
    except Exception as error:
        return jsonify({"error": str(error)}), 500

@application.route('/get_stream', methods=['GET'])
def stream():
    """
    Web route to get the processed image stream.
    Uses executor for non-blocking operation.
    """
    try:
        # Execute in thread pool to avoid blocking
        future = executor.submit(get_stream_data, application.config, class_labels)
        data, status_code = future.result(timeout=10)  # 10 second timeout
        return jsonify(data), status_code
    except Exception as e:
        return jsonify({"error": f"Stream processing failed: {str(e)}"}), 500

def startRobot():
    log_with_timestamp("startRobot thread has started.")
    global hat_found_and_intercepted
    hat_found_and_intercepted_ref = [hat_found_and_intercepted]

    turn_counter = 0
    last_obstacle_check = 0
    obstacle_check_interval = 1.0  # Check obstacles every 1 second instead of 0.5

    log_with_timestamp("Entering main control loop...")
    while thread_event.is_set():
        current_time = time.time()
        
        # Optimize obstacle checking frequency
        #if current_time - last_obstacle_check > obstacle_check_interval:
         #   if bypass_obstacle(application.config, min_distance_to_obstacle, angle_delta):
          #      last_obstacle_check = current_time
           #     time.sleep(0.3)  # Reduced sleep time
            #    continue
           # last_obstacle_check = current_time

        turn_counter = search_for_hat_step(
            application.config,
            turn_counter,
            class_labels,
            confidence_threshold,
            image_resolution_x,
            delta_threshold,
            hat_found_and_intercepted_ref
        )

        time.sleep(0.3)  # Reduced from 0.5 to 0.3 seconds

    hat_found_and_intercepted = hat_found_and_intercepted_ref[0]
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
   application.run(host="0.0.0.0", port=8080, threaded=True)
