from flask import Flask, render_template, jsonify, send_from_directory
from jinja2.exceptions import TemplateNotFound
import requests
import time
from lib.preprocessing import preprocess_encoded_image, preprocess_image_file
from lib.object_detection import detect_objects
from lib.object_rendering import add_model_info_to_image
import threading
from collections import namedtuple
import base64
import math
import random
import cv2
import numpy as np
import os


# Define variables for Flask proxy/web/application server
application = Flask(__name__)
application.config.from_object('config')
thread_event = threading.Event()

# Define model parameters
class_labels = ['Fedora',]
Coordinates = namedtuple('Coordinates', 'confidence_score x_upper_left y_upper_left x_lower_right y_lower_right object_class')

image_resolution_x = 640 # pixels; resolution of camera used in robot
confidence_threshold = 0.6 # e.g. 0.6 = 60%; confidence at which an object identified as hat is intercepted
delta_threshold = 280 # pixels; delta for standard fedora (defines minimum desired pixel size of fedora in image)
hat_found_and_intercepted = False # boolean; switch for a found and intercepted hat
# Define parameters for hat search and obstacle bypass algos
min_distance_to_obstacle = 300 # mm; distance at which the obstacle bypass mode is activated
angle_delta = 90 # deg; angle used for sidestepping obstacle

# Standard route
@application.route('/')
def index():
    return render_template('index.html')

# Route for starting the application
@application.route('/run', methods=['POST'])
def run():
    try:
        thread_event.set()

        thread = threading.Thread(target=startRobot)
        thread.start()

        return "Robot started"
    except Exception as error:
        return str(error)

# Route for stopping the application
@application.route('/stop', methods=['POST'])
def stop():
    try:
        thread_event.clear()

        return "Robot stopped"
    except Exception as error:
        return str(error)

# Route for getting a status of the application
@application.route('/status', methods=['POST'])
def status():
    response = requests.get(application.config['ROBOT_API'] + '/remote_status?user_key=' + application.config['ROBOT_NAME'], verify=False)
    return response.text


def draw_detections(image_np, detections, ratio, dwdh, class_labels):
    """
    Draws bounding boxes and labels on the image based on detected objects.
    Optimized for faster performance by vectorizing coordinate transformations.

    Args:
        image_np (numpy.ndarray): The original (or preprocessed) image in OpenCV format.
        detections (list): A list of detected objects, where each object is expected
                            to be a list/array-like with [x1, y1, x2, y2, confidence, class_id].
        ratio (float or tuple): Scaling ratio used during preprocessing. Can be a single float
                                (r) or a tuple (rw, rh).
        dwdh (tuple): Padding added during preprocessing (dw, dh).
        class_labels (list): A list of strings for class names.

    Returns:
        numpy.ndarray: The image with detections drawn on it.
    """
    if detections is None or len(detections) == 0:
        return image_np

    # Unpack dwdh (delta width, delta height)
    dw, dh = dwdh[0], dwdh[1]
    dw_half = dw / 2
    dh_half = dh / 2

    # Safely unpack ratio: it might be a single float or a tuple (r, r)
    if isinstance(ratio, (list, tuple)) and len(ratio) == 2:
        rw, rh = ratio[0], ratio[1]
    else: # Assume it's a single float, meaning uniform scaling
        rw, rh = ratio, ratio

    # Create a copy to draw on to avoid modifying the original image_np
    display_image = image_np.copy()

    original_h, original_w = image_np.shape[:2]
    
    # Calculate unpadded dimensions (where the actual image content sits within 640x640)
    unpadded_w = 640 - dw
    unpadded_h = 640 - dh

    # Handle potential division by zero if unpadded dimensions are zero
    if unpadded_w == 0 or unpadded_h == 0:
        print("Warning: Unpadded image dimensions are zero, cannot calculate scale factors.")
        return display_image # Return original image if scaling is problematic

    # Calculate scale factors from 640x640 (unpadded area) to original image size
    scale_x = original_w / unpadded_w
    scale_y = original_h / unpadded_h

    # Convert detections to a NumPy array for vectorized operations
    # Ensure detections are floats for calculations
    detections_np = np.array(detections, dtype=np.float32)

    # Extract coordinates, confidence, and class_id
    # detections_np[:, :4] contains [x1, y1, x2, y2]
    # detections_np[:, 4] contains confidence
    # detections_np[:, 5] contains class_id

    # Apply inverse padding and then scale using vectorized operations
    # Create copies to avoid modifying the original detections_np array in place if it's used elsewhere
    x1s = (detections_np[:, 0] - dw_half) * scale_x
    y1s = (detections_np[:, 1] - dh_half) * scale_y
    x2s = (detections_np[:, 2] - dw_half) * scale_x
    y2s = (detections_np[:, 3] - dh_half) * scale_y

    # Ensure coordinates are within image bounds using np.clip
    x1s = np.clip(x1s, 0, original_w - 1).astype(int)
    y1s = np.clip(y1s, 0, original_h - 1).astype(int)
    x2s = np.clip(x2s, 0, original_w - 1).astype(int)
    y2s = np.clip(y2s, 0, original_h - 1).astype(int)

    # Loop through each detection to draw (drawing functions are not easily vectorized)
    for i in range(len(detections_np)):
        x1, y1, x2, y2 = x1s[i], y1s[i], x2s[i], y2s[i]
        conf = detections_np[i, 4]
        class_id = int(detections_np[i, 5])

        # Draw rectangle
        cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green box

        # Put label
        label = f"{class_labels[class_id]}: {conf:.2f}" if class_id < len(class_labels) else f"Unknown: {conf:.2f}"
        
        # Determine text position to be above the box
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = x1
        text_y = y1 - 10 if y1 - 10 > text_size[1] else y1 + text_size[1] + 10 # Place above or below if no space

        cv2.putText(display_image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    return display_image

# Route for stopping the application
@application.route('/get_stream', methods=['GET'])
def stream():
    try:
        start_total_time = time.time() # Start total timer
        timing_details = {}
        start_fetch_time = time.time() 
        img_response = requests.get(application.config['ROBOT_API'] + '/camera'+ '?user_key=' + application.config['ROBOT_NAME'], verify=False)

        base64_robot_image = img_response.text

        image_data_np, ratio, dwdh = preprocess_encoded_image(base64_robot_image)
        end_fetch_time = time.time()
        timing_details['fetch_image'] = f"{end_fetch_time - start_fetch_time:.4f} seconds"

        # Send image to object detection endpoint
        print ('Sending image to inferencing')
        start_inferencing_time = time.time()
        objects = detect_objects(
            image_data_np,
            application.config['INFERENCING_API'] ,
            token=application.config['INFERENCING_API_TOKEN'],
            classes_count=len(class_labels),
            confidence_threshold=0.15,
            iou_threshold=0.4
        )
        end_inferencing_time = time.time()
        timing_details['inferencing'] = f"{end_inferencing_time - start_inferencing_time:.4f} seconds"


        start_rendering_time = time.time()

        # 4. Render detection boxes on the original image (or the raw image from robot)
        # To get the raw image from base64_robot_image for drawing:
        # Decode the base64 string back to bytes
        img_bytes = base64.b64decode(base64_robot_image)
        # Convert bytes to numpy array (image)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        # Decode image using OpenCV
        original_cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if original_cv_image is None:
            raise ValueError("Could not decode base64 robot image into OpenCV format.")

        # Draw detections on the original image
        image_with_detections = draw_detections(original_cv_image, objects, ratio, dwdh, class_labels)

        end_rendering_time = time.time()
        timing_details['rendering_boxes'] = f"{end_rendering_time - start_rendering_time:.4f} seconds"

        start_encoding_time = time.time()
        # 5. Encode the image with detections to Base64
        success, encoded_image = cv2.imencode(".jpeg", image_with_detections, [int(cv2.IMWRITE_JPEG_QUALITY), 75])

        if not success:
            raise ValueError("Failed to encode image with detections.")

        base64_string = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
        end_encoding_time = time.time()
        timing_details['encoding_image'] = f"{end_encoding_time - start_encoding_time:.4f} seconds"

        #print(base64_string)

        end_time = time.time() # End timer
        end_total_time = time.time() # End total timer
        timing_details['total_execution'] = f"{end_total_time - start_total_time:.4f} seconds"
        print(f"get_stream execution details: {timing_details}") # Log elapsed time


        return jsonify({"image": base64_string})
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error to robot or inferencing API: {e}")
        return jsonify({"error": "Failed to connect to image source or inference service. Check API URLs."}), 503
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error from API: {e}")
        return jsonify({"error": f"API returned an error: {e.response.status_code} - {e.response.text}"}), 500
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify({"error": f"Failed to get stream: {str(e)}"}), 500

# Main function that is called when the "Run" button is pressed
def startRobot():
    # Drop your code here
    # Initialize switch for identifying found and intercepted hats across functions
    global hat_found_and_intercepted
    hat_found_and_intercepted = False

    # Main loop running until one hat is properly identified and intercepted (or the app ist stopped)
    while thread_event.is_set() and not hat_found_and_intercepted:
        # Check for an obstacle directly in front and bypass it, if existing
        bypass_obstacle()

        # Search for the hat and intercrept it
        #search_for_hat()

    print('Done')


# Take a picture using the camera of the robot
def take_picture(image_file_name):
    # Get current camera image from robot
    print ('Taking picture from camera at ' + image_file_name)
    img_response = requests.get(application.config['ROBOT_API'] + '/camera'+ '?user_key=' + application.config['ROBOT_NAME'], verify=False)
    print ('    --> Response status code -> ', img_response.status_code)

    # Write image to file
    with open(image_file_name, "wb") as fh:
        fh.write(base64.urlsafe_b64decode(img_response.text))

    return img_response

# Take a picture using the camera of the robot and detect objects
def take_picture_and_detect_objects():
    # Example calling the ML inferencing endpoint for object detection

    # Get current camera image from robot
    img_response = take_picture('static/current_view.jpg')

    # Normalize image
    image_data, ratio, dwdh = preprocess_encoded_image(img_response.text)

    # Send image to object detection endpoint
    print ('Sending image to inferencing')
    objects = detect_objects(
        image_data,
        application.config['INFERENCING_API'] ,
        token=application.config['INFERENCING_API_TOKEN'],
        classes_count=len(class_labels),
        confidence_threshold=0.15,
        iou_threshold=0.2
    )

    # Add bounding box to image
    image_box_path = "static/current_view_box.jpg"
    # - Write original image to file
    with open(image_box_path, "wb") as fh:
        fh.write(base64.urlsafe_b64decode(img_response.text))

    # - Process image and add model info (bounding boxes & confidence scores)
    #   to image and write amended image back to file
    _, scaling, padding = preprocess_image_file(image_box_path)
    add_model_info_to_image(image_box_path, objects, scaling, padding, class_labels)

    return objects

# Find highest confidence score of identified objects
def find_highest_score(objects):
    # Initialize a variable to store the coordinates with the highest score
    detected_object = [0,0,0,0,0,0]

    # Iterate over the detected objects and update the highest score if necessary
    for o in objects:
        # Check if the current object is of the class 'Fedora' and has a higher score than the previous highest score
        if o[-1] == 0 and detected_object[-2] < o[-2]:
            # Update the detected object with the new highest score
            detected_object = o

    # Check if there was at least one object detected with a confidence score greater than 0
    if (detected_object[-2] > 0):
        # Create a Coordinates namedtuple with the information of the object with the highest score
        coordinates = Coordinates(detected_object[-2], detected_object[0], detected_object[1], detected_object[2], detected_object[3], detected_object[-1])
        return coordinates

    return

# Check for an obstacle directly in front and bypass it, if existing
def bypass_obstacle():
    # Determine if an obstacle is in sight
    obstacle_in_sight = distance_int() <= min_distance_to_obstacle

    # Only continue if an obstacle is ahead
    if not obstacle_in_sight:
        return

    # For debugging only
    take_picture('static/current_view.jpg')

    # Determine distance to obstacle
    distance_to_object = distance_int()

    print('### Bypass Obstacle Mode - START ###')

    # Turn left
    turn_left(angle_delta)

    # Determine if there is another obstacle is in sight
    obstacle_in_sight = distance_int() <= min_distance_to_obstacle
    print ('Got distance -> ', distance())

    # If no other obstacle is in the bypass direction, move a bit forward
    # and then go back again on course
    if not obstacle_in_sight:
        move_forward(20)
        turn_right(angle_delta)

    # Determine if original obsctacle is still in sight (after having turned back in original direction)
    obstacle_in_sight = distance_int() <= min_distance_to_obstacle
    print ('Got distance -> ', distance())

    # If original obstacle is not in sight anymore, move forward to bypass it
    if not obstacle_in_sight:
        # Move forward using the original distance to the obstacle and a buffer
        move_forward(math.ceil(distance_to_object / 10) + 40)

    print('### Bypass Obstacle Mode - END: SUCCESS! ###')

def search_for_hat():
    # Define switch for identifying found and intercepted hats across functions
    global hat_found_and_intercepted

    print('\n### Search For Hat Mode - START ###')

    # Circle, capture images, apply model and drive towards an identified object
    turn_counter = 0
    while thread_event.is_set():
        print('\n')

        # Take picture and find the object with the highest probabilty of being a hat
        objects = take_picture_and_detect_objects()
        coordinates = find_highest_score(objects)

        # Output distance from sensor
        print ('Got distance -> ', distance())

        # Check if there is an obstacle ahead
        if distance_int() <= min_distance_to_obstacle:
            print('### Search For Hat Mode - END: Obstacle detected! ###')
            return

        # Align to and drive towards identified object
        if coordinates and coordinates.confidence_score > confidence_threshold:
            print(f'''Object with highest score -> [
                confidence score: {coordinates.confidence_score},
                x upper left corner: {coordinates.x_upper_left},
                y upper left corner: {coordinates.y_upper_left},
                x lower right corner: {coordinates.x_lower_right},
                y lower right corner: {coordinates.y_lower_right},
                object class: {coordinates.object_class} ]''')

            # Align so that the most likely hat identified is in the center (within 20 pixels)
            center_x = (coordinates.x_upper_left + coordinates.x_lower_right) / 2
            print(f'center_x: {center_x}')

            # Center of hat needs to be within 20 pixels
            if abs(image_resolution_x/2-center_x) >= 20:
                if center_x < 320:
                    turn_left(10)
                else:
                    turn_right(10)

            # Determine size of the object in the image (not the real size!)
            delta = coordinates.x_lower_right - coordinates.x_upper_left
            print(f'delta: {delta}')

            # Move forward, if size of identified object in image is not big enough
            # (i.e. if it's not close enough)
            if delta < delta_threshold:
                move_forward(10)
            else:
                hat_found_and_intercepted = True
                print('### Search For Hat Mode - END: OJECT FOUND ! ###')
                return

        else:
            # Circle in case no hat could be identied
            if turn_counter <= 360:
                turn_right(10)
                turn_counter = turn_counter + 10
            else:
                # After a full circle, move forward and circle again to find the hat
                move_forward(40)
                turn_counter = 0

    print('### Search For Hat Mode - END ###')

# Move robot forward
def move_forward(length):
    response = requests.post(application.config['ROBOT_API'] + '/forward/' + str(length),data={"user_key": application.config['ROBOT_NAME']} ,verify=False)
    return response.text

# Move robot backward
def move_backward(length):
    response = requests.post(application.config['ROBOT_API'] + '/backward/' + str(length),data={"user_key": application.config['ROBOT_NAME']} ,verify=False)
    return response.text

# Turn robot left
def turn_left(degrees):
    response = requests.post(application.config['ROBOT_API'] + '/left/' + str(degrees),data={"user_key": application.config['ROBOT_NAME']} ,verify=False)
    return response.text

# Turn robot right
def turn_right(degrees):
    response = requests.post(application.config['ROBOT_API'] + '/right/' + str(degrees),data={"user_key": application.config['ROBOT_NAME']} ,verify=False)
    return response.text

# Obtain distance from robot's distance sensor
def distance():
    response = requests.get(application.config['ROBOT_API'] + '/distance' + '?user_key=' + application.config['ROBOT_NAME'] ,verify=False)
    return response.text

# Return distance as integer
def distance_int():
    return int(''.join(distance()))

# NEW: Generic route to serve other HTML files from the 'templates' folder.
@application.route('/<string:page_name>')
def serve_page(page_name):
    """
    Renders any .html file from the 'templates' folder.
    For example, a request to /about will attempt to render templates/about.html.
    """
    try:
        # Attempt to render the requested HTML file
        return render_template(f'{page_name}.html')
    except TemplateNotFound:
        # If the template file doesn't exist, return a 404 error
        return "<h1>Page not found</h1><p>The requested page does not exist.</p>", 404
    except Exception as e:
        # Handle other potential errors
        print(f"An error occurred while trying to serve page {page_name}: {e}")
        return "<h1>Server Error</h1><p>An internal error occurred.</p>", 500


# NEW: Route to serve static files (like svg) from the templates folder
@application.route('/templates/<path:filename>')
def serve_template_asset(filename):
    """
    Serves a static file from the 'templates' directory.
    This is used to serve the rh-logo.svg file.
    """
    return send_from_directory(os.path.join(application.root_path, 'templates'), filename)

# Main function that is called after app started
if __name__ == '__main__':
   application.run(host="0.0.0.0", port=8080)
