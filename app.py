from flask import Flask, render_template
import requests
import time
from lib.preprocessing import preprocess_encoded_image, preprocess_image_file
from lib.object_detection import detect_objects
from lib.object_rendering import add_model_info_to_image
import threading
from collections import namedtuple
import base64
import math

# Define variables for Flask proxy/web/application server
application = Flask(__name__)
application.config.from_object('config')
thread_event = threading.Event()

# Define model parameters
class_labels = ['Fedora',]
Coordinates = namedtuple('Coordinates', 'confidence_score x_upper_left y_upper_left x_lower_right y_lower_right object_class')

# Define parameters for hat search and obstacle bypass algos
min_distance_to_obstacle = 300 # mm; distance at which the obstacle bypass mode is activated
angle_delta = 90 # deg; angle used for sidestepping obstacle
image_resolution_x = 640 # pixels; resolution of camera used in robot
confidence_threshold = 0.6 # e.g. 0.6 = 60%; confidence at which an object identified as hat is intercepted
delta_threshold = 280 # pixels; delta for standard fedora (defines minimum desired pixel size of fedora in image)
hat_found_and_intercepted = False # boolean; switch for a found and intercepted hat

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

# Main function that is called when the "Run" button is pressed
def startRobot():
    # Drop your code here
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

# Main function that is called after app started
if __name__ == '__main__':
   application.run(host="0.0.0.0", port=8080)
