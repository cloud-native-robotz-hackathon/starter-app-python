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
    # Initialize switch for identifying found and intercepted hats across functions
    global hat_found_and_intercepted
    hat_found_and_intercepted = False

    # Main loop running until one hat is properly identified and intercepted (or the app ist stopped)
    while thread_event.is_set() and not hat_found_and_intercepted:
        # Check for an obstacle directly in front and bypass it, if existing
        bypass_obstacle()

        # Search for the hat and intercrept it
        search_for_hat()

    print('Done')

# Search for the hat and intercrept it
# Method: circle until hat is found and then move towards it. If
# no hat is found after a full turn, move forward and try again.
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
        move_forward(math.ceil(distance_to_object / 10) + 50)

    print('### Bypass Obstacle Mode - END: SUCCESS! ###')


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
