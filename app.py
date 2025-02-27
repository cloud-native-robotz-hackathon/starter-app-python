from flask import Flask, render_template
import requests
import time
from lib.preprocessing import preprocess_encoded_image, preprocess_image_file
from lib.object_detection import detect_objects
from lib.object_rendering import draw_boxes
import threading
from collections import namedtuple
import base64

application = Flask(__name__)
application.config.from_object('config')
thread_event = threading.Event()

class_labels = ['Fedora',]
Coordinates = namedtuple('Coordinates', 'confidence_score x_upper_left y_upper_left x_lower_right y_lower_right object_class')


min_distance_to_obstacle = 300 # mm
# min_distance_to_move_forward_after_turn = 600 # mm
angle_delta = 90 # deg
image_resolution_x = 640
confidence_threshold = 0.6 # 60%
hat_found_and_intercepted = False

@application.route('/')
def index():
    return render_template('index.html')


@application.route('/run', methods=['POST'])
def run():
    try:
        thread_event.set()

        thread = threading.Thread(target=startRobot)
        thread.start()

        return "Robot started"
    except Exception as error:
        return str(error)


@application.route('/stop', methods=['POST'])
def stop():
    try:
        thread_event.clear()

        return "Robot stopped"
    except Exception as error:
        return str(error)


@application.route('/status', methods=['POST'])
def status():
    response = requests.get(application.config['ROBOT_API'] + '/remote_status?user_key=' + application.config['ROBOT_NAME'], verify=False)
    return response.text


def startRobot():

    # move_forward(1) corresponds to 10 mm.
    # Strategy: if distance <20cm, go to obstacle_avoidance_mode
    # Assume plain objects


    # print ('\nInitial distance at 0°-> ', distance())
    # take_picture('static/view_initial.jpg')

    # move_backward(40)
    # turn_right(40)
    # turn_left(180)
    # return

    global hat_found_and_intercepted
    hat_found_and_intercepted = False

    while thread_event.is_set() and not hat_found_and_intercepted:
        #code...

        # Bypass obstacle, if directly in front
        bypass_obstacle()

        # Search for the hat: circle until hat is found and then move towards it.
        # If not hat is found after full turn, move forward and try again.
        search_for_hat()

    print('Done')

def search_for_hat():

    global hat_found_and_intercepted

    print('\n### Search For Hat Mode - START ###')

    turn_counter = 0
    while thread_event.is_set():
        print('\n')

        objects = take_picture_and_detect_objects()
        coordinates = find_highest_score(objects)

        # Output distance from sensor
        print ('Got distance -> ', distance())

        # Check if there is an obstacle ahead
        if distance_int() <= min_distance_to_obstacle:
            print('### Search For Hat Mode - END: Obstacle detected! ###')

            return

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
    
            if abs(image_resolution_x/2-center_x) >= 20: # center of hat needs to be within 20 pixels
                if center_x < 320: # TODO switch from pixels to variable that is fed from image information
                    turn_left(10)
                else:
                    turn_right(10)

            # Determine size of the object in the image (not the real size!)
            delta = coordinates.x_lower_right - coordinates.x_upper_left
            print(f'delta: {delta}')

            # hat_detection_threshold = 0.50 # i.e. 850%
            # delta_threshold = 150 # Mini-Hüte
            delta_threshold = 280 # 250 # Standard-Hüte
            if delta < delta_threshold:
                move_forward(10)
            else:
                hat_found_and_intercepted = True
                print('### Search For Hat Mode - END: OJECT FOUND ! ###')
                return

        else:
            if turn_counter <= 360:
                turn_right(10)
                turn_counter = turn_counter + 10
            else:
                # After a full circle, move forward and circle again to find the hat
                move_forward(40)
                turn_counter = 0

    

    print('### Search For Hat Mode - END ###')

    # dist = distance()
    # print ('Initial distance at 0°-> ', dist)
    # take_picture('static/view_initial.jpg')

    # turn_left(30)
    # dist = distance()
    # print ('Got distance at 30° left -> ', dist)
    # take_picture('static/view_30_l.jpg')

    # turn_left(30)
    # dist = distance()
    # print ('Got distance at 60° left -> ', dist)
    # take_picture('static/view_60_l.jpg')

    # turn_left(30)
    # dist = distance()
    # print ('Got distance at 90° left -> ', dist)
    # take_picture('static/view_90_l.jpg')

    # turn_right(120)
    # dist = distance()
    # print ('Got distance at 30° right -> ', dist)
    # take_picture('static/view_30_r.jpg')

    # turn_right(30)
    # dist = distance()
    # print ('Got distance at 60° right -> ', dist)
    # take_picture('static/view_60_r.jpg')

    # turn_right(30)
    # dist = distance()
    # print ('Got distance at 90° right -> ', dist)
    # take_picture('static/view_90_r.jpg')

    # turn_left(90)
    # dist = distance()
    # print ('Got final distance at 0° -> ', dist)
    # take_picture('static/view_final.jpg')



    # move_forward(2)
    # dist = distance()
    # print ('Got distance -> ', dist)

    # move_forward(2)
    # dist = distance()
    # print ('Got distance -> ', dist)

    # move_forward(2)
    # dist = distance()
    # print ('Got distance -> ', dist)



def bypass_obstacle():
    # Determine if an obstacle is in sight
    obstacle_in_sight = distance_int() <= min_distance_to_obstacle

    # Only continue of an obstacle is ahead
    if not obstacle_in_sight:
        return

    #debug
    take_picture('static/current_view.jpg')

    # Determine distance to obstacle
    distance_to_object = distance_int()

    # angle_int = 0

    print('### Bypass Obstacle Mode - START ###')

    # Turn left
    turn_left(angle_delta)

    # Determine if there is another obstacle is in sight
    obstacle_in_sight = distance_int() <= min_distance_to_obstacle
    print ('Got distance -> ', distance())

    if not obstacle_in_sight:
        # Move a bit forward and then go back again on course
        move_forward(20)
        turn_right(angle_delta)


    # Determine if original obsctacle is still in sight
    obstacle_in_sight = distance_int() <= min_distance_to_obstacle
    print ('Got distance -> ', distance())

    if not obstacle_in_sight:
        # Move forward to bypass original obstacle
        import math
        move_forward(math.ceil(distance_to_object / 10) + 50)

    




    # while obstacle_in_sight:

    #     # Turn left
    #     turn_left(angle_delta)
    #     angle_int += angle_delta # Build incremental angle to determine exit condition

    #     # Determine if obsctacle is still in sight
    #     obstacle_in_sight = distance_int() <= min_distance_to_move_forward_after_turn

    #     print ('Got distance -> ', distance())
        
    #     #debug
    #     take_picture('static/current_view.jpg')

    #     # emergency exit
    #     if angle_int > 360:
    #         print('### Bypass Obstacle Mode - END: NO ESCAPE! ###')
    #         return

    # # Move forward and the go back on course
    # move_forward(40)
    # turn_right(angle_int)



    print('### Bypass Obstacle Mode - END: SUCCESS! ###')



def take_picture(image_file_name):
    ## get current camera image from robot
    print ('Taking picture from camera at ' + image_file_name)
    img_response = requests.get(application.config['ROBOT_API'] + '/camera'+ '?user_key=' + application.config['ROBOT_NAME'], verify=False)
    print ('    --> Response status code -> ', img_response.status_code)

    with open(image_file_name, "wb") as fh:
        fh.write(base64.urlsafe_b64decode(img_response.text))

    return img_response


def take_picture_and_detect_objects():
    ## Example calling the ML inferencing endpoint for object detection

    ## get current camera image from robot
    img_response = take_picture('static/current_view.jpg')

    ## normalize image
    image_data, ratio, dwdh = preprocess_encoded_image(img_response.text)

    ## send image to object detection endpoint
    print ('Sending image to inferencing')
    objects = detect_objects(
        image_data,
        application.config['INFERENCING_API'] ,
        token=application.config['INFERENCING_API_TOKEN'],
        classes_count=len(class_labels),
        confidence_threshold=0.15,
        iou_threshold=0.2
    )

    ## Add bounding box to image
    image_box_path = "static/current_view_box.jpg"
    with open(image_box_path, "wb") as fh:
        fh.write(base64.urlsafe_b64decode(img_response.text))

    xxx, scaling, padding = preprocess_image_file(image_box_path)
    draw_boxes(image_box_path, objects, scaling, padding, class_labels)

    return objects


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


def move_forward(length):
    response = requests.post(application.config['ROBOT_API'] + '/forward/' + str(length),data={"user_key": application.config['ROBOT_NAME']} ,verify=False)
    return response.text


def move_backward(length):
    response = requests.post(application.config['ROBOT_API'] + '/backward/' + str(length),data={"user_key": application.config['ROBOT_NAME']} ,verify=False)
    return response.text


def turn_left(degrees):
    response = requests.post(application.config['ROBOT_API'] + '/left/' + str(degrees),data={"user_key": application.config['ROBOT_NAME']} ,verify=False)
    return response.text


def turn_right(degrees):
    response = requests.post(application.config['ROBOT_API'] + '/right/' + str(degrees),data={"user_key": application.config['ROBOT_NAME']} ,verify=False)
    return response.text

def distance():
    response = requests.get(application.config['ROBOT_API'] + '/distance' + '?user_key=' + application.config['ROBOT_NAME'] ,verify=False)
    return response.text

# Return distance as integer
def distance_int():
    return int(''.join(distance()))

if __name__ == '__main__':
   application.run(host="0.0.0.0", port=8080)
