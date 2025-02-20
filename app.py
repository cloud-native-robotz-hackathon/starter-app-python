from flask import Flask, render_template
import requests
import time
from lib.preprocessing import preprocess_encoded_image, preprocess_image_file
from lib.object_detection import detect_objects
import threading
from collections import namedtuple
import base64

application = Flask(__name__)
application.config.from_object('config')
thread_event = threading.Event()

class_labels = ['Fedora',]
Coordinates = namedtuple('Coordinates', 'confidence_score x_upper_left y_upper_left x_lower_right y_lower_right object_class')


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
    # Drop your code here
    dist = distance()
    print ('Got distance -> ', dist)
    print('Done')

    # Initialize a variable to store the coordinates of the identified
    hat_coordinates = None

    hat_aligned = False

    turn_counter = 0
    while thread_event.is_set():
        dist = distance()
        print ('Got distance -> ', dist)

        objects = take_picture_and_detect_objects()
        coordinates = find_highest_score(objects)

        if coordinates and coordinates.confidence_score > 0.5:
            print(f'''Object with highest score -> [
                confidence score: {coordinates.confidence_score},
                x upper left corner: {coordinates.x_upper_left},
                y upper left corner: {coordinates.y_upper_left},
                x lower right corner: {coordinates.x_lower_right},
                y lower right corner: {coordinates.y_lower_right},
                object class: {coordinates.object_class} ]''')

            # Determine size of the object in the image (not the real size!)
            delta = coordinates.x_lower_right - coordinates.x_upper_left
            print(f'delta: {delta}')

            # Check if one hat could be identified with sufficient probabilty
            hat_detection_threshold = 0.50 # i.e. 850%
            if coordinates.confidence_score > hat_detection_threshold:
                hat_coordinates = coordinates
                print('Hat identified!')

            if not hat_coordinates:
                turn_left(10)
                turn_counter = turn_counter + 10

            # If the hat was identified, align so that the most likely hat is in the center
            if hat_coordinates:
                center_x = (coordinates.x_upper_left + coordinates.x_lower_right) / 2
                print(f'center_x: {center_x}')
                # if center_x < 320: # TODO switch from pixels to variable that is fed from image information
                #     turn_left(int(abs(320-center_x)))
                # else:
                #     turn_right(int(abs(320-center_x)))

                if abs(640/2-center_x) < 20: # TODO make this a var
                    hat_aligned = True
                else:
                    if center_x < 320: # TODO switch from pixels to variable that is fed from image information
                        turn_left(10)
                    else:
                        turn_right(10)


            # Move forward to the hat if it was identified
            if hat_coordinates and hat_aligned:
                if (delta > 350 or int(dist) < 300):
                    print('Done - arrived at the object')
                    return
                else:
                    move_forward(10)





            # # Center object that is most likely a hat
            # center_x = (coordinates.x_upper_left + coordinates.x_lower_right) / 2
            # if abs(640/2-center_x) < 20: # TODO make this a var
            #     hat_aligned = True
            # else:
            #     if move_x < 320: # TODO switch from pixels to variable that is fed from image information
            #         turn_left(10)
            #     else:
            #         turn_right(10)



            # # Move forward to the hat if it was identified
            # if hat_coordinates:
            #     # Determine size of the object in the image (not the real size!)
            #     delta = coordinates.x_lower_right - coordinates.x_upper_left
            #     print(f'delta: {delta}')



            # if delta < 350:
            #     move_forward(10)
            # else:
            #     print('Done - arrived at the object')
            #     return

            # # Define stop criterium
            # if coordinates.confidence_score > 0.8 and (delta > 350 or int(dist) < 300):
            #     print('Done - arrived at the object')
            #     return
            # else:
            #     move_forward(5)


            # if int(dist) > 300: # TODO Add variable for stop distance (here 100 mm)
            #     move_forward(10)
            # else:
            #     print('Done - arrived at the object')
            #     return

        else:
            if turn_counter < 360:
                turn_left(10)
                turn_counter = turn_counter + 10
            else:
                print('Done - no object found')
                return

    print('Done')


def take_picture_and_detect_objects():
    ## Example calling the ML inferencing endpoint for object detection
    ## get current camera image from robot
    print ('Taking picture from camera')
    img_response = requests.get(application.config['ROBOT_API'] + '/camera'+ '?user_key=' + application.config['ROBOT_NAME'], verify=False)
    print ('Response status code -> ', img_response.status_code)

    with open("static/current_view.jpg", "wb") as fh:
        fh.write(base64.urlsafe_b64decode(img_response.text))

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

    ## Add boxes to image
    from lib.object_rendering import draw_boxes
    image_path = "static/current_view.jpg"
    xxx, scaling, padding = preprocess_image_file(image_path)
    draw_boxes(image_path, objects, scaling, padding, class_labels)


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


if __name__ == '__main__':
    # Set environment variables
    import os
    os.environ["PYTHONDONTWRITEBYTEC"] = "1"
    os.environ["PATH"] = os.environ["PATH"] + ":/home/user/.local/bin"

    # Run flask service
    application.run(host="0.0.0.0", port=8080, debug = True)
