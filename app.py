from flask import Flask, render_template
import requests
import time
from lib.preprocessing import preprocess_encoded_image
from lib.object_detection import detect_objects

application = Flask(__name__)
application.config.from_object('config')

class_labels = ['Fedora', 'Ball', 'Tennis ball']

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/run', methods=['POST'])
def run():
    ## Example GET invocations of the Robot API

    ## Connect and get the status of the robot
    # response = requests.get(application.config['ROBOT_API'] + '/status' + '?user_key=' + application.config['ROBOT_NAME'], verify=False)

    ## Get the current power of the robot
    # response = requests.get(application.config['ROBOT_API'] + '/power' + '?user_key=' + application.config['ROBOT_NAME'],verify=False)

    ## Example POST invocation of the Robot API for e.g. moving
    #response = requests.post(application.config['ROBOT_API'] + '/forward/10',data={"user_key": application.config['ROBOT_NAME']} ,verify=False)

    ## Example calling the ML inferencing endpoint for object detection
    ## get current camera image from robot
    #img_response = requests.get(application.config['ROBOT_API'] + '/camera'+ '?user_key=' + application.config['ROBOT_NAME'], verify=False)

    ## normalize image
    #image_data, ratio, dwdh = preprocess_encoded_image(img_response.text)

    ## send image to object detection endpoint
    #objects = detect_objects(
    #    image_data,
    #    application.config['INFERENCING_API'] ,
    #    token=application.config['INFERENCING_API_TOKEN'],
    #    classes_count=len(class_labels),
    #    confidence_threshold=0.15,
    #    iou_threshold=0.2
    #)

    ## objects will then contain a list if tensors with box coordinates, confidence score and class ( 0 = Fedora) of detected object (x left upper corner, y left upper corner,
    ## x right lower corner, confidence score, class)
    #print('objects found ->', objects)

    ## Get the detected fedora object with the highest confidence score
    #fedora = [0,0,0,0,0,0]

    #for o in objects:
    #    if o[-1] == 0 and fedora[-2] < o[-2]:
    #        fedora = o

    #if (fedora[-2] > 0):
    #    print('Object with highest score -> [ confidence score : {},  x left upper corner : {}, y left upper corner : {}, x right lower corner{}, y right lower corner{}'.format(fedora[-2], fedora[0], fedora[1],fedora[2],fedora[3],))

    return ('OK')


@application.route('/status', methods=['POST'])
def status():
    response = requests.get(application.config['ROBOT_API'] + '/remote_status?user_key=' + application.config['ROBOT_NAME'], verify=False)
    return response.text

if __name__ == '__main__':
   application.run(host="0.0.0.0", port=8080)
