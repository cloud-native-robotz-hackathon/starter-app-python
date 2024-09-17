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
    # response = requests.get('http://' + application.config['ROBOT_API'] + ':5000/distance' + '?user_key=' + application.config['ROBOT_NAME'], verify=False)
    # response = requests.get(application.config['URI'] + '/power' + '?user_key=' + application.config['ROBOT_NAME'],verify=False)

    ## Example POST invocation of the Robot API for e.g. moving
    # response = requests.post('http://' + application.config['ROBOT_API'] + ':5000/forward/10', verify=False)

    ## Example calling the ML inferencing endpoint for object detection
    ## get current camera image from robot
    #img_response = requests.get(application.config['ROBOT_API'] + '/camera', verify=False)
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
    ## objects will then contain a list if tensors with box coordinatas, confidence score and class of detected object
    #print('Detected {} obejects', len(objects))

    return ('OK')


@application.route('/status', methods=['POST'])
def status():
    response = requests.get(application.config['URI'] + '/remote_status?user_key=' + application.config['ROBOT_NAME'], verify=False)
    return response.text

if __name__ == '__main__':
   application.run(host="0.0.0.0", port=8080)
