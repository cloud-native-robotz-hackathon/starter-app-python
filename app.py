from flask import Flask, render_template
import requests
import time

application = Flask(__name__)
application.config.from_object('config')

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/run', methods=['POST'])
def run():
    # Example GET invocations of the Robot API       
    response = requests.get('http://' + application.config['URI'] + ':5000/distance', verify=False)  
    # response = requests.get(application.config['URI'] + '/power' + '?user_key=' + application.config['APITOKEN'],verify=False)
    
    # Example POST invocation of the Robot API for e.g. moving  
    response = requests.post('http://' + application.config['URI'] + ':5000/forward/10', verify=False)
    return response.text
    #return render_template('result.html', message=str(response.text))	

   
@application.route('/status', methods=['POST'])
def status():
    response = requests.get(application.config['URI'] + '/remote_status' , verify=False)
    return response.text
    #return render_template('result.html', message=str(response.text))

if __name__ == '__main__':
   application.run(host="0.0.0.0", port=8080)
