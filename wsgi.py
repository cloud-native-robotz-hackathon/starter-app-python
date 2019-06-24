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
    #response = requests.get(application.config['URI'] + '/distance' + '?user_key=' + application.config['APITOKEN'], verify=False)  
    # response = requests.get(application.config['URI'] + '/power' + '?user_key=' + application.config['APITOKEN'],verify=False)
    
    # Example POST invocation of the Robot API for e.g. moving  
    data = {'user_key': application.config['APITOKEN']} 
    response = requests.post(application.config['URI'] + '/forward/10', data=data, verify=False)
    return response.text
    #return render_template('result.html', message=str(response.text))
    
@application.route('/status', methods=['POST'])
def status():
    response = requests.get(application.config['URI'] + '/remote_status' + '?user_key=' + application.config['APITOKEN'], verify=False)
    return response.text
    #return render_template('result.html', message=str(response.text))

if __name__ == '__main__':
   application.run()
