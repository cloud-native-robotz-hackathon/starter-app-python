from flask import Flask, render_template
import requests
import time

application = Flask(__name__)
application.config.from_object('config')
headers = {'accept': 'text/html'}

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/run')
def run():
    # response = requests.get(application.config['URI'] + '/power' + '?user_key=' + application.config['APITOKEN'], headers=headers,verify=False)
    # Example GET invokation of the Robot API       
    response = requests.get(application.config['URI'] + '/distance' + '?user_key=' + application.config['APITOKEN'], headers=headers, verify=False)  
        
     # Example POST invokation of the Robot API       
     #response = requests.post(application.config['URI'] + '/forward/5' + '?user_key=' + application.config['APITOKEN'], headers=headers, verify=False)
    return render_template('result.html', message=str(response.text))
    
@application.route('/status')
def status():
    response = requests.get(application.config['URI'] + '/remote_status' + '?user_key=' + application.config['APITOKEN'], headers=headers, verify=False)
    return render_template('result.html', message=str(response.text))

if __name__ == '__main__':
   application.run()
