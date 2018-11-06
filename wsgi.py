from flask import Flask, render_template
import requests
application = Flask(__name__)

uri = 'http://hub-controller-live-hub-controller.apps-9d00.generic.opentlc.com/api/robot'

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/run')
def run():
    response = requests.get(uri + '/power')
     
    # Example GET invokation of the Robot API       
     #response = requests.get(uri + '/distance')  
        
     # Example POST invokation of the Robot API       
     #response = requests.get(uri + '/forward/5'
    #return response.text    
    return render_template('result.html', message=str(response.text))
    
@application.route('/status')
def status():
    response = requests.get(uri + '/status')
    return response.text

if __name__ == '__main__':
   application.run()
