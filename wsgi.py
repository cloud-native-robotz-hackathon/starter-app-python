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
    min_dist = 150
    noproblemo = True
    while noproblemo:
       response = requests.get(application.config['URI'] + '/distance' + '?user_key=' + application.config['APITOKEN'], headers=headers)
       print "Distance: " + response.text
       if int(response.text) > min_dist:
          response_drive = requests.post(application.config['URI'] + '/forward/10' + '?user_key=' + application.config['APITOKEN'], headers=headers)
          time.sleep(1)
       else:
          noproblemo = False
    
    # response = requests.get(application.config['URI'] + '/power' + '?user_key=' + application.config['APITOKEN'], headers=headers)
    # Example GET invokation of the Robot API       
     #response = requests.get(application.config['URI'] + '/distance' + '?user_key=' + application.config['APITOKEN'], headers=headers)  
        
     # Example POST invokation of the Robot API       
     #response = requests.post(application.config['URI'] + '/forward/5' + '?user_key=' + application.config['APITOKEN'], headers=headers)
    return render_template('result.html', message=str(response.text))
    
@application.route('/status')
def status():
    response = requests.get(application.config['URI'] + '/status' + '?user_key=' + application.config['APITOKEN'], headers=headers)
    return render_template('result.html', message=str(response.text))

if __name__ == '__main__':
   application.run()
