from flask import Flask, render_template
import requests

application = Flask(__name__)
application.config.from_object('config')

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/run')
def run():
    response = requests.get(application.config['URI'] + '/power')
     
    # Example GET invokation of the Robot API       
     #response = requests.get(application.config['URI'] + '/distance')  
        
     # Example POST invokation of the Robot API       
     #response = requests.get(application.config['URI'] + '/forward/5'
    #return response.text    
    return render_template('result.html', message=str(response.text))
    
@application.route('/status')
def status():
    response = requests.get(application.config['URI'] + '/status')
    #return response.text
    return render_template('result.html', message=str(response.text))

if __name__ == '__main__':
   application.run()
