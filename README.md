# Robot Controller Application Template - Python/Flask

## Running the App in DevSpaces

Start Workspace based in `https://github.com/cloud-native-robotz-hackathon/starter-app-python.git`

### via Task

F1 -> Tasks: Run Task -> devfile -> devfile: “Run the application"

### via Terminal

Install requirements
```bash
$ python -m venv ~/.venv/
$ source ~/.venv/bin/activate
$ cd starter-app-python

$ pip install -r requirements.txt
$ gunicorn -b 0.0.0.0:8080 app --reload
```



## Running the App Locally

Install the required packages

```bash
$ cd starter-app-python

$ pip install -r requirements.txt
$ gunicorn -b 0.0.0.0:8080 app --reload
```


In the config.py set to following variables

- `ROBOT_API_ENDPOINT` = IP or Hostname of edge-controller endpoint. If set ROBOT_API will be ignored and overwritten to: `http://${ROBOT_API_ENDPOINT}:5000/`
- `ROBOT_API` = URL of the robot REST Api, example: http://robot:5000/
- `API_TOKEN` = The token fo your robot/team
- `INFERENCING_API` = The Url of the object detection inferencing service


To run this application on your local host:

```bash
$ gunicorn -b 0.0.0.0:8080 app --reload
```

## Interacting with the Application Locally

To interact with your booster while it's running locally, you can either open the page at `http://localhost:5000` or use the `curl` command:

```bash
$ curl http://localhost:8080/status


$ curl -X POST http://localhost:8080/run

```


## Updating the Application
To update your application:

Changing python files will automatically reload the application. For chnages in static files (*.html) you will have to restart the app

## Running the Application on the OpenShift Cluster

The app is prepared with a S2I configuration. You can directly create a pipeline from this repo in the OpenSHift Developer View.

## Local build & push

```shell
export VERSION=$(date +%Y%m%d%H%M)

git tag $VERSION

export IMAGE="quay.io/cloud-native-robotz-hackathon/starter-app-python:${VERSION}"

podman rmi $IMAGE
podman manifest rm ${IMAGE}
podman build --platform linux/amd64,linux/arm64  --manifest ${IMAGE}  .
podman manifest push ${IMAGE}

git push --tags 
```
