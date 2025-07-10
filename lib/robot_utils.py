import requests
import time
import threading
import base64
import math
import cv2
import numpy as np
from PIL import UnidentifiedImageError
from .preprocessing import preprocess_encoded_image
from .object_detection import detect_objects
from collections import namedtuple

Coordinates = namedtuple('Coordinates', 'confidence_score x_upper_left y_upper_left x_lower_right y_lower_right object_class')

def log_with_timestamp(message):
    """Helper function to print messages with a timestamp for debugging."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] {message}")

def _write_file_thread_target(filename, data):
    """Target for the writer thread to perform the actual file I/O."""
    try:
        with open(filename, "wb") as f:
            f.write(data)
        log_with_timestamp(f"Async write to {filename} successful.")
    except Exception as e:
        log_with_timestamp(f"Async write to {filename} failed: {e}")

def write_file_async(filename, data):
    """Writes data to a file in a background thread to avoid blocking."""
    log_with_timestamp(f"Spawning background thread to write to {filename}.")
    thread = threading.Thread(target=_write_file_thread_target, args=(filename, data))
    thread.start()

def draw_detections(image_np, detections, ratio, dwdh, class_labels):
    """
    Draws bounding boxes and labels on the image based on detected objects.
    """
    if detections is None or len(detections) == 0:
        return image_np

    dw, dh = dwdh
    dw_half, dh_half = dw / 2, dh / 2

    if isinstance(ratio, (list, tuple)) and len(ratio) == 2:
        rw, rh = ratio
    else:
        rw, rh = ratio, ratio

    display_image = image_np.copy()
    original_h, original_w = image_np.shape[:2]
    
    unpadded_w = 640 - dw
    unpadded_h = 640 - dh

    if unpadded_w == 0 or unpadded_h == 0:
        print("Warning: Unpadded image dimensions are zero, cannot calculate scale factors.")
        return display_image

    scale_x = original_w / unpadded_w
    scale_y = original_h / unpadded_h

    detections_np = np.array(detections, dtype=np.float32)

    x1s = (detections_np[:, 0] - dw_half) * scale_x
    y1s = (detections_np[:, 1] - dh_half) * scale_y
    x2s = (detections_np[:, 2] - dw_half) * scale_x
    y2s = (detections_np[:, 3] - dh_half) * scale_y

    x1s = np.clip(x1s, 0, original_w - 1).astype(int)
    y1s = np.clip(y1s, 0, original_h - 1).astype(int)
    x2s = np.clip(x2s, 0, original_w - 1).astype(int)
    y2s = np.clip(y2s, 0, original_h - 1).astype(int)

    for i in range(len(detections_np)):
        x1, y1, x2, y2 = x1s[i], y1s[i], x2s[i], y2s[i]
        conf = detections_np[i, 4]
        class_id = int(detections_np[i, 5])

        cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{class_labels[class_id]}: {conf:.2f}" if class_id < len(class_labels) else f"Unknown: {conf:.2f}"
        
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = x1
        text_y = y1 - 10 if y1 - 10 > text_size[1] else y1 + text_size[1] + 10

        cv2.putText(display_image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    return display_image

def get_stream_data(config, class_labels):
    """
    Fetches an image from the robot, detects objects, draws boxes,
    and returns the processed image data as a dictionary and a status code.
    """
    try:
        img_response = requests.get(config['ROBOT_API'] + '/camera'+ '?user_key=' + config['ROBOT_NAME'], verify=False)
        base64_robot_image = img_response.text
        image_data_np, ratio, dwdh = preprocess_encoded_image(base64_robot_image)
        
        objects = detect_objects(
            image_data_np,
            config['INFERENCING_API'] ,
            token=config['INFERENCING_API_TOKEN'],
            classes_count=len(class_labels),
            confidence_threshold=0.15,
            iou_threshold=0.4
        )
        
        img_bytes = base64.b64decode(base64_robot_image)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        original_cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if original_cv_image is None:
            raise ValueError("Could not decode base64 robot image into OpenCV format.")

        image_with_detections = draw_detections(original_cv_image, objects, ratio, dwdh, class_labels)
        
        success, encoded_image = cv2.imencode(".jpeg", image_with_detections, [int(cv2.IMWRITE_JPEG_QUALITY), 75])

        if not success:
            raise ValueError("Failed to encode image with detections.")

        base64_string = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
        
        return {"image": base64_string}, 200
    except UnidentifiedImageError as e:
        log_with_timestamp(f"UnidentifiedImageError in get_stream_data: {e}")
        return {"error": "Cannot identify image file from robot response."}, 500
    except Exception as e:
        log_with_timestamp(f"An unexpected error occurred in get_stream_data: {e}")
        return {"error": f"Failed to get stream: {str(e)}"}, 500

def take_picture(config, image_file_name):
    log_with_timestamp(f"Executing take_picture for '{image_file_name}'...")
    img_response = requests.get(config['ROBOT_API'] + '/camera'+ '?user_key=' + config['ROBOT_NAME'], verify=False)
    
    image_bytes = base64.urlsafe_b64decode(img_response.text)
    write_file_async(image_file_name, image_bytes)
    
    log_with_timestamp("take_picture finished (file writing started in background).")
    return img_response

def take_picture_and_detect_objects(config, class_labels):
    log_with_timestamp("Executing take_picture_and_detect_objects...")
    img_response = take_picture(config, 'static/current_view.jpg')
    
    try:
        image_data_for_detection, ratio, dwdh = preprocess_encoded_image(img_response.text)
    except UnidentifiedImageError:
        log_with_timestamp("ERROR: Cannot identify image file from robot response.")
        return None
    
    log_with_timestamp("Detecting objects...")
    objects = detect_objects(
        image_data_for_detection,
        config['INFERENCING_API'],
        token=config['INFERENCING_API_TOKEN'],
        classes_count=len(class_labels),
        confidence_threshold=0.15,
        iou_threshold=0.2
    )
    log_with_timestamp(f"Detection finished. Found {len(objects) if objects is not None else 0} objects.")

    img_bytes = base64.b64decode(img_response.text)
    original_cv_image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    if original_cv_image is not None:
        image_with_detections = draw_detections(original_cv_image, objects, ratio, dwdh, class_labels)
        success, encoded_image_bytes = cv2.imencode(".jpg", image_with_detections)
        if success:
            write_file_async('static/current_view_box.jpg', encoded_image_bytes.tobytes())
    
    return objects

def find_highest_score(objects):
    if objects is None:
        return None
    detected_object = [0,0,0,0,0,0]
    for o in objects:
        if o[-1] == 0 and detected_object[-2] < o[-2]:
            detected_object = o
    if (detected_object[-2] > 0):
        return Coordinates(detected_object[-2], detected_object[0], detected_object[1], detected_object[2], detected_object[3], detected_object[-1])
    return None

def bypass_obstacle(config, min_distance_to_obstacle, angle_delta):
    log_with_timestamp("bypass_obstacle: Checking distance...")
    dist = distance_int(config)
    log_with_timestamp(f"bypass_obstacle: Distance is {dist}mm.")

    if dist <= min_distance_to_obstacle:
        log_with_timestamp("bypass_obstacle: Obstacle detected.")
        take_picture(config, 'static/current_view.jpg')
        distance_to_object = distance_int(config)
        turn_left(config, angle_delta)
        if distance_int(config) > min_distance_to_obstacle:
            move_forward(config, 20)
            turn_right(config, angle_delta)
        if distance_int(config) > min_distance_to_obstacle:
            move_forward(config, math.ceil(distance_to_object / 10) + 40)
        return True
    return False

def search_for_hat_step(config, turn_counter, class_labels, confidence_threshold, image_resolution_x, delta_threshold, hat_found_and_intercepted_ref):
    objects = take_picture_and_detect_objects(config, class_labels)
    
    if objects is None:
        log_with_timestamp("search_for_hat_step: Skipping due to image processing error.")
        time.sleep(1) 
        return turn_counter

    coordinates = find_highest_score(objects)

    if coordinates and coordinates.confidence_score > confidence_threshold:
        log_with_timestamp("Hat candidate found. Aligning and approaching.")
        center_x = (coordinates.x_upper_left + coordinates.x_lower_right) / 2
        
        if abs(image_resolution_x/2 - center_x) >= 20:
            if center_x < 320: turn_left(config, 10)
            else: turn_right(config, 9)
        else:
            delta = coordinates.x_lower_right - coordinates.x_upper_left
            if delta < delta_threshold:
                move_forward(config, 10)
            else:
                hat_found_and_intercepted_ref[0] = True
                log_with_timestamp('### Hat Intercepted! ###')
    else:
        log_with_timestamp("No hat found. Continuing search pattern.")
        if turn_counter <= 360:
            turn_right(config, 10)
            turn_counter += 10
        else:
            move_forward(config, 40)
            turn_counter = 0
            
    return turn_counter

def move_forward(config, length):
    log_with_timestamp(f"Sending command: move_forward({length})")
    requests.post(f"{config['ROBOT_API']}/forward/{length}",data={"user_key": config['ROBOT_NAME']} ,verify=False)
def move_backward(config, length):
    log_with_timestamp(f"Sending command: move_backward({length})")
    requests.post(f"{config['ROBOT_API']}/backward/{length}",data={"user_key": config['ROBOT_NAME']} ,verify=False)
def turn_left(config, degrees):
    log_with_timestamp(f"Sending command: turn_left({degrees})")
    requests.post(f"{config['ROBOT_API']}/left/{degrees}",data={"user_key": config['ROBOT_NAME']} ,verify=False)
def turn_right(config, degrees):
    log_with_timestamp(f"Sending command: turn_right({degrees})")
    requests.post(f"{config['ROBOT_API']}/right/{degrees}",data={"user_key": config['ROBOT_NAME']} ,verify=False)
def distance(config):
    return requests.get(f"{config['ROBOT_API']}/distance?user_key={config['ROBOT_NAME']}" ,verify=False).text
def distance_int(config):
    return int(''.join(distance(config)))
