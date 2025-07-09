import requests
import time
import threading
import base64
import math
import cv2
import numpy as np
from PIL import UnidentifiedImageError
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from .preprocessing import preprocess_encoded_image
from .object_detection import detect_objects
from collections import namedtuple

Coordinates = namedtuple('Coordinates', 'confidence_score x_upper_left y_upper_left x_lower_right y_lower_right object_class')

# Thread pool for I/O operations
io_executor = ThreadPoolExecutor(max_workers=3)

# Session for connection pooling
session = requests.Session()
session.mount('http://', requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=20))
session.mount('https://', requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=20))

# Cache for distance measurements (short duration)
distance_cache = {}
distance_cache_timeout = 0.5  # Cache for 0.5 seconds

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
    io_executor.submit(_write_file_thread_target, filename, data)

@lru_cache(maxsize=32)
def get_cached_scale_factors(original_w, original_h, unpadded_w, unpadded_h):
    """Cache scale factor calculations to avoid redundant computations."""
    if unpadded_w == 0 or unpadded_h == 0:
        return 1.0, 1.0
    return original_w / unpadded_w, original_h / unpadded_h

def draw_detections(image_np, detections, ratio, dwdh, class_labels):
    """
    Optimized version of draw_detections with caching and vectorized operations.
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

    # Use cached scale factors
    scale_x, scale_y = get_cached_scale_factors(original_w, original_h, unpadded_w, unpadded_h)

    if scale_x == 1.0 and scale_y == 1.0:
        print("Warning: Using default scale factors due to zero dimensions.")
        return display_image

    # Vectorized operations for better performance
    detections_np = np.array(detections, dtype=np.float32)

    x1s = (detections_np[:, 0] - dw_half) * scale_x
    y1s = (detections_np[:, 1] - dh_half) * scale_y
    x2s = (detections_np[:, 2] - dw_half) * scale_x
    y2s = (detections_np[:, 3] - dh_half) * scale_y

    x1s = np.clip(x1s, 0, original_w - 1).astype(int)
    y1s = np.clip(y1s, 0, original_h - 1).astype(int)
    x2s = np.clip(x2s, 0, original_w - 1).astype(int)
    y2s = np.clip(y2s, 0, original_h - 1).astype(int)

    # Batch drawing for better performance
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
    Optimized version with connection pooling and error handling.
    """
    try:
        # Use session for connection pooling
        img_response = session.get(
            config['ROBOT_API'] + '/camera' + '?user_key=' + config['ROBOT_NAME'], 
            verify=False, 
            timeout=10
        )
        
        if img_response.status_code != 200:
            raise ValueError(f"Failed to fetch image from robot: {img_response.status_code}")
            
        base64_robot_image = img_response.text
        
        # Process image in parallel with object detection
        def process_image():
            return preprocess_encoded_image(base64_robot_image)
        
        def detect_objects_async():
            image_data_np, _, _ = preprocess_encoded_image(base64_robot_image)
            return detect_objects(
                image_data_np,
                config['INFERENCING_API'],
                token=config['INFERENCING_API_TOKEN'],
                classes_count=len(class_labels),
                confidence_threshold=0.15,
                iou_threshold=0.4
            )
        
        # Execute preprocessing and detection in parallel
        future_image = io_executor.submit(process_image)
        future_objects = io_executor.submit(detect_objects_async)
        
        # Get results
        image_data_np, ratio, dwdh = future_image.result()
        objects = future_objects.result()
        
        # Decode original image
        img_bytes = base64.b64decode(base64_robot_image)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        original_cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if original_cv_image is None:
            raise ValueError("Could not decode base64 robot image into OpenCV format.")

        image_with_detections = draw_detections(original_cv_image, objects, ratio, dwdh, class_labels)
        
        # Use lower quality for faster encoding
        success, encoded_image = cv2.imencode(
            ".jpeg", 
            image_with_detections, 
            [int(cv2.IMWRITE_JPEG_QUALITY), 60]  # Reduced from 75 to 60
        )

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
    """Optimized with connection pooling and timeout."""
    log_with_timestamp(f"Executing take_picture for '{image_file_name}'...")
    
    try:
        img_response = session.get(
            config['ROBOT_API'] + '/camera' + '?user_key=' + config['ROBOT_NAME'], 
            verify=False, 
            timeout=10
        )
        
        if img_response.status_code != 200:
            log_with_timestamp(f"Failed to get image: {img_response.status_code}")
            return None
            
        image_bytes = base64.urlsafe_b64decode(img_response.text)
        write_file_async(image_file_name, image_bytes)
        
        log_with_timestamp("take_picture finished (file writing started in background).")
        return img_response
        
    except Exception as e:
        log_with_timestamp(f"Error in take_picture: {e}")
        return None

def take_picture_and_detect_objects(config, class_labels):
    """Optimized with parallel processing and error handling."""
    log_with_timestamp("Executing take_picture_and_detect_objects...")
    
    img_response = take_picture(config, 'static/current_view.jpg')
    if img_response is None:
        return None
    
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

    # Process image rendering in background
    def render_image():
        try:
            img_bytes = base64.b64decode(img_response.text)
            original_cv_image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

            if original_cv_image is not None:
                image_with_detections = draw_detections(original_cv_image, objects, ratio, dwdh, class_labels)
                success, encoded_image_bytes = cv2.imencode(".jpg", image_with_detections)
                if success:
                    write_file_async('static/current_view_box.jpg', encoded_image_bytes.tobytes())
        except Exception as e:
            log_with_timestamp(f"Error in image rendering: {e}")
    
    # Render image in background
    io_executor.submit(render_image)
    
    return objects

def find_highest_score(objects):
    """Optimized with early exit and vectorized operations."""
    if objects is None or len(objects) == 0:
        return None
    
    # Convert to numpy for vectorized operations
    objects_np = np.array(objects)
    
    # Filter for class 0 objects
    class_0_mask = objects_np[:, -1] == 0
    if not np.any(class_0_mask):
        return None
    
    class_0_objects = objects_np[class_0_mask]
    
    # Find highest confidence
    max_conf_idx = np.argmax(class_0_objects[:, -2])
    detected_object = class_0_objects[max_conf_idx]
    
    if detected_object[-2] > 0:
        return Coordinates(
            detected_object[-2], detected_object[0], detected_object[1], 
            detected_object[2], detected_object[3], detected_object[-1]
        )
    return None

def bypass_obstacle(config, min_distance_to_obstacle, angle_delta):
    """Optimized with caching and reduced API calls."""
    log_with_timestamp("bypass_obstacle: Checking distance...")
    
    dist = distance_int_cached(config)
    log_with_timestamp(f"bypass_obstacle: Distance is {dist}mm.")

    if dist <= min_distance_to_obstacle:
        log_with_timestamp("bypass_obstacle: Obstacle detected.")
        take_picture(config, 'static/current_view.jpg')
        
        # Use cached distance to avoid redundant calls
        distance_to_object = dist
        turn_left(config, angle_delta)
        
        new_distance = distance_int_cached(config)
        if new_distance > min_distance_to_obstacle:
            move_forward(config, 20)
            turn_right(config, angle_delta)
            
        final_distance = distance_int_cached(config)
        if final_distance > min_distance_to_obstacle:
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
            else: turn_right(config, 10)
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
    """Optimized with session pooling and timeout."""
    log_with_timestamp(f"Sending command: move_forward({length})")
    try:
        session.post(
            f"{config['ROBOT_API']}/forward/{length}",
            data={"user_key": config['ROBOT_NAME']},
            verify=False,
            timeout=5
        )
    except Exception as e:
        log_with_timestamp(f"Error in move_forward: {e}")

def move_backward(config, length):
    """Optimized with session pooling and timeout."""
    log_with_timestamp(f"Sending command: move_backward({length})")
    try:
        session.post(
            f"{config['ROBOT_API']}/backward/{length}",
            data={"user_key": config['ROBOT_NAME']},
            verify=False,
            timeout=5
        )
    except Exception as e:
        log_with_timestamp(f"Error in move_backward: {e}")

def turn_left(config, degrees):
    """Optimized with session pooling and timeout."""
    log_with_timestamp(f"Sending command: turn_left({degrees})")
    try:
        session.post(
            f"{config['ROBOT_API']}/left/{degrees}",
            data={"user_key": config['ROBOT_NAME']},
            verify=False,
            timeout=5
        )
    except Exception as e:
        log_with_timestamp(f"Error in turn_left: {e}")

def turn_right(config, degrees):
    """Optimized with session pooling and timeout."""
    log_with_timestamp(f"Sending command: turn_right({degrees})")
    try:
        session.post(
            f"{config['ROBOT_API']}/right/{degrees}",
            data={"user_key": config['ROBOT_NAME']},
            verify=False,
            timeout=5
        )
    except Exception as e:
        log_with_timestamp(f"Error in turn_right: {e}")

def distance(config):
    """Optimized with session pooling and timeout."""
    try:
        response = session.get(
            f"{config['ROBOT_API']}/distance?user_key={config['ROBOT_NAME']}",
            verify=False,
            timeout=5
        )
        return response.text
    except Exception as e:
        log_with_timestamp(f"Error in distance: {e}")
        return "0"

def distance_int(config):
    """Convert distance response to integer."""
    try:
        return int(''.join(distance(config)))
    except ValueError:
        return 0

def distance_int_cached(config):
    """Cache distance measurements to reduce API calls."""
    current_time = time.time()
    cache_key = config['ROBOT_NAME']
    
    if (cache_key not in distance_cache or 
        current_time - distance_cache[cache_key]['timestamp'] > distance_cache_timeout):
        
        distance_val = distance_int(config)
        distance_cache[cache_key] = {
            'value': distance_val,
            'timestamp': current_time
        }
        return distance_val
    
    return distance_cache[cache_key]['value']
