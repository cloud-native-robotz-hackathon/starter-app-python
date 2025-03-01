import random
import cv2
import numpy as np

# Add bounding boxes and confidence scores to an image
def add_model_info_to_image(image_path, model_output, scaling, padding, class_labels):
    # Read the original image
    image = cv2.imread(image_path)

    # Loop through identified objects and draw boxes & write confidence scores
    colors = {
        name: [
            random.randint(0, 255) for _ in range(3)
        ] for i, name in enumerate(class_labels)
    }
    for i, (x0, y0, x1, y1, score, cls_id) in enumerate(model_output):
        box = np.array([x0, y0, x1, y1])
        box -= np.array(padding*2)
        box /= scaling
        box = box.round().astype(np.int32).tolist()
        cls_id = int(cls_id)
        score = round(float(score), 3)
        name = class_labels[cls_id]
        color = colors[name]
        name += ' '+str(score)
        cv2.rectangle(image, box[:2], box[2:], color,2)
        cv2.putText(
            image,
            name,
            (box[0], box[1] - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            [0, 255, 0],
            thickness=2
        )
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Write amended image to file
    cv2.imwrite(image_path, img)
