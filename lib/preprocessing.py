from base64 import b64decode
from glob import glob
from io import BytesIO
from os import path
from pickle import dump
from functools import lru_cache

from PIL import Image, ImageOps
import numpy as np


def preprocess_image_file(image_path):
    image = Image.open(image_path)
    return _transform(image)


def preprocess_encoded_image(base64encoded_image):
    img_bytes = b64decode(base64encoded_image)
    image = Image.open(BytesIO(img_bytes))
    return _transform(image)


def preprocess_image_folder(data_folder='./data'):
    print('Commencing data preprocessing.')

    image_names, image_file_paths = _scan_images_folder(data_folder)

    image_data = [
        preprocess_image_file(image_path)[0] for image_path in image_file_paths
    ]
    with open(f'{data_folder}/images.pickle', 'wb') as outputfile:
        dump([image_names, image_data], outputfile)

    print('Data preprocessing done.')


def _scan_images_folder(images_folder):
    print(f'Scanning images folder {images_folder}.')

    image_file_paths = glob(path.join(images_folder, "*.jpg"))
    image_names = [
        file_path.split('/')[-1].rstrip('.jpg')
        for file_path in image_file_paths
    ]
    print(f'Found image files: {image_file_paths}.')
    print(f'Image names: {image_names}.')
    return image_names, image_file_paths


def _transform(image, image_size=640):
    """Optimized image transformation with improved memory efficiency."""
    image, ratio, dwdh = _letterbox_image(image, image_size, auto=False)
    
    # More efficient array operations
    image_array = np.array(image, dtype=np.float32)  # Convert to float32 immediately
    image_array /= 255.0  # Normalize to [0-1]
    
    # Transpose and expand dimensions in one go
    image_array = image_array.transpose((2, 0, 1))  # HWC->CHW for PyTorch model
    image_array = np.expand_dims(image_array, 0)  # Model expects an array of images
    
    # Ensure contiguous memory layout for better performance
    image_array = np.ascontiguousarray(image_array)
    
    return image_array, ratio, dwdh


@lru_cache(maxsize=64)
def _get_letterbox_params(width, height, image_size, auto=True, scaleup=True, stride=32):
    """Cache letterbox parameters for repeated image sizes."""
    shape = (width, height)
    new_shape = image_size
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    return new_unpad, r, (dw, dh)

def _letterbox_image(
        im, image_size, color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    """Optimized letterbox function with parameter caching."""
    # Get cached parameters
    new_unpad, r, (dw, dh) = _get_letterbox_params(
        im.size[0], im.size[1], image_size, auto, scaleup, stride
    )

    # Resize if needed
    if im.size != new_unpad[::-1]:
        im = im.resize(new_unpad, Image.BILINEAR)

    # Add border
    im = ImageOps.expand(im, border=(int(dw), int(dh)), fill=color)

    return im, r, (dw, dh)


if __name__ == '__main__':
    preprocess_image_folder(data_folder='/data')
