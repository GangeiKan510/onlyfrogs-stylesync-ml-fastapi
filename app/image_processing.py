from PIL import Image
import io
import numpy as np
import cv2


def process_image(image_bytes: bytes):
    # Open the image with PIL
    image = Image.open(io.BytesIO(image_bytes))

    # Convert to NumPy array for processing
    np_image = np.array(image)

    # Example: Convert image to grayscale using OpenCV
    gray_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)

    # Perform more complex operations here if needed
    # Save or return the processed image in desired format
    return gray_image
