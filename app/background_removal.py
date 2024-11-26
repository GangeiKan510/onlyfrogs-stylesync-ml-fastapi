# background_removal.py
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import requests
from fastapi import HTTPException
from rembg import remove
from PIL import Image
from io import BytesIO
from datetime import datetime
import pyrebase
from config import firebase_config  # Import firebase_config from config

firebase = pyrebase.initialize_app(firebase_config)
storage = firebase.storage()


# Initialize Firebase with the imported config
firebase = pyrebase.initialize_app(firebase_config)
storage = firebase.storage()


async def remove_background(image_url: str):
    try:
        response = requests.get(image_url)
        if response.status_code != 200:
            raise HTTPException(
                status_code=400, detail="Failed to download the image.")

        input_image = Image.open(BytesIO(response.content))
        output_image = remove(input_image)

        output_image = output_image.resize((512, 512))

        enhancer = ImageEnhance.Contrast(output_image)
        output_image = enhancer.enhance(2)

        output_image = output_image.convert("L")

        output_image_np = np.array(output_image)
        blurred_image = cv2.GaussianBlur(output_image_np, (5, 5), 0)
        output_image = Image.fromarray(blurred_image)

        output_image_io = BytesIO()
        output_image.save(output_image_io, format='PNG')
        output_image_io.seek(0)

        # Firebase upload
        filename = f"output_images/{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        storage.child(filename).put(output_image_io)

        file_url = storage.child(filename).get_url(None)
        return {"message": "Image processed and uploaded successfully.", "file_url": file_url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
