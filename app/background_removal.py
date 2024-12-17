import requests
from fastapi import HTTPException
from rembg import remove
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from datetime import datetime
import pyrebase
from config import firebase_config  # Import firebase_config from config

# Initialize Firebase with the imported config
firebase = pyrebase.initialize_app(firebase_config)
storage = firebase.storage()


def preprocess_image(image: Image.Image, max_size: int = 512):
    """
    Resize the image to fit within the specified max_size while preserving aspect ratio.
    """
    original_width, original_height = image.size
    if max(original_width, original_height) > max_size:
        scaling_factor = max_size / max(original_width, original_height)
        new_size = (
            int(original_width * scaling_factor),
            int(original_height * scaling_factor)
        )
        # Use LANCZOS for high-quality resizing
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    return image


async def remove_background(image_url: str):
    try:
        response = requests.get(image_url)
        if response.status_code != 200:
            raise HTTPException(
                status_code=400, detail="Failed to download the image."
            )

        try:
            input_image = Image.open(BytesIO(response.content))
            if input_image.format == "WEBP":
                input_image = input_image.convert("RGBA")
        except UnidentifiedImageError:
            raise HTTPException(
                status_code=400, detail="Unsupported image format.")

        resized_image = preprocess_image(input_image)

        output_image = remove(resized_image)

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
