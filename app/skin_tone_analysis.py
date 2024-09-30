import cv2
import numpy as np
from PIL import Image
import requests
from fastapi import HTTPException
from sklearn.cluster import KMeans
from io import BytesIO


async def analyze_skin_tone(image_url: str):
    try:
        result = get_skin_tone(image_url)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def get_skin_tone(image_url: str):
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        img = np.array(img)

        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        elif len(img.shape) == 2 or img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            return {"error": "No faces detected"}

        x, y, w, h = faces[0]
        face_roi = img[y:y+h, x:x+w]
        h_roi, w_roi = face_roi.shape[:2]
        skin_sample_region = face_roi[int(
            0.4*h_roi):int(0.6*h_roi), int(0.3*w_roi):int(0.7*w_roi)]
        pixels = skin_sample_region.reshape(-1, 3)

        kmeans = KMeans(n_clusters=1)
        kmeans.fit(pixels)
        dominant_color = kmeans.cluster_centers_[0]

        skin_tone_rgb = [int(dominant_color[2]), int(
            dominant_color[1]), int(dominant_color[0])]
        skin_tone_hex = "#{:02x}{:02x}{:02x}".format(
            int(dominant_color[2]), int(dominant_color[1]), int(dominant_color[0]))

        season, sub_season = classify_season_and_sub_season(skin_tone_rgb)

        complement_colors = generate_complement_colors(skin_tone_rgb)

        return {
            "skin_tone": skin_tone_hex,
            "season": season,
            "sub_season": sub_season,
            "complements": complement_colors
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def classify_season_and_sub_season(skin_tone_rgb):
    # Same classification logic as in your original Django code
    pass


def generate_complement_colors(skin_tone_rgb):
    # Generate complement colors based on the same logic
    pass
