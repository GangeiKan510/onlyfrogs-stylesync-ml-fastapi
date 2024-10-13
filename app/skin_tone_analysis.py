import cv2
import numpy as np
from PIL import Image
import requests
from fastapi import HTTPException
from sklearn.cluster import KMeans
from io import BytesIO


def process_image(image_bytes: bytes):
    image = Image.open(BytesIO(image_bytes))

    np_image = np.array(image)

    if np_image.shape[2] == 4:
        np_image = cv2.cvtColor(np_image, cv2.COLOR_RGBA2RGB)
    elif len(np_image.shape) == 2 or np_image.shape[2] == 1:
        np_image = cv2.cvtColor(np_image, cv2.COLOR_GRAY2RGB)

    return np_image


async def analyze_skin_tone(image_url: str):
    try:
        response = requests.get(image_url)
        if response.status_code != 200:
            raise HTTPException(
                status_code=400, detail="Failed to download the image.")

        processed_image = process_image(response.content)

        result = get_skin_tone_from_processed_image(processed_image)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def get_skin_tone_from_processed_image(processed_image: np.ndarray):
    try:
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        if len(faces) == 0:
            raise HTTPException(
                status_code=400, detail="No faces detected. Please retake the image with a clear face.")

        if len(faces) > 1:
            raise HTTPException(
                status_code=400, detail="Multiple faces detected. Please ensure only one face is visible.")

        x, y, w, h = faces[0]

        img_height, img_width = processed_image.shape[:2]
        face_area = w * h
        img_area = img_height * img_width

        if face_area / img_area < 0.05:
            raise HTTPException(
                status_code=400, detail="Face is too small. Please retake the image with a closer view of your face.")

        face_roi = processed_image[y:y + h, x:x + w]
        h_roi, w_roi = face_roi.shape[:2]
        skin_sample_region = face_roi[int(
            0.4 * h_roi):int(0.6 * h_roi), int(0.3 * w_roi):int(0.7 * w_roi)]
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

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Internal error during processing: {str(e)}")


def classify_season_and_sub_season(skin_tone_rgb):
    hsv = cv2.cvtColor(np.uint8([[skin_tone_rgb]]), cv2.COLOR_BGR2HSV)[0][0]
    hue, saturation, value = hsv
    warm_undertone = (hue < 30 or hue > 150)

    if value > 200:
        depth = "light"
    elif value < 85:
        depth = "deep"
    else:
        depth = "medium"

    if saturation > 150:
        chroma = "bright"
    else:
        chroma = "soft"

    if warm_undertone:
        if depth == "light":
            if chroma == "bright":
                return ("spring", "light spring")
            else:
                return ("autumn", "soft autumn")
        elif depth == "deep":
            return ("autumn", "deep autumn")
        else:
            if chroma == "bright":
                return ("spring", "warm spring")
            else:
                return ("autumn", "warm autumn")
    else:
        if depth == "light":
            if chroma == "soft":
                return ("summer", "light summer")
            else:
                return ("winter", "bright winter")
        elif depth == "deep":
            return ("winter", "deep winter")
        else:
            if chroma == "soft":
                return ("summer", "cool summer")
            else:
                return ("winter", "cool winter")


def generate_complement_colors(skin_tone_rgb):
    complement_colors = []
    for i in range(12):
        complementary_hue = (cv2.cvtColor(
            np.uint8([[skin_tone_rgb]]), cv2.COLOR_BGR2HSV)[0][0][0] + (i * 15)) % 180

        saturation = np.random.randint(70, 180)
        value = np.random.randint(100, 250)

        complement_color = cv2.cvtColor(np.uint8(
            [[[complementary_hue, saturation, value]]]), cv2.COLOR_HSV2BGR)[0][0]
        complement_color_hex = "#{:02x}{:02x}{:02x}".format(
            int(complement_color[2]), int(complement_color[1]), int(complement_color[0]))
        complement_colors.append(complement_color_hex)

    neutral_colors = ['#000000', '#808080', '#A9A9A9', '#D3D3D3']
    complement_colors.extend(neutral_colors)

    return complement_colors
