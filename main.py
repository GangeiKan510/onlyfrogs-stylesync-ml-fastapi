# main.py
from app.skin_tone_analysis import analyze_skin_tone
from app.background_removal import remove_background
from fastapi import FastAPI, Form
from config import firebase_config  # Import firebase_config from config

# Initialize FastAPI
app = FastAPI()

# Define routes


@app.get('/')
async def home():
    return "Stylesync FastAPI Server"


@app.post("/remove-background/")
async def remove_background_route(image_url: str = Form(...)):
    return await remove_background(image_url)


@app.post("/analyze-skin-tone/")
async def analyze_skin_tone_route(image_url: str = Form(...)):
    return await analyze_skin_tone(image_url)
