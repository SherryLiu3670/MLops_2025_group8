import threading
import time
import numpy as np
import requests
from PIL import Image
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes
import numpy as np
import av

import pdb

# Set backend API URL
API_URL = "http://localhost:8000/label/"

# Streamlit UI
st.title("Fruit and Vegetable Classification App")

# Create a placeholder to display the prediction
prediction_placeholder = st.empty()
prediction_placeholder.text("Prediction: N/A")

prediction = "N/A"

# Initialize the WebRTC streamer
def transform(frame: av.VideoFrame):

    global prediction

    # Convert the frame to an image
    img = frame.to_ndarray(format="bgr24")

    # resize it to 224x224 and save the img as jpeg
    img = cv2.resize(img, (224, 224))
    cv2.imwrite("img.jpg", img)

    # Make API request
    files = {"data": open("img.jpg", "rb")}
    response = requests.post(API_URL, files=files)

    if response.status_code == 200:
        prediction = response.json().get("prediction", "N/A")

webrtc_streamer(
    key="streamer",
    video_frame_callback=transform,
    sendback_audio=False
    )

# setup a thread to update prediction_placeholder with the prediction every 1 second

# Function to update the UI with predictions every 1 second
while True:
    prediction_placeholder.text(f"Prediction: {prediction}")
    time.sleep(1)  # Update every 1 second
