import os
import requests
import streamlit as st
from PIL import Image
import cv2
import numpy as np

import pdb

# Set backend API URL
API_BASE_URL = os.getenv("API_URL", "https://fruit-and-vegetable-api-34394117935.europe-west1.run.app/")
API_URL = f"http://localhost:8080/label"

# Streamlit UI
st.title("Fruit and Vegetable Classification App")

# Create a placeholder to display the prediction
prediction_placeholder = st.empty()
prediction_placeholder.text("Prediction: N/A")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded file as a PIL image
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert the image to a numpy array and resize it to 224x224
    image_np = np.array(image)
    image_resized = cv2.resize(image_np, (224, 224))

    # Save the resized image to a temporary file
    temp_image_path = "temp_img.jpg"
    cv2.imwrite(temp_image_path, image_resized)

    # Make API request
    with open(temp_image_path, "rb") as file:
        files = {"data": file}
        response = requests.post(API_URL, files=files)

    # Display the prediction result
    if response.status_code == 200:
        prediction = response.json().get("prediction", "N/A")
        prediction_placeholder.text(f"Prediction: {prediction}")
    else:
        st.error("Failed to get prediction from the API.")