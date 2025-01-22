from contextlib import asynccontextmanager
import sys
import hydra
from hydra.utils import get_original_cwd
import os
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image

import onnxruntime as ort

current_module_path = os.path.abspath(__file__)
current_module_dir = os.path.dirname(current_module_path)

# Define a function to initialize Hydra configuration
def initialize_hydra():
    hydra.initialize(config_path="../../configs")
    return hydra.compose(config_name="config", overrides=["experiment=api"])

sess_options = ort.SessionOptions()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up the model on startup and shutdown."""
    global model, device, gen_kwargs, cfg, target_labels

    # Initialize Hydra configuration
    cfg = initialize_hydra()

    # Configure device
    device = cfg.experiment.device

    original_working_directory = os.getcwd()
    model_checkpoint = os.path.join(original_working_directory, cfg.experiment.modelpath)
    # scrap off .pth extension and add .onnx
    model_checkpoint = model_checkpoint[:-4] + ".onnx"
    print(f"Loading model from: {model_checkpoint}")

    # Add the current module directory to the Python path
    sys.path.insert(0, current_module_dir)

    # Inferece session
    model = ort.InferenceSession(model_checkpoint)

    # Load target labels
    target_labels = cfg.dataset["labels"]

    # Generation parameters
    gen_kwargs = {"max_length": 16, "num_beams": 8, "num_return_sequences": 1}

    yield

    print("Cleaning up...")
    del model, device, gen_kwargs, cfg, target_labels


# Create the FastAPI application with the lifespan context
app = FastAPI(lifespan=lifespan)


@app.post("/label/")
async def label(data: UploadFile = File(...)):
    """Generate a prediction label for an image."""
    global cfg, target_labels  # Ensure configuration is accessible

    # Load the uploaded image
    i_image = Image.open(data.file)
    if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")

    # Convert the PIL image to a numpy array
    image = np.array(i_image)
    
    # Resize the image if desired input resolution is provided
    if "desired_input_resolution" in cfg.model:
        image = Image.fromarray(image).resize(cfg.model.desired_input_resolution)
        image = np.array(image)

    # Prepare the input data
    image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
    # append additional dimension to the image
    image = image[None, ...]
    input_data = {"input.1": image}
    

    # Predict with the model
    y_pred = model.run(None, input_data)[0]

    # Map prediction to the target labels
    # get argument of max value from the prediction
    pred = target_labels[int(np.argmax(y_pred))]

    return {"prediction": pred}
