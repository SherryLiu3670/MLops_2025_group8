from contextlib import asynccontextmanager
import sys
import hydra
from hydra.utils import get_original_cwd
import os
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from PIL import Image
from pydantic import BaseModel

import onnxruntime as ort
import wandb

current_module_path = os.path.abspath(__file__)
current_module_dir = os.path.dirname(current_module_path)


def initialize_hydra():
    hydra.initialize(config_path="../../configs")
    return hydra.compose(config_name="config", overrides=["checkpoint=resnet18"])

sess_options = ort.SessionOptions()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up the model on startup and shutdown."""
    global model, device, gen_kwargs, cfg, target_labels

    cfg = initialize_hydra()
    device = cfg.hyperparams.device

    target_labels = cfg.dataset["labels"]

    gen_kwargs = {"max_length": 16, "num_beams": 8, "num_return_sequences": 1}

    yield

    print("Cleaning up...")
    del model, device, gen_kwargs, cfg, target_labels


# Create the FastAPI application with the lifespan context
app = FastAPI(lifespan=lifespan)


@app.post("/label/")
async def label(data: UploadFile = File(...), model_type: str = Form(...)):
    """Generate a prediction label for an image."""
    global cfg, target_labels  # Ensure configuration is accessible

    if model_type == 'resnet18':
        overrides = ["checkpoint=resnet18"]
    elif model_type == 'resnet34':
        overrides = ["checkpoint=resnet34"]
    else:
        overrides = ["checkpoint=mobilenet"]

    cfg = hydra.compose(config_name="config", overrides=overrides)

    # Load the uploaded image
    i_image = Image.open(data.file)
    if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")
    image = np.array(i_image)
    
    # Resize the image if desired input resolution is provided
    if "desired_input_resolution" in cfg.model:
        image = Image.fromarray(image).resize(cfg.model.desired_input_resolution)
        image = np.array(image)

    # Prepare the input data
    image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
    image = image[None, ...]
    input_data = {"input.1": image}

    run = wandb.init(project="fruit_vegetable_classification", job_type="test", resume="allow")
    artifact = run.use_artifact(cfg.checkpoint.modelpath, type='model')
    artifact_dir = artifact.download("./models")  

    model_checkpoint = os.path.join(artifact_dir, cfg.checkpoint.name[:-4] + ".onnx")
    print(f"Loading model from: {model_checkpoint}")
    model = ort.InferenceSession(model_checkpoint, sess_options)  

    y_pred = model.run(None, input_data)[0]
    pred = target_labels[int(np.argmax(y_pred))]

    return {"prediction": pred}
