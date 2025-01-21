from contextlib import asynccontextmanager
import sys
import hydra
from hydra.utils import get_original_cwd
import os
import torch
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile
from PIL import Image

current_module_path = os.path.abspath(__file__)
current_module_dir = os.path.dirname(current_module_path)

# Define a function to initialize Hydra configuration
def initialize_hydra():
    hydra.initialize(config_path="../../configs")
    return hydra.compose(config_name="config", overrides=["experiment=api"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up the model on startup and shutdown."""
    global model, device, gen_kwargs, cfg, target_labels

    # Initialize Hydra configuration
    cfg = initialize_hydra()

    # Configure device
    device = cfg.experiment.device
    if device != "cpu" and not torch.cuda.is_available():
        device = "cpu"

    original_working_directory = os.getcwd()
    model_checkpoint = os.path.join(original_working_directory, cfg.experiment.modelpath)
    print(f"Loading model from: {model_checkpoint}")

    # Add the current module directory to the Python path
    sys.path.insert(0, current_module_dir)

    # Configure the model with additional attributes
    cfg.model.model_config.input_channels = cfg.dataset.input_channels
    model = hydra.utils.instantiate(cfg.model.model_config).to(device)

    # Load model weights
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    model.eval()

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

    # Convert the image to a tensor
    image_tensor = transforms.ToTensor()(i_image).unsqueeze(0)

    # Resize the image if desired input resolution is provided
    if "desired_input_resolution" in cfg.model:
        image_tensor = transforms.Resize(cfg.model.desired_input_resolution)(image_tensor)
    image_tensor = image_tensor.to(device)

    # Predict with the model
    y_pred = model(image_tensor)

    # Map prediction to the target labels
    pred = target_labels[y_pred.argmax(dim=1).item()]
    return {"prediction": pred}
