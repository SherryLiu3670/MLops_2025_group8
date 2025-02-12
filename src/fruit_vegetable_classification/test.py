"""Test a model on a dataset."""
import os

import hydra
import torch
import torchvision.transforms as transforms
from hydra.utils import get_original_cwd
from torch.utils.data import DataLoader

import data
import wandb

from loguru import logger 


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def test(cfg) -> None:
    """Test a model on a dataset."""
    device = cfg.hyperparams.device
    batch_size = cfg.hyperparams.batch_size

    logger.info("Testing the model")

    # Initialize W&B
    run = wandb.init(project="fruit_vegetable_classification", job_type="test")

    # Use the artifact from W&B
    artifact = run.use_artifact(cfg.checkpoint.modelpath, type='model')
    artifact_dir = artifact.download("./models")  

    # Load the model
    model_checkpoint = os.path.join(artifact_dir, cfg.checkpoint.name)

    # Load the model
    cfg.model.model_config.input_channels = cfg.dataset.input_channels
    cfg.model.model_config.num_classes = cfg.dataset.num_classes
    model = hydra.utils.instantiate(cfg.model.model_config).to(device)
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    model.eval()

    # Load the test dataset and create dataloader
    test_dataset_class = getattr(data, cfg.dataset.test_class)
    desired_resolution = cfg.model.desired_input_resolution if "desired_input_resolution" in cfg.model else None
    test_set = test_dataset_class(**cfg.dataset.process_config, desired_resolution=desired_resolution)
    test_dataloader = DataLoader(test_set, batch_size=batch_size)

    # Initialize the loss function
    loss_fn = hydra.utils.instantiate(cfg.loss)

    total_loss = 0.0
    total_accuracy = 0.0

    with torch.no_grad():
        for img, target in test_dataloader:
            img, target = img.to(device), target.to(device)

            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            total_loss += loss.item()

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            total_accuracy += accuracy

    avg_loss = total_loss / len(test_dataloader)
    avg_accuracy = total_accuracy / len(test_dataloader)

    logger.info(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {avg_accuracy:.4f}")


if __name__ == "__main__":
    test()
