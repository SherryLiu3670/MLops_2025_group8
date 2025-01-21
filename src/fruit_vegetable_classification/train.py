"""Train a model for classification task."""
from typing import Dict, List
import hydra
import matplotlib.pyplot as plt
import torch
import typer
import data
import os

import torchvision.transforms as transforms
from torch.profiler import profile, record_function, ProfilerActivity
import torch.utils.tensorboard
from loguru import logger 
import wandb

import omegaconf

def train_fn(cfg, stats_callback=None) -> None:
    """Train a model for classification task."""
    lr = cfg.experiment.lr
    batch_size = cfg.experiment.batch_size
    epochs = cfg.experiment.epochs
    
    device = cfg.experiment.device
    # check if the device in configuration is supported otherwise fallback to cpu
    if device != "cpu" and not torch.cuda.is_available():
        device = "cpu"

    wandb.init(
        project="food_classification",
        name=cfg.experiment.name,
    )
    wandb.config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    logger.info("Initialized wandb")
    
    #lr: float = 1e-3, batch_size: int = 32, epochs: int = 10
    logger.info("Training started")
    logger.info(f"Learning rate: {lr}, Batch size: {batch_size}, Epochs: {epochs}, Device: {device}")

    # instead of fixating the model, we can use the config file to define the model
    # replace models input_channels parameter with the number of channels in the dataset keeping hydra.utils.instantiate(cfg.model) as is
    cfg.model.model_config.input_channels = cfg.dataset.input_channels
    model = hydra.utils.instantiate(cfg.model.model_config).to(device)

    # preparing training dataset
    if "desired_input_resolution" in cfg.model:
        transform = transforms.Compose(
            [
                transforms.Resize((cfg.model.desired_input_resolution)),  # Resize to datasets' native resolution
            ]
        )
    else:
        transform = None

    train_dataset_class = getattr(data, cfg.dataset.train_class)
    train_set = train_dataset_class(**cfg.dataset.process_config, transform=transform)

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    validate_dataset_class = getattr(data, cfg.dataset.val_class)
    validate_set = validate_dataset_class(**cfg.dataset.process_config)
    validate_dataloader = torch.utils.data.DataLoader(validate_set, batch_size=batch_size)

    loss_fn = hydra.utils.instantiate(cfg.loss)
    optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters())

    best_validation_loss = float('inf')
    statistics = {"train_loss": [], "train_accuracy": [], "validation_loss": [], "validation_accuracy": []}

    with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
    ) as prof:
        for epoch in range(epochs):
            # training step
            model.train()
            for i, (img, target) in enumerate(train_dataloader):
                with record_function("dataloader_time"):
                    img, target = img.to(device), target.to(device)
                optimizer.zero_grad()
                with record_function("model_forward"):
                    y_pred = model(img)
                with record_function("loss_calculation"):
                    loss = loss_fn(y_pred, target)
                with record_function("backward_pass"):
                    loss.backward()
                with record_function("optimizer_step"):
                    optimizer.step()
                statistics["train_loss"].append(loss.item())

                accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
                statistics["train_accuracy"].append(accuracy)

                if i % 100 == 0:
                    logger.info(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

                prof.step()    
            
            # Validation step (every epoch)
            model.eval()
            epoch_validation_loss = 0.0
            epoch_validation_accuracy = 0.0

            with torch.no_grad():
                for img_val, target_val in validate_dataloader:
                    img_val, target_val = img_val.to(device), target_val.to(device)

                    y_pred_val = model(img_val)
                    loss_val = loss_fn(y_pred_val, target_val)
                    epoch_validation_loss += loss_val.item()

                    accuracy_val = (y_pred_val.argmax(dim=1) == target_val).float().mean().item()
                    epoch_validation_accuracy += accuracy_val

            epoch_validation_loss /= len(validate_dataloader)
            epoch_validation_accuracy /= len(validate_dataloader)

            statistics["validation_loss"].append(epoch_validation_loss)
            statistics["validation_accuracy"].append(epoch_validation_accuracy)

            logger.info(f"Epoch {epoch} - Validation Loss: {epoch_validation_loss:.4f}, Validation Accuracy: {epoch_validation_accuracy:.4f}")
            wandb.log({"epoch": epoch + 1, "validation_loss": epoch_validation_loss, "validation_accuracy": epoch_validation_accuracy})

            # Save the best model
            if epoch_validation_loss < best_validation_loss:
                best_validation_loss = epoch_validation_loss
                torch.save(model.state_dict(), "best_model.pth")
                logger.info(f"New best model saved with validation loss: {best_validation_loss:.4f}")
        
        torch.save(model.state_dict(), "last_model.pth")

    if stats_callback:
        stats_callback(statistics)

    logger.info("Training complete")
    wandb.finish()
    logger.info(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))    

    # torch.save(model.state_dict(), "../../../models/model.pth")
    # fig, axs = plt.subplots(2, 2, figsize=(15, 5))
    # axs[0][0].plot(statistics["train_loss"])
    # axs[0][0].set_title("Train loss")
    # axs[1][0].plot(statistics["train_accuracy"])
    # axs[1][0].set_title("Train accuracy")
    # axs[0][1].plot(statistics["validation_loss"])
    # axs[0][1].set_title("Validation loss")
    # axs[1][1].plot(statistics["validation_accuracy"])
    # axs[1][1].set_title("Validation accuracy")
    # fig.savefig("../../../reports/figures/training_statistics.png")


@hydra.main(config_path="../../configs", config_name="config")
def train(cfg) -> None:
    """Train a model for classification task."""
    train_fn(cfg)


if __name__ == "__main__":
    train()
