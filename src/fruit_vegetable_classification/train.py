"""Train a model for classification task."""
from typing import Dict, List
import hydra
import matplotlib.pyplot as plt
import torch

import data
from loguru import logger 
import wandb
from sklearn.metrics import precision_score, recall_score, f1_score

import omegaconf

def train_fn(cfg, stats_callback=None) -> None:
    """Train a model for classification task."""
    lr = cfg.hyperparams.lr
    batch_size = cfg.hyperparams.batch_size
    epochs = cfg.hyperparams.epochs

    device = cfg.hyperparams.device
    # check if the device in configuration is supported otherwise fallback to cpu
    if device != "cpu" and not torch.cuda.is_available():
        device = "cpu"

    # generate model name
    model_name = f"{cfg.model.model_config.model_type}_lr{lr}_batch{batch_size}_epochs{epochs}"

    # Initialize wandb
    wandbcfg = wandb.config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    wandb.init(
        project="fruit_vegetable_classification",
        name=model_name,
        config=wandbcfg,
    )
    logger.info("Initialized wandb")
    
    # Log the training configuration
    logger.info("Training configuration details:")
    logger.info(f"Learning rate: {lr}, Batch size: {batch_size}, Epochs: {epochs}, Device: {device}")

    # instead of fixating the model, we can use the config file to define the model
    # replace models input_channels parameter with the number of channels in the dataset keeping hydra.utils.instantiate(cfg.model) as is
    # replace models num_classes parameter with the number of classes in the dataset keeping hydra.utils.instantiate(cfg.model) as is
    cfg.model.model_config.input_channels = cfg.dataset.input_channels
    cfg.model.model_config.num_classes = cfg.dataset.num_classes
    model = hydra.utils.instantiate(cfg.model.model_config).to(device)

    # Load the train dataset and create dataloader
    train_dataset_class = getattr(data, cfg.dataset.train_class)
    desired_resolution = cfg.model.desired_input_resolution if "desired_input_resolution" in cfg.model else None
    train_set = train_dataset_class(**cfg.dataset.process_config, desired_resolution=desired_resolution)
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # Load the validation dataset and create dataloader
    validate_dataset_class = getattr(data, cfg.dataset.val_class)
    validate_set = validate_dataset_class(**cfg.dataset.process_config, desired_resolution=desired_resolution)
    validate_dataloader = torch.utils.data.DataLoader(validate_set, batch_size=batch_size)

    # Initialize the loss function and optimizer
    loss_fn = hydra.utils.instantiate(cfg.loss)
    optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters())

    # Training loop
    logger.info("Starting training")
    best_validation_loss = float('inf')
    best_epoch = -1
    statistics = {"train_loss": [], "train_accuracy": [], 
                  "validation_loss": [], "validation_accuracy": [],
                  "validation_precision": [], "validation_recall": [],"validation_f1": [],}
        
    for epoch in range(epochs):
        # training step
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(device), target.to(device)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % 100 == 0:
                logger.info(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

        # Validation step (every epoch)
        model.eval()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_precision = 0.0
        epoch_recall = 0.0
        epoch_f1 = 0.0

        with torch.no_grad():
            for img_val, target_val in validate_dataloader:
                img_val, target_val = img_val.to(device), target_val.to(device)

                y_pred_val = model(img_val)
                loss_val = loss_fn(y_pred_val, target_val)
                epoch_loss += loss_val.item()

                y_pred_classes = y_pred_val.argmax(dim=1)
                accuracy_val = (y_pred_classes == target_val).float().mean().item()
                epoch_accuracy += accuracy_val

                epoch_precision += precision_score(target_val.cpu().numpy(),y_pred_classes.cpu().numpy(),average="macro", zero_division=1)
                epoch_recall += recall_score(target_val.cpu().numpy(),y_pred_classes.cpu().numpy(),average="macro", zero_division=1)
                epoch_f1 += f1_score(target_val.cpu().numpy(),y_pred_classes.cpu().numpy(),average="macro", zero_division=1)
                

        epoch_loss       /= len(validate_dataloader)
        epoch_accuracy   /= len(validate_dataloader)
        epoch_precision  /= len(validate_dataloader)
        epoch_recall     /= len(validate_dataloader)
        epoch_f1         /= len(validate_dataloader)

        statistics["validation_loss"].append(epoch_loss)
        statistics["validation_accuracy"].append(epoch_accuracy)
        statistics["validation_precision"].append(epoch_precision)
        statistics["validation_recall"].append(epoch_recall)
        statistics["validation_f1"].append(epoch_f1)

        logger.info(
            f"Epoch {epoch} - Validation Summary: "
            f"Loss: {epoch_loss:.4f}, "
            f"Accuracy: {epoch_accuracy:.4f}, "
            f"Precision: {epoch_precision:.4f}, "
            f"Recall: {epoch_recall:.4f}, "
            f"F1 Score: {epoch_f1:.4f}"
        )
        wandb.log({
            "epoch": epoch + 1,
            "validation_loss": epoch_loss,
            "validation_accuracy": epoch_accuracy,
            "validation_precision": epoch_precision,
            "validation_recall": epoch_recall,
            "validation_f1": epoch_f1,
        })

        # Save the best model
        if epoch_loss < best_validation_loss:
            best_epoch = epoch
            best_validation_loss = epoch_loss
            torch.save(model.state_dict(), "best_model.pth")
            logger.info(f"New best model saved with validation loss: {best_validation_loss:.4f}")

    # Save the last model        
    torch.save(model.state_dict(), "last_model.pth")
    
    # Log the best model to wandb
    best_model_artifact = wandb.Artifact(name=f"{model_name}_best", type="model",
                                         metadata={"loss":      statistics["validation_loss"][best_epoch],
                                                   "accuracy":  statistics["validation_accuracy"][best_epoch],
                                                   "precision": statistics["validation_precision"][best_epoch],
                                                   "recall":    statistics["validation_recall"][best_epoch],
                                                   "f1":        statistics["validation_f1"][best_epoch],
                                                   "epoch":     best_epoch + 1,
                                                   }
    )
    best_model_artifact.add_file("best_model.pth")
    wandb.log_artifact(best_model_artifact)
    
    # Log the last model to wandb
    last_model_artifact = wandb.Artifact(name=f"{model_name}_last", type="model",
                                         metadata={"loss":      statistics["validation_loss"][-1],
                                                   "accuracy":  statistics["validation_accuracy"][-1],
                                                   "precision": statistics["validation_precision"][-1],
                                                   "recall":    statistics["validation_recall"][-1],
                                                   "f1":        statistics["validation_f1"][-1],
                                                   "epoch":      epochs,
                                                   }
    )
    last_model_artifact.add_file("last_model.pth")
    wandb.log_artifact(last_model_artifact)

    if stats_callback:
        stats_callback(statistics)

    logger.info("Training complete")
    wandb.finish()

    if not stats_callback:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].plot(statistics["train_loss"])
        axs[0].set_title("Train loss")
        axs[1].plot(statistics["train_accuracy"])
        axs[1].set_title("Train accuracy")
        fig.savefig("training_statistics.png")
        
        fig_val, axs_val = plt.subplots(2, 3, figsize=(15, 10))
        axs_val[0, 0].plot(statistics["validation_loss"])
        axs_val[0, 0].set_title("Validation Loss")
        axs_val[0, 1].plot(statistics["validation_accuracy"])
        axs_val[0, 1].set_title("Validation Accuracy")
        axs_val[0, 2].plot(statistics["validation_precision"])
        axs_val[0, 2].set_title("Validation Precision")
        axs_val[1, 0].plot(statistics["validation_recall"])
        axs_val[1, 0].set_title("Validation Recall")
        axs_val[1, 1].plot(statistics["validation_f1"])
        axs_val[1, 1].set_title("Validation F1-Score")
        axs_val[1, 2].axis("off") 
        fig_val.savefig("validation_statistics.png")

@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def train(cfg) -> None:
    """Train a model for classification task."""
    train_fn(cfg)

if __name__ == "__main__":
    train()
