import hydra
import matplotlib.pyplot as plt
import torch
import typer
import data
# from . import data
# from . import model as models

import pdb

@hydra.main(config_path="../../configs", config_name="config")
def train(cfg) -> None:
    """Train a model on MNIST."""

    lr=cfg.experiment.lr
    batch_size=cfg.experiment.batch_size
    epochs=cfg.experiment.epochs
    
    device = cfg.experiment.device
    # check if the device in configuration is supported otherwise fallback to cpu
    if device != "cpu" and not torch.cuda.is_available():
        device = "cpu"
    
    #lr: float = 1e-3, batch_size: int = 32, epochs: int = 10
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    # instead of fixating the model, we can use the config file to define the model
    model = hydra.utils.instantiate(cfg.model).to(device)

    # preparing training dataset   
    train_dataset_class = getattr(data, cfg.dataset.train_class)
    train_set = train_dataset_class(**cfg.dataset.processed_files)
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    validate_dataset_class = getattr(data, cfg.dataset.test_class)
    validate_set = validate_dataset_class(**cfg.dataset.processed_files)
    validate_dataloader = torch.utils.data.DataLoader(validate_set, batch_size=batch_size)

    loss_fn = hydra.utils.instantiate(cfg.loss)
    optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters())

    best_validation_loss = float('inf')
    statistics = {"train_loss": [], "train_accuracy": [], "validation_loss": [], "validation_accuracy": []}
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
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")
        
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

        print(f"Epoch {epoch} - Validation Loss: {epoch_validation_loss:.4f}, Validation Accuracy: {epoch_validation_accuracy:.4f}")

        # Save the best model
        if epoch_validation_loss < best_validation_loss:
            best_validation_loss = epoch_validation_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"New best model saved with validation loss: {best_validation_loss:.4f}")
    
    torch.save(model.state_dict(), "last_model.pth")

    print("Training complete")
    # torch.save(model.state_dict(), "../../../models/model.pth")
    fig, axs = plt.subplots(2, 2, figsize=(15, 5))
    axs[0][0].plot(statistics["train_loss"])
    axs[0][0].set_title("Train loss")
    axs[1][0].plot(statistics["train_accuracy"])
    axs[1][0].set_title("Train accuracy")
    axs[0][1].plot(statistics["validation_loss"])
    axs[0][1].set_title("Validation loss")
    axs[1][1].plot(statistics["validation_accuracy"])
    axs[1][1].set_title("Validation accuracy")
    fig.savefig("../../../reports/figures/training_statistics.png")


if __name__ == "__main__":
    train()