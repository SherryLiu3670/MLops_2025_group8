import hydra
import matplotlib.pyplot as plt
import torch
import typer
import data
import model as models

@hydra.main(config_path="../../configs", config_name="config")
def train(cfg) -> None:
    """Train a model on MNIST."""

    lr=cfg.experiment.lr
    batch_size=cfg.experiment.batch_size
    epochs=cfg.experiment.epochs
    
    device = cfg.experiment.device
    
    #lr: float = 1e-3, batch_size: int = 32, epochs: int = 10
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    # instead of fixating the model, we can use the config file to define the model
    model = hydra.utils.instantiate(cfg.model).to(device)

    # preparing training dataset   
    train_dataset_class = getattr(data, cfg.dataset.train_class)
    train_set = train_dataset_class(**cfg.dataset.processed_files)
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    loss_fn = hydra.utils.instantiate(cfg.loss)
    optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters())

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
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

    print("Training complete")
    torch.save(model.state_dict(), "../../../models/model.pth")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("../../../reports/figures/training_statistics.png")


if __name__ == "__main__":
    train()