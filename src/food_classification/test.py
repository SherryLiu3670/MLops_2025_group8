import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import hydra
import data

@hydra.main(config_path="../../configs", config_name="config")
def test(cfg) -> None:
    """Test a model on a dataset."""
    device = cfg.experiment.device
    batch_size = cfg.experiment.batch_size

    print("Testing the model")
    current_working_directory = os.getcwd()
    model_checkpoint = Path(current_working_directory + "/../14-52-09/best_model.pth")
    model = hydra.utils.instantiate(cfg.model).to(device)
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    model.eval()

    test_dataset_class = getattr(data, cfg.dataset.test_class)
    test_set = test_dataset_class(**cfg.dataset.processed_files)
    test_dataloader = DataLoader(test_set, batch_size=batch_size)

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

    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {avg_accuracy:.4f}")


if __name__ == "__main__":
    test()