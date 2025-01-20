import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
import hydra

def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images."""
    return (images - images.mean()) / images.std()

class MNISTTrainDataset(Dataset):
    """Custom dataset for training."""

    def __init__(self, **preprocessed_dict) -> None:

        output_folder = preprocessed_dict["output_folder"]
        train_images_file = preprocessed_dict["train_images_file"]
        train_target_file = preprocessed_dict["train_target_file"]
        project_root = os.getcwd()
        output_path = os.path.join(project_root, output_folder)
        train_images_path = os.path.abspath(os.path.join(output_path, train_images_file))
        train_target_path = os.path.abspath(os.path.join(output_path, train_target_file))
        # train_images_path = f"../../../{output_folder}/{train_images_file}"
        # train_target_path = f"../../../{output_folder}/{train_target_file}"

        if not os.path.exists(train_images_path) or not os.path.exists(train_target_path):
            raise FileNotFoundError("Preprocessing step should be executed first")
        
        train_images = torch.load(train_images_path)
        train_target = torch.load(train_target_path)

        self.train_images = train_images
        self.train_target = train_target

    def __len__(self) -> int:
        """Return the length of the training dataset."""
        return len(self.train_images)

    def __getitem__(self, index: int):
        """Return a given sample from the training dataset."""
        return self.train_images[index], self.train_target[index]

class MNISTTestDataset(Dataset):
    """Custom dataset for testing."""

    def __init__(self, **preprocessed_dict) -> None:

        output_folder = preprocessed_dict["output_folder"]
        test_images_file = preprocessed_dict["train_images_file"]
        test_target_file = preprocessed_dict["train_target_file"]
        project_root = os.getcwd()
        output_path = os.path.join(project_root, output_folder)
        test_images_path = os.path.abspath(os.path.join(output_path, test_images_file))
        test_target_path = os.path.abspath(os.path.join(output_path, test_target_file))
        

        # test_images_path = f"../../../{output_folder}/{test_images_file}"
        # test_target_path = f"../../../{output_folder}/{test_target_file}"

            
        if not os.path.exists(test_images_path) or not os.path.exists(test_target_path):
            raise FileNotFoundError("Preprocessing step should be executed first")
        
        test_images = torch.load(test_images_path)
        test_target = torch.load(test_target_path)

        self.test_images = test_images
        self.test_target = test_target

    def __len__(self) -> int:
        """Return the length of the testing dataset."""
        return len(self.test_images)

    def __getitem__(self, index: int):
        """Return a given sample from the testing dataset."""
        return self.test_images[index], self.test_target[index]


class MNISTDataset:
    """Class to preprocess and hold train and test datasets."""

    def __init__(self, raw_data_path: Path) -> None:
        self.data_path = raw_data_path
        self.train_set = None
        self.test_set = None

    def preprocess(self, output_folder: Path, 
                   train_images_file: Path, train_target_file: Path, 
                   test_images_file: Path, test_target_file: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        # Read all the pt files in the raw data folder
        train_images, train_target = [], []
        for i in range(6):
            train_images.append(torch.load(f"{self.data_path}/train_images_{i}.pt"))
            train_target.append(torch.load(f"{self.data_path}/train_target_{i}.pt"))
        train_images = torch.cat(train_images)
        train_target = torch.cat(train_target)

        test_images = torch.load(f"{self.data_path}/test_images.pt")
        test_target = torch.load(f"{self.data_path}/test_target.pt")

        train_images = train_images.unsqueeze(1).float()
        test_images = test_images.unsqueeze(1).float()

        train_target = train_target.long()
        test_target = test_target.long()

        train_images = normalize(train_images)
        test_images = normalize(test_images)

        os.makedirs(output_folder, exist_ok=True)

        torch.save(train_images, f"{output_folder}/{train_images_file}")
        torch.save(train_target, f"{output_folder}/{train_target_file}")
        torch.save(test_images, f"{output_folder}/{test_images_file}")
        torch.save(test_target, f"{output_folder}/{test_target_file}")

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def preprocess(cfg) -> None:
    print("Preprocessing data...")
    dataset_class = globals()[cfg.dataset.preprocess_class]
    dataset = dataset_class(cfg.dataset.data_dir)
    dataset.preprocess(**cfg.dataset.processed_files)
    print("Data preprocessed!")
    
if __name__ == "__main__":
    preprocess()
