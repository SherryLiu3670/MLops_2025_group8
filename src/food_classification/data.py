import os
from pathlib import Path
import shutil
import torch
from torch.utils.data import Dataset
from hydra.utils import get_original_cwd
import torchvision.transforms as T
from PIL import Image
import hydra
import kagglehub
import yaml


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images."""
    return (images - images.mean()) / images.std()


class MNISTTrainDataset(Dataset):
    """Custom dataset for training."""

    def __init__(self, **preprocessed_dict) -> None:
        output_folder = preprocessed_dict["output_folder"]
        train_images_file = preprocessed_dict["train_images_file"]
        train_target_file = preprocessed_dict["train_target_file"]
        project_root = get_original_cwd()
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
        test_images_file = preprocessed_dict["test_images_file"]
        test_target_file = preprocessed_dict["test_target_file"]
        project_root = get_original_cwd()
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

    def preprocess(
        self,
        output_folder: Path,
        train_images_file: Path,
        train_target_file: Path,
        test_images_file: Path,
        test_target_file: Path,
    ) -> None:
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


class FruitTestDataset(Dataset):
    """Custom dataset for testing."""

    def __init__(self, **preprocessed_dict) -> None:
        output_folder = preprocessed_dict["output_folder"]
        test_images_file = preprocessed_dict["test_images_file"]
        test_target_file = preprocessed_dict["test_target_file"]

        test_images_path = f"../../../{output_folder}/{test_images_file}"
        test_target_path = f"../../../{output_folder}/{test_target_file}"

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


class FruitValDataset(Dataset):
    """Custom dataset for validation."""

    def __init__(self, **preprocessed_dict) -> None:
        output_folder = preprocessed_dict["output_folder"]
        val_images_file = preprocessed_dict["val_images_file"]
        val_target_file = preprocessed_dict["val_target_file"]

        val_images_path = f"../../../{output_folder}/{val_images_file}"
        val_target_path = f"../../../{output_folder}/{val_target_file}"

        if not os.path.exists(val_images_path) or not os.path.exists(val_target_path):
            raise FileNotFoundError("Preprocessing step should be executed first")

        val_images = torch.load(val_images_path)
        val_targets = torch.load(val_target_path)

        self.val_images = val_images
        self.val_targets = val_targets

    def __len__(self) -> int:
        return len(self.val_images)

    def __getitem__(self, index: int):
        return self.val_images[index], self.val_targets[index]


class FruitTrainDataset(Dataset):
    """Custom dataset for training."""

    def __init__(self, **preprocessed_dict) -> None:
        output_folder = preprocessed_dict["output_folder"]
        train_images_file = preprocessed_dict["train_images_file"]
        train_target_file = preprocessed_dict["train_target_file"]

        train_images_path = f"../../../{output_folder}/{train_images_file}"
        train_target_path = f"../../../{output_folder}/{train_target_file}"

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


class FruitDataset:
    """Class to preprocess and hold train and test datasets."""

    def __init__(self, raw_data_path: Path) -> None:
        self.data_path = Path(raw_data_path)
        self.train_set = None
        self.test_set = None
        self.validation_set = None
        self.kaggle_dataset_identifier = "kritikseth/fruit-and-vegetable-image-recognition/versions/8"
        self.class_names = None

    def download_data(self, output_folder: Path) -> None:
        print("Downloading data...")
        try:
            # downloading dataset from kaggle
            path = kagglehub.dataset_download(self.kaggle_dataset_identifier)
            # moving the downloaded dataset to the output folder
            shutil.move(path, str(output_folder))
            print(f"Data downloaded and saved to {str(output_folder)}")

        except Exception as e:
            print(f"Error downloading data: {e}")

    # fetch labels from the folder names
    def fetch_labels(self, directory_path: Path, partial: bool = False) -> None:
        folder_names = [
            folder for folder in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, folder))
        ]

        # Create a data configuration dictionary
        if partial is False:
            data_config = {"labels": folder_names}
        else:
            data_config = {"labels": folder_names[:2]}

        # Path to the YAML file
        yaml_file_path = "configs/dataset/Fruit.yaml"

        # Load the existing YAML data
        with open(yaml_file_path, "r") as f:
            fruit_config = yaml.safe_load(f)

        # Ensure the file content is a dictionary
        if fruit_config is None:
            fruit_config = {}

        # Remove the "labels" key if it exists
        fruit_config.pop("labels", None)

        # Append with the new labels
        fruit_config.update(data_config)

        # Write the updated content back to the YAML file
        with open(yaml_file_path, "w") as f:
            yaml.dump(fruit_config, f)

        return data_config["labels"]

    def _read_images_from_folder(self, folder: Path) -> tuple:
        images_list = []
        targets_list = []

        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}

        transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
            ]
        )

        for cls_name in self.class_names:
            cls_dir = folder / cls_name
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(".jpg"):
                    img_path = cls_dir / img_name
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = transform(img)  # (3,224,224)
                    images_list.append(img_tensor)
                    targets_list.append(class_to_idx[cls_name])

        images_tensor = torch.stack(images_list, dim=0)  # (N, 3, 224, 224)
        targets_tensor = torch.tensor(targets_list, dtype=torch.long)  # (N,)
        return images_tensor, targets_tensor

    def preprocess(
        self,
        output_folder: str,
        train_images_file: str,
        train_target_file: str,
        val_images_file: str,
        val_target_file: str,
        test_images_file: str,
        test_target_file: str,
        partial: bool,
    ) -> None:
        if not os.path.exists(self.data_path):
            self.download_data(self.data_path)
        else:
            print(
                "Skipping download as data already exists. If you want to re-download, delete the raw dataset folder."
            )

        self.class_names = self.fetch_labels(os.path.join(self.data_path, "train"), partial=partial)

        # 1) train
        train_folder = self.data_path / "train"
        train_images, train_targets = self._read_images_from_folder(train_folder)

        # 2) validation
        val_folder = self.data_path / "validation"
        val_images, val_targets = self._read_images_from_folder(val_folder)

        # 3) test
        test_folder = self.data_path / "test"
        test_images, test_targets = self._read_images_from_folder(test_folder)

        train_images = train_images.float()
        val_images = val_images.float()
        test_images = test_images.float()

        train_images = normalize(train_images)
        val_images = normalize(val_images)
        test_images = normalize(test_images)

        os.makedirs(output_folder, exist_ok=True)

        torch.save(train_images, os.path.join(output_folder, train_images_file))
        torch.save(train_targets, os.path.join(output_folder, train_target_file))

        torch.save(val_images, os.path.join(output_folder, val_images_file))
        torch.save(val_targets, os.path.join(output_folder, val_target_file))

        torch.save(test_images, os.path.join(output_folder, test_images_file))
        torch.save(test_targets, os.path.join(output_folder, test_target_file))


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def preprocess(cfg) -> None:
    print("Preprocessing data...")
    dataset_class = globals()[cfg.dataset.preprocess_class]
    dataset = dataset_class(cfg.dataset.data_dir)
    dataset.preprocess(**cfg.dataset.process_config)
    print("Data preprocessed!")


if __name__ == "__main__":
    preprocess()
