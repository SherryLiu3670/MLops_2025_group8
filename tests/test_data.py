import os
import torch
import pytest
from hydra import initialize, compose
from data import MNISTDataset, FruitVegetableDataset
from torch.utils.data import DataLoader


@pytest.fixture
def mnist_cfg():
    """Load the MNIST config."""
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config", overrides=["dataset=mnist"])
    return cfg


@pytest.fixture
def fruit_veg_cfg():
    """Load the FruitVegetable config."""
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config", overrides=["dataset=fruit_vegetable"])
    return cfg


def test_mnist_preprocessing(mnist_cfg):
    """Test MNIST data preprocessing."""
    try:
        mnist_dataset = MNISTDataset(mnist_cfg.dataset.data_dir)
        mnist_dataset.preprocess(**mnist_cfg.dataset.process_config)
        # Check if preprocessed files exist
        output_folder = os.path.join(os.getcwd(), mnist_cfg.dataset.process_config.output_folder)
        assert os.path.exists(os.path.join(output_folder, mnist_cfg.dataset.process_config.train_images_file)), \
            "Train images file is missing."
        assert os.path.exists(os.path.join(output_folder, mnist_cfg.dataset.process_config.train_target_file)), \
            "Train targets file is missing."
        assert os.path.exists(os.path.join(output_folder, mnist_cfg.dataset.process_config.test_images_file)), \
            "Test images file is missing."
        assert os.path.exists(os.path.join(output_folder, mnist_cfg.dataset.process_config.test_target_file)), \
            "Test targets file is missing."
    except Exception as e:
        pytest.fail(f"MNIST preprocessing failed: {e}")


def test_fruit_veg_preprocessing(fruit_veg_cfg):
    """Test FruitVegetable data preprocessing."""
    try:
        fruit_veg_dataset = FruitVegetableDataset(fruit_veg_cfg.dataset.data_dir)
        fruit_veg_dataset.preprocess(**fruit_veg_cfg.dataset.process_config)
        # Check if preprocessed files exist
        output_folder = os.path.join(os.getcwd(), fruit_veg_cfg.dataset.process_config.output_folder)
        assert os.path.exists(os.path.join(output_folder, fruit_veg_cfg.dataset.process_config.train_images_file)), \
            "Train images file is missing."
        assert os.path.exists(os.path.join(output_folder, fruit_veg_cfg.dataset.process_config.train_target_file)), \
            "Train targets file is missing."
        assert os.path.exists(os.path.join(output_folder, fruit_veg_cfg.dataset.process_config.val_images_file)), \
            "Validation images file is missing."
        assert os.path.exists(os.path.join(output_folder, fruit_veg_cfg.dataset.process_config.val_target_file)), \
            "Validation targets file is missing."
        assert os.path.exists(os.path.join(output_folder, fruit_veg_cfg.dataset.process_config.test_images_file)), \
            "Test images file is missing."
        assert os.path.exists(os.path.join(output_folder, fruit_veg_cfg.dataset.process_config.test_target_file)), \
            "Test targets file is missing."
    except Exception as e:
        pytest.fail(f"FruitVegetable preprocessing failed: {e}")


def test_mnist_dataloader(mnist_cfg):
    """Test MNIST data loading."""
    try:
        from data import MNISTTrainDataset, MNISTTestDataset
        train_set = MNISTTrainDataset(**mnist_cfg.dataset.process_config)
        test_set = MNISTTestDataset(**mnist_cfg.dataset.process_config)

        train_loader = DataLoader(train_set, batch_size=mnist_cfg.experiment.batch_size)
        test_loader = DataLoader(test_set, batch_size=mnist_cfg.experiment.batch_size)

        assert len(train_loader) > 0, "MNIST training DataLoader should not be empty."
        assert len(test_loader) > 0, "MNIST testing DataLoader should not be empty."
    except Exception as e:
        pytest.fail(f"MNIST DataLoader test failed: {e}")


def test_fruit_veg_dataloader(fruit_veg_cfg):
    """Test FruitVegetable data loading."""
    try:
        from data import FruitVegetableTrainDataset, FruitVegetableValDataset, FruitVegetableTestDataset

        train_set = FruitVegetableTrainDataset(**fruit_veg_cfg.dataset.process_config)
        val_set = FruitVegetableValDataset(**fruit_veg_cfg.dataset.process_config)
        test_set = FruitVegetableTestDataset(**fruit_veg_cfg.dataset.process_config)

        train_loader = DataLoader(train_set, batch_size=fruit_veg_cfg.experiment.batch_size)
        val_loader = DataLoader(val_set, batch_size=fruit_veg_cfg.experiment.batch_size)
        test_loader = DataLoader(test_set, batch_size=fruit_veg_cfg.experiment.batch_size)

        assert len(train_loader) > 0, "FruitVegetable training DataLoader should not be empty."
        assert len(val_loader) > 0, "FruitVegetable validation DataLoader should not be empty."
        assert len(test_loader) > 0, "FruitVegetable testing DataLoader should not be empty."
    except Exception as e:
        pytest.fail(f"FruitVegetable DataLoader test failed: {e}")
