import torch
import pytest
from hydra import initialize, compose
import os
import data

from tests import _PROJECT_ROOT


@pytest.fixture
def cfg():
    """Load the config file."""
    # using pytest.fixture to load the config file
    with initialize(version_base=None, config_path="../configs"):
        # override dataset in the main config file
        cfg = compose(config_name="config", overrides=["dataset=mnist"])
    return cfg


def test_data_processing(cfg):
    """Test that the data processing is set up correctly."""
    try:
        data_processing_class = getattr(data, cfg.dataset.preprocess_class)
        dataset = data_processing_class(cfg.dataset.data_dir)
        assert dataset is not None, "Data processing should not be None."
        dataset.preprocess(**cfg.dataset.process_config)
        # assert that cfg.dataset.process_config.output_folder has been created
        assert os.path.exists(
            os.path.join(_PROJECT_ROOT, cfg.dataset.process_config.output_folder)
        ), "Preprocessed dataset folder should exist."

    except Exception as e:
        pytest.fail(f"Data processing failed: {e}")


def test_data_loading(cfg):
    """Test that the datasets and dataloaders are set up correctly."""
    try:
        train_dataset_class = getattr(data, cfg.dataset.train_class)
        train_set = train_dataset_class(**cfg.dataset.process_config)
        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=cfg.hyperparams.batch_size)

        validate_dataset_class = getattr(data, cfg.dataset.test_class)
        validate_set = validate_dataset_class(**cfg.dataset.process_config)
        validate_dataloader = torch.utils.data.DataLoader(validate_set, batch_size=cfg.hyperparams.batch_size)

        assert len(train_dataloader) > 0, "Training DataLoader should not be empty."
        assert len(validate_dataloader) > 0, "Validation DataLoader should not be empty."
    except Exception as e:
        pytest.fail(f"Data loading failed: {e}")
