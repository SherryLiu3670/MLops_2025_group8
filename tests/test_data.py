import torch
import pytest
from hydra import initialize, compose
import os

import importlib.util
from tests import _SRC_ROOT

# using importlib to load the data.py as module
spec = importlib.util.spec_from_file_location("data", os.path.join(_SRC_ROOT, "data.py"))
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


@pytest.fixture
def cfg():
    """Load the config file."""
    # using pytest.fixture to load the config file
    with initialize(version_base=None, config_path="../configs"):
        # override dataset in the main config file
        cfg = compose(config_name="config", overrides=["dataset=mnist"])
    return cfg


def test_data_loading(cfg):
    """Test that the datasets and dataloaders are set up correctly."""
    try:
        train_dataset_class = getattr(data, cfg.dataset.train_class)
        train_set = train_dataset_class(**cfg.dataset.process_config)
        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=cfg.experiment.batch_size)

        validate_dataset_class = getattr(data, cfg.dataset.test_class)
        validate_set = validate_dataset_class(**cfg.dataset.process_config)
        validate_dataloader = torch.utils.data.DataLoader(validate_set, batch_size=cfg.experiment.batch_size)

        assert len(train_dataloader) > 0, "Training DataLoader should not be empty."
        assert len(validate_dataloader) > 0, "Validation DataLoader should not be empty."
    except Exception as e:
        pytest.fail(f"Data loading failed: {e}")
