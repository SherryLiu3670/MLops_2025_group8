import pytest
import torch
import hydra
from hydra import compose, initialize
import train
import data
import os


config_path = "configs"
@pytest.fixture
def cfg():
    # using pytest.fixture to load the config file
    with hydra.initialize_config_dir(config_dir=os.path.abspath(config_path)):
        cfg = hydra.compose(config_name="config.yaml")
    return cfg

def test_device_selection(cfg):
    """Test that the device falls back to CPU if CUDA is unavailable."""
    # with initialize(config_path="../../configs"):
    cfg.experiment.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not torch.cuda.is_available():
        assert cfg.experiment.device != "cuda", "Device should fall back to CPU when CUDA is unavailable."


def test_model_initialization(cfg):
    """Test that the model is initialized correctly."""
    # with initialize(config_path="../../configs"):
        # cfg = compose(config_name="config")
    model = torch.nn.Module()
    cfg.experiment.device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = hydra.utils.instantiate(cfg.model).to(cfg.experiment.device)
        assert model is not None, "Model should be initialized successfully."
    except Exception as e:
        pytest.fail(f"Model initialization failed: {e}")


def test_data_loading(cfg):
    """Test that the datasets and dataloaders are set up correctly."""
    # with initialize(config_path="../../configs"):
    #     cfg = compose(config_name="config")
    try:
        train_dataset_class = getattr(data, cfg.dataset.train_class)
        train_set = train_dataset_class(**cfg.dataset.processed_files)
        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=cfg.experiment.batch_size)

        validate_dataset_class = getattr(data, cfg.dataset.test_class)
        validate_set = validate_dataset_class(**cfg.dataset.processed_files)
        validate_dataloader = torch.utils.data.DataLoader(validate_set, batch_size=cfg.experiment.batch_size)

        assert len(train_dataloader) > 0, "Training DataLoader should not be empty."
        assert len(validate_dataloader) > 0, "Validation DataLoader should not be empty."
    except Exception as e:
        pytest.fail(f"Data loading failed: {e}")


def test_training_loop(cfg):
    """Test one epoch of training and validation."""
    # with initialize(config_path="../../configs"):
    #     cfg = compose(config_name="config")
    cfg.experiment.epochs = 2  # Run for only one epoch for testing purposes

    try:
        train.train(cfg)
        output_path = os.getcwd()
        print(output_path)
        assert os.path.exists(os.path.join(output_path,"best_model.pth")), "Best model should be saved after training."
    except Exception as e:
        pytest.fail(f"Training loop failed: {e}")


def test_training_statistics_plot(cfg):
    """Test that the training statistics plot is generated."""
    # with initialize(config_path="../../configs"):
    #     cfg = compose(config_name="config")
    cfg.experiment.epochs = 1  # Run for only one epoch for testing purposes

    try:
        train.train(cfg)
        output_path = os.getcwd()
        assert os.path.exists(os.path.join(output_path,"reports/figures/training_statistics.png")), "Training statistics plot should be generated."
    except Exception as e:
        pytest.fail(f"Training statistics plot generation failed: {e}")