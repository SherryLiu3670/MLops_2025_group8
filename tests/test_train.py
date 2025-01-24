import pytest
import torch
import hydra
import os
from hydra import compose, initialize
import matplotlib

from unittest.mock import MagicMock, patch

@pytest.fixture
def cfg():
    """Load the config file."""
    # using pytest.fixture to load the config file
    with initialize(version_base=None, config_path="../configs"):
        # override dataset in the main config file
        cfg = compose(config_name="config", overrides=["dataset=mnist"])
    return cfg

def test_device_selection(cfg):
    """Test that the device falls back to CPU if CUDA is unavailable."""
    # with initialize(config_path="../../configs"):
    cfg.hyperparams.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not torch.cuda.is_available():
        assert cfg.hyperparams.device != "cuda", "Device should fall back to CPU when CUDA is unavailable."


def test_model_initialization(cfg):
    """Test that the model is initialized correctly."""
    # with initialize(config_path="../../configs"):
        # cfg = compose(config_name="config")
    model = torch.nn.Module()
    cfg.hyperparams.device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        cfg.model.model_config.input_channels = cfg.dataset.input_channels
        model = hydra.utils.instantiate(cfg.model.model_config).to(cfg.hyperparams.device)
        assert model is not None, "Model should be initialized successfully."
    except Exception as e:
        pytest.fail(f"Model initialization failed: {e}")

@patch("wandb.init", MagicMock())
@patch("wandb.log_artifact", MagicMock())
@patch("wandb.log", MagicMock())
@patch("wandb.Artifact.add_file", MagicMock())
@patch("torch.save", MagicMock())
@patch("matplotlib.pyplot.subplots", MagicMock(return_value=(MagicMock(), MagicMock())))
@patch("hydra.core.hydra_config.HydraConfig.get", MagicMock())
@patch("generate_onnx.main", MagicMock())
def test_training_loop(cfg):
    """Test one epoch of training and validation."""
    # with initialize(config_path="../../configs"):
    #     cfg = compose(config_name="config")
    cfg.hyperparams.epochs = 1  # Run for only one epoch for testing purposes
    captured_statistics = None
    def my_callback(statistics):
        nonlocal captured_statistics
        captured_statistics = statistics
    
    import train
    train.train_fn(cfg, stats_callback=my_callback)
    # Ensure captured_statistics is valid
    assert isinstance(captured_statistics, dict), "Captured statistics must be a dictionary."
    assert "validation_loss" in captured_statistics, "'validation_loss' key must be in captured statistics."
    assert "validation_accuracy" in captured_statistics, "'validation_accuracy' key must be in captured statistics."
    
    # Validation checks
    assert isinstance(captured_statistics["validation_loss"][0], (float, int)), "Validation loss must be a number."
    assert captured_statistics["validation_loss"][0] < float("inf"), "Validation loss should be computed."
    assert isinstance(captured_statistics["validation_accuracy"][0], (float, int)), "Validation accuracy must be a number."
    assert captured_statistics["validation_accuracy"][0] >= 0, "Validation accuracy should be computed."
    
    # except Exception as e:
    #     pytest.fail(f"Training loop failed: {e}")


# def test_training_statistics_plot(cfg):
#     """Test that the training statistics plot is generated."""
#     # with initialize(config_path="../../configs"):
#     #     cfg = compose(config_name="config")
#     cfg.hyperparams.epochs = 1  # Run for only one epoch for testing purposes

#     try:
#         train.train(cfg)
#         assert os.path.exists("reports/figures/training_statistics.png"), "Training statistics plot should be generated."
#     except Exception as e:
#         pytest.fail(f"Training statistics plot generation failed: {e}")

