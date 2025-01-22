import pytest
import torch
from model import MyAwesomeModel, ResNetModel, MobileNetModel

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

def test_my_awesome_model_initialization():
    """Test MyAwesomeModel initialization."""
    try:
        model = MyAwesomeModel(
            input_channels=1,
            conv_layers=[(16, 3, 1), (32, 3, 1)],
            fc_layers=[128, 10],
            activation="ReLU",
            dropout_p=0.3,
        )
        assert isinstance(model, MyAwesomeModel), "MyAwesomeModel initialization failed."
    except Exception as e:
        pytest.fail(f"MyAwesomeModel initialization test failed: {e}")


def test_my_awesome_model_forward():
    """Test MyAwesomeModel forward pass."""
    model = MyAwesomeModel(
        input_channels=1,
        conv_layers=[(16, 3, 1), (32, 3, 1)],
        fc_layers=[128, 10],
    )
    dummy_input = torch.randn(8, 1, 28, 28)  # Batch of 8, 1 channel, 28x28 images
    try:
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        assert output.shape == (8, 10), f"Unexpected output shape: {output.shape}"
    except Exception as e:
        pytest.fail(f"Forward pass failed: {e}")


def test_resnet_model_initialization():
    """Test ResNetModel initialization."""
    try:
        model = ResNetModel(input_channels=1, model_type="resnet18", num_classes=10, pretrained=False)
        assert isinstance(model, ResNetModel), "ResNetModel initialization failed."
    except Exception as e:
        pytest.fail(f"ResNetModel initialization test failed: {e}")


def test_resnet_model_forward():
    """Test ResNetModel forward pass."""
    model = ResNetModel(input_channels=1, model_type="resnet18", num_classes=10, pretrained=False)
    dummy_input = torch.randn(4, 1, 224, 224)  # Batch of 4, 1 channel, 224x224 images
    output = model(dummy_input)
    assert output.shape == (4, 10), f"Unexpected output shape: {output.shape}"


def test_mobilenet_model_initialization():
    """Test MobileNetModel initialization."""
    try:
        model = MobileNetModel(input_channels=1, model_type="mobilenetV2", num_classes=10, pretrained=False)
        assert isinstance(model, MobileNetModel), "MobileNetModel initialization failed."
    except Exception as e:
        pytest.fail(f"MobileNetModel initialization test failed: {e}")


def test_mobilenet_model_forward():
    """Test MobileNetModel forward pass."""
    model = MobileNetModel(input_channels=1, model_type="mobilenetV2", num_classes=10, pretrained=False)
    dummy_input = torch.randn(2, 1, 224, 224)  # Batch of 2, 1 channel, 224x224 images
    output = model(dummy_input)
    assert output.shape == (2, 10), f"Unexpected output shape: {output.shape}"


def test_my_awesome_model_num_parameters():
    """Test the number of parameters in MyAwesomeModel."""
    model = MyAwesomeModel(
        input_channels=1,
        conv_layers=[(16, 3, 1), (32, 3, 1)],
        fc_layers=[128, 10],
    )
    num_params = sum(p.numel() for p in model.parameters())
    assert num_params > 0, "MyAwesomeModel should have more than 0 parameters."


def test_resnet_model_num_parameters():
    """Test the number of parameters in ResNetModel."""
    model = ResNetModel(input_channels=1, model_type="resnet18", num_classes=10, pretrained=False)
    num_params = sum(p.numel() for p in model.parameters())
    assert num_params > 0, "ResNetModel should have more than 0 parameters."


def test_mobilenet_model_num_parameters():
    """Test the number of parameters in MobileNetModel."""
    model = MobileNetModel(input_channels=1, model_type="mobilenetV2", num_classes=10, pretrained=False)
    num_params = sum(p.numel() for p in model.parameters())
    assert num_params > 0, "MobileNetModel should have more than 0 parameters."
