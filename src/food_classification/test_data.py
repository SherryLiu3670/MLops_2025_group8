import pytest
import torch
from food_classification.data import normalize

def test_normalize():
    """
    Test that normalize outputs zero-mean and unit-std (approximately).
    """
    images = torch.randn(8, 3, 224, 224)  # Simulate a batch of images
    normed = normalize(images)
    mean = normed.mean().item()
    std = normed.std().item()
    
    assert abs(mean) < 1e-5, f"Mean is not approximately 0, got {mean}"
    assert abs(std - 1.0) < 1e-5, f"Std is not approximately 1, got {std}"