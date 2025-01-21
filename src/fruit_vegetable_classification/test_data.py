import os
from pathlib import Path

import pytest
import torch
from fruit_vegetable_classification.data import (
    normalize,
    MNISTTrainDataset,
    MNISTTestDataset,
    FruitVegetableTrainDataset,
    FruitVegetableTestDataset,
    FruitVegetableValDataset,
)


@pytest.mark.parametrize("shape", [(8, 3, 224, 224), (16, 1, 28, 28)])
def test_normalize(shape):

    images = torch.randn(*shape)
    normed = normalize(images)
    mean = normed.mean().item()
    std = normed.std().item()

    assert abs(mean) < 1e-5, f"Mean is not approximately 0, got {mean}"
    assert abs(std - 1.0) < 1e-5, f"Std is not approximately 1, got {std}"


@pytest.fixture
def mnist_sample_data(tmp_path):
    
    train_images = torch.randn(10, 1, 28, 28)
    train_targets = torch.randint(low=0, high=10, size=(10,))

    
    train_images_file = "train_images.pt"
    train_target_file = "train_targets.pt"

    
    train_images_path = tmp_path / train_images_file
    train_targets_path = tmp_path / train_target_file

    torch.save(train_images, train_images_path)
    torch.save(train_targets, train_targets_path)

    return {
        "train_images": train_images,
        "train_targets": train_targets,
        "train_images_file": train_images_file,
        "train_target_file": train_target_file,
        "output_folder": str(tmp_path),
    }


def test_mnist_train_dataset_file_not_found():

    with pytest.raises(FileNotFoundError):
        
        MNISTTrainDataset(
            output_folder="non_existent_folder",
            train_images_file="dummy.pt",
            train_target_file="dummy.pt"
        )


def test_mnist_train_dataset_len_and_getitem(mnist_sample_data):
 
    
    preprocessed_dict = {
        "output_folder": mnist_sample_data["output_folder"],
        "train_images_file": mnist_sample_data["train_images_file"],
        "train_target_file": mnist_sample_data["train_target_file"],
    }
    dataset = MNISTTrainDataset(**preprocessed_dict)

    assert len(dataset) == 10, "MNISTTrainDataset 10"

    first_item = dataset[0]
    assert isinstance(first_item, tuple), "getitem is (image_tensor, target)"
    assert first_item[0].shape == (1, 28, 28), "size (1, 28, 28)"
    assert 0 <= first_item[1].item() < 10, "label [0, 9] "


@pytest.fixture
def mnist_test_sample_data(tmp_path):
 
    test_images = torch.randn(5, 1, 28, 28)
    test_targets = torch.randint(low=0, high=10, size=(5,))

    test_images_file = "test_images.pt"
    test_target_file = "test_targets.pt"

    test_images_path = tmp_path / test_images_file
    test_targets_path = tmp_path / test_target_file

    torch.save(test_images, test_images_path)
    torch.save(test_targets, test_targets_path)

    return {
        "test_images": test_images,
        "test_targets": test_targets,
        "test_images_file": test_images_file,
        "test_target_file": test_target_file,
        "output_folder": str(tmp_path),
    }


def test_mnist_test_dataset_file_not_found():

    with pytest.raises(FileNotFoundError):
        MNISTTestDataset(
            output_folder="non_existent_folder",
            test_images_file="dummy.pt",
            test_target_file="dummy.pt"
        )


def test_mnist_test_dataset_len_and_getitem(mnist_test_sample_data):
 
    preprocessed_dict = {
        "output_folder": mnist_test_sample_data["output_folder"],
        "test_images_file": mnist_test_sample_data["test_images_file"],
        "test_target_file": mnist_test_sample_data["test_target_file"],
    }

    dataset = MNISTTestDataset(**preprocessed_dict)
    assert len(dataset) == 5, "MNISTTestDataset 长度应为 5"

    first_item = dataset[0]
    assert isinstance(first_item, tuple)
    assert first_item[0].shape == (1, 28, 28)
    assert 0 <= first_item[1].item() < 10


@pytest.fixture
def fruit_veg_sample_data(tmp_path):
 
    train_images = torch.randn(6, 3, 224, 224)
    train_targets = torch.randint(0, 5, (6,))  # 假设有 5 个类别
    val_images = torch.randn(4, 3, 224, 224)
    val_targets = torch.randint(0, 5, (4,))
    test_images = torch.randn(3, 3, 224, 224)
    test_targets = torch.randint(0, 5, (3,))

    output_folder = tmp_path
    train_images_file = "fruit_train_images.pt"
    train_target_file = "fruit_train_targets.pt"
    val_images_file = "fruit_val_images.pt"
    val_target_file = "fruit_val_targets.pt"
    test_images_file = "fruit_test_images.pt"
    test_target_file = "fruit_test_targets.pt"

    torch.save(train_images, output_folder / train_images_file)
    torch.save(train_targets, output_folder / train_target_file)
    torch.save(val_images, output_folder / val_images_file)
    torch.save(val_targets, output_folder / val_target_file)
    torch.save(test_images, output_folder / test_images_file)
    torch.save(test_targets, output_folder / test_target_file)

    return {
        "train_images": train_images,
        "train_targets": train_targets,
        "val_images": val_images,
        "val_targets": val_targets,
        "test_images": test_images,
        "test_targets": test_targets,
        "output_folder": str(output_folder),
        "train_images_file": train_images_file,
        "train_target_file": train_target_file,
        "val_images_file": val_images_file,
        "val_target_file": val_target_file,
        "test_images_file": test_images_file,
        "test_target_file": test_target_file,
    }


def test_fruit_vegetable_train_dataset_file_not_found():
 
    with pytest.raises(FileNotFoundError):
        FruitVegetableTrainDataset(
            output_folder="not_exist",
            train_images_file="dummy.pt",
            train_target_file="dummy.pt"
        )


def test_fruit_vegetable_train_dataset_len_and_getitem(fruit_veg_sample_data):

    preprocessed_dict = {
        "output_folder": fruit_veg_sample_data["output_folder"],
        "train_images_file": fruit_veg_sample_data["train_images_file"],
        "train_target_file": fruit_veg_sample_data["train_target_file"],
    }
    dataset = FruitVegetableTrainDataset(**preprocessed_dict)
    assert len(dataset) == 6

    x, y = dataset[0]
    assert x.shape == (3, 224, 224)
    assert 0 <= y.item() < 5


def test_fruit_vegetable_val_dataset_file_not_found():
 
    with pytest.raises(FileNotFoundError):
        FruitVegetableValDataset(
            output_folder="not_exist",
            val_images_file="dummy.pt",
            val_target_file="dummy.pt"
        )


def test_fruit_vegetable_val_dataset_len_and_getitem(fruit_veg_sample_data):
 
    preprocessed_dict = {
        "output_folder": fruit_veg_sample_data["output_folder"],
        "val_images_file": fruit_veg_sample_data["val_images_file"],
        "val_target_file": fruit_veg_sample_data["val_target_file"],
    }
    dataset = FruitVegetableValDataset(**preprocessed_dict)
    assert len(dataset) == 4

    x, y = dataset[0]
    assert x.shape == (3, 224, 224)
    assert 0 <= y.item() < 5


def test_fruit_vegetable_test_dataset_file_not_found():
 
    with pytest.raises(FileNotFoundError):
        FruitVegetableTestDataset(
            output_folder="not_exist",
            test_images_file="dummy.pt",
            test_target_file="dummy.pt"
        )


def test_fruit_vegetable_test_dataset_len_and_getitem(fruit_veg_sample_data):
 
    preprocessed_dict = {
        "output_folder": fruit_veg_sample_data["output_folder"],
        "test_images_file": fruit_veg_sample_data["test_images_file"],
        "test_target_file": fruit_veg_sample_data["test_target_file"],
    }
    dataset = FruitVegetableTestDataset(**preprocessed_dict)
    assert len(dataset) == 3

    x, y = dataset[0]
    assert x.shape == (3, 224, 224)
    assert 0 <= y.item() < 5
