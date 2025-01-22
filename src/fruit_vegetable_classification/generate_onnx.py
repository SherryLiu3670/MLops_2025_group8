import hydra
import torch
import os

import pdb

def main(cfg):
    """Convert a PyTorch model to ONNX format."""

    # load the model using pth_path
    cfg.model.model_config.input_channels = cfg.dataset.input_channels
    model = hydra.utils.instantiate(cfg.model.model_config)

    input_res = cfg.model.desired_input_resolution if "desired_input_resolution" in cfg.model else [224, 224]
    input_shape = [cfg.dataset.input_channels, *input_res]

    original_working_directory = os.getcwd()
    model_checkpoint = os.path.join(original_working_directory, cfg.experiment.modelpath)
    model.load_state_dict(torch.load(model_checkpoint))
    # scrap off .pth extension and add .onnx
    output_path = model_checkpoint[:-4] + ".onnx"

    # Set the model to inference mode
    model.eval()

    # Create dummy input data
    dummy_input = torch.randn(1, *input_shape)

    # Export the model to ONNX
    torch.onnx.export(model, dummy_input, output_path)

if __name__ == "__main__":
    overrides = [
        "experiment=api"
    ]

    # using pytest.fixture to load the config file
    with hydra.initialize(version_base=None, config_path="../../configs"):
        # override dataset in the main config file
        cfg = hydra.compose(config_name="config", overrides=overrides)

    main(cfg)