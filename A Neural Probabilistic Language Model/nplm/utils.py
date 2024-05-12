"""
Contains various utility functions for PyTorch model training and saving.
"""

from pathlib import Path

import torch


def save_model(model: torch.nn.Module, model_name: str):

    # Create target directory
    target_dir_path = Path("Neural_Probabilistic_Language_Model/models")
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"
    ), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)


def load_model(model, file_name, device=None):

    # Determine if CUDA is available and set the default device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model directory
    target_dir_path = Path("models")
    # Ensure the directory exists
    target_dir_path.mkdir(parents=True, exist_ok=True)
    # Model load path
    path = target_dir_path / file_name

    # Load the model state dict with the appropriate map location
    model_state = torch.load(path, map_location=device)

    # Load the state dict into the model
    model.load_state_dict(model_state)

    # Move model to the device
    model.to(device)
    model.eval()

    return model
