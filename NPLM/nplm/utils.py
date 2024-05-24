"""
Contains various utility functions for PyTorch model training and saving.
"""

from pathlib import Path

import torch


def save_model(model: torch.nn.Module, model_name: str):
    """
    Save a PyTorch model to a file.

    Args:
        model (torch.nn.Module): The PyTorch model to be saved.
        model_name (str): The name of the file to save the model to. The file should end with either ".pth" or ".pt".

    Returns:
        None

    Raises:
        AssertionError: If the model_name does not end with either ".pth" or ".pt".

    Description:
        This function saves a PyTorch model to a file. It creates a target directory if it does not exist and creates a model save path based on the provided model_name. The model's state_dict() is saved to the file.

        The target directory is created using the following path: "Neural_Probabilistic_Language_Model/models".

        The model_name should end with either ".pth" or ".pt". If it does not, an AssertionError is raised.

        The model save path is created by appending the model_name to the target directory path.

        The model's state_dict() is saved to the model save path using torch.save().

        The function prints a message indicating where the model is being saved.
    """
    # Create target directory
    target_dir_path = Path("NPLM/models")
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
    """
    Load a PyTorch model from a file.

    Args:
        model (torch.nn.Module): The PyTorch model to be loaded.
        file_name (str): The name of the file to load the model from. The file should end with either ".pth" or ".pt".
        device (torch.device, optional): The device (CPU or GPU) on which the model will be loaded. If not provided, the device will be determined automatically based on the availability of CUDA. Defaults to None.

    Returns:
        torch.nn.Module: The loaded PyTorch model.

    Description:
        This function loads a PyTorch model from a file. It creates a target directory if it does not exist and creates a model load path based on the provided file_name. The model's state_dict() is loaded from the file.

        The target directory is created using the following path: "models".

        The file_name should end with either ".pth" or ".pt". If it does not, an AssertionError is raised.

        The model load path is created by appending the file_name to the target directory path.

        The model's state_dict() is loaded from the model load path using torch.load().

        The model's state_dict() is loaded into the provided model using model.load_state_dict().

        The model is moved to the specified device using model.to(device).

        The model is set to evaluation mode using model.eval().

        The loaded model is returned.
    """
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
