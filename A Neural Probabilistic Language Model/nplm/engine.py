"""
Contains functions for training and testing a PyTorch model.
"""

from typing import Dict, List

import torch

from tqdm.auto import tqdm


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Trains a PyTorch model for one step using the given data loader, loss function, optimizer, and device.

    Parameters:
        model (torch.nn.Module): The PyTorch model to be trained.
        dataloader (torch.utils.data.DataLoader): The data loader containing the training data.
        loss_fn (torch.nn.Module): The loss function used to calculate the loss.
        optimizer (torch.optim.Optimizer): The optimizer used to update the model's parameters.
        device (torch.device): The device (CPU or GPU) on which the training will be performed.

    Returns:
        float: The average loss per batch.

    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss = 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    return train_loss


def eval_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> float:
    """
    Evaluates the model using the provided data loader and loss function.

    Parameters:
        model (torch.nn.Module): The model to be evaluated.
        dataloader (torch.utils.data.DataLoader): The data loader containing evaluation data.
        loss_fn (torch.nn.Module): The loss function used to calculate the evaluation loss.
        device (torch.device): The device (CPU or GPU) on which the evaluation will be performed.

    Returns:
        float: The average evaluation loss.
    """
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss = 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    return test_loss


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    dev: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
):
    """
    Trains a PyTorch model using the given data loaders, optimizer, loss function, number of epochs, and device.

    Parameters:
        model (torch.nn.Module): The PyTorch model to be trained.
        train_dataloader (torch.utils.data.DataLoader): The data loader containing the training data.
        dev (torch.utils.data.DataLoader): The data loader containing the development data.
        optimizer (torch.optim.Optimizer): The optimizer used to update the model's parameters.
        loss_fn (torch.nn.Module): The loss function used to calculate the loss.
        epochs (int): The number of epochs to train the model.
        device (torch.device): The device (CPU or GPU) on which the training will be performed.

    Returns:
        None
    """
    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        dev_loss = eval_model(
            model=model, dataloader=dev, loss_fn=loss_fn, device=device
        )

        # Print out what's happening
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"dev_loss: {dev_loss:.4f} | "
        )
