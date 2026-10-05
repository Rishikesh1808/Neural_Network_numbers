"""
Training and Evaluation Framework for DigitRecognizer.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import DEVICE, LEARNING_RATE, EPOCHS, MODEL_PATH
from model import DigitRecognizer
from dataset import get_kaggle_data_loaders, get_data_loaders


def get_loss_function() -> nn.Module:
    """Returns the loss function (Negative Log-Likelihood Loss for Log-Softmax outputs)."""
    return nn.NLLLoss()


def get_optimizer(model: nn.Module, lr: float = LEARNING_RATE) -> torch.optim.Optimizer:
    """Returns the optimizer (Adam)."""
    return torch.optim.Adam(model.parameters(), lr=lr)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """
    Trains the model for a single epoch.
    Returns: average training loss.
    """
    model.train()
    total_loss = 0.0

    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple[float, float]:
    """
    Evaluates the model on test/validation data.
    Returns: (average_loss, accuracy_percentage)
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    avg_loss = test_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def train(
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    save_path: str = MODEL_PATH,
    device: torch.device = DEVICE
):
    """
    Runs the complete training and validation pipeline across epochs
    and saves the model weights to disk.
    """
    print("=" * 55)
    print(" Starting Model Training")
    print("=" * 55)
    import time
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        print(f" Device        : {device} ({gpu_name})")
    else:
        print(f" Device        : {device}")
    print(f" Epochs        : {epochs}")
    print(f" Learning Rate : {lr}")
    print(f" Save Path     : {save_path}")
    print("=" * 55)

    # 1. Prepare data loaders
    if os.path.exists("./data/train.csv"):
        print("\nLoading Kaggle CSV dataset...")
        train_loader, val_loader = get_kaggle_data_loaders()
    else:
        print("\nLoading torchvision MNIST dataset...")
        train_loader, val_loader = get_data_loaders()

    # 2. Initialize model, criterion, and optimizer
    model = DigitRecognizer().to(device)
    criterion = get_loss_function()
    optimizer = get_optimizer(model, lr=lr)

    best_val_acc = 0.0
    start_total_time = time.perf_counter()

    # 3. Training loop
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start

        print(
            f"Epoch {epoch:2d}/{epochs:2d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:5.2f}% | "
            f"Time: {epoch_time:.2f}s"
        )

        # Save the best model weights
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best model to {save_path} (Acc: {val_acc:.2f}%)")

    total_time = time.perf_counter() - start_total_time
    print("\n" + "=" * 55)
    print(" Training Complete!")
    print(f" Total Time             : {total_time:.2f}s ({total_time / epochs:.2f}s / epoch)")
    print(f" Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f" Weights saved at       : {save_path}")
    print("=" * 55)


if __name__ == "__main__":
    train()
