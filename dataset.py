"""
Dataset loader and preprocessing for MNIST handwritten digits (0-9).
Supports two sources:
  1. torchvision MNIST (auto-download)
  2. Kaggle Digit Recognizer CSVs (train.csv / test.csv)

Note: Uses numpy for CSV reading to avoid Windows Application Control DLL restrictions.
"""
from typing import Any
import torch
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

from config import BATCH_SIZE, DATA_DIR


# --- Transforms ---------------------------------------------------------------

def get_transforms() -> transforms.Compose:
    """
    Standard preprocessing transforms for MNIST digits:
    - ToTensor: converts PIL Image (H x W x C) [0,255] -> FloatTensor [0.0,1.0]
    - Normalize: standardizes with MNIST dataset mean=0.1307, std=0.3081
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


# --- Source 1: torchvision MNIST auto-download --------------------------------

def get_data_loaders(
    batch_size: int = BATCH_SIZE,
    data_dir: str = DATA_DIR,
    download: bool = True
) -> tuple[DataLoader, DataLoader]:
    """
    Loads MNIST dataset via torchvision and returns train & test DataLoaders.

    Args:
        batch_size (int): Samples per batch.
        data_dir (str): Directory where MNIST data will be stored/cached.
        download (bool): Download from internet if not already present.

    Returns:
        tuple[DataLoader, DataLoader]: (train_loader, test_loader)
    """
    transform = get_transforms()

    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=download, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=download, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# --- Source 2: Kaggle Digit Recognizer CSV format ----------------------------

class KaggleMNISTDataset(Dataset):
    """
    PyTorch Dataset for the Kaggle Digit Recognizer CSV format.
    Uses numpy instead of pandas to avoid Windows DLL policy restrictions.

    CSV layout:
      train.csv  -> 'label' column (0-9) + pixel0..pixel783 (int, 0-255)
      test.csv   -> pixel0..pixel783 only (no label column)

    Args:
        csv_path  (str) : Path to train.csv or test.csv.
        is_test   (bool): Set True when loading test.csv (no label column).
        transform       : Optional torchvision transform to apply.
    """
    def __init__(self, csv_path: str, is_test: bool = False, transform=None):
        print(f"Loading CSV: {csv_path} ...")
        # numpy reads CSV natively without pandas DLL dependencies
        data: Any = np.genfromtxt(csv_path, delimiter=",", skip_header=1, dtype=np.float32)

        if not is_test:
            self.labels: Any = data[:, 0].astype(np.int64)   # first column = label
            self.pixels: Any = data[:, 1:]                    # pixel0..pixel783
        else:
            self.labels: Any = None
            self.pixels: Any = data                           # all columns are pixels

        self.is_test   = is_test
        self.transform = transform
        print(f"  Loaded {len(self.pixels):,} samples.")

    def __len__(self) -> int:
        return len(self.pixels)

    def __getitem__(self, idx: int):
        pixels = self.pixels[idx].astype(np.uint8).reshape(28, 28)
        image  = Image.fromarray(pixels, mode="L")       # Pillow grayscale image

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])(image)

        if not self.is_test and self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return image, label
        return image


def get_kaggle_data_loaders(
    train_csv: str = "./data/train.csv",
    test_csv:  str = "./data/test.csv",
    batch_size: int = BATCH_SIZE,
    val_split: float = 0.1
) -> tuple[DataLoader, DataLoader]:
    """
    Builds train and validation DataLoaders from Kaggle MNIST CSV files.
    10% of train.csv is held out as a validation set by default.

    Args:
        train_csv  (str)  : Path to Kaggle train.csv.
        test_csv   (str)  : Path to Kaggle test.csv (not used for training).
        batch_size (int)  : Samples per batch.
        val_split  (float): Fraction of training data for validation.

    Returns:
        tuple[DataLoader, DataLoader]: (train_loader, val_loader)
    """
    from torch.utils.data import random_split

    transform    = get_transforms()
    full_dataset = KaggleMNISTDataset(train_csv, is_test=False, transform=transform)

    val_size   = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    print(f"  Training samples   : {train_size:,}")
    print(f"  Validation samples : {val_size:,}")

    return train_loader, val_loader


# --- Quick sanity check -------------------------------------------------------

if __name__ == "__main__":
    import os

    if os.path.exists("./data/train.csv"):
        print("=== Kaggle CSV Mode ===")
        train_loader, val_loader = get_kaggle_data_loaders()
        images, labels = next(iter(train_loader))
        print(f"Batch shape : {images.shape}")
        print(f"Labels      : {labels[:10].tolist()}")
    else:
        print("=== torchvision MNIST Mode (auto-download) ===")
        train_loader, test_loader = get_data_loaders()
        images, labels = next(iter(train_loader))
        print(f"Batch shape : {images.shape}")
        print(f"Labels      : {labels[:10].tolist()}")
