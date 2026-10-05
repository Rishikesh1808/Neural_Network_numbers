"""
Configuration and hyperparameters for the Handwritten Digit Recognition model.
Values are loaded from the .env file if present, otherwise fall back to defaults.
"""
import os
import torch
from dotenv import load_dotenv

# Load .env file from the project root
load_dotenv()

# --- Network Architecture ---
INPUT_SIZE    = 28 * 28  # 784 pixels for 28x28 grayscale image
HIDDEN_SIZE_1 = 128      # First hidden layer neurons
HIDDEN_SIZE_2 = 64       # Second hidden layer neurons
OUTPUT_SIZE   = 10       # 10 classes (digits 0 through 9)

# --- Hyperparameters (loaded from .env, with defaults) ---
BATCH_SIZE    = int(os.getenv("BATCH_SIZE",    64))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.001))
EPOCHS        = int(os.getenv("EPOCHS",        10))

# --- Paths ---
DATA_DIR   = os.getenv("DATA_DIR", "./data")
MODEL_PATH = os.getenv("MODEL_PATH", "model.pth")

# --- Kaggle Credentials (used by kaggle CLI automatically if set in environment) ---
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME", "")
KAGGLE_KEY      = os.getenv("KAGGLE_KEY", "")

# --- Compute Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    print("=" * 40)
    print(" Active Configuration")
    print("=" * 40)
    print(f"  Device        : {DEVICE}")
    print(f"  Data Dir      : {DATA_DIR}")
    print(f"  Batch Size    : {BATCH_SIZE}")
    print(f"  Learning Rate : {LEARNING_RATE}")
    print(f"  Epochs        : {EPOCHS}")
    print(f"  Kaggle User   : {KAGGLE_USERNAME or '(not set)'}")
    print(f"  Kaggle Key    : {'*' * len(KAGGLE_KEY) if KAGGLE_KEY else '(not set)'}")
    print("=" * 40)
