# Handwritten Digit Recognizer: End-to-End Architecture & Guide

Welcome to the comprehensive technical documentation for the **Digit Recognizer** neural network project. This guide covers the complete mathematical and computational architecture, file workflows, commands to run every component, and step-by-step instructions on deploying your saved weights in other projects.

---

## 1. High-Level Project Overview & Workflow

This project is a complete Deep Learning pipeline designed to recognize handwritten digits ($0$ through $9$) from 28x28 grayscale images (MNIST dataset format).

```mermaid
flowchart LR
    A["Raw Input Data\n(Kaggle train.csv or torchvision)"] --> B["dataset.py\n(Preprocessing & DataLoader)"]
    B --> C["model.py\n(DigitRecognizer MLP)"]
    C --> D["train.py\n(Adam Optimizer + NLLLoss)"]
    D --> E["model.pth\n(State Dict Weights)"]
    E --> F["predict.py / check.py\n(Inference on single image/pixels)"]
```

### The Pipeline Lifecycle:
1. **Configuration ([`config.py`](file:///c:/Users/rishi/OneDrive/Projects/Neural%20Network/config.py))**: Central source of truth for hyperparameters (batch size, learning rate, layer dimensions, device detection).
2. **Data Ingestion ([`dataset.py`](file:///c:/Users/rishi/OneDrive/Projects/Neural%20Network/dataset.py))**: Converts tabular CSV rows (784 pixel columns) or downloaded image datasets into normalized PyTorch tensors.
3. **Model Design ([`model.py`](file:///c:/Users/rishi/OneDrive/Projects/Neural%20Network/model.py))**: Defines the Multilayer Perceptron (MLP) architecture with non-linear activations.
4. **Pre-Training Verification ([`test_sanity.py`](file:///c:/Users/rishi/OneDrive/Projects/Neural%20Network/test_sanity.py))**: Verifies the pipeline math and random baseline accuracy (~10%) before training.
5. **Training Engine ([`train.py`](file:///c:/Users/rishi/OneDrive/Projects/Neural%20Network/train.py))**: Runs forward/backward propagation, calculates loss, updates weights, and saves the best model state to [`model.pth`](file:///c:/Users/rishi/OneDrive/Projects/Neural%20Network/model.pth).
6. **Inference ([`predict.py`](file:///c:/Users/rishi/OneDrive/Projects/Neural%20Network/predict.py) & [`check.py`](file:///c:/Users/rishi/OneDrive/Projects/Neural%20Network/check.py))**: Loads [`model.pth`](file:///c:/Users/rishi/OneDrive/Projects/Neural%20Network/model.pth) to predict digits from raw pixel arrays or image files.

---

## 2. Deep Dive: Neural Network Architecture ([`model.py`](file:///c:/Users/rishi/OneDrive/Projects/Neural%20Network/model.py))

The model is a **Feedforward Artificial Neural Network (Multilayer Perceptron / MLP)**.

### Mathematical Breakdown
* **Input Layer**: $28 \times 28 = 784$ features. Each feature corresponds to a pixel value scaled between roughly $-0.42$ and $+2.82$ (normalized).
* **Hidden Layer 1 (`fc1`)**: Fully connected linear transformation:
  $$z_1 = x W_1^T + b_1 \quad \text{where } W_1 \in \mathbb{R}^{128 \times 784}, b_1 \in \mathbb{R}^{128}$$
  Followed by the **ReLU (Rectified Linear Unit)** activation function:
  $$a_1 = \max(0, z_1)$$
* **Hidden Layer 2 (`fc2`)**: Fully connected linear transformation:
  $$z_2 = a_1 W_2^T + b_2 \quad \text{where } W_2 \in \mathbb{R}^{64 \times 128}, b_2 \in \mathbb{R}^{64}$$
  Followed by **ReLU**:
  $$a_2 = \max(0, z_2)$$
* **Output Layer (`fc3`)**: Fully connected linear transformation down to 10 classes:
  $$z_3 = a_2 W_3^T + b_3 \quad \text{where } W_3 \in \mathbb{R}^{10 \times 64}, b_3 \in \mathbb{R}^{10}$$
* **Log-Softmax Output**:
  $$\text{LogSoftmax}(z_3)_i = \log \left( \frac{e^{(z_3)_i}}{\sum_{j=0}^{9} e^{(z_3)_j}} \right)$$

### Parameter Calculation

| Layer | Weight Matrix | Bias Vector | Total Parameters |
|---|---|---|---|
| **`fc1`** | $784 \times 128 = 100,352$ | $128$ | $100,480$ |
| **`fc2`** | $128 \times 64 = 8,192$ | $64$ | $8,256$ |
| **`fc3`** | $64 \times 10 = 640$ | $10$ | $650$ |
| **Total** | | | **109,386 parameters (437.5 KB)** |

### Why Log-Softmax + NLLLoss?
Using `F.log_softmax` in the model paired with `nn.NLLLoss` (Negative Log-Likelihood Loss) in [`train.py`](file:///c:/Users/rishi/OneDrive/Projects/Neural%20Network/train.py) is mathematically identical to `CrossEntropyLoss`, but provides **numerical stability** by avoiding underflow/overflow issues with floating point exponents.

---

## 3. Data Processing Pipeline ([`dataset.py`](file:///c:/Users/rishi/OneDrive/Projects/Neural%20Network/dataset.py))

### Normalization Formula
Raw pixels range from $[0, 255]$:
1. `transforms.ToTensor()` converts integers in $[0, 255]$ to floats in $[0.0, 1.0]$.
2. `transforms.Normalize((0.1307,), (0.3081,))` applies standard normal standardization using the global MNIST statistics ($\mu = 0.1307, \sigma = 0.3081$):
   $$\hat{x} = \frac{x - 0.1307}{0.3081}$$
   - Black background ($0$) maps to $\approx -0.424$
   - Pure white strokes ($1.0$) map to $\approx +2.821$
   - This zero-centered input prevents neuron saturation and accelerates gradient descent convergence.

### Kaggle CSV vs torchvision Loader
[`dataset.py`](file:///c:/Users/rishi/OneDrive/Projects/Neural%20Network/dataset.py) features a dual-loader pattern:
- If [`data/train.csv`](file:///c:/Users/rishi/OneDrive/Projects/Neural%20Network/data/train.csv) exists, it uses `np.genfromtxt` to load the 42,000 tabular rows, holding out 10% (4,200 samples) as validation data.
- If CSVs are absent, it auto-downloads torchvision's standard MNIST dataset.

---

## 4. File-by-File Breakdown & Execution Commands

Always run using the virtual environment interpreter (`.venv\Scripts\python.exe`) or after activating your virtual environment.

### 1. [`config.py`](file:///c:/Users/rishi/OneDrive/Projects/Neural%20Network/config.py)
* **What it does**: Reads `.env` values, defines layer dimensions, batch size, learning rate, and auto-detects CUDA hardware vs CPU fallback.
* **Command**:
  ```powershell
  .venv\Scripts\python.exe config.py
  ```
* **Output**: Prints the active configuration, compute device, and hyperparameters.

---

### 2. [`model.py`](file:///c:/Users/rishi/OneDrive/Projects/Neural%20Network/model.py)
* **What it does**: Contains the [`DigitRecognizer`](file:///c:/Users/rishi/OneDrive/Projects/Neural%20Network/model.py#L11) class definition and parameter counting utility.
* **Command**:
  ```powershell
  .venv\Scripts\python.exe model.py
  ```
* **Output**: Prints the layer architecture, total trainable parameters, and performs a dry-run test with a dummy batch tensor.

---

### 3. [`dataset.py`](file:///c:/Users/rishi/OneDrive/Projects/Neural%20Network/dataset.py)
* **What it does**: Preprocessing transforms and [`KaggleMNISTDataset`](file:///c:/Users/rishi/OneDrive/Projects/Neural%20Network/dataset.py#L68) custom PyTorch `Dataset` implementation.
* **Command**:
  ```powershell
  .venv\Scripts\python.exe dataset.py
  ```
* **Output**: Verifies dataset loading, splits 37,800 train / 4,200 val samples, and inspects the first batch shape.

---

### 4. [`test_sanity.py`](file:///c:/Users/rishi/OneDrive/Projects/Neural%20Network/test_sanity.py)
* **What it does**: Pre-training sanity check. Evaluates 5 batches on an untrained model to verify that initial loss is $\approx \ln(10) \approx 2.302$ and accuracy is $\approx 10\%$.
* **Command**:
  ```powershell
  .venv\Scripts\python.exe test_sanity.py
  ```
* **Output**: Detailed batch loss reports and verification pass/fail status.

---

### 5. [`train.py`](file:///c:/Users/rishi/OneDrive/Projects/Neural%20Network/train.py)
* **What it does**: Executes model training across epochs using the Adam optimizer. Computes validation accuracy at each epoch, records execution times, and saves the best model weights to [`model.pth`](file:///c:/Users/rishi/OneDrive/Projects/Neural%20Network/model.pth) using `torch.save()`.
* **Command**:
  ```powershell
  .venv\Scripts\python.exe train.py
  ```
* **Output**: Epoch-by-epoch loss, validation accuracy, epoch timings, and notification when a new best model is saved.

---

### 6. [`predict.py`](file:///c:/Users/rishi/OneDrive/Projects/Neural%20Network/predict.py)
* **What it does**: Inference helper module. Provides [`load_model()`](file:///c:/Users/rishi/OneDrive/Projects/Neural%20Network/predict.py#L13), [`predict_tensor()`](file:///c:/Users/rishi/OneDrive/Projects/Neural%20Network/predict.py#L25), and [`predict_image()`](file:///c:/Users/rishi/OneDrive/Projects/Neural%20Network/predict.py#L47) (to classify any `.png` or `.jpg` file from disk).
* **Command**:
  ```powershell
  .venv\Scripts\python.exe predict.py
  ```
* **Output**: Demonstrates prediction function with probabilities for digits 0-9.

---

### 7. [`check.py`](file:///c:/Users/rishi/OneDrive/Projects/Neural%20Network/check.py)
* **What it does**: Direct single-sample verification script. Takes 784 raw pixel values, normalizes them, loads [`model.pth`](file:///c:/Users/rishi/OneDrive/Projects/Neural%20Network/model.pth), and outputs the predicted digit with percentage probabilities.
* **Command**:
  ```powershell
  .venv\Scripts\python.exe check.py
  ```
* **Output**: Shows the predicted digit along with a visual confidence bar for all 10 digits.

---

## 5. How to Use `model.pth` in Other Projects

[`model.pth`](file:///c:/Users/rishi/OneDrive/Projects/Neural%20Network/model.pth) stores the PyTorch **state dictionary** (`state_dict`), which contains only the learned weights and biases (tensors), keeping the file lightweight and decoupled from code paths.

To use these weights in any external application (such as a FastAPI backend, Flask web app, or desktop GUI):

### Step 1: Copy two files into your new project
1. [`model.pth`](file:///c:/Users/rishi/OneDrive/Projects/Neural%20Network/model.pth) (the trained weights)
2. The model definition class ([`DigitRecognizer`](file:///c:/Users/rishi/OneDrive/Projects/Neural%20Network/model.py#L11))

### Step 2: Standalone Inference Code in the New Project

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

# 1. Define identical architecture
class DigitRecognizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# 2. Load model and load weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DigitRecognizer().to(device)

state_dict = torch.load("model.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()  # IMPORTANT: Set to evaluation mode!

# 3. Prediction helper function
def predict_digit(image_or_pixels):
    """
    image_or_pixels: Can be a file path, PIL Image, or list of 784 integers.
    """
    if isinstance(image_or_pixels, (list, np.ndarray)):
        arr = np.array(image_or_pixels, dtype=np.uint8).reshape(28, 28)
        image = Image.fromarray(arr, mode="L")
    elif isinstance(image_or_pixels, str):
        image = Image.open(image_or_pixels).convert("L").resize((28, 28))
    else:
        image = image_or_pixels

    # Same normalization as training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        log_probs = model(tensor)
        probabilities = torch.exp(log_probs).squeeze(0)
        predicted_class = int(torch.argmax(log_probs, dim=1).item())

    return predicted_class, probabilities.tolist()

# Example Usage:
# digit, probs = predict_digit("sample_digit.png")
# print(f"Predicted Digit: {digit}")
```

### Important Production Rules for External Use
1. **Always call `model.eval()`**: Ensures layers behave in deterministic evaluation mode.
2. **Use `torch.no_grad()`**: Disables autograd gradient tracking during inference to reduce memory usage and double execution speed.
3. **Use `map_location=device`**: Prevents runtime crashes when running on a CPU machine after training on a GPU (or vice versa).
4. **Maintain exact normalization values**: Ensure `(0.1307,), (0.3081,)` are applied to the input; otherwise, the inputs will be out of distribution and accuracy will degrade.
