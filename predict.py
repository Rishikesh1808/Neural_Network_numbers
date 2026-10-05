"""
Inference / Prediction Module for Handwritten Digits (0-9).
"""
import torch
import torch.nn as nn
from PIL import Image

from config import DEVICE
from model import DigitRecognizer
from dataset import get_transforms


def load_model(weights_path: str | None = None, device: torch.device = DEVICE) -> DigitRecognizer:
    """
    Initializes the model architecture and optionally loads trained weights.
    """
    model = DigitRecognizer().to(device)
    if weights_path:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_tensor(model: nn.Module, tensor: torch.Tensor, device: torch.device = DEVICE) -> tuple[int, list[float]]:
    """
    Runs inference on an input tensor.
    
    Args:
        model: DigitRecognizer instance.
        tensor: Image tensor of shape (1, 1, 28, 28) or (1, 784).
        device: Torch compute device.
        
    Returns:
        tuple[int, list[float]]: (predicted_digit, class_probabilities)
    """
    model.eval()
    tensor = tensor.to(device)
    with torch.no_grad():
        log_probs = model(tensor)
        probabilities = torch.exp(log_probs).squeeze(0).tolist()
        predicted_class = int(torch.argmax(log_probs, dim=1).item())
        
    return predicted_class, probabilities


def predict_image(model: nn.Module, image_path: str, device: torch.device = DEVICE) -> tuple[int, list[float]]:
    """
    Preprocesses a handwritten digit image file from disk and predicts its digit.
    
    Args:
        model: DigitRecognizer instance.
        image_path: Path to the image file.
        device: Torch compute device.
    """
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image = image.resize((28, 28))               # Resize to 28x28
    
    transform = get_transforms()
    tensor = transform(image).unsqueeze(0)       # Shape: (1, 1, 28, 28)
    
    return predict_tensor(model, tensor, device=device)


if __name__ == "__main__":
    print("=" * 50)
    print(" Inference Module Base Frame")
    print("=" * 50)
    
    # Initialize base model (untrained)
    model = load_model()
    
    # Test with a synthetic 28x28 sample digit tensor
    dummy_digit = torch.randn(1, 1, 28, 28)
    pred_digit, probs = predict_tensor(model, dummy_digit)
    
    print(f"Sample prediction with untrained base frame:")
    print(f"Predicted Digit : {pred_digit}")
    print(f"Class probabilities (sum={sum(probs):.2f}):")
    for digit, prob in enumerate(probs):
        bar = "#" * int(prob * 30)
        print(f"  Digit {digit}: {prob * 100:5.2f}% | {bar}")
