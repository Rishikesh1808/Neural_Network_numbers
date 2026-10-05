"""
Base Neural Network Architecture for Handwritten Digit Classification (0-9).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import INPUT_SIZE, HIDDEN_SIZE_1, HIDDEN_SIZE_2, OUTPUT_SIZE, DEVICE


class DigitRecognizer(nn.Module):
    """
    A feed-forward neural network for classifying 28x28 handwritten digits (0-9).
    
    Architecture:
        Input (784) -> Linear(128) -> ReLU -> Linear(64) -> ReLU -> Linear(10) -> Log-Softmax
    """
    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size_1: int = HIDDEN_SIZE_1,
        hidden_size_2: int = HIDDEN_SIZE_2,
        output_size: int = OUTPUT_SIZE
    ):
        super(DigitRecognizer, self).__init__()
        
        # Layer definitions
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28) or (batch_size, 784).
            
        Returns:
            torch.Tensor: Log-probabilities of shape (batch_size, 10).
        """
        # Flatten image from (batch_size, 1, 28, 28) or any shape to (batch_size, 784)
        x = x.view(x.size(0), -1)
        
        # Hidden Layer 1 with ReLU activation
        x = F.relu(self.fc1(x))
        
        # Hidden Layer 2 with ReLU activation
        x = F.relu(self.fc2(x))
        
        # Output Layer (logits)
        x = self.fc3(x)
        
        # Numerical stability: log_softmax for use with NLLLoss
        return F.log_softmax(x, dim=1)


def get_model_summary(model: nn.Module) -> str:
    """Returns a string summary of the model layers and total parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    summary = [
        "=" * 45,
        " DigitRecognizer Architecture",
        "=" * 45,
        str(model),
        "-" * 45,
        f"Total Parameters:      {total_params:,}",
        f"Trainable Parameters:  {trainable_params:,}",
        "=" * 45
    ]
    return "\n".join(summary)


if __name__ == "__main__":
    # Test the base framework with a dummy input (batch of 2 images of 28x28)
    model = DigitRecognizer().to(DEVICE)
    print(get_model_summary(model))
    
    dummy_input = torch.randn(2, 1, 28, 28, device=DEVICE)
    with torch.no_grad():
        output = model(dummy_input)
        
    print(f"\nDevice: {DEVICE}")
    print(f"Dummy Input shape:  {dummy_input.shape}")
    print(f"Output shape:       {output.shape} (batch_size, num_classes)")
    print(f"Sample prediction (log-probabilities):\n{output}")
    print(f"Predicted classes (untrained): {output.argmax(dim=1).tolist()}")
