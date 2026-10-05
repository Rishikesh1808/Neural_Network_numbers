"""
Pre-training sanity test for the DigitRecognizer pipeline.
Verifies that data loading, model forward pass, and loss computation all work
correctly BEFORE any training occurs.
Expected accuracy: ~10% (random chance across 10 classes).
"""
import torch

from config import DEVICE, EPOCHS, BATCH_SIZE, LEARNING_RATE
from model import DigitRecognizer, get_model_summary
from dataset import get_kaggle_data_loaders
from train import get_loss_function, get_optimizer


def run_sanity_test(num_batches: int = 5):
    """
    Runs a quick sanity check on the untrained model:
      1. Loads a few batches from the Kaggle dataset
      2. Runs a forward pass through the untrained model
      3. Computes loss (should be ~log(10) ~ 2.302 for random weights)
      4. Checks predictions and accuracy (expected ~10%)
    """
    print("=" * 55)
    print(" PRE-TRAINING SANITY TEST")
    print("=" * 55)

    # Step 1: Load config
    print(f"\n[1/4] Configuration")
    print(f"  Device        : {DEVICE}")
    print(f"  Batch Size    : {BATCH_SIZE}")
    print(f"  Learning Rate : {LEARNING_RATE}")
    print(f"  Epochs        : {EPOCHS}")

    # Step 2: Load model
    print(f"\n[2/4] Model Architecture")
    model = DigitRecognizer().to(DEVICE)
    print(get_model_summary(model))

    # Step 3: Load data
    print(f"\n[3/4] Loading Dataset")
    train_loader, val_loader = get_kaggle_data_loaders()
    criterion = get_loss_function()

    # Step 4: Forward pass on a few batches
    print(f"\n[4/4] Forward Pass Test ({num_batches} batches)")
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss    = 0.0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            if batch_idx >= num_batches:
                break

            images  = images.to(DEVICE)
            labels  = labels.to(DEVICE)

            # Forward pass
            outputs = model(images)

            # Loss (untrained random weights -> expect ~2.302)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Accuracy
            preds = outputs.argmax(dim=1)
            total_correct += preds.eq(labels).sum().item()
            total_samples += labels.size(0)

            print(f"  Batch {batch_idx+1}: loss={loss.item():.4f} | "
                  f"preds={preds[:8].tolist()} | "
                  f"truth={labels[:8].tolist()}")

    avg_loss = total_loss / num_batches
    accuracy = 100.0 * total_correct / total_samples

    print("\n" + "=" * 55)
    print(" RESULTS (Untrained Model — Random Baseline)")
    print("=" * 55)
    print(f"  Batches tested  : {num_batches}")
    print(f"  Samples tested  : {total_samples}")
    print(f"  Avg Loss        : {avg_loss:.4f}  (expected ~2.302)")
    print(f"  Accuracy        : {accuracy:.2f}%  (expected ~10%)")
    print("=" * 55)

    # Sanity assertions
    assert avg_loss < 4.0,   "Loss is too high — something is wrong with the model output."
    assert accuracy < 30.0,  "Accuracy suspiciously high for an untrained model."
    print("\n  All sanity checks passed! Pipeline is correct.")
    print("  Ready to train.\n")


if __name__ == "__main__":
    run_sanity_test(num_batches=5)
