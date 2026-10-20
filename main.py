# main.py
# Entry point: connects dataset → model → train → evaluate

import torch
import torch.optim as optim

from dataset  import get_dataloaders
from model    import get_model
from train    import train
from evaluate import evaluate


# ── Config ──────────────────────────────────
BATCH_SIZE = 64
LR         = 0.001
EPOCHS     = 25
# ────────────────────────────────────────────


def main():
    #check whether GPU is available or not if not run using CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load data
    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

    # 2. Build model
    model = get_model(num_classes=10).to(device)
    print(model)

    # 3. Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 4. Train
    train(model, train_loader, test_loader, optimizer, device, epochs=EPOCHS)

    # 5. Final test accuracy
    final_acc = evaluate(model, test_loader, device)
    print(f"\nFinal Test Accuracy: {final_acc:.2f}%")


if __name__ == '__main__':
    main()
