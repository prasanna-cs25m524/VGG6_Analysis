import torch
import torch.optim as optim
import wandb

from dataset  import get_dataloaders
from model    import get_model
from train    import train
from evaluate import evaluate

BATCH_SIZE = 64
LR         = 0.001
EPOCHS     = 25

def main():
    # Check whether GPU is available or not; if not, run using CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize Weights & Biases (W&B) run
    wandb.init(
        project="CS6886W-VGG6",
        name="baseline_adam_relu",
        config={
            "architecture": "VGG-6",
            "dataset": "CIFAR-10",
            "activation": "ReLU",
            "optimizer": "Adam",
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS
        }
    )

    # Load data
    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

    # Build model
    model = get_model(num_classes=10).to(device)
    print(model)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Train model (logs to W&B automatically after each epoch)
    train(model, train_loader, test_loader, optimizer, device, epochs=EPOCHS)

    # Final test accuracy
    final_acc = evaluate(model, test_loader, device)
    print(f"\nFinal Test Accuracy: {final_acc:.2f}%")
    wandb.summary["final_test_acc"] = final_acc

    # Close W&B run
    wandb.finish()


if __name__ == '__main__':
    main()
