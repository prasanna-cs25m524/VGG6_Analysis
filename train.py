import torch
import torch.nn as nn
import wandb
from evaluate import evaluate

def train(model, train_loader, test_loader, optimizer, device, epochs=25):
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Compute epoch metrics
        train_acc = evaluate(model, train_loader, device)
        test_acc  = evaluate(model, test_loader,  device)
        avg_loss  = running_loss / len(train_loader)

        # Log metrics automatically to W&B (generates live Loss & Accuracy curves)
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_acc": train_acc,
            "val_acc": test_acc
        })

        print(f"Epoch [{epoch}/{epochs}]  Loss: {avg_loss:.4f}  "
              f"Train Acc: {train_acc:.2f}%  Test Acc: {test_acc:.2f}%")
