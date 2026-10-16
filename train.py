# train.py
# Responsible for: one full training loop over all epochs

import torch
import torch.nn as nn
from evaluate import evaluate


def train(model, train_loader, test_loader, optimizer, device, epochs=10):
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

        # Compute accuracies after each epoch
        train_acc = evaluate(model, train_loader, device)
        test_acc  = evaluate(model, test_loader,  device)
        avg_loss  = running_loss / len(train_loader)

        print(f"Epoch [{epoch}/{epochs}]  Loss: {avg_loss:.4f}  "
              f"Train Acc: {train_acc:.2f}%  Test Acc: {test_acc:.2f}%")
