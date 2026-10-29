import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from evaluate import evaluate

#config optimizer depend on the configuration
def get_optimizer(name, model, lr):
    name = name.lower()
    params = model.parameters()

    if name == 'adam':
        return optim.Adam(params, lr=lr)
    elif name == 'sgd':
        return optim.SGD(params, lr=lr)
    elif name == 'nesterov_sgd':
        return optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
    elif name == 'adagrad':
        return optim.Adagrad(params, lr=lr)
    elif name == 'rmsprop':
        return optim.RMSprop(params, lr=lr)
    elif name == 'nadam':
        return optim.NAdam(params, lr=lr)
    else:
        raise ValueError(f"You gave wrong optimizer i.e : '{name}'. "
                         f"Supported ones are: ['sgd', 'nesterov_sgd',"
						 f"'adam', 'adagrad', 'rmsprop', 'nadam']")


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

        # Compute validation loss
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for v_images, v_labels in test_loader:
                v_images, v_labels = v_images.to(device), v_labels.to(device)
                v_outputs = model(v_images)
                val_loss_sum += criterion(v_outputs, v_labels).item()
        val_loss = val_loss_sum / len(test_loader)

        # Log metrics automatically to W&B (generates live Loss & Accuracy curves)
        wandb.log({
            "epoch": int(epoch),
            "train_loss": float(avg_loss),
            "val_loss": float(val_loss),
            "train_acc": float(train_acc),
            "val_acc": float(test_acc)
        }, commit=True)

        print(f"Epoch [{epoch}/{epochs}]  Loss: {avg_loss:.4f}  Val Loss: {val_loss:.4f}  "
              f"Train Acc: {train_acc:.2f}%  Test Acc: {test_acc:.2f}%")

    return avg_loss, val_loss
