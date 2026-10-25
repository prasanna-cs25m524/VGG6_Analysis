import os
import sys
import torch
import wandb

from dataset  import get_dataloaders
from model    import get_model
from train    import train, get_optimizer
from evaluate import evaluate


# Baseline Config (without --sweep option)
ACTIVATION = 'relu'      # ['relu', 'sigmoid', 'tanh', 'silu', 'gelu']
OPTIMIZER  = 'adam'      # ['sgd', 'nesterov_sgd', 'adam', 'adagrad', 'rmsprop', 'nadam']
BATCH_SIZE = 64
LR         = 0.001
EPOCHS     = 10

# W&B Sweep Config (With --sweep option)
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val_acc', 'goal': 'maximize'},
    'parameters': {
        'activation': {'values': ['relu', 'sigmoid', 'tanh', 'silu', 'gelu']},
        'optimizer':  {'values': ['sgd', 'nesterov_sgd', 'adam', 'adagrad', 'rmsprop', 'nadam']},
        'lr':         {'values': [0.1, 0.01, 0.001, 0.0001]},
        'batch_size': {'values': [32, 64, 128, 256]},
        'epochs':     {'value': 10}
    }
}


def run_experiment():
    # Check whether GPU is available or not; if not, run using CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # If running inside a sweep, wandb has already initialized with sweep config
    # else its baseline run so need to make config
    if wandb.run is None:
        wandb.init(
            project="CS6886W-VGG6",
            name=f"baseline_{OPTIMIZER}_{ACTIVATION}",
            config={
                "architecture": "VGG-6",
                "dataset": "CIFAR-10",
                "activation": ACTIVATION,
                "optimizer": OPTIMIZER,
                "lr": LR,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS
            }
        )

    cfg = wandb.config

    # Load data
    train_loader, test_loader = get_dataloaders(batch_size=cfg.batch_size)

    # Build model with configurable activation
    model = get_model(num_classes=10, activation=cfg.activation).to(device)

    # Optimizer using factory function
    optimizer = get_optimizer(cfg.optimizer, model, lr=cfg.lr)

    # Train model
    train(model, train_loader, test_loader, optimizer, device, epochs=cfg.epochs)

    # Final test accuracy
    final_acc = evaluate(model, test_loader, device)
    print(f"\nFinal Test Accuracy: {final_acc:.2f}%")
    wandb.summary["final_test_acc"] = final_acc

    # Close W&B run
    wandb.finish()


def main():
    # Automated W&B Login
    api_key = os.environ.get("WANDB_API_KEY", "YOUR_WANDB_API_KEY_HERE")
    if api_key and api_key != "YOUR_WANDB_API_KEY_HERE":
        wandb.login(key=api_key)
    else:
        wandb.login()

    if '--sweep' in sys.argv:
        sweep_id = wandb.sweep(sweep_config, project="CS6886W-VGG6")
        print(f"\nStarting W&B Sweep with ID: {sweep_id}")
        wandb.agent(sweep_id, function=run_experiment, count=20)
    else:
        # Single Baseline run
        run_experiment()


if __name__ == '__main__':
    main()
