import os
import argparse
from datetime import datetime
import torch
import wandb

from dataset  import get_dataloaders
from model    import get_model
from train    import train, get_optimizer
from evaluate import evaluate


# Default Baseline Config
DEFAULT_ACTIVATION = 'relu'      # ['relu', 'sigmoid', 'tanh', 'silu', 'gelu']
DEFAULT_OPTIMIZER  = 'adam'      # ['sgd', 'nesterov_sgd', 'adam', 'adagrad', 'rmsprop', 'nadam']
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR         = 0.001
DEFAULT_EPOCHS     = 10

# W&B Sweep Config (used only when --sweep is passed)
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


def run_experiment(manual_config=None):
    # Check whether GPU is available or not; if not, run using CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # If running as manual run (not inside a sweep), initialize W&B with manual config
    if wandb.run is None:
        cfg_dict = manual_config or {
            "activation": DEFAULT_ACTIVATION,
            "optimizer": DEFAULT_OPTIMIZER,
            "lr": DEFAULT_LR,
            "batch_size": DEFAULT_BATCH_SIZE,
            "epochs": DEFAULT_EPOCHS
        }
        wandb.init(
            project="CS6886W-VGG6",
            name=f"manual_{cfg_dict['optimizer']}_{cfg_dict['activation']}",
            config={
                "architecture": "VGG-6",
                "dataset": "CIFAR-10",
                **cfg_dict
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

    # Evaluation
    train_acc = evaluate(model, train_loader, device)
    test_acc  = evaluate(model, test_loader, device)
    print(f"\nFinal Train Accuracy: {train_acc:.2f}% | Final Test Accuracy: {test_acc:.2f}%")

    # Save Model inside data/trained_models/ folder
    checkpoint_dir = 'data/trained_models'
    os.makedirs(checkpoint_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id    = wandb.run.id if (wandb.run and wandb.run.id) else "NA"

    # Filename format matching W&B run ID (or NA), timestamp, configs, and accuracies
    model_filename = (
        f"{run_id}_{timestamp}_{cfg.activation}_{cfg.optimizer}_"
        f"lr-{cfg.lr}_bs-{cfg.batch_size}_epochs-{cfg.epochs}_"
        f"acc-train-{train_acc:.2f}_acc-test-{test_acc:.2f}.pth"
    )
    save_path = os.path.join(checkpoint_dir, model_filename)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved: {save_path}")

    if wandb.run:
        wandb.summary["final_train_acc"]  = train_acc
        wandb.summary["final_test_acc"]   = test_acc
        wandb.summary["model_checkpoint"] = model_filename
        wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="VGG-6 on CIFAR-10 Training & Sweeps")
    parser.add_argument('--sweep', action='store_true', help='Launch W&B automated sweep')
    parser.add_argument('--count', type=int, default=20, help='No of sweep trials (default: 20)')

    # Custom manual run arguments (ignored if --sweep is provided)
    parser.add_argument('--activation', type=str, default=DEFAULT_ACTIVATION,
                        choices=['relu', 'sigmoid', 'tanh', 'silu', 'gelu'],
                        help=f'Activation function (default: {DEFAULT_ACTIVATION})')
    parser.add_argument('--optimizer', type=str, default=DEFAULT_OPTIMIZER,
                        choices=['sgd', 'nesterov_sgd', 'adam', 'adagrad', 'rmsprop', 'nadam'],
                        help=f'Optimizer (default: {DEFAULT_OPTIMIZER})')
    parser.add_argument('--lr', type=float, default=DEFAULT_LR,
                        help=f'Learning rate (default: {DEFAULT_LR})')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'Batch size (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help=f'Number of training epochs (default: {DEFAULT_EPOCHS})')

    args = parser.parse_args()

    # Automated W&B Login
    api_key = os.environ.get("WANDB_API_KEY", "YOUR_WANDB_API_KEY_HERE")
    if api_key and api_key != "YOUR_WANDB_API_KEY_HERE":
        wandb.login(key=api_key)
    else:
        wandb.login()

    if args.sweep:
        sweep_id = wandb.sweep(sweep_config, project="CS6886W-VGG6")
        print(f"\nStarting W&B Sweep with ID: {sweep_id}")
        wandb.agent(sweep_id, function=run_experiment, count=args.count)
    else:
        # Manual run using CLI arguments (or defaults if not specified)
        manual_config = {
            "activation": args.activation,
            "optimizer": args.optimizer,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs
        }
        run_experiment(manual_config=manual_config)


if __name__ == '__main__':
    main()
