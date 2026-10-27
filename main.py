import os
import sys
import random
import argparse
import numpy as np
import torch
import wandb

from datetime import datetime
from dataset  import get_dataloaders
from model    import get_model
from train    import train, get_optimizer
from evaluate import evaluate


# Default Baseline Config
DEFAULT_PROJECT    = 'CS6886W-VGG6'
DEFAULT_SEED       = 42
DEFAULT_ACTIVATION = 'relu'
DEFAULT_OPTIMIZER  = 'adam'
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR         = 0.001
DEFAULT_EPOCHS     = 30

# W&B Sweep Config (used only when --sweep is passed)
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val_acc', 'goal': 'maximize'},
    'parameters': {
        'activation': {'values': ['relu', 'sigmoid', 'tanh', 'silu', 'gelu']},
        'optimizer':  {'values': ['sgd', 'nesterov_sgd', 'adam', 'adagrad', 'rmsprop', 'nadam']},
        'lr':         {'values': [0.1, 0.01, 0.001, 0.0001]},
        'batch_size': {'values': [32, 64, 128, 256]},
        'epochs':     {'value': 30}
    }
}

# Set seed to reproduce the result later
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def run_experiment(manual_config=None, project=DEFAULT_PROJECT):
	#Checking and settig device whether gpu available or not
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # If running as manual run (i.e not inside a sweep), initialize W&B with manual config
    if wandb.run is None:
        cfg_dict = manual_config or {
            "seed": DEFAULT_SEED,
            "activation": DEFAULT_ACTIVATION,
            "optimizer": DEFAULT_OPTIMIZER,
            "lr": DEFAULT_LR,
            "batch_size": DEFAULT_BATCH_SIZE,
            "epochs": DEFAULT_EPOCHS
        }
        run_name = f"{cfg_dict['optimizer']}_{cfg_dict['activation']}_lr{cfg_dict['lr']}_bs{cfg_dict['batch_size']}"
        wandb.init(
            project=project,
            name=run_name,
            config={
                "architecture": "VGG-6",
                "dataset": "CIFAR-10",
                **cfg_dict
            }
        )

    cfg = wandb.config

    # Explicitly set dynamic run name so W&B dashboard always displays active config
    run_name = f"{cfg.optimizer}_{cfg.activation}_lr{cfg.lr}_bs{cfg.batch_size}"
    if wandb.run:
        wandb.run.name = run_name

    # Load data
    train_loader, test_loader = get_dataloaders(batch_size=cfg.batch_size)

    # Build model with configurable activation
    model = get_model(num_classes=10, activation=cfg.activation).to(device)
    print(model)

    # Configure Optimizer
    optimizer = get_optimizer(cfg.optimizer, model, lr=cfg.lr)

    # Print active experiment parameters before starting training
    print(f"\n{'='*60}")
    print(f"Starting Run Configuration:")
    print(f"  • Activation   : {cfg.activation}")
    print(f"  • Optimizer    : {cfg.optimizer}")
    print(f"  • Learning Rate: {cfg.lr}")
    print(f"  • Batch Size   : {cfg.batch_size}")
    print(f"  • Epochs       : {cfg.epochs}")
    print(f"{'='*60}\n")

    # Train model
    train(model, train_loader, test_loader, optimizer, device, epochs=cfg.epochs)

    # Evaluation of accuracy
    train_acc = evaluate(model, train_loader, device)
    test_acc  = evaluate(model, test_loader, device)
    print(f"\nFinal Train Accuracy: {train_acc:.2f}% | Final Test Accuracy: {test_acc:.2f}%")

    # Save Model Checkpoint in data/trained_models/
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
    print(f"✓ Model checkpoint saved: {save_path}")

    # Log summary metrics to W&B
    if wandb.run:
        wandb.summary["final_train_acc"]  = train_acc
        wandb.summary["final_test_acc"]   = test_acc
        wandb.summary["model_checkpoint"] = model_filename
        wandb.finish()


def main():
	# Support for several user input so that we can run the analysys in 
	# Sweep mode or manual mode with config or manual mode with default values
    parser = argparse.ArgumentParser(description="VGG-6 on CIFAR-10 Training & Sweeps")
    parser.add_argument('--sweep', action='store_true',
						help='Launch W&B automated sweep')
    parser.add_argument('--count', type=int, default=20,
						help='Number of sweep trials (default: 20)')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
						help=f'Random seed (default: {DEFAULT_SEED})')
    parser.add_argument('--project', type=str, default=DEFAULT_PROJECT,
						help=f'W&B project name (default: {DEFAULT_PROJECT})')

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

    # Add random seed for exact reproducibility across runs
    set_seed(args.seed)

    # Automated W&B Login for ease :p
    api_key = os.environ.get("WANDB_API_KEY", "YOUR_WANDB_API_KEY_HERE")
    if api_key and api_key != "YOUR_WANDB_API_KEY_HERE":
        wandb.login(key=api_key)
    else:
        wandb.login()

    if args.sweep:
        sweep_id = wandb.sweep(sweep_config, project=args.project)
        print(f"\nStarting W&B Sweep with ID: {sweep_id} in project: {args.project}")
        wandb.agent(sweep_id, function=run_experiment, count=args.count)
    else:
        # Manual run using CLI arguments (or defaults if not specified)
        manual_config = {
            "seed": args.seed,
            "activation": args.activation,
            "optimizer": args.optimizer,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs
        }
        run_experiment(manual_config=manual_config, project=args.project)


if __name__ == '__main__':
    main()
