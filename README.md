# **CS6886W: Systems Engineering for Deep Learning** — Assignment 1

**Department of Computer Science and Engineering, IIT Madras**  
**Semester**: Fall 2026 (July – November 2026)  
**Student Name**: Prasanna JS  
**Roll Number**: CS25M524  

---

## Section 1: Setup and Project Details

### 1.1 Project Overview
This project explores the performance of the **VGG-6** convolutional neural network architecture on the **CIFAR-10** image classification dataset. The project is organized in a clean, modular fashion to systematically evaluate how different architectural and hyperparameter choices impact model convergence and accuracy, using **Weights & Biases (W&B)** for experiment tracking and automated hyperparameter sweeps.

### 1.2 Directory Structure
```text
VGG6_Analysis/
├── dataset.py            # Data loading, normalization, and full data augmentation pipeline
├── model.py              # VGG-6 network architecture with configurable activation functions
├── train.py              # Training loop, optimizer factory, and live W&B logging
├── evaluate.py           # Evaluation logic for top-1 classification accuracy
├── main.py               # Unified execution script (supports single runs, CLI overrides, and W&B sweep)
└── README.md             # Project documentation and execution instructions
```
*(Note: At runtime, a `data/` directory is automatically created to store the CIFAR-10 dataset, and `data/trained_models/` stores saved model checkpoints).*

### 1.3 Environment & Dependencies
- **Python**: 3.10+ recommended
- **Hardware**: NVIDIA GPU with CUDA support recommended (e.g., Google Colab T4) or CPU

Required packages:
```bash
pip install torch torchvision wandb
```

#### W&B Setup & Auto-Login
To allow automated metric logging without interactive terminal prompts, `main.py` supports silent auto-login. You can either:
1. Set the environment variable:
   ```bash
   export WANDB_API_KEY="YOUR_WANDB_API_KEY_HERE"
   ```
   *(Inside Google Colab: `os.environ["WANDB_API_KEY"] = "YOUR_WANDB_API_KEY_HERE"`)*
2. Or configure `api_key` in `main.py` directly.

### 1.4 Seed Configuration (Reproducibility)
To guarantee full determinism and exact reproducibility across runs, the random seed is fixed via `set_seed(42)` at the beginning of `main.py`:
- `random.seed(42)`: Seeds Python's built-in random module.
- `np.random.seed(42)`: Seeds NumPy random generator (used in Cutout augmentation).
- `torch.manual_seed(42)` & `torch.cuda.manual_seed_all(42)`: Seeds PyTorch CPU and GPU RNGs.
- `torch.backends.cudnn.deterministic = True`: Enforces deterministic cuDNN convolution algorithms.

You can also pass a custom seed via CLI: `--seed <int>` (default: `42`).

### 1.5 How to Run

#### Option A: Baseline Training (Default Configuration)
Runs the baseline model (Adam, ReLU, batch size 64, lr 0.001, epochs 10, seed 42) and logs loss/accuracy curves to W&B:
```bash
python3 main.py
```

#### Option B: Manual Training with Custom Parameters
You can override any parameter directly via command-line arguments without editing code:
```bash
python3 main.py --activation gelu --optimizer nesterov_sgd --lr 0.01 --batch_size 128 --epochs 25 --seed 42
```
Available CLI options:
- `--activation`: `relu`, `sigmoid`, `tanh`, `silu`, `gelu` (default: `relu`)
- `--optimizer`: `sgd`, `nesterov_sgd`, `adam`, `adagrad`, `rmsprop`, `nadam` (default: `adam`)
- `--lr`: float learning rate (default: `0.001`)
- `--batch_size`: batch size (default: `64`)
- `--epochs`: number of epochs (default: `10`)
- `--seed`: random seed (default: `42`)

#### Option C: Automated Hyperparameter Sweep
Launches an automated Bayesian optimization sweep using W&B across 5 activations, 6 optimizers, 4 learning rates, and 4 batch sizes:
```bash
python3 main.py --sweep
```

### 1.6 Model Checkpoints & W&B Mapping
After each training trial, the trained model weights (`.pth`) are automatically saved into `data/trained_models/` with the filename structured as:
```text
{run_id}_{timestamp}_{activation}_{optimizer}_lr-{lr}_bs-{bs}_epochs-{epochs}_acc-train-{train_acc}_acc-test-{test_acc}.pth
```
- When using W&B, `{run_id}` matches the unique 8-character W&B run ID (e.g., `98uat2wj`).
- If running without W&B, `{run_id}` is set to `NA`.
- The exact filename is logged to the W&B dashboard summary table under the `model_checkpoint` column, providing an instant 1-to-1 link between any curve or plot and its saved weights on disk.

### 1.7 Modular Component Responsibilities
- `dataset.py`: Loads CIFAR-10 (50k train, 10k test), applies data augmentations (Random Crop padding 4, Random Horizontal Flip, Random Rotation 5°, Color Jitter, Normalization, Cutout), and provides DataLoaders.
- `model.py`: Constructs the VGG-6 network (`[64, 64, 'M', 128, 128, 'M']`) with a `get_activation` factory supporting **ReLU**, **Sigmoid**, **Tanh**, **SiLU**, and **GELU**.
- `train.py`: Provides `get_optimizer` factory supporting **SGD**, **Nesterov-SGD**, **Adam**, **Adagrad**, **RMSprop**, and **Nadam**, executes forward/backward passes, and logs live metrics (`train_loss`, `train_acc`, `val_acc`) to W&B.
- `evaluate.py`: Calculates classification top-1 accuracy over any DataLoader in inference mode (`torch.no_grad()`).
- `main.py`: Unified entry point with seed configuration, CLI argument parser, and automated W&B login. Detects compute device (CUDA/CPU), runs manual experiments, and orchestrates automated W&B sweeps via `--sweep`.

---

## Section 2: Introduction & SOP

> **[TODO]**: To be added.

---

## Section 3: Analysis

> **[TODO]**: To be added.

---

## Section 4: Experimental Results & Benchmarks

This section tracks the systematic benchmarking of VGG-6 on CIFAR-10 across different architectural components and hyperparameter settings. All runs are logged via Weights & Biases with saved model checkpoints in `data/trained_models/`.

### 4.1 Varying Optimizers
In this set of experiments, the activation function (`relu`), batch size (`64`), learning rate (`0.001`), and training duration (`30` epochs) are kept constant while evaluating different optimization algorithms.

| Run # | Optimizer | Learning Rate | Batch Size | Epochs | Train Acc (%) | Test Acc (%) | Comments / Observations |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | `adam` (Baseline) | 0.001 | 64 | 30 | — | — | Standard baseline adaptive moment estimation |
| 2 | `sgd` | 0.001 | 64 | 30 | — | — | Standard stochastic gradient descent |
| 3 | `nesterov_sgd` | 0.001 | 64 | 30 | — | — | SGD with momentum=0.9 and Nesterov acceleration |
| 4 | `adagrad` | 0.001 | 64 | 30 | — | — | Adaptive subgradient with parameter-wise learning rates |
| 5 | `rmsprop` | 0.001 | 64 | 30 | — | — | Root Mean Square Propagation (alpha=0.99) |
| 6 | `nadam` | 0.001 | 64 | 30 | — | — | Nesterov-accelerated Adaptive Moment Estimation |

---

### 4.2 Varying Activation Functions
In this set of experiments, the optimizer (`adam`), learning rate (`0.001`), batch size (`64`), and training duration (`30` epochs) are held constant while evaluating different non-linear activation functions across all convolutional and linear layers.

| Run # | Activation | Optimizer | Learning Rate | Batch Size | Epochs | Train Acc (%) | Test Acc (%) | Comments / Observations |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| 1 | `relu` (Baseline) | adam | 0.001 | 64 | 30 | — | — | Standard non-saturating piecewise linear activation |
| 2 | `sigmoid` | adam | 0.001 | 64 | 30 | — | — | Saturating activation; sensitive to vanishing gradients |
| 3 | `tanh` | adam | 0.001 | 64 | 30 | — | — | Zero-centered saturating non-linearity |
| 4 | `silu` (Swish) | adam | 0.001 | 64 | 30 | — | — | Smooth non-monotonic gating function |
| 5 | `gelu` | adam | 0.001 | 64 | 30 | — | — | Gaussian Error Linear Unit non-linearity |

---

### 4.3 Varying Learning Rate and Epochs
In this set of experiments, we test sensitivity to learning rate values, mini-batch sizes, and training duration on the baseline architecture.

| Run # | Learning Rate | Batch Size | Epochs | Optimizer | Activation | Train Acc (%) | Test Acc (%) | Comments / Observations |
| :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :--- |
| 1 | 0.1 | 64 | 30 | adam | relu | — | — | High learning rate trial |
| 2 | 0.01 | 64 | 30 | adam | relu | — | — | Moderate learning rate trial |
| 3 | 0.001 (Baseline) | 64 | 30 | adam | relu | — | — | Default baseline learning rate |
| 4 | 0.0001 | 64 | 30 | adam | relu | — | — | Lower learning rate; slower convergence |
| 5 | 0.001 | 32 | 30 | adam | relu | — | — | Small batch size (frequent updates) |
| 6 | 0.001 | 128 | 30 | adam | relu | — | — | Moderate batch size |
| 7 | 0.001 | 256 | 30 | adam | relu | — | — | Large batch size (smoother gradients) |
| 8 | 0.001 | 64 | 10 | adam | relu | — | — | Short training horizon (10 epochs) |
| 9 | 0.001 | 64 | 40 | adam | relu | — | — | Extended training horizon (40 epochs) |

---

### 4.4 Optimal Model Run
> **[TODO]**: To be finalized after analyzing comparative runs from Sections 4.1 to 4.3 and automated sweep outputs. The best-performing combination of activation, optimizer, learning rate, and batch size will be evaluated here for peak accuracy.

| Run # | Configuration (Activation + Optimizer + LR + BS) | Epochs | Train Acc (%) | Test Acc (%) | Comments / Key Findings |
| :---: | :--- | :---: | :---: | :---: | :--- |
| Best | *TBD (e.g., gelu / silu + adam / nadam, lr=..., bs=...)* | 40 | — | — | Selected best hyperparameter configuration |

