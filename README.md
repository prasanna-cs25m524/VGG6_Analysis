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
Runs the baseline model (Adam, ReLU, batch size 64, lr 0.001, epochs 30, seed 42) and logs loss/accuracy curves to W&B:
```bash
python3 main.py
```

#### Option B: Manual Training with Custom Parameters
You can override any parameter directly via command-line arguments without editing code:
```bash
python3 main.py --activation gelu --optimizer nesterov_sgd --lr 0.01 --batch_size 128 --epochs 30 --seed 42 --project CS6886W-VGG6
```
Available CLI options:
- `--activation`: `relu`, `sigmoid`, `tanh`, `silu`, `gelu` (default: `relu`)
- `--optimizer`: `sgd`, `nesterov_sgd`, `adam`, `adagrad`, `rmsprop`, `nadam` (default: `adam`)
- `--lr`: float learning rate (default: `0.001`)
- `--batch_size`: batch size (default: `64`)
- `--epochs`: number of epochs (default: `30`)
- `--seed`: random seed (default: `42`)
- `--project`: W&B project name to group and superimpose runs (default: `CS6886W-VGG6`)

#### Option C: Automated Hyperparameter Sweep
Launches an automated Bayesian optimization sweep using W&B across 5 activations, 6 optimizers, 4 learning rates, and 4 batch sizes:
```bash
python3 main.py --sweep --count 20 --project CS6886W-VGG6
```
Sweep CLI options:
- `--sweep`: flag to enable automated sweep agent mode
- `--count`: maximum number of trials/runs to execute (default: `20`)
- `--project`: target W&B project for the sweep (default: `CS6886W-VGG6`)

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

| Run # | Optimizer | Activation | Learning Rate | Batch Size | Epochs | Train Acc (%) | Test Acc (%) | Comments / Observations |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | `adam` (Baseline) | relu | 0.001 | 64 | 30 | 78.79% | 81.03% | Best optimizer overall, learned fast and stayed stable |
| 2 | `sgd` | relu | 0.001 | 64 | 30 | 53.80% | 56.90% | Pretty bad at this low LR, converged way too slow without momentum |
| 3 | `nesterov_sgd` | relu | 0.001 | 64 | 30 | 66.18% | 70.07% | Adding momentum helped a lot, gave ~13% jump over basic SGD |
| 4 | `adagrad` | relu | 0.001 | 64 | 30 | 59.51% | 62.83% | Learning flattened early because step size decayed too quickly |
| 5 | `rmsprop` | relu | 0.001 | 64 | 30 | 73.48% | 75.94% | Decent score, but loss curve was noticeably more bumpy than Adam |
| 6 | `nadam` | relu | 0.001 | 64 | 30 | 77.06% | 78.27% | Solid second place, momentum lookahead gave steady progress |

---

### 4.2 Varying Activation Functions
In this set of experiments, the optimizer (`adam`), learning rate (`0.001`), batch size (`64`), and training duration (`30` epochs) are held constant while evaluating different non-linear activation functions across all convolutional and linear layers.

| Run # | Activation | Optimizer | Learning Rate | Batch Size | Epochs | Train Acc (%) | Test Acc (%) | Comments / Observations |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | `relu` (Baseline) | adam | 0.001 | 64 | 30 | 78.79% | 81.03% | Solid standard baseline, fast and easy to train |
| 2 | `sigmoid` | adam | 0.001 | 64 | 30 | 39.79% | 49.61% | Failed badly due to vanishing gradients, didn't even cross 50% |
| 3 | `tanh` | adam | 0.001 | 64 | 30 | 71.37% | 75.81% | Better than sigmoid, but still saturated and fell behind ReLU |
| 4 | `silu` (Swish) | adam | 0.001 | 64 | 30 | 80.12% | 82.55% | Worked really well, smooth curve helped beat normal ReLU |
| 5 | `gelu` | adam | 0.001 | 64 | 30 | 80.80% | 83.04% | Best activation tested, gave the highest accuracy at 83% |

---

### 4.3 Varying Learning Rate and Epochs
In this set of experiments, we test sensitivity to learning rate values, mini-batch sizes, and training duration on the baseline architecture.

| Run # | Learning Rate | Batch Size | Epochs | Optimizer | Activation | Train Acc (%) | Test Acc (%) | Comments / Observations |
| :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :--- |
| 1 | 0.1 | 64 | 30 | adam | relu | 45.18% | 48.80% | Way too high, loss kept jumping and barely learned |
| 2 | 0.01 | 64 | 30 | adam | relu | 81.65% | 82.90% | Surprising winner here, Adam handled 0.01 really well |
| 3 | 0.001 (Baseline) | 64 | 30 | adam | relu | 78.79% | 81.03% | Default baseline, safe and stable |
| 4 | 0.0001 | 64 | 30 | adam | relu | 68.64% | 73.32% | Too small, needed way more epochs to catch up |
| 5 | 0.001 | 32 | 30 | adam | relu | 79.31% | 82.04% | Good accuracy, noisy updates helped prevent overfitting |
| 6 | 0.001 | 128 | 30 | adam | relu | 78.69% | 81.57% | Ran faster, accuracy stayed pretty close to 64 |
| 7 | 0.001 | 256 | 30 | adam | relu | 75.17% | 78.58% | Fastest per epoch, but fewer updates made it lose ~2.5% |
| 8 | 0.001 | 64 | 10 | adam | relu | 69.89% | 73.67% | Stopped too early while still actively learning |
| 9 | 0.001 | 64 | 40 | adam | relu | 81.13% | 83.29% | More training time helped push past 83% |

---

### 4.4 Optimal Model Runs (30 Epochs)
Based on the observations from Sections 4.1 to 4.3, we tested two optimal combination runs at 30 epochs:

| Run # | Configuration (Activation + Optimizer + LR + BS) | Epochs | Train Acc (%) | Test Acc (%) | Comments / Key Findings |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | `gelu` + `adam` (lr=0.01, bs=64) | 30 | 82.49% | 84.17% | Highest accuracy across all 30-epoch tests by combining best activation and best LR |
| 2 | `relu` + `nesterov_sgd` (lr=0.01, bs=64) | 30 | 76.66% | 79.41% | Giving SGD a higher LR (0.01) brought it up from 70% to almost 80% |

---

### 4.5 Extended Training (200 Epochs)
To see how far the top architectures can go when given full convergence time, we ran extended 200-epoch training across 7 systematic configurations:

| Run # | Configuration (Activation + Optimizer + LR + BS) | Epochs | Train Acc (%) | Test Acc (%) | Train Loss | Val Loss | Comments / Key Findings |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | `gelu` + `nadam` (lr=0.001, bs=32) | 200 | 90.55% | 88.30% | 0.3034 | 0.4076 | All-time highest test accuracy! Smaller batch gave regularizing noise and lowest val loss |
| 2 | `gelu` + `nadam` (lr=0.01, bs=64) | 200 | 90.49% | 88.14% | 0.2973 | 0.4483 | Strongest 64-batch run, Nadam lookahead handled 0.01 significantly better than Adam |
| 3 | `gelu` + `nadam` (lr=0.001, bs=64) | 200 | 89.69% | 87.81% | 0.3111 | 0.4165 | Steady, smooth convergence throughout all 200 epochs |
| 4 | `relu` + `adam` (lr=0.001, bs=64) | 200 | 88.52% | 87.77% | 0.3320 | 0.4268 | Baseline gained +6.74% with extended training; converged solidly |
| 5 | `gelu` + `adam` (lr=0.001, bs=64) | 200 | 90.06% | 87.74% | 0.3115 | 0.4161 | Broke 90% train accuracy cleanly with stable validation loss |
| 6 | `gelu` + `adam` (lr=0.01, bs=64) | 200 | 88.56% | 87.29% | 0.3059 | 0.4929 | Fast early on, but constant 0.01 without scheduler caused minor late-stage jitter |
| 7 | `gelu` + `nadam` (lr=0.01, bs=32) | 200 | 88.17% | 86.46% | 0.3184 | 0.5163 | High LR combined with small batch caused too much gradient variance, hurting final test score |



