# **CS6886W: Systems Engineering for Deep Learning** - Assignment 1

**Department of Computer Science and Engineering, IIT Madras**  
**Semester**: Fall 2026 (July – November 2026)  
**Name**: Prasanna JS  
**Roll Number**: CS25M524  

---

## Section 1: Setup and Analysis Details

### 1.1 Analysis overview
In this Assignment I explored the performance of the **VGG-6** CNN architecture with **CIFAR-10** dataset. I have evaluated how different configuration impact model convergence and accuracy, Used **Weights & Biases (W&B)** for visualization of experiments.

### 1.2 Code Tree Structure
```text
VGG6_Analysis/
├── data/trained_models   # Trained PyTorch models files for diffrent iteration
├── data/screenshots      # Screenshots of graphs (Yet to upload)
├── dataset.py            # Data loading, normalization, and full data augmentation related code
├── model.py              # VGG-6 CNN with option to config activation function
├── train.py              # Training loop with W&B logging related implementation
├── evaluate.py           # Accuracy calculation (Just added placeholder for future)
├── main.py               # Main function where accept diffrent command from the user or to set default values
└── README.md             # Project Doc, ofcourse this doc :p
```
*(Note: At runtime, in `data/` directory CIFAR-10 dataset is automatically downloaded, and inside `data/trained_models/` we store model files).*

### 1.3 Environment Setup & Dependencies
- **Python**: 3.10+ 
- **Hardware**: NVIDIA GPU with CUDA support (For This Assignment i have used Google colab and Kaggle)

Required packages:
```bash
pip install torch torchvision wandb
```

#### W&B Setup & Auto-Login & Seed related user configs
To allow automated logging and log visualization `main.py` supports silent auto-login. You can either:
1. Set the environment variable:
   ```bash
   export WANDB_API_KEY="YOUR_WANDB_API_KEY_HERE"
   ```
   *(Inside Google Colab: `os.environ["WANDB_API_KEY"] = "YOUR_WANDB_API_KEY_HERE"`)*
2. Or configure `api_key` in `main.py` directly by hardcoding.

### 1.4 Seed Configuration (To reproduce identical result)
To guarantee exact reproducibility across runs, the seed value (which ideally should be random) is fixed via `set_seed(42)`:
We can also pass a custom seed via CLI: `--seed <int>` (And default val is: `42`).

### 1.5 How to Run

#### Option A: Baseline Training (Default Configuration)
Runs the baseline model (Adam, ReLU, batch size 64, lr 0.001, epochs 30, seed 42) and logs loss/accuracy curves to W&B:
```bash
python3 main.py
```

#### Option B: Manual Training with Custom user provided parameters via cmd line arguments
You can override any parameter directly via cmd line arguments without editing code:
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
- `--project`: W&B project name to redirect logs (default: `CS6886W-VGG6`)

#### Option C: Automated Sweep run
This is an automated optimization sweep using W&B across 5 activations, 6 optimizers, 4 learning rates, and 4 batch sizes:
```bash
python3 main.py --sweep --count 20 --project CS6886W-VGG6
```
Sweep CLI options:
- `--sweep`: flag to enable automated sweep agent mode
- `--count`: maximum number of trials/runs to execute (default: `20`)
- `--project`: W&B project name to redirect logs (default: `CS6886W-VGG6`)

### 1.6 Trained Model Saving & W&B log Mapping
After each training trial, the trained model weights (`.pth`) are automatically saved into `data/trained_models/` with the filename structured as:
```text
{run_id}_{timestamp}_{activation}_{optimizer}_lr-{lr}_bs-{bs}_epochs-{epochs}_acc-train-{train_acc}_acc-test-{test_acc}.pth
```
- When using W&B, `{run_id}` matches the unique 8-character W&B run ID.
- If running without W&B, `{run_id}` is set to `NA`.

## Section 4: Experimental Results & Benchmarks

This section I tabulated the benchmark of VGG-6 with CIFAR-10 across different configurations.

### 4.1 Varying Optimizers
In this set of experiments, the activation function (`relu`), batch size (`64`), learning rate (`0.001`), and training duration (`30` epochs) are kept same to evaluate the effect of optimization algorithms.

| Run # | Optimizer | Activation | Learning Rate | Batch Size | Epochs | Train Acc (%) | Test Acc (%) | Comments / Observations |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | `adam` (Baseline) | relu | 0.001 | 64 | 30 | 78.79% | 81.03% | Overall this is Best optimizer, I see it converged fast and stayed stable |
| 2 | `sgd` | relu | 0.001 | 64 | 30 | 53.80% | 56.90% | This one is pretty bad at low learning rate this converged way too slow |
| 3 | `nesterov_sgd` | relu | 0.001 | 64 | 30 | 66.18% | 70.07% | Here Adding momentum helped a lot, gave good improvement over basic SGD |
| 4 | `adagrad` | relu | 0.001 | 64 | 30 | 59.51% | 62.83% | Learning got flattened so fast (Need to understand why) |
| 5 | `rmsprop` | relu | 0.001 | 64 | 30 | 73.48% | 75.94% | This has good accurecy but I noticed loss curve is not smooth as compared to Adam |
| 6 | `nadam` | relu | 0.001 | 64 | 30 | 77.06% | 78.27% |  This gave gave steady progress, but in comparison to Adam this is having low accurecy |

> **Note on SGD & Learning Rate Sensitivity:**  
> Since standard SGD struggled at `lr=0.001`, we also tested `nesterov_sgd` with a higher learning rate of `0.01` (`relu`, bs 64, 30 epochs). Accuracy jumped from 70.07% to **79.41%** (Train Acc: 76.66%). This confirms that SGD is heavily dependent on a larger learning rate, though it still fell short of Adam's 81.03%.

---

### 4.2 Varying Activation Functions
In this set of experiments, the optimizer (`adam`), learning rate (`0.001`), batch size (`64`), and training duration (`30` epochs) are kept same to evaluate the effect of different activation functions.

| Run # | Activation | Optimizer | Learning Rate | Batch Size | Epochs | Train Acc (%) | Test Acc (%) | Comments / Observations |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | `relu` (Baseline) | adam | 0.001 | 64 | 30 | 78.79% | 81.03% | Solid and fast one and easy to train but its has a issue that relu only outputs non-negative numbers |
| 2 | `sigmoid` | adam | 0.001 | 64 | 30 | 39.79% | 49.61% | This one is very bad may be due to vanishing gradients, didn't even cross 50% |
| 3 | `tanh` | adam | 0.001 | 64 | 30 | 71.37% | 75.81% | Better than sigmoid, but still not good as relu |
| 4 | `silu` | adam | 0.001 | 64 | 30 | 80.12% | 82.55% | Worked really well, smooth curve and beat the relu |
| 5 | `gelu` | adam | 0.001 | 64 | 30 | 80.80% | 83.04% | Best activation tested, gave the highest accuracy at 83% |

---

### 4.3 Varying Learning Rate, Batch Size, and Epochs
In this set of experiments, we kept the baseline configuration (Adam + ReLU) and evaluated how varying the learning rate, mini-batch size, and training epochs affects convergence.

#### A. Learning Rate Exploration (Fixed: BS=64, Epochs=30)
| Run # | Learning Rate | Batch Size | Epochs | Optimizer | Activation | Train Acc (%) | Test Acc (%) | Comments / Observations |
| :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :--- |
| 1 | 0.001 (Baseline) | 64 | 30 | adam | relu | 78.79% | 81.03% | Standard baseline reference; very stable convergence |
| 2 | 0.1 | 64 | 30 | adam | relu | 45.18% | 48.80% | Learning rate is way too high; loss oscillated wildly and failed to settle |
| 3 | 0.01 | 64 | 30 | adam | relu | 81.65% | 82.90% | Surprising winner! Beat the baseline comfortably |
| 4 | 0.0001 | 64 | 30 | adam | relu | 68.64% | 73.32% | Too small step size; convergence was too sluggish within 30 epochs |

#### B. Batch Size Exploration (Fixed: LR=0.001, Epochs=30)
| Run # | Learning Rate | Batch Size | Epochs | Optimizer | Activation | Train Acc (%) | Test Acc (%) | Comments / Observations |
| :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :--- |
| 1 | 0.001 (Baseline) | 64 | 30 | adam | relu | 78.79% | 81.03% | Standard baseline reference; very stable convergence |
| 5 | 0.001 | 32 | 30 | adam | relu | 79.31% | 82.04% | Best accuracy in batch variations, but took longer to train per epoch |
| 6 | 0.001 | 128 | 30 | adam | relu | 78.69% | 81.57% | Trained faster on GPU; accuracy stayed close to batch size 64 |
| 7 | 0.001 | 256 | 30 | adam | relu | 75.17% | 78.58% | Fastest execution, but fewer gradient updates caused accuracy to drop |

#### C. Epoch Duration Exploration (Fixed: LR=0.001, BS=64)
| Run # | Learning Rate | Batch Size | Epochs | Optimizer | Activation | Train Acc (%) | Test Acc (%) | Comments / Observations |
| :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :--- |
| 1 | 0.001 (Baseline) | 64 | 30 | adam | relu | 78.79% | 81.03% | Standard baseline reference; 30 epochs reached good convergence |
| 8 | 0.001 | 64 | 10 | adam | relu | 69.89% | 73.67% | Stopped too early while loss curve was still actively descending |
| 9 | 0.001 | 64 | 40 | adam | relu | 81.13% | 83.29% | Extra 10 epochs pushed past 83% without any signs of overfitting |


---

### 4.4 Promising Candidates Extended Evaluation (100–200 Epochs)
Based on the individual parameter benchmarks from Sections 4.1 to 4.3, clear patterns emerged:
- **Activations:** `gelu` (83.04%) and `silu` (82.55%) easily outperformed standard `relu` (81.03%) by preventing dying neurons.
- **Optimizers:** `adam` (81.03%) and `nadam` (78.27%) were the undisputed top-tier adaptive optimizers.
- **Learning Rate & Batch Size:** `0.01` gave fast early convergence (e.g. `gelu + adam, lr=0.01` achieved **84.17%** at 30 epochs), while `0.001` was rock solid. Smaller batches (`32` and `64`) provided useful regularizing noise compared to `256`.

We took these top candidate components and ran extended training (100 to 200 epochs) to push model convergence to its full potential:

| Run # | Configuration (Activation + Optimizer + LR + BS) | Epochs | Train Acc (%) | Test Acc (%) | Comments / Key Findings |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | `gelu` + `nadam` (lr=0.001, bs=32) | 200 | 90.55% | 88.30% | Highest accuracy across the entire project; smaller batch regularized smoothly |
| 2 | `gelu` + `nadam` (lr=0.01, bs=64) | 200 | 90.49% | 88.14% | Strongest 64-batch run; Nadam lookahead handled 0.01 very well |
| 3 | `silu` + `adam` (lr=0.01, bs=64) | 100 | 88.29% | 87.91% | SiLU with Adam at lr=0.01 reached almost 88% in just 100 epochs |
| 4 | `gelu` + `nadam` (lr=0.001, bs=64) | 200 | 89.69% | 87.81% | Steady and clean convergence throughout |
| 5 | `gelu` + `adam` (lr=0.001, bs=64) | 200 | 90.06% | 87.74% | Broke 90% train accuracy with low validation loss |
| 6 | `silu` + `nadam` (lr=0.001, bs=32) | 100 | 88.17% | 87.44% | Stable run, proving SiLU closely tracks GELU over long horizons |

---

### 4.5 Optimal Model Selection
From all experiments conducted across all horizons and combinations, the single best configuration is:

| Run # | Configuration (Activation + Optimizer + LR + BS) | Epochs | Train Acc (%) | Test Acc (%) | Comments / Key Findings |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | `gelu` + `nadam` (lr=0.001, bs=32) | 200 | 90.55% | 88.30% | **Overall Project Champion.** Reached the highest test accuracy (88.30%) and lowest validation loss (0.4076) across the entire project by combining smooth non-linear gradients with Nesterov lookahead momentum and small-batch regularization noise. |

---

### 4.6 Extended Runs That Didn't Help (Diminishing Returns & Instability)
Not all combinations or longer runs led to improvements. Here are the runs where increasing epochs or tuning parameters didn't help or caused issues:

| Run # | Configuration (Activation + Optimizer + LR + BS) | Epochs | Train Acc (%) | Test Acc (%) | Train Loss | Val Loss | Why It Didn't Help |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | `gelu` + `nadam` (lr=0.01, bs=32) | 200 | 88.17% | 86.46% | 0.3184 | 0.5163 | Pairing high LR (0.01) with noisy small batch (32) caused excessive gradient variance; val loss blew up to 0.5163 |
| 2 | `gelu` + `adam` (lr=0.01, bs=64) | 200 | 88.56% | 87.29% | 0.3059 | 0.4929 | Fast at 30 epochs (84.17%), but over 200 epochs without an LR decay scheduler it oscillated near the minimum |
| 3 | `relu` + `adam` (lr=0.001, bs=64) | 200 | 88.52% | 87.77% | 0.3320 | 0.4268 | Baseline gained accuracy with epochs, but plateaued early due to dying neurons and couldn't match GELU/Nadam |



