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

### 4.3 Varying Learning Rate, batch size and Epochs (combined 3 as there were less entries)
In this set of experiments, kept the baseline config and varied learning rate values, batch sizes, and no of epochs on the baseline.

| Run # | Learning Rate | Batch Size | Epochs | Optimizer | Activation | Train Acc (%) | Test Acc (%) | Comments / Observations |
| :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :--- |
| 1 | 0.001 (Baseline) | 64 | 30 | adam | relu | 78.79% | 81.03% | This is default baseline and this one is very stable |
| 2 | 0.1 | 64 | 30 | adam | relu | 45.18% | 48.80% | I think this is too high value, I see loss kept fluctuating so much |
| 3 | 0.01 | 64 | 30 | adam | relu | 81.65% | 82.90% | This one is better than baseline |
| 4 | 0.0001 | 64 | 30 | adam | relu | 68.64% | 73.32% | I think its too small value and convergence is too slow |

| Run # | Learning Rate | Batch Size | Epochs | Optimizer | Activation | Train Acc (%) | Test Acc (%) | Comments / Observations |
| :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :--- |
| 1 | 0.001 (Baseline) | 64 | 30 | adam | relu | 78.79% | 81.03% | This is default baseline and this one is very stable |
| 5 | 0.001 | 32 | 30 | adam | relu | 79.31% | 82.04% | Got Good accuracy and this is the best in this section, but too slow traiining |
| 6 | 0.001 | 128 | 30 | adam | relu | 78.69% | 81.57% | Ran faster, accuracy is similar to batch size 64 |
| 7 | 0.001 | 256 | 30 | adam | relu | 75.17% | 78.58% | Fastest but accurecy dropped |

| Run # | Learning Rate | Batch Size | Epochs | Optimizer | Activation | Train Acc (%) | Test Acc (%) | Comments / Observations |
| :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :--- |
| 8 | 0.001 (Baseline)| 64 | 10 | adam | relu | 69.89% | 73.67% | This is default baseline and this one is very stable |
| 9 | 0.001 | 64 | 40 | adam | relu | 81.13% | 83.29% | More training time helped to get 2-3% good accurecy |


---

### 4.5 Cross-Configuration Explorations & Promising Candidates
Looking at the individual parameter benchmarks from Sections 4.1 to 4.4, clear patterns emerged:
- **Activations:** `gelu` (83.04%) and `silu` (82.55%) easily outperformed standard `relu` (81.03%) by preventing dying neurons.
- **Optimizers:** `adam` (81.03%) and `nadam` (78.27%) were the undisputed top-tier adaptive optimizers.
- **Learning Rate & Batch Size:** `0.01` gave fast early convergence, while `0.001` was rock solid. Smaller batches (`32` and `64`) provided useful regularizing noise compared to `256`.

Based on these promising candidates, we cross-tested combinations to see how these top components interact together:

| Run # | Configuration (Activation + Optimizer + LR + BS) | Epochs | Train Acc (%) | Test Acc (%) | Comments / Key Findings |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | `gelu` + `adam` (lr=0.01, bs=64) | 30 | 82.49% | 84.17% | Combining top activation with higher LR gave a solid +3.14% jump over baseline |
| 2 | `gelu` + `nadam` (lr=0.01, bs=64) | 30 | 83.18% | 85.17% | Pairing GELU with Nadam at lr=0.01 produced our highest 30-epoch score (+4.14% over baseline) |
| 3 | `silu` + `adam` (lr=0.01, bs=64) | 100 | 88.29% | 87.91% | SiLU performed impressively at lr=0.01, reaching nearly 88% by epoch 100 |
| 4 | `silu` + `nadam` (lr=0.001, bs=32) | 100 | 88.17% | 87.44% | Stable, smooth convergence, staying within half a percent of GELU |

---

### 4.6 Optimal Model Selection (30 Epochs)
Comparing all combinations within the standard 30-epoch constraint, the single best configuration is:

| Run # | Configuration (Activation + Optimizer + LR + BS) | Epochs | Train Acc (%) | Test Acc (%) | Comments / Key Findings |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | `gelu` + `nadam` (lr=0.01, bs=64) | 30 | 83.18% | 85.17% | **Clear Winner.** Reached 85.17% test accuracy (+4.14% over baseline) by combining smooth non-linear gradients with Nesterov momentum lookahead and 0.01 step size. |

---

### 4.7 Extended Training (200 Epochs)
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



