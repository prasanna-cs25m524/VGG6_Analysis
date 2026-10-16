CS6886W: Systems Engineering for Deep Learning — Assignment 1

Department of Computer Science and Engineering, IIT Madras
Semester: Fall 2026 (July – November 2026)
Student Name: Prasanna JS
Roll Number: CS25M524

Section 1: Setup and Project Details
1.1 Project Overview

This project explores the performance of the VGG-6 convolutional neural network architecture on the CIFAR-10 image classification dataset. The project is organized in a clean, modular fashion to systematically evaluate how different architectural and hyperparameter choices impact model convergence and accuracy.

1.2 Directory Structure
text
VGG6_A1/
├── dataset.py            # Data loading, normalization, and data augmentation pipeline
├── model.py              # VGG-6 network architecture definition
├── train.py              # Model training loop, optimization, and metric recording
├── evaluate.py           # Evaluation logic for top-1 classification accuracy
├── main.py               # Main execution and coordination script
└── README.md             # Project documentation and execution instructions

(Note: When you run main.py, a data/ directory will automatically be created to store the CIFAR-10 dataset).

1.3 Environment & Dependencies
Python: 3.10+ recommended
Hardware: NVIDIA GPU with CUDA support recommended (e.g., Google Colab T4) or CPU

Required packages:

bash
pip install torch torchvision
1.4 How to Run
Option A: Running on Google Colab (Terminal / Notebook)
Upload the VGG6_A1 directory files into your Colab runtime /content directory.
In a Colab notebook cell, ensure GPU is selected (Runtime -> Change runtime type -> T4 GPU).
Run:
bash
!python3 main.py
Option B: Running Locally (Terminal)

Navigate into the project directory and execute main.py:

bash
cd VGG6_A1
python3 main.py
1.5 Modular Component Responsibilities
dataset.py: Loads CIFAR-10 (50k train, 10k test), applies data augmentations (Random Crop with padding 4, Random Horizontal Flip, Normalization), and generates batch loaders.
model.py: Constructs the VGG-6 network ([64, 64, 'M', 128, 128, 'M']), applies global average pooling, and projects to 10 class logits.
train.py: Handles forward pass, cross-entropy loss computation, gradient backpropagation, and parameter updates.
evaluate.py: Calculates evaluation accuracy over any dataloader in evaluation mode (torch.no_grad()).
main.py: Glues all components together, detects the compute device (CUDA/CPU), runs training, and outputs final test accuracy.
Section 2: Introduction & SOP

[TODO]: To be added.

Section 3: Analysis

[TODO]: To be added.
