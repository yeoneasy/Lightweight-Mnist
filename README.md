# **Lightweight Deep Learning with SWA, Pruning, and Quantization**

This project demonstrates a lightweight deep learning pipeline using **Stochastic Weight Averaging (SWA)**, **Pruning**, and **Quantization** on the MNIST dataset. The goal is to optimize model size and computational efficiency while maintaining high accuracy.

---

## **Features**
- **SWA (Stochastic Weight Averaging)**:
  - Improves generalization by averaging weights over multiple epochs.
- **Pruning**:
  - Reduces model size by removing less important parameters using L1 unstructured pruning.
- **Quantization**:
  - Converts model weights to lower precision (INT8), reducing memory usage.
- **Performance Metrics**:
  - Evaluation includes accuracy and model size.

---

## **Dataset**
The project uses the [MNIST Dataset](http://yann.lecun.com/exdb/mnist/), which contains 28x28 grayscale images of handwritten digits (0-9). Each image is normalized to [0, 1].

---

## **Requirements**
To run the project, the following libraries are required:

- Python 3.8+
- PyTorch 1.11+
- torchvision
- `thop` (for parameter calculation)
- `torch.quantization`
- Matplotlib
- NumPy
- Pandas

Install dependencies:
```bash
pip install torch torchvision thop matplotlib numpy pandas
