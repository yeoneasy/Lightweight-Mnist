
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

## **Code Execution Process**

The following steps outline the workflow of the project:

### **1. Data Preparation**
- Load the MNIST dataset, which contains 28x28 grayscale images of handwritten digits.
- Normalize the dataset to scale pixel values between 0 and 1.
- Convert the data into 4-dimensional tensors with a shape of `(batch_size, channels, height, width)` for CNN compatibility.

### **2. Model Training**
- Train a Convolutional Neural Network (CNN) model using the training data.
- Use **AdamW optimizer** with a learning rate of `0.01` to minimize the **Cross-Entropy Loss**.
- Perform training for 10 epochs, using a validation dataset to monitor performance during training.

### **3. Stochastic Weight Averaging (SWA)**
- Apply SWA starting from the 5th epoch (`--swa_start 5`).
- Average the model weights over multiple epochs to improve generalization.

### **4. Pruning**
- Perform **L1 unstructured pruning** on convolutional layers to reduce the number of parameters.
- Pruning removes 50% (`--pruning_amount 0.5`) of the least important weights based on their L1 norm.
- Update the BatchNorm statistics to adjust for pruned weights.

### **5. Quantization**
- Convert the pruned model's weights to **INT8** format using dynamic quantization.
- Quantization reduces memory usage and improves inference speed while maintaining accuracy.

### **6. Model Evaluation**
- Evaluate both the original and optimized models on the validation dataset.
- Compare the accuracy, model size, and parameter count.
- For optimized models, approximate the FLOPs and evaluate their inference speed on CPU.

### **7. Results Visualization**
- Generate comparison plots to visualize the difference between the original and optimized models in terms of accuracy, model size, and parameter count.

---

### **Execution Command**
To reproduce the entire pipeline, follow these steps:
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/your-repository.git
   cd your-repository
   
---

## **Table of Contents**
1. [Dataset](#dataset)
2. [Requirements](#requirements)
3. [Usage](#usage)
4. [Results](#results)
5. [Performance Metrics](#performance-metrics)
6. [References](#references)

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
```

---

## **Usage**

### 1. **Train the Original Model**
Run the script to train the base CNN model:
```bash
python train.py --epochs 10 --lr 0.01
```

### 2. **Apply SWA, Pruning, and Quantization**
Execute the following command to apply SWA, pruning, and quantization:
```bash
python lightweight.py --swa_start 5 --pruning_amount 0.5
```

### 3. **Evaluate the Optimized Model**
Run the script to evaluate the pruned and quantized model:
```bash
python evaluate.py
```

---

## **Results**

| **Metric**       | **Original Model** | **Optimized Model** |
|-------------------|--------------------|---------------------|
| Accuracy         | 99.5%             | 99.5%              |
| Model Size (MB)  | 25.08 MB          | 6.26 MB            |

---

## **Performance Metrics**

### 1. **Accuracy**
- The optimized model achieves comparable accuracy to the original model.

### 2. **Model Size**
- Pruning and quantization reduce the model size by approximately **75%**.

---

## **References**

1. Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D., & Wilson, A. G. "Averaging Weights Leads to Wider Optima in Deep Learning." 2018. [Link](https://arxiv.org/abs/1803.05407)
2. Han, S., Pool, J., Tran, J., & Dally, W. "Learning both Weights and Connections for Efficient Neural Networks." 2015. [Link](https://arxiv.org/abs/1506.02626)
3. Jacob, B., et al. "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference." 2018. [Link](https://arxiv.org/abs/1712.05877)
4. Frantar, E., Singh, S. P., & Alistarh, D. "Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning." [Link](https://arxiv.org/abs/2210.03887)
