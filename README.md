# MLOps Lab Assignment - 1 — B22BB038

**Name:** Shivam Khanchandani
**Roll Number:** B22BB038

## Links

- **WANDB visualizations:** [View Assignment on WandB](https://wandb.ai/b22bb038-indian-institute-of-technology-jodhpur/CIFAR10_Assignment_Lab2/reports/ML-DL-Ops-Lab-2-Worksheet-Report--VmlldzoxNTgxMTU5MQ)

- **GitHub Repository:** [View Assignment on GitHub](https://github.com/SK-038/MLOps-ShivamKhanchandani-B22BB038/tree/lab_assignment_1_B22BB038)

---

## 1. Objective
The objective of this lab was to:
* Train a Convolutional Neural Network (CNN) on the **CIFAR-10 dataset**.
* Implement a **custom PyTorch dataloader**.
* Compute **FLOPs** for the selected model.
* Train the model for **25 epochs**.
* Visualize **Gradient flow** and **Weight update flow**.
* Log all metrics and visualizations using **Weights & Biases (WandB)**.

---

## 2. Dataset
We used the **CIFAR-10 dataset**, which contains:
* 60,000 RGB images of size **32×32**.
* 10 classes (airplane, car, bird, cat, etc.).
* 50,000 training images and 10,000 test images.

---

## 3. Model Architecture
A **ResNet-18** model was chosen for its ability to handle deep feature extraction through residual blocks.
* **Key components:** Residual blocks, Batch normalization, and ReLU activations.

---

## 4. Custom DataLoader Implementation
A custom `Dataset` class was implemented to manually handle data loading.
* **Augmentation:** Random crop, horizontal flip, and CIFAR-10 normalization.

---

## 5. FLOPs and Complexity
Complexity was measured using the `thop` library:
* **Total FLOPs:** 37.22 Million (0.037 GFLOPs).
* **Total Parameters:** 11.18 Million.

---

## 6. Training Setup
| Hyperparameter | Value |
| -------------- | ---------------- |
| Optimizer      | Adam |
| Learning Rate  | 0.001 |
| Batch Size     | 128 |
| Epochs         | 25 |
| Loss           | CrossEntropyLoss |

---

## 7. Visualization Observations
### **Gradient Flow**
* Gradients were monitored periodically via WandB bar charts.
* **Observation:** ResNet skip-connections maintained stable gradients across all 18 layers, preventing vanishing gradient issues.

### **Weight Update Flow**
* Tracks the magnitude of change in parameters ($||W_{new} - W_{old}||$).
* **Observation:** Large updates occurred initially, with magnitude decreasing and stabilizing as the model converged towards 25 epochs.

---

## 8. Results and Findings
* **Accuracy:** Training accuracy showed steady growth, reaching over 85% by the final epoch.
* **Validation:** Validation accuracy was tracked side-by-side to monitor generalization.
* **Convergence:** Model stability was achieved after ~15-20 epochs.
