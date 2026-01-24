# DL-Ops Assignment 1 Results

## Q1)A) ResNet Performance on MNIST and FashionMNIST
> **Note:** Models trained with `pretrained=False` and `USE_AMP=True`. Experiments varied by `pin_memory` and `epochs`, the best accuracy obtained has been noted among different experiment results.

### Dataset: MNIST
| Batch Size | Optimizer | Learning Rate | ResNet-18 Accuracy (%) | ResNet-50 Accuracy (%) |
| :--- | :--- | :--- | :--- | :--- |
| 16 | SGD | 0.001 | 97.44 | 95.47 |
| 16 | SGD | 0.0001 |77.62 |42.52 |
| 16 | Adam | 0.001 | 99.16|98.76 |
| 16 | Adam | 0.0001 |99.3 |99.01 |
| 32 | SGD | 0.001 |96.21 | 95.07| 
| 32 | SGD | 0.0001 | 63.64|33.53 |
| 32 | Adam | 0.001 | 99.0| 98.55|
| 32 | Adam | 0.0001 | 99.14| 98.94|

### Dataset: FashionMNIST
| Batch Size | Optimizer | Learning Rate | ResNet-18 Accuracy (%) | ResNet-50 Accuracy (%) |
| :--- | :--- | :--- | :--- | :--- |
| 16 | SGD | 0.001 | | |
| 16 | SGD | 0.0001 | | |
| 16 | Adam | 0.001 | | |
| 16 | Adam | 0.0001 | | |
| 32 | SGD | 0.001 | | |
| 32 | SGD | 0.0001 | | |
| 32 | Adam | 0.001 | | |
| 32 | Adam | 0.0001 | | |

---

## Q1)B) SVM Classifier Results
> **Requirement:** Testing Classification Accuracy and training time in ms.

| Dataset | Kernel | Test Accuracy (%) | Train Time (ms) |
| :--- | :--- | :--- | :--- |
| MNIST | poly |97.71 |288743.64 |
| MNIST | rbf |97.92 |268346.94 |
| FashionMNIST | poly | 86.30| 473309.68|
| FashionMNIST | rbf |88.29 |389151.25 |

---

## Q2) Hardware Performance Analysis (FashionMNIST)
> **Requirement:** Comparison of CPU vs GPU performance including FLOPs. All were run for 5 epochs

| Compute | Batch Size | Optimizer | Learning Rate | Model | Test Accuracy (%) | Train Time (ms) | FLOPs (GFLOPs) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CPU | 16 | SGD | 0.001 | ResNet-18 |84.69 |506547 | 0.0332|
| CPU | 16 | SGD | 0.001 | ResNet-50 | 90.73 |1898165 |0.0788 |
| GPU | 16 | SGD | 0.001 | ResNet-18 | 85.08|446587 | 0.0332|
| GPU | 16 | SGD | 0.001 | ResNet-50 |90.65 | 1907845| 0.0788|
| CPU | 16 | Adam | 0.001 | ResNet-18 | 92.16|456461 |0.0332 |
| CPU | 16 | Adam | 0.001 | ResNet-50 | 93.54| 2508762 | 0.0788|
| GPU | 16 | Adam | 0.001 | ResNet-18 | 92.07| 408916|0.0332 |
| GPU | 16 | Adam | 0.001 | ResNet-50 | 93.42| 2376544|0.0788 |
