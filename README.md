# DL-Ops Assignment 1 Results

## Q1)A) ResNet Performance on MNIST and FashionMNIST
> **Note:** Models trained with `pretrained=False` and `USE_AMP=True`. Experiments varied by `pin_memory` and `epochs`.

### Dataset: MNIST
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
| MNIST | poly | | |
| MNIST | rbf | | |
| FashionMNIST | poly | | |
| FashionMNIST | rbf | | |

---

## Q2) Hardware Performance Analysis (FashionMNIST)
> **Requirement:** Comparison of CPU vs GPU performance including FLOPs.

| Compute | Batch Size | Optimizer | Learning Rate | Model | Test Accuracy (%) | Train Time (ms) | FLOPs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CPU | 16 | SGD | 0.001 | ResNet-18 | | | |
| CPU | 16 | SGD | 0.001 | ResNet-50 | | | |
| GPU | 16 | SGD | 0.001 | ResNet-18 | | | |
| GPU | 16 | SGD | 0.001 | ResNet-50 | | | |
