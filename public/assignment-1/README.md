# DL-Ops Assignment 1 Results

## Q1)A) ResNet Performance on MNIST and FashionMNIST
> **Note:** Models trained with `pretrained=False` and `USE_AMP=True`. Experiments varied by `pin_memory` and `epochs'; the best accuracy obtained is noted among the different experiment results.

### Dataset: MNIST
| Batch Size | Optimizer | Learning Rate | ResNet-18 Accuracy (%) | No. of Epochs | pin memory| ResNet-50 Accuracy (%) | No. of Epochs | pin memory|
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 16 | SGD | 0.001 | 97.44 |5 | True | 97.36 |5 |True |
| 16 | SGD | 0.0001 |77.62 |5 |False | 53.98 |5 |False |
| 16 | Adam | 0.001 | 99.16 |3 |False | 98.91 | 5|True |
| 16 | Adam | 0.0001 |99.3 |5 |True |99.15 |5 |True |
| 32 | SGD | 0.001 |96.21 |5 |True | 95.07|5 |True | 
| 32 | SGD | 0.0001 | 63.64| 5|True |40.69 | 5|True |
| 32 | Adam | 0.001 | 99.0| 5 |False| 98.77| 5|False |
| 32 | Adam | 0.0001 | 99.14| 5|True | 98.94|3 |True |

### Dataset: FashionMNIST
| Batch Size | Optimizer | Learning Rate | ResNet-18 Accuracy (%) | No. of Epochs | pin memory| ResNet-50 Accuracy (%) | No. of Epochs | pin memory|
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 16 | SGD | 0.001 | 88.01| 5 | False |87.56 |5 |False |
| 16 | SGD | 0.0001 | 74.56| 5 | True | 74.43| 5|True |
| 16 | Adam | 0.001 | 93.46| 5 | False | 90.05| 5 |True |
| 16 | Adam | 0.0001 | 93.26| 5 | True |91.59 |5 |False |
| 32 | SGD | 0.001 |80.36 |5 | True | 79.87|5 |True |
| 32 | SGD | 0.0001 | 69.39| 5 | False | 67.32|5 |True |
| 32 | Adam | 0.001 | 92.57| 5 | False | 91.67| 5| False|
| 32 | Adam | 0.0001 |92.62 | 3 | False | 90.98| 5|True |

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
> **Requirement:** Comparison of CPU vs GPU performance, including FLOPs. All were run for 5 epochs

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
