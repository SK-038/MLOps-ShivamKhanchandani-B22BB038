# Assignment 4: Optimizing Transformer Translation with Ray Tune + Optuna

**Name:** [Your Name]  
**Roll Number:** [Your Roll Number]

---

## 1. Objective
The objective of this assignment is to optimize a custom PyTorch Transformer model for **English-to-Hindi** translation. By replacing fixed hyperparameters with a **Ray Tune + Optuna** search workflow, the goal is to match or exceed the baseline BLEU score in significantly fewer than 100 epochs.

---

## 2. Part 1: Baseline Execution
The baseline was established by training the model using the original `en_to_hi.ipynb` without any modifications to the architecture or hyperparameters.

| Metric | Baseline Value |
| --- | --- |
| GPU | [e.g., NVIDIA RTX A5000] |
| Total Training Time (100 epochs) | [Insert Your Time, e.g., 40 mins] |
| Final Training Loss | [Insert Your Loss] |
| Final BLEU Score (NLTK) | [Insert Your BLEU, e.g., 0.6347] |

**Baseline Weights Saved:** `transformer_translation_final.pth`.

---

## 3. Part 2: Refactor for Ray Tune + Optuna

### 2.1 Ray-Compatible Training Function
The training loop was refactored into a `train_tune(config)` function. It initializes the model, optimizer, and loss criterion directly from the `config` dictionary provided by the tuner. Metrics are reported per epoch to allow for real-time tracking:


## 3. Part 2: Refactor for Ray Tune + Optuna
To optimize the model, 5 key hyperparameters were varied. We ensured that the `d_model` remained divisible by the chosen `num_heads` to maintain the validity of the Transformer architecture.

* **Learning Rate (lr):** `tune.loguniform(1e-5, 1e-3)` 
* **Batch Size:** `tune.choice([16, 32, 64])` 
* **Number of Attention Heads:** `4, 8` 
* **FeedForward Dimension (d_ff):** `1024, 2048` 
* **Dropout Rate:** `0.1 to 0.4` 

### 2.3 Configuration & Best Result
The **ASHA Scheduler** was utilized to implement early stopping, terminating underperforming trials after a grace period of 5 epochs to save computational resources.

**Best Configuration Found:** 
```json
{
  "lr": 0.00012752212408379722,
  "batch_size": 16,
  "num_heads": 4,
  "d_ff": 1024,
  "dropout": 0.2810754004322514,
  "max_epochs": 40
}