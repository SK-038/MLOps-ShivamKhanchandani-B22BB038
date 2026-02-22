# Assignment 3: Fine-tuning ModernBERT on GLUE MRPC

**Name:** Shivam Khanchandani  
**Task:** Paraphrase classification on the Microsoft Research Paraphrase Corpus (MRPC)

##  Links

- **GitHub Repository:** https://github.com/SK-038/MLOps-ShivamKhanchandani-B22BB038/tree/assignment-3
- **Hugging Face Model:** https://huggingface.co/Shivam-K/ML-Ops-Assignment-3

---

##  Objective

Fine-tune a pre-trained **ModernBERT** model on the **GLUE MRPC** task, evaluate validation performance, and containerize the evaluation workflow using Docker.

**Evaluation Metrics:**
- Accuracy
- F1 score

## Model and Task

- **Model:** `answerdotai/ModernBERT-base` (sequence classification)
- **Dataset:** GLUE MRPC
- **Problem type:** Binary classification (paraphrase / not paraphrase)
- **Why this model:** ModernBERT is a recent, highly optimized architectural update to the classic BERT model. It was chosen because it offers superior computational efficiency, better handling of longer contexts, and faster inference times while maintaining state-of-the-art accuracy on GLUE benchmark tasks.

---

## Training Configuration

| Parameter | Value |
| --- | --- |
| `output_dir` | `ModernBERT_mrpc_ft` |
| `per_device_train_batch_size` | 32 |
| `num_train_epochs` | 2 |
| `learning_rate` | 8e-5 |
| `lr_scheduler_type` | linear |
| `optim` | `adamw_torch` |
| `bf16` | `True` |

**Tokenizer/Config Alignment Note:**
During training, tokenizer special tokens differed from model/generation config. Configs were aligned automatically:
- Updated keys: `eos_token_id`, `bos_token_id`
- Updated values: `{'eos_token_id': None, 'bos_token_id': None}`

---

## Training Progress & Evaluation Metrics

- Total optimization steps: **230/230**
- Total training time: **~4m 29s**
- Epochs completed: **2/2**

| Epoch | Training Loss | Validation Loss | 
| ---: | ---: | ---: |
| 1 | 0.507908 | 0.339598 |
| 2 | 0.263230 | 0.292411 |

### Final Result (Hub Model Evaluation)
After 2 epochs of fine-tuning ModernBERT on MRPC, the model was pulled directly from the Hugging Face Hub and re-evaluated, confirming the weights were successfully synced:

- **Accuracy:** `0.8823529411764706` (88.23%)
- **F1:** `0.9175257731958762` (91.75%)

These results indicate strong paraphrase detection performance, with F1 exceeding 0.90 on validation.

---

## Docker Image Build & Run Instructions

The final Docker image is configured to be **Evaluation-Only** and runs automatically on startup. Because the evaluation simply fetches the publicly hosted model from the Hugging Face Hub, no private access tokens are required at runtime.

**1. Build the Docker Image:**
```bash
docker build -t ml-ops-assignment-3:v1 .
```  

**2. Run Docker container:**
```bash
docker run ml-ops-assignment-3:v1