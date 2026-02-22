import os
import torch
import numpy as np
from functools import partial
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Helper Functions 
def get_label_maps(raw_datasets, train_ds_name):
    labels = raw_datasets[train_ds_name].features["label"]
    id2label = {idx: name.upper() for idx, name in enumerate(labels.names)} if hasattr(labels, "names") else None
    label2id = {name.upper(): idx for idx, name in enumerate(labels.names)} if hasattr(labels, "names") else None
    return id2label, label2id

def preprocess_function(examples, tokenizer, task_inputs):
    inps = [examples[inp] for inp in task_inputs]
    return tokenizer(*inps, truncation=True)

def compute_metrics(eval_pred, task_metrics):
    predictions, labels = eval_pred
    metrics_d = {}
    for metric_func in task_metrics:
        score = metric_func(np.argmax(predictions, axis=-1), labels)
        metrics_d[metric_func.__name__] = score[0] if isinstance(score, tuple) else score
    return metrics_d

# Main Execution
def main():
    task = "mrpc"
    hf_repo_id = "Shivam-K/ML-Ops-Assignment-3" 
    
    print(f"Loading data for {task}...")
    raw_datasets = load_dataset("glue", task)
    train_ds_name, valid_ds_name = "train", "validation"
    task_inputs = ["sentence1", "sentence2"]
    
    id2label, label2id = get_label_maps(raw_datasets, train_ds_name)
    
    print(f"Loading tokenizer and model from Hub: {hf_repo_id}...")
    # Pull tokenizer and model 
    tokenizer = AutoTokenizer.from_pretrained(hf_repo_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_repo_id, num_labels=2, id2label=id2label, label2id=label2id
    )
    
    tokenized_datasets = raw_datasets.map(
        partial(preprocess_function, tokenizer=tokenizer, task_inputs=task_inputs), 
        batched=True
    )
    
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    eval_dataset = tokenized_datasets[valid_ds_name]
    all_logits = []
    all_true_labels = []

    print("Running evaluation...")
    for example in eval_dataset:
        ex_1_text = example[task_inputs[0]]
        ex_2_text = example[task_inputs[1]] if len(task_inputs) > 1 else None

        if ex_2_text:
            inf_inputs = tokenizer(ex_1_text, ex_2_text, return_tensors="pt")
        else:
            inf_inputs = tokenizer(ex_1_text, return_tensors="pt")

        inf_inputs = {k: v.to(device) for k, v in inf_inputs.items()}

        with torch.no_grad():
            logits = model(**inf_inputs).logits

        all_logits.append(logits.cpu().numpy())
        all_true_labels.append(example["label"])

    all_logits_array = np.vstack(all_logits)
    all_true_labels_array = np.array(all_true_labels)

    eval_metrics = compute_metrics((all_logits_array, all_true_labels_array), [accuracy_score, f1_score])

    print("Final Evaluation Metrics (Hub Model):")
    print(eval_metrics)

if __name__ == "__main__":
    main()