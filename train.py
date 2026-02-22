import os
import numpy as np
from functools import partial
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from huggingface_hub import login

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

class MetricsCallback(TrainerCallback):
    def __init__(self):
        self.training_history = {"train": [], "eval": []}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if "loss" in logs:  
                self.training_history["train"].append(logs)
            elif "eval_loss" in logs:  
                self.training_history["eval"].append(logs)

# Main Execution 
def main():
    # 1. Authenticate to Hugging Face
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not input in environment yet.")
    
    login(hf_token)

    task = "mrpc"
    checkpoint = "answerdotai/ModernBERT-base"
    hf_repo_id = "Shivam-K/ML-Ops-Assignment-3" 

    print("Loading and preparing data...")
    raw_datasets = load_dataset("glue", task)
    train_ds_name, valid_ds_name = "train", "validation"
    task_inputs = ["sentence1", "sentence2"]
    
    id2label, label2id = get_label_maps(raw_datasets, train_ds_name)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
    tokenized_datasets = raw_datasets.map(
        partial(preprocess_function, tokenizer=tokenizer, task_inputs=task_inputs), 
        batched=True
    )
    
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=2, id2label=id2label, label2id=label2id
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=f"ModernBERT_{task}_ft",
        learning_rate=8e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        bf16=True, 
        push_to_hub=True, 
        hub_model_id=hf_repo_id, 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets[train_ds_name],
        eval_dataset=tokenized_datasets[valid_ds_name],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, task_metrics=[accuracy_score, f1_score]),
    )

    trainer.add_callback(MetricsCallback())

    print("Starting training...")
    trainer.train()
    
    print(f"Pushing model and tokenizer to Hub: {hf_repo_id}...")
    trainer.push_to_hub()
    print("Training and push complete.")

if __name__ == "__main__":
    main()