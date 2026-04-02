# train_roberta.py
# Fine-tunes RoBERTa-base for single-label securitization classification
# on the "Text" column → predicts "Securitization_Text"
# Labels: high | moderate | low | not applicable
# Located in: /Data Analysis/

import os
import json
import time
import platform
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset, Features, Value, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# Paths aligned across all scripts
# Use the common root for all data analysis results
PROJECT_ROOT = "/Users/davidluu/Library/Mobile Documents/com~apple~CloudDocs/ACADEMIA/me/Publications/Securitizing the Global South in a Bipolar World Order A Multimodal Analysis of US and Chinese News Media/Data Analysis"
DATA_DIR    = os.path.join(PROJECT_ROOT, "results")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "roberta")
README_MD   = os.path.join(PROJECT_ROOT, "READme.md")

os.makedirs(os.path.join(RESULTS_DIR, "trained_model"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "logs"),          exist_ok=True)

# Label configuration
LABELS   = ["high", "moderate", "low", "not applicable"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
NUM_LABELS = len(LABELS)

# Model
MODEL_NAME = "roberta-base"
MAX_LENGTH = 512

# Helpers
def get_ver(mod, attr="__version__"):
    try:
        return getattr(mod, attr)
    except Exception:
        return "n/a"

def hms(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def log_to_readme(action: str, started_at: float, notes: str = ""):
    try:
        import transformers, datasets, sklearn, numpy, pandas as _pd
    except Exception:
        transformers = datasets = sklearn = numpy = _pd = None
    finished_at = time.time()
    lines = [
        f"## [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {action}",
        f"- Duration: {hms(finished_at - started_at)}",
        f"- Host: {platform.node()} — Python {platform.python_version()} ({platform.platform()})",
        "- Package versions:",
        f"  - torch: {get_ver(torch)}",
        f"  - transformers: {get_ver(transformers)}",
        f"  - datasets: {get_ver(datasets)}",
        f"  - scikit-learn: {get_ver(sklearn)}",
        f"  - pandas: {get_ver(_pd)}",
        f"  - numpy: {get_ver(numpy)}",
    ]
    if notes:
        lines.append(f"- Notes: {notes}")
    lines.append("\n---\n")
    try:
        with open(README_MD, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except Exception:
        pass

def read_csv_robust(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="utf-8", errors="replace")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

def tokenize_batch(batch):
    return tokenizer(
        batch["Text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_micro = f1_score(labels, preds, average="micro", zero_division=0)
    return {
        "f1_weighted": f1_weighted,
        "f1_macro":    f1_macro,
        "f1_micro":    f1_micro,
        "precision_weighted": precision,
        "recall_weighted":    recall,
    }

# Main
if __name__ == "__main__":
    t0 = time.time()

    # Load data
    train_path = os.path.join(DATA_DIR, "train_text.csv")
    test_path  = os.path.join(DATA_DIR, "test_text.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Train/test CSVs not found in {DATA_DIR}. Run data_preparation.py first."
        )

    train_df = read_csv_robust(train_path)
    test_df  = read_csv_robust(test_path)

    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")

    # Ensure label_id column present
    if "label_id" not in train_df.columns:
        train_df["label_id"] = train_df["Securitization_Text"].str.strip().str.lower().map(LABEL2ID)
        test_df["label_id"]  = test_df["Securitization_Text"].str.strip().str.lower().map(LABEL2ID)

    # Drop any rows where label mapping failed
    train_df = train_df.dropna(subset=["label_id"]).reset_index(drop=True)
    test_df  = test_df.dropna(subset=["label_id"]).reset_index(drop=True)
    train_df["label_id"] = train_df["label_id"].astype(int)
    test_df["label_id"]  = test_df["label_id"].astype(int)

    # Build datasets
    def to_hf_dataset(df: pd.DataFrame) -> Dataset:
        return Dataset.from_dict({
            "Text":    df["Text"].astype(str).tolist(),
            "labels":  df["label_id"].tolist(),
        })

    train_ds = to_hf_dataset(train_df)
    test_ds  = to_hf_dataset(test_df)

    train_ds = train_ds.map(tokenize_batch, batched=True, remove_columns=["Text"])
    test_ds  = test_ds.map(tokenize_batch,  batched=True, remove_columns=["Text"])

    # Model and device
    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()         else
        "cpu"
    )
    print(f"Using device: {device}")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    # Training arguments
    # The final model is saved here for finetuned_analysis.py
    save_path = os.path.join(RESULTS_DIR, "trained_model", "roberta_text")
    os.makedirs(save_path, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=os.path.join(RESULTS_DIR, "checkpoints"),
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch",  # FIXED: Renamed from evaluation_strategy to avoid TypeError
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        logging_dir=os.path.join(RESULTS_DIR, "logs"),
        logging_steps=50,
        report_to="none",
        push_to_hub=False,
        fp16=torch.cuda.is_available(),   # only on CUDA
        max_grad_norm=1.0,
        dataloader_pin_memory=(device.type == "cuda"),
    )

    # Class weights
    # Look for an externally computed class_weights JSON.
    weights_path = os.path.join(DATA_DIR, "class_weights_text.json")
    class_weights = None
    if os.path.exists(weights_path):
        try:
            with open(weights_path, "r", encoding="utf-8") as f:
                cw = json.load(f)
            class_weights = torch.tensor([cw[str(i)] for i in range(NUM_LABELS)], dtype=torch.float).to(device)
            print(f"Loaded class weights from {weights_path}")
        except Exception as e:
            print(f"Failed to load class weights: {e}")

    # Weighted trainer
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    # Trainer instantiation
    trainer_class = WeightedTrainer if class_weights is not None else Trainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    print("\nStarting RoBERTa fine-tuning on Text → Securitization_Text …")
    trainer.train()

    # Evaluation on test set
    print("\nEvaluating on test set …")
    pred_output = trainer.predict(test_ds)
    preds  = np.argmax(pred_output.predictions, axis=-1)
    labels = test_df["label_id"].values

    print("\nClassification Report")
    report = classification_report(labels, preds, target_names=LABELS, zero_division=0)
    print(report)

    f1_w = f1_score(labels, preds, average="weighted", zero_division=0)
    f1_m = f1_score(labels, preds, average="macro",    zero_division=0)

    # Save report
    report_path = os.path.join(RESULTS_DIR, "roberta_text_evaluation.txt")
    with open(report_path, "w", encoding="utf-8") as fp:
        fp.write("RoBERTa — Text Securitization Classification\n" + "=" * 60 + "\n" + report)
        fp.write(f"\nWeighted F1: {f1_w:.4f}  |  Macro F1: {f1_m:.4f}\n")

    # Save model
    model.to("cpu").save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    with open(os.path.join(save_path, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(LABEL2ID, f, indent=2)
    with open(os.path.join(save_path, "id2label.json"), "w", encoding="utf-8") as f:
        json.dump(ID2LABEL, f, indent=2)
    print(f"Model saved → {save_path}")

    log_to_readme(
        action="Train RoBERTa (Text → Securitization_Text)",
        started_at=t0,
        notes=f"Weighted F1={f1_w:.4f} | saved={save_path}",
    )
    print("\nRoBERTa training complete.")
