# train_clip.py
# Fine-tunes CLIP (openai/clip-vit-base-patch32) with a linear classification
# head for single-label securitization classification.
# Input:  "Image" column (URL string) → downloads and encodes image (cached to disk)
# Output: predicts "Securitization_Visual"
# Labels: high | moderate | low | not applicable
# Located in: /Data Handling/

import os
import io
import json
import time
import platform
import warnings
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight
from torchvision import transforms

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = (
    "/Users/davidluu/Library/Mobile Documents/com~apple~CloudDocs/ACADEMIA/me/"
    "Publications/Securitizing the Global South in a Bipolar World Order "
    "A Multimodal Analysis of US and Chinese News Media/Data Handling"
)
DATA_DIR = "/Users/davidluu/Library/Mobile Documents/com~apple~CloudDocs/ACADEMIA/me/Publications/Securitizing the Global South in a Bipolar World Order A Multimodal Analysis of US and Chinese News Media/Data Analysis/results"
RESULTS_DIR = os.path.join(BASE_DIR, "results", "clip")
README_MD   = os.path.join(BASE_DIR, "READme.md")
IMAGE_CACHE_DIR = os.path.join(DATA_DIR, "image_cache")

os.makedirs(os.path.join(RESULTS_DIR, "trained_model"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "logs"),          exist_ok=True)
os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)

# ── Label config ──────────────────────────────────────────────────────────────
LABELS    = ["high", "moderate", "low", "not applicable"]
LABEL2ID  = {l: i for i, l in enumerate(LABELS)}
ID2LABEL  = {i: l for l, i in LABEL2ID.items()}
NUM_LABELS = len(LABELS)

# ── Model ─────────────────────────────────────────────────────────────────────
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
IMAGE_TIMEOUT   = 10      # seconds per image download
BATCH_SIZE      = 16
NUM_EPOCHS      = 5
LEARNING_RATE   = 2e-5
CLIP_LR         = 2e-5
IMAGE_PLACEHOLDER = "placeholder"  # used when image cannot be fetched
EARLY_STOPPING_PATIENCE = 3  # set to 0 or None to disable early stopping

# ── Helpers ───────────────────────────────────────────────────────────────────
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
        import transformers, sklearn, numpy, pandas as _pd
    except Exception:
        transformers = sklearn = numpy = _pd = None
    finished_at = time.time()
    lines = [
        f"## [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {action}",
        f"- Duration: {hms(finished_at - started_at)}",
        f"- Host: {platform.node()} — Python {platform.python_version()} ({platform.platform()})",
        "- Package versions:",
        f"  - torch: {get_ver(torch)}",
        f"  - transformers: {get_ver(transformers)}",
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

def _url_to_filename(url: str) -> str:
    """Deterministic filename for cache based on SHA1 of url."""
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return f"{h}.jpg"

def fetch_image_to_cache(url: str, cache_dir: str, timeout: int = IMAGE_TIMEOUT) -> Image.Image | None:
    """
    Fetch an image from a URL, save to disk cache, and return a PIL Image.
    If the image is already cached on disk, load from disk instead of re-downloading.
    """
    if not isinstance(url, str) or not url.strip().lower().startswith("http"):
        return None

    fname = _url_to_filename(url)
    fpath = os.path.join(cache_dir, fname)

    # If cached file exists, try to open it
    if os.path.exists(fpath):
        try:
            with open(fpath, "rb") as f:
                img = Image.open(io.BytesIO(f.read())).convert("RGB")
            return img
        except Exception:
            # cached file corrupted: remove and re-download
            try:
                os.remove(fpath)
            except Exception:
                pass

    # Download and write to cache
    try:
        resp = requests.get(url.strip(), timeout=timeout, stream=True)
        resp.raise_for_status()
        content = resp.content
        # Try to write the content to file (atomic-ish)
        tmp_path = fpath + ".tmp"
        with open(tmp_path, "wb") as tmpf:
            tmpf.write(content)
        os.replace(tmp_path, fpath)
        img = Image.open(io.BytesIO(content)).convert("RGB")
        return img
    except Exception:
        # on failure create a placeholder image file so we don't repeatedly retry remote URL
        try:
            placeholder_path = os.path.join(cache_dir, IMAGE_PLACEHOLDER + ".jpg")
            if not os.path.exists(placeholder_path):
                Image.new("RGB", (224, 224), color=(255, 255, 255)).save(placeholder_path, format="JPEG")
            with open(placeholder_path, "rb") as pf:
                return Image.open(io.BytesIO(pf.read())).convert("RGB")
        except Exception:
            return None

# ── CLIP classifier model ─────────────────────────────────────────────────────
class CLIPClassifier(nn.Module):
    """CLIP visual encoder + linear classification head."""
    def __init__(self, clip_model: CLIPModel, num_labels: int):
        super().__init__()
        self.clip       = clip_model
        # clip_model.config.projection_dim is typically the projection dimension
        hidden_dim      = getattr(clip_model.config, "projection_dim", None)
        if hidden_dim is None:
            # fallback: try to find visual_projection weight size
            try:
                hidden_dim = clip_model.visual_projection.weight.shape[0]
            except Exception:
                hidden_dim = 512
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_labels),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Extract visual features from CLIP's vision encoder
        vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
        # Some CLIP variants may expose pooler_output, otherwise use last_hidden_state mean
        pooled = getattr(vision_outputs, "pooler_output", None)
        if pooled is None:
            # last_hidden_state: (B, seq_len, hidden)
            last = getattr(vision_outputs, "last_hidden_state", None)
            if last is not None:
                pooled = last.mean(dim=1)
            else:
                raise RuntimeError("Unexpected CLIP vision output structure.")
        projected = self.clip.visual_projection(pooled)  # (B, proj_dim)
        logits    = self.classifier(projected)
        return logits

# ── PyTorch dataset ───────────────────────────────────────────────────────────
class ImageDataset(TorchDataset):
    def __init__(self, image_list: list, label_list: list, processor: CLIPProcessor, transform=None):
        self.images    = image_list    # list of PIL.Image or None
        self.labels    = label_list    # list of int
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        if img is None:
            # Fallback: blank white image
            img = Image.new("RGB", (224, 224), color=(255, 255, 255))
        
        # Apply data augmentation if provided
        if self.transform:
            img = self.transform(img)

        # processor returns batched tensors; we squeeze the batch dim
        inputs = self.processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # (3, H, W)
        return pixel_values, torch.tensor(self.labels[idx], dtype=torch.long)

# ── Image prefetch (disk-cached) ───────────────────────────────────────────────
def prefetch_images(urls: list, cache_dir: str = IMAGE_CACHE_DIR) -> list:
    """
    Prefetch images and cache to disk. Returns list of PIL.Image objects (or placeholders).
    The cache avoids re-fetching across multiple runs and epochs.
    """
    images = []
    total  = len(urls)
    failed = 0
    for i, url in enumerate(urls):
        # progress every 50 or on the last
        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  Fetching images: {i+1}/{total} (failed: {failed})")
        img = None
        try:
            img = fetch_image_to_cache(url, cache_dir)
        except Exception:
            img = None
        if img is None:
            failed += 1
        images.append(img)
    print(f"  Done. {total - failed}/{total} images fetched or placeholder-loaded successfully.")
    return images

# ── Training / evaluation loop ────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for pixels, labels in loader:
        pixels, labels = pixels.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(pixels)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(-1) == labels).sum().item()
        total      += len(labels)
    return total_loss / total if total > 0 else 0.0, correct / total if total > 0 else 0.0

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for pixels, labels in loader:
        pixels = pixels.to(device)
        logits = model(pixels)
        preds  = logits.argmax(-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()

    # ── Load data ─────────────────────────────────────────────────────────────
    train_path = os.path.join(DATA_DIR, "train_visual.csv")
    test_path  = os.path.join(DATA_DIR, "test_visual.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Train/test CSVs not found. Run data_preparation.py first.")

    train_df = read_csv_robust(train_path)
    test_df  = read_csv_robust(test_path)
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")

    if "label_id" not in train_df.columns:
        train_df["label_id"] = train_df["Securitization_Visual"].str.strip().str.lower().map(LABEL2ID)
        test_df["label_id"]  = test_df["Securitization_Visual"].str.strip().str.lower().map(LABEL2ID)

    train_df = train_df.dropna(subset=["label_id"]).reset_index(drop=True)
    test_df  = test_df.dropna(subset=["label_id"]).reset_index(drop=True)
    train_df["label_id"] = train_df["label_id"].astype(int)
    test_df["label_id"]  = test_df["label_id"].astype(int)

    # ── Download & cache images ────────────────────────────────────────────────
    print("\nPrefetching training images (cached to disk) …")
    train_images = prefetch_images(train_df["Image"].tolist(), cache_dir=IMAGE_CACHE_DIR)
    print("Prefetching test images (cached to disk) …")
    test_images  = prefetch_images(test_df["Image"].tolist(),  cache_dir=IMAGE_CACHE_DIR)

    # ── Processor & model ─────────────────────────────────────────────────────
    device    = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()         else
        "cpu"
    )
    print(f"\nUsing device: {device}")

    processor  = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    classifier = CLIPClassifier(clip_model, NUM_LABELS).to(device)

    # ── Data Augmentation ─────────────────────────────────────────────────────
    # Standard resizing/normalization handled by CLIPProcessor later, 
    # but geometric and color transforms help small datasets generalize.
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
    ])

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_dataset = ImageDataset(train_images, train_df["label_id"].tolist(), processor, transform=train_transform)
    test_dataset  = ImageDataset(test_images,  test_df["label_id"].tolist(),  processor, transform=None)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Optimiser: different LR for encoder vs. head ──────────────────────────
    optimizer = torch.optim.AdamW([
        {"params": classifier.clip.parameters(),       "lr": CLIP_LR},
        {"params": classifier.classifier.parameters(), "lr": LEARNING_RATE},
    ], weight_decay=0.01)

    # Cosine annealing LR scheduler over epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, NUM_EPOCHS))

    # ── Address Class Imbalance via Weighted Loss ─────────────────────────────
    # Calculate weights based on actual training data distribution
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_df["label_id"]),
        y=train_df["label_id"]
    )
    weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"\nCalculated class weights: {weight_tensor.tolist()} (Order: {LABELS})")
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    # ── Training loop with per-epoch validation & checkpointing ──────────────
    print(f"\nStarting CLIP fine-tuning ({NUM_EPOCHS} epochs) …")
    best_val_f1 = -1.0
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        loss, acc = train_epoch(classifier, train_loader, optimizer, criterion, device)
        val_preds, val_labels = evaluate(classifier, test_loader, device)
        val_f1 = f1_score(val_labels, val_preds, average="weighted", zero_division=0)

        print(f"  Epoch {epoch}/{NUM_EPOCHS} — loss: {loss:.4f} | train acc: {acc:.4f} | val F1: {val_f1:.4f}")

        # Checkpoint best model (by weighted val F1)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_no_improve = 0
            # save best
            save_path = os.path.join(RESULTS_DIR, "trained_model", "best_clip_visual")
            os.makedirs(save_path, exist_ok=True)
            try:
                classifier.clip.save_pretrained(save_path)
                # Save classification head weights separately
                torch.save(classifier.classifier.state_dict(), os.path.join(save_path, "classifier_head.pt"))
                with open(os.path.join(save_path, "label2id.json"), "w") as f:
                    json.dump(LABEL2ID, f, indent=2)
                with open(os.path.join(save_path, "id2label.json"), "w") as f:
                    json.dump(ID2LABEL, f, indent=2)
                print(f"    New best val F1: {best_val_f1:.4f} — model saved → {save_path}")
            except Exception as e:
                print(f"    Warning: failed to save checkpoint: {e}")
        else:
            epochs_no_improve += 1
            print(f"    (no improvement for {epochs_no_improve} epoch(s))")

        # Scheduler step (per epoch)
        try:
            scheduler.step()
        except Exception:
            pass

        # Early stopping
        if EARLY_STOPPING_PATIENCE and epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered (no improvement for {epochs_no_improve} epochs). Stopping at epoch {epoch}.")
            break

    # ── Final Evaluation on test set using best checkpoint if available ───────
    print("\nEvaluating on test set …")
    # If we saved a best checkpoint, try to load its head weights into current model for final eval
    best_save_path = os.path.join(RESULTS_DIR, "trained_model", "best_clip_visual")
    if os.path.isdir(best_save_path):
        try:
            # reload CLIP backbone (in case it was overwritten) and classifier head
            classifier.clip = CLIPModel.from_pretrained(best_save_path)
            head_path = os.path.join(best_save_path, "classifier_head.pt")
            if os.path.exists(head_path):
                classifier.classifier.load_state_dict(torch.load(head_path, map_location=device))
            classifier.to(device)
            print(f"Loaded best checkpoint from epoch {best_epoch} for final evaluation.")
        except Exception as e:
            print(f"  Warning: failed to load best checkpoint for eval: {e}")

    preds, labels = evaluate(classifier, test_loader, device)

    print("\n── Classification Report ──────────────────────────────────────────")
    report = classification_report(labels, preds, target_names=LABELS, zero_division=0)
    print(report)

    f1_w = f1_score(labels, preds, average="weighted", zero_division=0)
    f1_m = f1_score(labels, preds, average="macro",    zero_division=0)
    print(f"Weighted F1: {f1_w:.4f}  |  Macro F1: {f1_m:.4f}")

    # Save report
    report_path = os.path.join(RESULTS_DIR, "clip_visual_evaluation.txt")
    with open(report_path, "w") as fp:
        fp.write("CLIP — Visual Securitization Classification\n")
        fp.write("=" * 60 + "\n")
        fp.write(report)
        fp.write(f"\nWeighted F1: {f1_w:.4f}  |  Macro F1: {f1_m:.4f}\n")
    print(f"Evaluation report saved → {report_path}")

    # ── Save final model (latest) ────────────────────────────────────────────
    final_save_path = os.path.join(RESULTS_DIR, "trained_model", "clip_visual")
    os.makedirs(final_save_path, exist_ok=True)
    try:
        classifier.clip.save_pretrained(final_save_path)
        processor.save_pretrained(final_save_path)
        torch.save(classifier.classifier.state_dict(), os.path.join(final_save_path, "classifier_head.pt"))
        with open(os.path.join(final_save_path, "label2id.json"), "w") as f:
            json.dump(LABEL2ID, f, indent=2)
        with open(os.path.join(final_save_path, "id2label.json"), "w") as f:
            json.dump(ID2LABEL, f, indent=2)
        print(f"Final model (latest state) saved → {final_save_path}")
    except Exception as e:
        print(f"Warning: failed to save final model: {e}")

    log_to_readme(
        action="Train CLIP (Image URL → Securitization_Visual)",
        started_at=t0,
        notes=f"Weighted F1={f1_w:.4f} | Macro F1={f1_m:.4f} | best_epoch={best_epoch} | saved_best={best_save_path} | Imbalance handled with ClassWeighting | Augmentation applied",
    )
    print("\nCLIP training complete.")