# finetuned_analysis.py
# Applies fine-tuned RoBERTa (Text) and CLIP (Visual) models to clustered CSVs.
# Optimized for high-success polite local image downloading and memory efficiency.

import os
import io
import time
import glob
import warnings
import re
import random
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

# ── Silence noisy warnings ─────────────────────────────────────────────────────
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
PROJECT_ROOT = (
    "/Users/davidluu/Library/Mobile Documents/com~apple~CloudDocs/ACADEMIA/me/"
    "Publications/Securitizing the Global South in a Bipolar World Order "
    "A Multimodal Analysis of US and Chinese News Media"
)

DATA_DIR    = os.path.join(PROJECT_ROOT, "Data Handling")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_final_analysis")
os.makedirs(RESULTS_DIR, exist_ok=True)

# New directory for polite, local image caching
IMAGE_DOWNLOAD_DIR = os.path.join(DATA_DIR, "downloaded_images")
os.makedirs(IMAGE_DOWNLOAD_DIR, exist_ok=True)

ROBERTA_MODEL_DIR = os.path.join(
    PROJECT_ROOT, "Data Analysis", "results", "roberta", "trained_model", "roberta_text"
)
CLIP_MODEL_DIR = os.path.join(
    PROJECT_ROOT, "Data Handling", "results", "clip", "trained_model", "clip_visual"
)
README_MD = os.path.join(PROJECT_ROOT, "READme.md")

TARGET_FILENAMES = {
    "cluster_2_g_zh_hy_clean.csv", "cluster_2_g_us_hy_clean.csv",
    "cluster_2_f_zh_hy_clean.csv", "cluster_2_f_us_hy_clean.csv",
    "cluster_2_e_zh_hy_clean.csv", "cluster_2_e_us_hy_clean.csv",
    "cluster_2_d_zh_hy_clean.csv", "cluster_2_d_us_hy_clean.csv",
    "cluster_2_c_zh_hy_clean.csv", "cluster_2_c_us_hy_clean.csv",
    "cluster_2_b_zh_hy_clean.csv", "cluster_2_b_us_hy_clean.csv",
    "cluster_2_a_zh_hy_clean.csv", "cluster_2_a_us_hy_clean.csv",
    "cluster_1_a_gn_hy_clean.csv", "cluster_1_a_gs_hy_clean.csv",
}

LABELS    = ["high", "moderate", "low", "not applicable"]
LABEL2ID  = {l: i for i, l in enumerate(LABELS)}
ID2LABEL  = {i: l for l, i in LABEL2ID.items()}

# Mapping: low=0.0, moderate=0.5, high=1.0. 9=np.nan (Excluded from mean calculation)
LABEL2SCORE   = {"high": 3, "moderate": 2, "low": 1, "not applicable": 9}
SCORE_STD_MAP = {1: 0.0, 2: 0.5, 3: 1.0, 9: np.nan}

_IMAGE_UNAVAILABLE = "not applicable"

BATCH_SIZE    = 16  # Moderate batch size for CLIP stability
IMAGE_TIMEOUT = 20

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
    "Connection": "keep-alive",
}

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def hms(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def log_to_readme(action: str, started_at: float, notes: str = ""):
    finished_at = time.time()
    lines = [f"## [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {action}",
             f"- Duration: {hms(finished_at - started_at)}", f"- Notes: {notes}", "\n---\n"]
    try:
        with open(README_MD, "a", encoding="utf-8") as f: f.write("\n".join(lines))
    except: pass

def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(_HEADERS)
    retry = Retry(total=3, backoff_factor=1, status_forcelist={429, 500, 502, 503, 504})
    session.mount("http://", HTTPAdapter(max_retries=retry))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session

def label_to_score(label):
    if pd.isna(label): return np.nan
    val = LABEL2SCORE.get(str(label).strip().lower())
    return SCORE_STD_MAP.get(val, np.nan)

# ══════════════════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════════════════

def load_roberta(model_dir: str, device: torch.device):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    return AutoTokenizer.from_pretrained(model_dir), AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()

def load_clip_classifier(model_dir: str, device: torch.device):
    from transformers import CLIPProcessor, CLIPModel
    class CLIPClassifier(nn.Module):
        def __init__(self, clip_model, num_labels):
            super().__init__()
            self.clip = clip_model
            self.classifier = nn.Sequential(
                nn.Linear(clip_model.config.projection_dim, 256), 
                nn.ReLU(), 
                nn.Dropout(0.1), 
                nn.Linear(256, num_labels)
            )
        def forward(self, pixel_values):
            v_out = self.clip.vision_model(pixel_values=pixel_values)
            projected = self.clip.visual_projection(v_out.pooler_output)
            return self.classifier(projected)

    proc = CLIPProcessor.from_pretrained(model_dir)
    base = CLIPModel.from_pretrained(model_dir)
    clf = CLIPClassifier(base, len(LABELS))
    clf.classifier.load_state_dict(torch.load(os.path.join(model_dir, "classifier_head.pt"), map_location="cpu"))
    return proc, clf.to(device).eval()

# ══════════════════════════════════════════════════════════════════════════════
# LOCAL CACHING & INFERENCE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def pre_download_images_politely(df, fname, session):
    """
    Downloads images sequentially with polite delays. Saves them as compressed JPEGs.
    Returns a list of local file paths corresponding to the DataFrame rows.
    """
    local_paths = []
    has_x_col = "X" in df.columns
    safe_fname = fname.replace(".csv", "")
    
    urls = df["Image"].fillna("").tolist()
    
    for idx, url in tqdm(enumerate(urls), total=len(urls), desc=f"  Downloading [{fname[:15]}]", leave=False):
        row_id = str(df.iloc[idx]["X"]) if has_x_col else str(idx)
        save_path = os.path.join(IMAGE_DOWNLOAD_DIR, f"{safe_fname}_row_{row_id}.jpg")
        local_paths.append(save_path)
        
        # Skip if already downloaded from a previous run
        if os.path.exists(save_path):
            continue
            
        if not isinstance(url, str) or not url.startswith("http"):
            continue
            
        try:
            # Polite break: random sleep between 1.0 and 2.5 seconds
            time.sleep(random.uniform(1.0, 2.5))
            
            r = session.get(url, timeout=IMAGE_TIMEOUT, stream=True)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            
            # Create compromised version: resize down (CLIP only needs 224x224) to save disk space
            img.thumbnail((300, 300))
            img.save(save_path, "JPEG", quality=75)
            
        except Exception:
            pass # Fails quietly; Inference will handle missing files as "not applicable"
            
    return local_paths

def process_visual_from_local(df, processor, classifier, device, fname):
    """
    Reads images exclusively from the downloaded local files.
    """
    results = []
    paths = df["Local_Image_Path"].tolist()
    
    for i in tqdm(range(0, len(paths), BATCH_SIZE), desc=f"  Visual Infer [{fname[:15]}]", leave=False):
        batch_paths = paths[i : i + BATCH_SIZE]
        batch_images = []
        valid_indices = []
        
        for idx, path in enumerate(batch_paths):
            if os.path.exists(path):
                try:
                    img = Image.open(path).convert("RGB")
                    batch_images.append(img)
                    valid_indices.append(idx)
                except Exception:
                    pass
        
        batch_preds = [_IMAGE_UNAVAILABLE] * len(batch_paths)
        
        if valid_indices:
            try:
                inputs = processor(images=batch_images, return_tensors="pt").to(device)
                with torch.no_grad():
                    logits = classifier(inputs["pixel_values"])
                preds = logits.argmax(-1).cpu().numpy()
                for idx_in_valid, pred_id in enumerate(preds):
                    batch_preds[valid_indices[idx_in_valid]] = ID2LABEL[pred_id]
            except Exception:
                # Fallback: Process one by one if batch fails
                for v_idx in valid_indices:
                    try:
                        single_input = processor(images=[batch_images[valid_indices.index(v_idx)]], return_tensors="pt").to(device)
                        with torch.no_grad():
                            single_logit = classifier(single_input["pixel_values"])
                        batch_preds[v_idx] = ID2LABEL[int(single_logit.argmax(-1).cpu().numpy()[0])]
                    except Exception:
                        pass
        
        results.extend(batch_preds)
        
    return results

def process_text_streaming(df, tokenizer, model, device, fname):
    results = []
    texts = df["Text"].fillna("").tolist()
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"  Text Infer   [{fname[:15]}]", leave=False):
        batch = texts[i : i + BATCH_SIZE]
        enc = tokenizer(batch, truncation=True, padding=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        results.extend([ID2LABEL[int(p)] for p in logits.argmax(-1).cpu().numpy()])
    return results

# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    start_time = time.time()
    files = sorted([p for p in glob.glob(os.path.join(DATA_DIR, "*_clean.csv")) if os.path.basename(p) in TARGET_FILENAMES])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    rt_tok, rt_mod = load_roberta(ROBERTA_MODEL_DIR, device)
    cp_proc, cp_clf = load_clip_classifier(CLIP_MODEL_DIR, device)
    session = make_session()

    stats = []
    for path in tqdm(files, desc="Overall Progress"):
        fn = os.path.basename(path)
        df = pd.read_csv(path)
        
        # 1. Download images politely to local disk first
        df["Local_Image_Path"] = pre_download_images_politely(df, fn, session)
        
        # 2. Text Analysis
        df["Securitization_Text"] = process_text_streaming(df, rt_tok, rt_mod, device, fn)
        
        # 3. Visual Analysis (using the local files we just downloaded)
        df["Securitization_Visual"] = process_visual_from_local(df, cp_proc, cp_clf, device, fn)
        
        # 4. Apply Scoring (np.nan handles "not applicable" correctly in .mean())
        df["Score_Text"]   = df["Securitization_Text"].apply(label_to_score)
        df["Score_Visual"] = df["Securitization_Visual"].apply(label_to_score)
        
        out_name = re.sub(r"_clean\.csv$", "_labeled.csv", fn)
        df.to_csv(os.path.join(RESULTS_DIR, out_name), index=False)

        # 5. Group statistics
        grp = "Other"
        if "_us_" in fn: grp = "USA"
        elif "_zh_" in fn: grp = "China"
        elif "_gs_" in fn: grp = "Global South"
        elif "_gn_" in fn: grp = "Global North"

        stats.append({
            "file": fn, "group": grp,
            "txt_avg": df["Score_Text"].mean(), 
            "vis_avg": df["Score_Visual"].mean()
        })

    # Final Report
    sum_df = pd.DataFrame(stats)
    print("\n" + "="*40)
    print("FINAL RESEARCH SUMMARY")
    print("="*40)
    final_grp = sum_df.groupby("group")[["txt_avg", "vis_avg"]].mean()
    print(final_grp)
    
    log_to_readme("Multimodal analysis complete (Polite Local Caching)", start_time, f"Processed {len(files)} files.")
    print(f"\nTotal Processing Time: {hms(time.time() - start_time)}")