# data_preparation.py
# Prepares train/test splits for the modality-specific training datasets.
# Modalities: Text (RoBERTa), Visual (CLIP)
# Located in: /Data Handling/

import os
import time
import platform
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
BASE_DIR = (
    "/Users/davidluu/Library/Mobile Documents/com~apple~CloudDocs/ACADEMIA/me/"
    "Publications/Securitizing the Global South in a Bipolar World Order "
    "A Multimodal Analysis of US and Chinese News Media/Data Handling"
)
# Source files (2000 rows) are in BASE_DIR
DATA_DIR     = BASE_DIR
# Output train/test files must go into /data for compatibility with training scripts
RESULTS_DIR  = os.path.join("/Users/davidluu/Library/Mobile Documents/com~apple~CloudDocs/ACADEMIA/me/Publications/Securitizing the Global South in a Bipolar World Order A Multimodal Analysis of US and Chinese News Media/Data Analysis", "results")
README_MD    = os.path.join(BASE_DIR, "READme.md")

os.makedirs(DATA_DIR,        exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Label configuration
# Valid labels across all annotation columns
VALID_LABELS   = {"high", "moderate", "low", "not applicable"}
LABEL2ID       = {"high": 0, "moderate": 1, "low": 2, "not applicable": 3}
ID2LABEL       = {v: k for k, v in LABEL2ID.items()}

# Numeric scores used for aggregate securitization scoring (manuscript p. 7)
LABEL2SCORE    = {"high": 3, "moderate": 2, "low": 1, "not applicable": 9}

# ADDED: Map the numeric outputs from ChatGPT (1, 2, 3, 9) back to the text labels
NUMERIC_TO_TEXT = {
    "3": "high", "3.0": "high",
    "2": "moderate", "2.0": "moderate",
    "1": "low", "1.0": "low",
    "9": "not applicable", "9.0": "not applicable"
}

# Modality specifications
MODALITIES = [
    {
        "name":        "text",
        "src_file":    "securitization_news_articles_text_training.csv",
        "label_col":   "Securitization_Text",
        "input_cols":  ["Text"],
    },
    {
        "name":        "visual",
        "src_file":    "securitization_news_articles_visual_training.csv",
        "label_col":   "Securitization_Visual",
        "input_cols":  ["Image"],
    },
]

REQUIRED_BASE_COLS = ["X", "Date", "URL", "Text", "Image"]

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
        import sklearn, numpy, pandas as _pd
    except Exception:
        sklearn = numpy = _pd = None
    finished_at = time.time()
    lines = [
        f"## [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {action}",
        f"- Duration: {hms(finished_at - started_at)}",
        f"- Host: {platform.node()} — Python {platform.python_version()} ({platform.platform()})",
        "- Package versions:",
        f"  - scikit-learn: {get_ver(sklearn)}",
        f"  - pandas: {get_ver(_pd)}",
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
    """Try common encodings to avoid UnicodeDecodeError."""
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="utf-8", errors="replace")

def validate_and_clean(df: pd.DataFrame, label_col: str, modality_name: str, required_inputs: list) -> pd.DataFrame:
    """Validate columns, normalise label strings, drop invalid rows."""
    # Check base columns
    for col in REQUIRED_BASE_COLS:
        if col not in df.columns:
            raise ValueError(f"[{modality_name}] Missing required column: '{col}'")
    if label_col not in df.columns:
        raise ValueError(f"[{modality_name}] Missing label column: '{label_col}'")

    # Normalise labels
    df[label_col] = df[label_col].astype(str).str.strip().str.lower()
    
    # Translate numeric labels from CSVs into standard text labels
    df[label_col] = df[label_col].replace(NUMERIC_TO_TEXT)

    before = len(df)
    df = df[df[label_col].isin(VALID_LABELS)].copy()
    after  = len(df)
    if before != after:
        print(f"  [{modality_name}] Dropped {before - after} rows with invalid/missing labels.")

    # Add integer label id for training
    df["label_id"] = df[label_col].map(LABEL2ID)

    # Only enforce inputs that are required for this modality.
    if "Text" in required_inputs:
        if "Text" not in df.columns:
            raise ValueError(f"[{modality_name}] 'Text' required but not present in dataframe.")
        before = len(df)
        df = df[df["Text"].notna() & (df["Text"].str.strip() != "")]
        after = len(df)
        if before != after:
            print(f"  [{modality_name}] Dropped {before - after} rows with missing/empty Text.")

    if "Image" in required_inputs:
        if "Image" not in df.columns:
            raise ValueError(f"[{modality_name}] 'Image' required but not present in dataframe.")
        df["image_missing"] = df["Image"].fillna("").astype(str).str.strip() == ""
    else:
        if "Image" in df.columns:
            df["image_missing"] = df["Image"].fillna("").astype(str).str.strip() == ""

    df = df.reset_index(drop=True)
    return df

# Main
if __name__ == "__main__":
    t0 = time.time()
    summary_lines = []

    import json
    import numpy as np

    for spec in MODALITIES:
        name      = spec["name"]
        src_file  = spec["src_file"]
        label_col = spec["label_col"]
        required_inputs = spec.get("input_cols", [])
        src_path  = os.path.join(DATA_DIR, src_file)

        print(f"\n{'='*60}")
        print(f"  Processing modality: {name.upper()}")
        print(f"  Source: {src_file}")
        print(f"{'='*60}")

        if not os.path.exists(src_path):
            print(f"  [WARNING] File not found, skipping: {src_path}")
            continue

        df = read_csv_robust(src_path)
        print(f"  Loaded {len(df)} rows, {len(df.columns)} columns.")

        df = validate_and_clean(df, label_col, name, required_inputs)
        print(f"  After cleaning: {len(df)} rows.")

        # Label distribution
        print("  Label distribution:")
        for lbl, cnt in df[label_col].value_counts().items():
            pct = cnt / len(df) * 100 if len(df) > 0 else 0.0
            print(f"    {lbl:20s}: {cnt:5d} ({pct:.1f}%)")

        # Compute class weights
        try:
            from sklearn.utils.class_weight import compute_class_weight
            classes_all = np.array([0, 1, 2, 3])
            y = df["label_id"].values
            try:
                weights = compute_class_weight("balanced", classes=classes_all, y=y)
            except ValueError:
                present = np.unique(y)
                weights_present = compute_class_weight("balanced", classes=present, y=y)
                weights = np.zeros(len(classes_all), dtype=float)
                for cls, w in zip(present, weights_present):
                    weights[int(cls)] = float(w)
            weights_path = os.path.join(RESULTS_DIR, f"class_weights_{name}.json")
            with open(weights_path, "w", encoding="utf-8") as f:
                json.dump({str(i): float(w) for i, w in enumerate(weights)}, f, indent=2)
            print(f"  Class weights saved → {weights_path}")
        except Exception as e:
            print(f"  [WARNING] Could not compute class weights for modality '{name}': {e}")

        # 80 / 20 stratified split
        if len(df) == 0:
            print(f"  [WARNING] No rows available after cleaning for modality '{name}'; skipping split/save.")
            summary_lines.append(f"{name}: total=0, train=0, test=0")
            continue

        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df["label_id"],
        )
        train_df = train_df.reset_index(drop=True)
        test_df  = test_df.reset_index(drop=True)

        # Save to the /data subfolder so training scripts find them
        train_out = os.path.join(RESULTS_DIR, f"train_{name}.csv")
        test_out  = os.path.join(RESULTS_DIR, f"test_{name}.csv")
        train_df.to_csv(train_out, index=False)
        test_df.to_csv(test_out,   index=False)

        print(f"  Train: {len(train_df)} rows → {train_out}")
        print(f"  Test:  {len(test_df)} rows  → {test_out}")

        summary_lines.append(
            f"{name}: total={len(df)}, train={len(train_df)}, test={len(test_df)}"
        )

    # Also copy the label maps to RESULTS_DIR for downstream scripts
    maps_path = os.path.join(RESULTS_DIR, "label_maps.json")
    with open(maps_path, "w", encoding="utf-8") as f:
        json.dump({
            "label2id": LABEL2ID,
            "id2label": {str(k): v for k, v in ID2LABEL.items()},
            "label2score": LABEL2SCORE
        }, f, indent=2)
    print(f"\nLabel maps saved → {maps_path}")

    print("\nData preparation complete.")
    log_to_readme(
        action="Data preparation (train/test splits)",
        started_at=t0,
        notes="; ".join(summary_lines),
    )
