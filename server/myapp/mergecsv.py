# yourapp/mergecsv.py
from pathlib import Path
from django.conf import settings
from django.utils.text import slugify
import pandas as pd
import time

# Base dirs
MEDIA_ROOT = Path(settings.MEDIA_ROOT).resolve()
UPLOAD_DIR = (MEDIA_ROOT / "uploads")
MERGE_DIR  = (MEDIA_ROOT / "mergefiles")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MERGE_DIR.mkdir(parents=True, exist_ok=True)

PRED_DERIVED_DROP = {
    "Confidence", "Match",
    "Prob_CANDIDATE", "Prob_CONFIRMED",
    "Prob_FALSE POSITIVE", "Prob_FALSE_POSITIVE",
}

PREFERRED_PREFIX = ["kepid", "kepoi_name", "kepler_name"]

def _resolve_csv(filename: str) -> Path:
    """
    Resolve a CSV path safely within MEDIA_ROOT.
    Tries uploads/, then mergefiles/, then treats it as a relative path under MEDIA_ROOT.
    """
    filename = str(filename).strip()
    cand = [
        (UPLOAD_DIR / filename),
        (MERGE_DIR  / filename),
        (MEDIA_ROOT / filename),
    ]
    for p in cand:
        p = p.resolve()
        if p.exists() and p.is_file() and p.suffix.lower() == ".csv":
            # Ensure no path escape
            if str(p).startswith(str(MEDIA_ROOT)):
                return p
    raise FileNotFoundError(f"File not found or not a .csv under media: {filename}")

def _clean_columns(cols): return [str(c).strip() for c in cols]

def _preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = _clean_columns(df.columns)

    drops = [c for c in df.columns if c in PRED_DERIVED_DROP]
    if drops:
        df = df.drop(columns=drops, errors="ignore")

    if "Predicted_Class" in df.columns:
        if "koi_disposition" in df.columns:
            df["koi_disposition"] = df["koi_disposition"].fillna(df["Predicted_Class"])
            df = df.drop(columns=["Predicted_Class"])
        else:
            df = df.rename(columns={"Predicted_Class": "koi_disposition"})

    if "koi_disposition" not in df.columns and "koi_pdisposition" in df.columns:
        df["koi_disposition"] = df["koi_pdisposition"]

    return df

def _canonical_column_order(all_cols: list[str]) -> list[str]:
    cols_set = set(all_cols)
    ordered = []

    # 1) Preferred prefix
    for c in PREFERRED_PREFIX:
        if c in cols_set:
            ordered.append(c)

    remaining = [c for c in all_cols if c not in ordered]

    # 2) koi_disposition & koi_pdisposition placement
    kd_present  = "koi_disposition"  in remaining
    kpd_present = "koi_pdisposition" in remaining
    if kd_present:  remaining.remove("koi_disposition")
    if kpd_present: remaining.remove("koi_pdisposition")

    if kd_present:
        if "kepler_name" in ordered:
            ordered.insert(ordered.index("kepler_name") + 1, "koi_disposition")
        else:
            ordered.append("koi_disposition")

    if kpd_present:
        if "koi_disposition" in ordered:
            ordered.insert(ordered.index("koi_disposition") + 1, "koi_pdisposition")
        else:
            ordered.append("koi_pdisposition")

    ordered.extend(sorted([c for c in remaining if c not in ordered]))

    # Dedup keep order
    seen, final = set(), []
    for c in ordered:
        if c not in seen:
            final.append(c); seen.add(c)
    for c in all_cols:
        if c not in seen:
            final.append(c); seen.add(c)
    return final

def merge_csvs(
    file_a: str,
    file_b: str,
    *,
    dedupe: bool = True,
    output_name: str | None = None,
    output_dir: Path | None = None,          # <-- NEW
) -> tuple[Path, int]:
    """
    Merge two CSVs with preprocessing:
      - Drop prediction-derived cols
      - Rename Predicted_Class -> koi_disposition (if present)
      - Align schemas (union of columns) with canonical ordering
    Saves to `output_dir` (defaults to MERGE_DIR).
    Returns (out_path, total_rows).
    """
    p_a = _resolve_csv(file_a)
    p_b = _resolve_csv(file_b)

    try:
        df_a = pd.read_csv(p_a, comment="#")
        df_b = pd.read_csv(p_b, comment="#")
    except Exception as e:
        raise ValueError(f"Failed to read CSVs: {e}")

    df_a = _preprocess_df(df_a)
    df_b = _preprocess_df(df_b)

    all_cols = sorted(set(df_a.columns) | set(df_b.columns))
    canon = _canonical_column_order(all_cols)
    df_a = df_a.reindex(columns=canon)
    df_b = df_b.reindex(columns=canon)

    try:
        merged = pd.concat([df_a, df_b], ignore_index=True)
        if dedupe:
            merged = merged.drop_duplicates().reset_index(drop=True)
    except Exception as e:
        raise ValueError(f"Failed to merge: {e}")

    if not output_name:
        ts = time.strftime("%Y%m%d-%H%M%S")
        stem_a = slugify(p_a.stem)
        stem_b = slugify(p_b.stem)
        output_name = f"merged__{stem_a}__{stem_b}__{ts}.csv"

    save_dir = (output_dir or MERGE_DIR)     # <-- default to mergefiles
    save_dir.mkdir(parents=True, exist_ok=True)

    out_path = (save_dir / output_name)
    try:
        merged.to_csv(out_path, index=False)
    except Exception as e:
        raise IOError(f"Failed to save merged CSV: {e}")

    return out_path, int(len(merged))
