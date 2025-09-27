# yourapp/merge.py
from pathlib import Path
from django.conf import settings
from django.utils.text import slugify
import pandas as pd
import time

UPLOAD_DIR = (settings.MEDIA_ROOT / "uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Columns to drop if present (from predictions/augmented CSVs)
PRED_DERIVED_DROP = {
    #"koi_disposition",           # drop existing label to rebuild from Predicted_Class
    "Confidence",
    "Match",
    "Prob_CANDIDATE",
    "Prob_CONFIRMED",
    "Prob_FALSE POSITIVE",
}

# Preferred front-order (and a rule to place koi_disposition)
PREFERRED_PREFIX = ["kepid", "kepoi_name", "kepler_name"]
PLACE_AFTER = ("kepler_name", "koi_disposition")       # insert koi_disposition after kepler_name
PLACE_BEFORE = ("koi_pdisposition", "koi_disposition") # ensure koi_disposition before koi_pdisposition

def _resolve_csv(filename: str) -> Path:
    """Resolve a CSV filename inside media/uploads safely."""
    p = (UPLOAD_DIR / filename).resolve()
    if UPLOAD_DIR.resolve() not in p.parents:
        raise PermissionError("Invalid file path")
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"File not found: {filename}")
    if p.suffix.lower() != ".csv":
        raise ValueError(f"Only .csv allowed: {filename}")
    return p

def _clean_columns(cols):
    """Strip whitespace and keep exact names; return list."""
    return [str(c).strip() for c in cols]

def _preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Strip whitespace from column names
    - Drop prediction-derived columns if present
    - Rename Predicted_Class -> koi_disposition (if exists)
    """
    df = df.copy()
    df.columns = _clean_columns(df.columns)

    # Drop any derived columns that may exist
    drops = [c for c in df.columns if c in PRED_DERIVED_DROP]
    if drops:
        df = df.drop(columns=drops, errors="ignore")

    # Rename Predicted_Class to koi_disposition (if present)
    if "Predicted_Class" in df.columns:
        # If koi_disposition exists (rare after drop), fill missing from Predicted_Class then drop Predicted_Class
        if "koi_disposition" in df.columns:
            # Fill NA only
            df["koi_disposition"] = df["koi_disposition"].fillna(df["Predicted_Class"])
            df = df.drop(columns=["Predicted_Class"])
        else:
            df = df.rename(columns={"Predicted_Class": "koi_disposition"})

    return df

def _canonical_column_order(all_cols: list[str]) -> list[str]:
    """
    Build a canonical order:
    1) kepid, kepoi_name, kepler_name (if present)
    2) koi_disposition (placed after kepler_name if both present)
    3) koi_pdisposition (after koi_disposition if both present)
    4) everything else sorted alphabetically
    """
    cols_set = set(all_cols)
    ordered = []

    # 1) Preferred prefix in order if present
    for c in PREFERRED_PREFIX:
        if c in cols_set:
            ordered.append(c)

    # Remaining pool
    remaining = [c for c in all_cols if c not in ordered]

    # 2) Ensure koi_disposition position relative to kepler_name and koi_pdisposition
    # First, pull koi_disposition and koi_pdisposition out of remaining (if present)
    kd_present = "koi_disposition" in remaining
    kpd_present = "koi_pdisposition" in remaining

    if kd_present:
        remaining.remove("koi_disposition")
    if kpd_present:
        remaining.remove("koi_pdisposition")

    # Insert koi_disposition after kepler_name if kepler_name already placed
    if kd_present:
        if "kepler_name" in ordered:
            insert_at = ordered.index("kepler_name") + 1
            ordered.insert(insert_at, "koi_disposition")
        else:
            # if kepler_name not present, just append early
            ordered.append("koi_disposition")

    # Then koi_pdisposition after koi_disposition (if both), else later
    if kpd_present:
        if "koi_disposition" in ordered:
            insert_at = ordered.index("koi_disposition") + 1
            ordered.insert(insert_at, "koi_pdisposition")
        else:
            ordered.append("koi_pdisposition")

    # 4) Everything else sorted
    tail = sorted([c for c in remaining if c not in ordered])
    ordered.extend(tail)

    # Finally ensure we didn't miss any (preserve original if any duplicates)
    seen = set()
    final = []
    for c in ordered:
        if c not in seen:
            final.append(c)
            seen.add(c)
    for c in all_cols:
        if c not in seen:
            final.append(c)
            seen.add(c)

    return final

def merge_csvs(file_a: str,
               file_b: str,
               *,
               dedupe: bool = True,
               output_name: str | None = None) -> tuple[Path, int]:
    """
    Merge two CSVs with preprocessing:
      - Drop prediction-derived cols if present
      - Rename 'Predicted_Class' -> 'koi_disposition' if present
      - Align schemas (union of columns)
      - Enforce canonical column order (koi_disposition placement)

    Returns (out_path, total_rows).
    """
    p_a = _resolve_csv(file_a)
    p_b = _resolve_csv(file_b)

    # Load & preprocess
    try:
        df_a = pd.read_csv(p_a, comment="#")
        df_b = pd.read_csv(p_b, comment="#")
    except Exception as e:
        raise ValueError(f"Failed to read CSVs: {e}")

    df_a = _preprocess_df(df_a)
    df_b = _preprocess_df(df_b)

    # Align schemas (union) and order canonically
    all_cols = sorted(set(df_a.columns) | set(df_b.columns))
    canon = _canonical_column_order(all_cols)
    df_a = df_a.reindex(columns=canon)
    df_b = df_b.reindex(columns=canon)

    # Merge
    try:
        merged = pd.concat([df_a, df_b], ignore_index=True)
        if dedupe:
            merged = merged.drop_duplicates().reset_index(drop=True)
    except Exception as e:
        raise ValueError(f"Failed to merge: {e}")

    # Save
    if not output_name:
        ts = time.strftime("%Y%m%d-%H%M%S")
        stem_a = slugify(p_a.stem)
        stem_b = slugify(p_b.stem)
        output_name = f"merged__{stem_a}__{stem_b}__{ts}.csv"

    out_path = (UPLOAD_DIR / output_name)
    try:
        merged.to_csv(out_path, index=False)
    except Exception as e:
        raise IOError(f"Failed to save merged CSV: {e}")

    return out_path, int(len(merged))
