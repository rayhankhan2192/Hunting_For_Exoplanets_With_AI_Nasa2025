# k2prediction.py
import os
import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import joblib
import numpy as np
import pandas as pd
from django.conf import settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Avoid duplicate handlers under Django autoreload
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

MEDIA_ROOT = Path(settings.MEDIA_ROOT).resolve()
MODELS_DIR = MEDIA_ROOT / "models"  # same convention as koiprediction.py

# Cache: (satellite, model_type) -> (model, meta_dict, feat_cfg_dict)
_CACHE: Dict[Tuple[str, str], Tuple[Any, dict, dict]] = {}

# Some linear models require explicit scaling if not in a pipeline (kept for parity/future use)
_SCALE_MODELS = {"LogisticRegression", "SVC", "LinearSVC", "SGDClassifier"}


# ------------------------- Feature engineering: K2 ------------------------- #
def add_k2_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Engineered features used in your K2 notebook."""
    out = frame.copy()
    if {"pl_rade", "pl_radj"}.issubset(out.columns):
        denom = (out["pl_radj"] * 11.209).replace(0, np.nan)
        out["radius_consistency"] = out["pl_rade"] / denom
    if {"pl_bmasse", "pl_rade"}.issubset(out.columns):
        denom = (out["pl_rade"] ** 3).replace(0, np.nan)
        out["bulk_density_proxy"] = out["pl_bmasse"] / denom
    if {"pl_orbper", "pl_orbsmax"}.issubset(out.columns):
        denom = (out["pl_orbsmax"] ** 3).replace(0, np.nan)
        out["kepler_ratio"] = (out["pl_orbper"] ** 2) / denom
    if {"pl_eqt", "st_teff"}.issubset(out.columns):
        denom = out["st_teff"].replace(0, np.nan)
        out["temp_ratio"] = out["pl_eqt"] / denom
    if "pl_insol" in out.columns:
        out["log_insol"] = np.log10(out["pl_insol"].clip(lower=0) + 1)
    if {"st_mass", "st_rad"}.issubset(out.columns):
        denom = (out["st_rad"] ** 3).replace(0, np.nan)
        out["stellar_density_proxy"] = out["st_mass"] / denom
    if "sy_dist" in out.columns:
        out["log_distance"] = np.log10(out["sy_dist"].clip(lower=0) + 1)
    if "pl_orbper" in out.columns:
        out["log_period"] = np.log10(out["pl_orbper"].clip(lower=0) + 1)
    return out.replace([np.inf, -np.inf], np.nan)


# ---------------------------- helpers & loaders --------------------------- #
def _is_pipeline(model) -> bool:
    try:
        from sklearn.pipeline import Pipeline as SkPipeline  # type: ignore
        return isinstance(model, SkPipeline)
    except Exception:
        return False


def _unwrap_estimator(model):
    """Return final estimator if model is a sklearn Pipeline; else identity."""
    if _is_pipeline(model):
        return getattr(model, "steps", [])[-1][1]
    return model


def _requires_scaling(model) -> bool:
    est = _unwrap_estimator(model)
    return est.__class__.__name__ in _SCALE_MODELS


def _artifact_paths(satellite: str, model_type: str) -> Tuple[Path, Path, Path]:
    """
    Resolve expected file paths inside MEDIA_ROOT/models:
      {SAT}_model_{mt}.joblib
      {SAT}_{mt}_metadata.json
      {SAT}_{mt}_selected_features.json
    """
    sat = (satellite or "K2").upper().strip()
    mt = (model_type or "rf").lower().strip()
    model_file = MODELS_DIR / f"{sat}_model_{mt}.joblib"
    meta_file = MODELS_DIR / f"{sat}_{mt}_metadata.json"
    feats_file = MODELS_DIR / f"{sat}_{mt}_selected_features.json"
    return model_file, meta_file, feats_file


def _verify_artifacts_exist(*paths: Path):
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing artifact(s): {', '.join(missing)} in {MODELS_DIR}")


def load_artifacts(satellite: str = "K2", model_type: str = "rf"):
    """
    Load and cache (model, metadata, features_config) for a given satellite+model_type.

    Expected files in MEDIA_ROOT/models:
      {SAT}_model_{mt}.joblib
      {SAT}_{mt}_metadata.json              -> {"label_set": [...]}
      {SAT}_{mt}_selected_features.json     -> EITHER:
           {"numeric_features":[...], "categorical_features":[...]}
        OR ["colA","colB", ...]   (a plain JSON array of selected features)
    """
    sat = (satellite or "K2").upper().strip()
    mt = (model_type or "rf").lower().strip()
    key = (sat, mt)
    if key in _CACHE:
        return _CACHE[key]

    model_path, meta_path, feats_path = _artifact_paths(sat, mt)
    _verify_artifacts_exist(model_path, meta_path, feats_path)

    logger.info(f"Loading {sat} artifacts: model_type={mt}")
    model = joblib.load(model_path)

    # metadata.json must be a dict with label_set
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if not isinstance(meta, dict) or "label_set" not in meta or not isinstance(meta["label_set"], list):
        raise ValueError(f"{meta_path.name} must contain a dict with 'label_set': [...]")

    # selected_features.json can be either dict OR list
    with open(feats_path, "r", encoding="utf-8") as f:
        feats = json.load(f)

    # If it's a list, normalize to {"selected_features": [...]}
    if isinstance(feats, list):
        feats = {"selected_features": feats}
    elif not isinstance(feats, dict):
        raise ValueError(f"{feats_path.name} must be a dict or a list")

    # Accept both 'categorical_features' and the common typo 'categororical_features'
    categorical = feats.get("categorical_features", feats.get("categororical_features", []))
    numeric = feats.get("numeric_features", [])
    selected = feats.get("selected_features", [])

    # Build a unified config
    feats_cfg = {
        "selected_features": selected if isinstance(selected, list) else [],
        "numeric_features": numeric if isinstance(numeric, list) else [],
        "categorical_features": categorical if isinstance(categorical, list) else [],
    }

    _CACHE[key] = (model, meta, feats_cfg)
    return _CACHE[key]



def _predict_proba_safe(model, X2d: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Robustly obtain class index + probability vector.
    - use predict_proba if available
    - else decision_function -> softmax
    - else hard predict -> one-hot
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X2d)[0]
        idx = int(np.argmax(probs))
        return idx, probs

    if hasattr(model, "decision_function"):
        df = model.decision_function(X2d)
        df = np.atleast_2d(df)
        exps = np.exp(df - np.max(df, axis=1, keepdims=True))
        probs = (exps / np.sum(exps, axis=1, keepdims=True))[0]
        idx = int(np.argmax(probs))
        return idx, probs

    idx = int(model.predict(X2d)[0])
    est = _unwrap_estimator(model)
    n_classes = len(getattr(est, "classes_", [])) or 2
    probs = np.zeros(n_classes, dtype=float)
    probs[idx] = 1.0
    return idx, probs


def _feature_order(model, feats_cfg: dict, engineered_df: pd.DataFrame) -> List[str]:
    """
    Determine feature order:
      1) If selected_features provided -> use that
      2) Else concat numeric_features + categorical_features
      3) Else model.feature_names_in_ if present
      4) Else fall back to engineered_df.columns
    """
    if feats_cfg.get("selected_features"):
        return list(feats_cfg["selected_features"])

    nc = feats_cfg.get("numeric_features", [])
    cc = feats_cfg.get("categorical_features", [])
    if nc or cc:
        return list(nc) + list(cc)

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    return list(engineered_df.columns)


def _apply_satellite_features(df_in: pd.DataFrame, satellite: str) -> pd.DataFrame:
    """Hook to apply per-satellite feature engineering; defaults to K2 behavior for 'K2'."""
    sat = (satellite or "K2").upper().strip()
    if sat == "K2":
        return add_k2_features(df_in)
    # For other satellites you can add branches later (KOI/TESS/TOI custom FEs).
    return df_in.copy()


def _prepare_features(df_in: pd.DataFrame, model, feats_cfg: dict, satellite: str) -> pd.DataFrame:
    """
    Apply satellite-specific features, ensure columns, order them, and sanitize NaNs/infs.
    """
    df_feat = _apply_satellite_features(df_in.copy(), satellite)
    desired_cols = _feature_order(model, feats_cfg, df_feat)

    # ensure columns exist
    for c in desired_cols:
        if c not in df_feat.columns:
            df_feat[c] = np.nan

    X = df_feat[desired_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X


def predict_from_csv(
    csv_path: str,
    satellite: str = "K2",
    model_type: str = "rf",
    row_numbers: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Predict labels for a CSV using {SAT}_model_{model_type}.joblib and JSON sidecars.

    Returns a DataFrame with:
      - pl_name (if present)
      - hostname (if present)
      - row_number_1based
      - actual_class (from CSV 'disposition' or 'koi_disposition' if present)
      - pred_label
      - pred_index
      - Confidence (max prob across classes, if probs available)
      - match (Yes/No/N/A)
      - Prob_<CLASS> columns (one per label in metadata label_set)
    """
    # Load artifacts and metadata
    model, meta, feats_cfg = load_artifacts(satellite, model_type)
    label_set: List[str] = meta["label_set"]

    # Read CSV
    df = pd.read_csv(csv_path, low_memory=False)

    # Choose rows
    if row_numbers is None:
        idx_list = list(range(len(df)))
        df_sel = df
    else:
        valid = [r for r in row_numbers if 0 <= r < len(df)]
        if not valid:
            return pd.DataFrame()
        idx_list = valid
        df_sel = df.iloc[idx_list].copy()

    # Prepare features
    X = _prepare_features(df_sel, model, feats_cfg, satellite)
    # For Pipelines, keep as DataFrame; for bare estimators, ndarray is fine
    X2d = X if _is_pipeline(model) else X.values.astype(float)

    # Predict hard labels (indices)
    if not hasattr(model, "predict"):
        raise AttributeError("Loaded model does not implement .predict")
    y_idx = np.asarray(model.predict(X2d)).astype(int).ravel()

    # ---- Batch probabilities (Pipeline-safe) ----
    proba: Optional[np.ndarray] = None
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X2d)  # batch, preserves DF if pipeline
        elif hasattr(model, "decision_function"):
            dfm = model.decision_function(X2d)
            dfm = np.atleast_2d(dfm)
            exps = np.exp(dfm - np.max(dfm, axis=1, keepdims=True))
            proba = exps / np.sum(exps, axis=1, keepdims=True)
        else:
            proba = None
    except Exception as e:
        logger.warning(f"Probability computation failed, continuing without probs: {e}")
        proba = None

    # Map indices → labels (guarded)
    def _label_of(i: int) -> str:
        try:
            return label_set[i]
        except Exception:
            return str(i)

    y_lab = [_label_of(int(i)) for i in y_idx]

    # ---- Build output ----
    out = pd.DataFrame({
        "row_number_1based": [i + 1 for i in idx_list],
        "pred_index": y_idx,
        "pred_label": y_lab,
    })

    # Keep original casing/spaces for probability headers
    if isinstance(proba, np.ndarray) and proba.ndim == 2:
        n_cols = proba.shape[1]
        labs = label_set
        if len(labs) != n_cols:
            logger.warning(f"Probability columns ({n_cols}) != label_set size ({len(label_set)}). Aligning best-effort.")
            labs = [label_set[i] if i < len(label_set) else f"class_{i}" for i in range(n_cols)]

        for j, lab in enumerate(labs):
            out[f"Prob_{lab}"] = proba[:, j].astype(float)

        out["Confidence"] = proba.max(axis=1).astype(float)

    # Add identifiers if present
    for col in ["pl_name", "hostname"]:
        if col in df_sel.columns:
            out[col] = df_sel[col].values

    # Add actual_class if available
    actual_col = None
    for candidate in ["disposition", "koi_disposition"]:
        if candidate in df_sel.columns:
            actual_col = candidate
            break
    if actual_col:
        out["actual_class"] = df_sel[actual_col].values
    else:
        out["actual_class"] = "Unknown"

    # Add match
    out["match"] = [
        "N/A" if a == "Unknown" else ("Yes" if a == p else "No")
        for a, p in zip(out["actual_class"], out["pred_label"])
    ]

    # Reorder nicely as requested
    front = [
        c for c in [
            "pl_name", "hostname", "row_number_1based",
            "actual_class", "pred_label", "pred_index",
            "Confidence", "match"
        ] if c in out.columns
    ]
    prob_cols = [c for c in out.columns if c.startswith("Prob_")]
    rest = [c for c in out.columns if c not in front + prob_cols]
    out = out[front + prob_cols + rest]

    return out
