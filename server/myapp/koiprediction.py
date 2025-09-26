import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

BASE_DIR = Path(__file__).resolve().parents[2]
SAVED_MODEL_DIR = BASE_DIR / "savedmodel"

MODEL_FILE = SAVED_MODEL_DIR / "KOI_model_xgb.joblib"
SCALER_FILE = SAVED_MODEL_DIR / "KOI_feature_scaler_xgb.joblib"
ENCODER_FILE = SAVED_MODEL_DIR / "KOI_target_encoder_xgb.joblib"

_MODEL, _SCALER, _ENCODER = None, None, None

# Models that require scaling 
_SCALE_MODELS = {"LogisticRegression", "SVC", "LinearSVC", "SGDClassifier"}


def _requires_scaling(model) -> bool:
    return model.__class__.__name__ in _SCALE_MODELS


def load_artifacts():
    """Load KOI model, scaler, and target encoder (cached)."""
    global _MODEL, _SCALER, _ENCODER
    if _MODEL is not None:
        return _MODEL, _SCALER, _ENCODER

    _MODEL = joblib.load(MODEL_FILE)
    _SCALER = joblib.load(SCALER_FILE)
    _ENCODER = joblib.load(ENCODER_FILE)

    return _MODEL, _SCALER, _ENCODER


def _feature_order(model, scaler, engineered_df: pd.DataFrame):
    """
    Determine the exact feature order used at training time.
    Prefer model.feature_names_in_; fall back to scaler.feature_names_in_;
    finally fall back to current engineered columns (least ideal).
    """
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(scaler, "feature_names_in_"):
        return list(scaler.feature_names_in_)
    return list(engineered_df.columns) 


def preprocess_single_row(row_dict, model, scaler):
    """Preprocess a single row for prediction (keeps your original math)."""
    try:
        df_row = pd.DataFrame([row_dict])

        if "koi_period" in df_row.columns and "koi_prad" in df_row.columns:
            p, r = df_row["koi_period"].iloc[0], df_row["koi_prad"].iloc[0]
            if pd.notna(p) and pd.notna(r) and r != 0:
                df_row["period_to_radius_ratio"] = p / (r**3)

        if "koi_depth" in df_row.columns and "koi_prad" in df_row.columns:
            d, r = df_row["koi_depth"].iloc[0], df_row["koi_prad"].iloc[0]
            if pd.notna(d) and pd.notna(r) and r != 0:
                df_row["depth_to_radius_sq"] = d / (r**2)

        if "koi_duration" in df_row.columns and "koi_period" in df_row.columns:
            dur, p = df_row["koi_duration"].iloc[0], df_row["koi_period"].iloc[0]
            if pd.notna(dur) and pd.notna(p) and p != 0:
                df_row["duration_period_ratio"] = dur / p

        if "koi_teq" in df_row.columns and "koi_steff" in df_row.columns:
            teq, st = df_row["koi_teq"].iloc[0], df_row["koi_steff"].iloc[0]
            if pd.notna(teq) and pd.notna(st) and st != 0:
                df_row["temp_ratio"] = teq / st

        # Build in the exact order used at training
        feature_names = _feature_order(model, scaler, df_row)
        vals = []
        for f in feature_names:
            if f in df_row.columns:
                v = df_row[f].iloc[0]
                if pd.isna(v) or np.isinf(v):
                    v = 0
                vals.append(v)
            else:
                vals.append(0)

        X_df = pd.DataFrame([vals], columns=feature_names)

        # Apply scaling ONLY if the model requires it (Logistic/SVM)
        if _requires_scaling(model):
            X_in = scaler.transform(X_df)           # ndarray (1, n)
        else:
            X_in = X_df.values                      # ndarray (1, n)

        return X_in[0] 
    except Exception as e:
        print(f" Preprocessing error: {str(e)}")
        return None


def predict_from_csv(csv_file_path, row_numbers=None):
    """
    Predict exoplanet classification for specific rows (or all rows if None).
    Returns a DataFrame with predictions.
    """
    model, scaler, encoder = load_artifacts()

    df = pd.read_csv(csv_file_path, comment="#")

    # If no row range given → predict all
    if row_numbers is None:
        row_numbers = list(range(len(df)))

    # Validate row numbers
    valid_rows = [r for r in row_numbers if 0 <= r < len(df)]
    if not valid_rows:
        return pd.DataFrame()

    selected_rows = df.iloc[valid_rows].copy()

    predictions_results = []
    for original_idx, row in selected_rows.iterrows():
        row_dict = row.to_dict()

        processed_data = preprocess_single_row(row_dict, model, scaler)
        if processed_data is None:
            continue

        # model expects 2D
        prediction = model.predict([processed_data])[0]
        probabilities = model.predict_proba([processed_data])[0]

        predicted_class = encoder.inverse_transform([prediction])[0]
        prob_dict = {encoder.classes_[i]: float(prob) for i, prob in enumerate(probabilities)}

        actual_class = row.get("koi_disposition", "Unknown")

        result = {
            "Row_Number": original_idx,
            "KEP_ID": row.get("kepid", "Unknown"),
            "KOI_Name": row.get("kepoi_name", "Unknown"),
            "Kepler_Name": row.get("kepler_name", "Unknown"),
            "Actual_Class": actual_class,
            "Predicted_Class": predicted_class,
            "Confidence": float(probabilities.max()),
            "Match": ("Yes" if actual_class == predicted_class else "No" if actual_class != "Unknown" else "N/A"),
        }
        for cls, p in prob_dict.items():
            result[f"Prob_{cls}"] = p

        predictions_results.append(result)

    return pd.DataFrame(predictions_results)