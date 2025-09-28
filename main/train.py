import logging, os, joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from django.conf import settings

# Support both "python -m main.train" and direct script execution
try:
    from .dataloader import load_data
    from .preprocess import preprocess_features, create_pipeline
    from .model import get_model
except ImportError:
    from dataloader import load_data
    from preprocess import preprocess_features, create_pipeline
    from model import get_model


# logging stays the same
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

import os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from django.conf import settings

MEDIA_ROOT = Path(settings.MEDIA_ROOT).resolve()
MEDIA_URL  = settings.MEDIA_URL  # e.g. "/media/"

MODELS_DIR = MEDIA_ROOT / "models"
PLOTS_DIR  = MEDIA_ROOT / "plots"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def _to_media_url(abs_path: Path) -> str:
    rel = abs_path.resolve().relative_to(MEDIA_ROOT)
    return (MEDIA_URL.rstrip("/") + "/" + str(rel).replace("\\", "/"))

def process_k2(data_path, model_type, satellite):
    """
    Train/eval on K2 and return:
    {
      "accuracy": float,
      "cv_mean": float,
      "cv_std": float,
      "auc_score": float or NaN,
      "cm_image_path": str,
      "cm_image_url": str,
      "model_path": str,
      "model_url": str
    }
    """
    import time
    import numpy as np
    import pandas as pd
    import joblib
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
    from sklearn.pipeline import Pipeline

    logger.info("Processing K2 dataset...")

    df = load_data(data_path, "K2")
    if df is None:
        logger.error("Dataset for K2 not found.")
        return

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {len(df.columns)}")
    df.columns = df.columns.str.strip()
    logger.info(f"Columns after stripping: {df.columns.tolist()}")

    X, y = preprocess_features(df, "K2")
    if X is None or (hasattr(X, "empty") and X.empty):
        logger.warning("No preprocessing steps applied for K2. Skipping training.")
        return
    if y is None:
        logger.error("Target variable 'disposition' is missing!")
        return

    CLASS_ORDER = ["FALSE POSITIVE", "CANDIDATE", "CONFIRMED"]
    CLASS_TO_ID = {c: i for i, c in enumerate(CLASS_ORDER)}
    logger.info(f"Disposition counts:\n{y.value_counts(dropna=False)}")

    # encode target & drop NaNs
    y = y.map(CLASS_TO_ID)
    keep = y.notna()
    X, y = X.loc[keep].copy(), y.loc[keep].astype(int)

    numeric_cols = [c for c in X.columns if X[c].dtype != "object"]
    categorical_cols = [c for c in X.columns if X[c].dtype == "object"]

    preprocessor = create_pipeline(numeric_cols, categorical_cols)
    clf = get_model(model_type=model_type)

    pipe = Pipeline([("preprocessor", preprocessor), ("clf", clf)])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    logger.info(f"Training {model_type} model for {satellite}")
    pipe.fit(X_train, y_train)

    logger.info(f"Evaluating {satellite} model")
    y_pred = pipe.predict(X_val)

    # Cross-val on the full pipe (avoids leakage)
    cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
    cv_mean, cv_std = float(cv_scores.mean()), float(cv_scores.std())

    # AUC if available
    auc_score = float("nan")
    y_proba = None
    try:
        y_proba = pipe.predict_proba(X_val)
    except Exception:
        try:
            scores = pipe.decision_function(X_val)
            if scores.ndim == 1:
                y_proba = np.vstack([1/(1+np.exp(scores)), 1/(1+np.exp(-scores))]).T
            else:
                e = np.exp(scores - scores.max(axis=1, keepdims=True))
                y_proba = e / e.sum(axis=1, keepdims=True)
        except Exception:
            pass

    if y_proba is not None:
        if len(np.unique(y)) > 2:
            auc_score = float(roc_auc_score(y_val, y_proba, multi_class="ovr", average="macro"))
        else:
            auc_score = float(roc_auc_score(y_val, y_proba[:, 1]))

    acc = float(accuracy_score(y_val, y_pred))

    logger.info("\n=== VALIDATION REPORT ===")
    logger.info("\n" + classification_report(y_val, y_pred, target_names=CLASS_ORDER, digits=4))

    cm = confusion_matrix(y_val, y_pred)
    logger.info("\n=== CONFUSION MATRIX ===")
    logger.info(f"\n{pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER)}")

    # Save model under MEDIA_ROOT/models
    model_path = MODELS_DIR / f"{satellite}_model_{model_type}.joblib"
    joblib.dump(pipe, model_path)
    logger.info(f"{satellite} model saved as {model_path}")

    # Save CM image under MEDIA_ROOT/plots
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    cm_img_path = PLOTS_DIR / f"cm_{satellite}_{model_type}_{timestamp}.png"

    plt.figure(figsize=(6, 5), dpi=160)
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix – {satellite} / {model_type}")
    plt.colorbar()
    ticks = np.arange(cm.shape[0])
    plt.xticks(ticks, CLASS_ORDER, rotation=45, ha="right")
    plt.yticks(ticks, CLASS_ORDER)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center", fontsize=10)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(cm_img_path, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved confusion matrix image at: {cm_img_path}")

    return {
        "accuracy": acc,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "auc_score": auc_score,
        "cm_image_path": str(cm_img_path),
        "cm_image_url": _to_media_url(cm_img_path),
        "model_path": str(model_path),
        "model_url": _to_media_url(model_path),
    }


# def process_koi(data_path, model_type, satellite):
#     """Process KOI dataset, train and evaluate the model."""
#     logger.info("Processing KOI dataset...")

#     df = load_data(data_path, "KOI")
#     if df is None:
#         logger.error("Dataset for KOI not found.")
#         return
#     logger.info(f"Dataset shape: {df.shape}")
#     logger.info(f"Columns: {len(df.columns)}")

#     x, y, target_encoder = preprocess_features(df, "KOI")
#     if x.empty: # No preprocessing steps applied
#         logger.warning("No preprocessing steps applied for KOI. Skipping training.")
#         return
#     if y is None:
#         logger.error("Target variable 'disposition' is missing!")
#         return
#     X_train, X_test, y_train, y_test = train_test_split(
#         x, y, test_size=0.2, random_state=42, stratify=y
#     )
    
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     logger.info(f"   Data split completed:")
#     logger.info(f"   Training set: {X_train.shape[0]} samples")
#     logger.info(f"   Test set: {X_test.shape[0]} samples")

#     model = get_model(model_type=model_type)
    

#     # Train the model
#     logger.info(f"Training {model_type} model for {satellite}")
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     y_pred_proba = model.predict_proba(X_test)
#     cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

#     accuracy = np.mean(y_pred == y_test)
#     cv_mean = cv_scores.mean()
#     cv_std = cv_scores.std()

#     if len(np.unique(y)) > 2:
#         auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
#     else:
#         auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
    
#     logger.info(f"Test Accuracy: {accuracy:.4f}")
#     logger.info(f"Cross-validation Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
#     logger.info(f"ROC-AUC Score: {auc_score:.4f}")

#     # Evaluate the model
#     logger.info(f"Evaluating {satellite} model")
#     preds = model.predict(X_test)
#     logger.info("\n=== VALIDATION REPORT ===")
#     logger.info(f"\n{classification_report(y_test, preds, target_names=target_encoder.classes_, digits=4)}")

#     cm = confusion_matrix(y_test, preds)
#     logger.info("\n=== CONFUSION MATRIX ===")
#     logger.info(f"\n{pd.DataFrame(cm, index=target_encoder.classes_, columns=target_encoder.classes_)}")

#     model_dir = '../savedmodel'
#     if not os.path.exists(model_dir):
#         logger.info(f"Creating directory {model_dir}")
#         os.makedirs(model_dir)


#     os.makedirs(SAVEDMODEL_DIR, exist_ok=True)
#     model_path = os.path.join(SAVEDMODEL_DIR, f"{satellite}_model_{model_type}.joblib")
#     target_encoder_path = os.path.join(SAVEDMODEL_DIR, f"{satellite}_target_encoder_{model_type}.joblib")
#     feature_scaler_path = os.path.join(SAVEDMODEL_DIR, f"{satellite}_feature_scaler_{model_type}.joblib")
    
#     joblib.dump(model, model_path)
#     joblib.dump(target_encoder, target_encoder_path)
#     joblib.dump(scaler, feature_scaler_path)
#     logger.info(f"{satellite} model saved as {model_path}")
#     return str(model_path)
def process_koi(data_path, model_type, satellite):
    """
    Train/eval on KOI and return:
    {
      "accuracy": float,
      "cv_mean": float,
      "cv_std": float,
      "auc_score": float or NaN,
      "cm_image_path": str,
      "cm_image_url": str,
      "model_path": str,
      "model_url": str
    }
    """
    import time
    import numpy as np
    import pandas as pd
    import joblib
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

    logger.info("Processing KOI dataset...")

    df = load_data(data_path, "KOI")
    if df is None:
        logger.error("Dataset for KOI not found.")
        return
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {len(df.columns)}")

    X, y, target_encoder = preprocess_features(df, "KOI")
    if X is None or (hasattr(X, "empty") and X.empty):
        logger.warning("No preprocessing steps applied for KOI. Skipping training.")
        return
    if y is None:
        logger.error("Target variable 'disposition' is missing!")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    logger.info("   Data split completed:")
    logger.info(f"   Training set: {X_train.shape[0]} samples")
    logger.info(f"   Test set: {X_test.shape[0]} samples")

    model = get_model(model_type=model_type)

    # CV with scaler+model to avoid leakage
    cv_pipe = make_pipeline(StandardScaler(), get_model(model_type=model_type))
    cv_scores = cross_val_score(cv_pipe, X, y, cv=5, scoring="accuracy")
    cv_mean, cv_std = float(cv_scores.mean()), float(cv_scores.std())

    logger.info(f"Training {model_type} model for {satellite}")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # AUC
    y_proba = None
    try:
        y_proba = model.predict_proba(X_test_scaled)
    except Exception:
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X_test_scaled)
            if scores.ndim == 1:
                y_proba = np.vstack([1/(1+np.exp(scores)), 1/(1+np.exp(-scores))]).T
            else:
                e = np.exp(scores - scores.max(axis=1, keepdims=True))
                y_proba = e / e.sum(axis=1, keepdims=True)

    accuracy = float(accuracy_score(y_test, y_pred))
    if y_proba is not None:
        if len(np.unique(y)) > 2:
            auc_score = float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"))
        else:
            auc_score = float(roc_auc_score(y_test, y_proba[:, 1]))
    else:
        auc_score = float("nan")

    logger.info("\n=== MODEL MATRIX ===")
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Cross-validation Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
    logger.info(f"ROC-AUC Score: {auc_score:.4f}")

    logger.info("\n=== VALIDATION REPORT ===")
    try:
        logger.info("\n" + classification_report(
            y_test, y_pred, target_names=list(target_encoder.classes_), digits=4
        ))
        labels = list(target_encoder.classes_)
    except Exception:
        logger.info("\n" + classification_report(y_test, y_pred, digits=4))
        labels = None

    cm = confusion_matrix(y_test, y_pred)
    logger.info("\n=== CONFUSION MATRIX ===")
    try:
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        logger.info(f"\n{cm_df}")
    except Exception:
        logger.info(f"\n{pd.DataFrame(cm)}")

    # Save artifacts to MEDIA_ROOT
    model_path = MODELS_DIR / f"{satellite}_model_{model_type}.joblib"
    target_encoder_path = MODELS_DIR / f"{satellite}_target_encoder_{model_type}.joblib"
    feature_scaler_path = MODELS_DIR / f"{satellite}_feature_scaler_{model_type}.joblib"

    joblib.dump(model, model_path)
    if target_encoder is not None:
        joblib.dump(target_encoder, target_encoder_path)
    joblib.dump(scaler, feature_scaler_path)
    logger.info(f"{satellite} model saved as {model_path}")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    cm_img_path = PLOTS_DIR / f"cm_{satellite}_{model_type}_{timestamp}.png"

    plt.figure(figsize=(6, 5), dpi=160)
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix – {satellite} / {model_type}")
    plt.colorbar()
    ticks = np.arange(cm.shape[0])
    tick_labels = labels if labels is not None else [str(i) for i in ticks]
    plt.xticks(ticks, tick_labels, rotation=45, ha="right")
    plt.yticks(ticks, tick_labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center", fontsize=10)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(cm_img_path, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved confusion matrix image at: {cm_img_path}")

    return {
        "accuracy": accuracy,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "auc_score": auc_score,
        "cm_image_path": str(cm_img_path),
        "cm_image_url": _to_media_url(cm_img_path),
        "model_path": str(model_path),
        "model_url": _to_media_url(model_path),
    }



def main(data_path, satellite="K2", model_type="rf"):
    """Main function to load data, preprocess, train model, and evaluate."""
    if satellite == "K2":
        #process_k2(data_path, model_type, satellite)
        return process_k2(data_path, model_type, satellite)
    elif satellite == "KOI":
        return process_koi(data_path, model_type, satellite)
    else:
        logger.error(f"Unknown satellite: {satellite}")
        return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train exoplanet model")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the dataset CSV file")
    parser.add_argument("--satellite", type=str, default="K2", help="Satellite name (K2, TOI, KOI)")
    parser.add_argument("--model", type=str, choices=["rf", "xgb", "dt", "grdb", "logreg", "svm"], default="rf", help="Model type (rf, xgb, dt)")

    args = parser.parse_args()

    main(args.data_path, args.satellite, args.model)
