import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

try:
    from django.conf import settings
    DJANGO_OK = True
except Exception:
    DJANGO_OK = False

if DJANGO_OK and getattr(settings, "CONFIGURED", False):
    MEDIA_ROOT = Path(settings.MEDIA_ROOT).resolve()
    MEDIA_URL  = settings.MEDIA_URL
else:
    MEDIA_ROOT = Path(os.getcwd()).joinpath("media").resolve()
    MEDIA_URL  = "/media/"

MODELS_DIR = MEDIA_ROOT / "models"
PLOTS_DIR  = MEDIA_ROOT / "plots"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)
try:
    from .dataloader import load_data
    from .preprocess import preprocess_features
    from .model import train_all
except ImportError:
    from dataloader import load_data
    from preprocess import preprocess_features
    from model import train_all


def _to_media_url(abs_path: Path) -> str:
    rel = abs_path.resolve().relative_to(MEDIA_ROOT)
    return (MEDIA_URL.rstrip("/") + "/" + str(rel).replace("\\", "/"))


def _save_confusion_images(cm: np.ndarray, labels, dataset_tag: str, model_name: str):
    """Save raw & normalized CM images. Return (raw_path, norm_path)."""
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Raw CM
    cm_img_path = PLOTS_DIR / f"cm_{dataset_tag}_{model_name}_{timestamp}.png"
    plt.figure(figsize=(6, 5), dpi=160)
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix – {dataset_tag}/{model_name}")
    plt.colorbar()
    ticks = np.arange(cm.shape[0])
    tick_labels = labels if labels is not None else [str(i) for i in ticks]
    plt.xticks(ticks, tick_labels, rotation=45, ha="right")
    plt.yticks(ticks, tick_labels)
    # counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center", fontsize=10)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(cm_img_path, bbox_inches="tight")
    plt.close()

    # Normalized CM
    with np.errstate(all="ignore"):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    cm_norm_img_path = PLOTS_DIR / f"cm_{dataset_tag}_{model_name}_{timestamp}_normalized.png"
    fig, ax = plt.subplots(figsize=(6.5, 6.2), dpi=180)
    im = ax.imshow(cm_norm, vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_title(f"Normalized Confusion Matrix – {dataset_tag}/{model_name}", pad=12)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    # gridlines
    ax.set_xticks(np.arange(-.5, cm.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, cm.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6, alpha=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)
    # colorbar with % ticks
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("Proportion per true class", rotation=90, labelpad=10)
    cbar_ticks = np.linspace(0, 1, 6)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f"{t:.0%}" for t in cbar_ticks])
    # annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm_norm[i, j]
            if v < 0.005:
                continue
            txt_color = "white" if v >= 0.6 else "#111"
            ax.text(j, i, f"{v:.2%}\n({cm[i, j]})",
                    ha="center", va="center", fontsize=9, color=txt_color, linespacing=1.1)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(cm_norm_img_path, bbox_inches="tight")
    plt.close()

    return cm_img_path, cm_norm_img_path


def process_koi_trainall(data_path: str):
    """
    Trains all models on KOI, selects the best by CV mean (tie-break on test acc),
    saves the BEST model + plots, and returns artifact paths/metrics.
    """
    dataset_tag = "KOI"
    logger.info("Processing KOI dataset...")
    df = load_data(data_path, dataset_tag)
    if df is None:
        logger.error("Dataset for KOI not found.")
        return

    logger.info(f"Dataset shape: {df.shape} | Columns: {len(df.columns)}")
    X, y, target_encoder = preprocess_features(df, dataset_tag)
    if X is None or (hasattr(X, "empty") and X.empty):
        logger.warning("No preprocessing steps applied for KOI. Skipping training.")
        return
    if y is None:
        logger.error("Target variable is missing!")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    models = train_all()  


    results = {}
    for name, base_model in models.items():
        logger.info(f"Training {name} ...")

        if name in {"SVM", "Logistic Regression"}:
            model = make_pipeline(StandardScaler(), base_model)
            X_train_fit, X_test_eval = X_train, X_test
        else:
            model = base_model
            X_train_fit, X_test_eval = X_train, X_test
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        except Exception as e:
            logger.warning(f"CV failed for {name}: {e}")
            cv_scores = np.array([np.nan] * 5)
        model.fit(X_train_fit, y_train)
        y_pred = model.predict(X_test_eval)
        try:
            y_pred_proba = model.predict_proba(X_test_eval)
        except Exception:
            logger.warning(f"{name} has no predict_proba; AUC may be NaN.")
            y_pred_proba = None
        accuracy = np.mean(y_pred == y_test)
        cv_mean = float(np.nanmean(cv_scores))
        cv_std = float(np.nanstd(cv_scores))
        try:
            if y_pred_proba is None:
                auc_score = float('nan')
            elif len(np.unique(y)) > 2:
                auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
            else:
                auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
        except Exception as e:
            logger.warning(f"AUC failed for {name}: {e}")
            auc_score = float('nan')

        logger.info(f"{name} -> Test Acc: {accuracy:.4f} | CV: {cv_mean:.4f} ± {cv_std:.4f} | AUC: {auc_score:.4f}")

        results[name] = {
            "model": model,
            "accuracy": accuracy,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "auc_score": auc_score,
            "y_pred": y_pred,
        }

    def select_best_model(item):
        name, r = item
        return (np.nan_to_num(r["cv_mean"], nan=-1.0), r["accuracy"])

    best_name, best = sorted(results.items(), key=select_best_model, reverse=True)[0]
    best_model = best["model"]
    y_pred = best["y_pred"]

    logger.info(f"Best model: {best_name} | CV {best['cv_mean']:.4f} ± {best['cv_std']:.4f} | Test Acc {best['accuracy']:.4f}")
    try:
        labels = list(target_encoder.classes_)
    except Exception:
        labels = None
    cm = confusion_matrix(y_test, y_pred)
    cm_img_path, cm_norm_img_path = _save_confusion_images(cm, labels, dataset_tag, best_name)

    model_path = MODELS_DIR / f"{dataset_tag}_model_{best_name}.joblib"
    target_encoder_path = MODELS_DIR / f"{dataset_tag}_target_encoder_{best_name}.joblib"

    import joblib
    joblib.dump(best_model, model_path)
    if target_encoder is not None:
        joblib.dump(target_encoder, target_encoder_path)

    logger.info(f"Saved best model: {model_path}")
    logger.info(f"Saved encoders: {target_encoder_path}")
    logger.info(f"Saved CM images:\n - {cm_img_path}\n - {cm_norm_img_path}")

    return {
        "accuracy": best["accuracy"],
        "cv_mean": best["cv_mean"],
        "cv_std": best["cv_std"],
        "auc_score": best["auc_score"],
        "cm_image_path": str(cm_img_path),
        "cm_image_url": _to_media_url(cm_img_path),
        "cm_norm_image_path": str(cm_norm_img_path),
        "cm_norm_image_url": _to_media_url(cm_norm_img_path),
        "model_path": str(model_path),
        "model_url": _to_media_url(model_path),
    }


def main(data_path, satellite="K2", is_trainall: bool = False):
    """Main function to load data, preprocess, train model, and evaluate."""
    if is_trainall:
        if satellite == "K2":
            pass
        elif satellite == "KOI":
            return process_koi_trainall(data_path)
        else:
            logger.error(f"Unknown satellite: {satellite}")
            return None