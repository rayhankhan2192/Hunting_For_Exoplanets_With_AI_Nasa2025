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


# def process_k22(data_path, model_type, satellite):
#     """
#     Train/eval on K2 and return:
#     {
#       "accuracy": float,
#       "cv_mean": float,
#       "cv_std": float,
#       "auc_score": float or NaN,
#       "cm_image_path": str,
#       "cm_image_url": str,
#       "model_path": str,
#       "model_url": str
#     }
#     """
#     import time
#     import numpy as np
#     import pandas as pd
#     import joblib
#     import matplotlib.pyplot as plt
#     from sklearn.model_selection import train_test_split, cross_val_score
#     from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
#     from sklearn.pipeline import Pipeline
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.ensemble import RandomForestClassifier

#     SELECT_K = 40
#     MIN_ONEHOT_FREQ = 10
#     TEST_SIZE = 0.2
#     N_SPLITS = 5
#     RANDOM_STATE = 42

#     logger.info("Processing K2 dataset...")
#     df = load_data(data_path, "K2")
#     if df is None:
#         logger.error("Dataset for K2 not found.")
#         return  
#     logger.info(f"Dataset shape: {df.shape}")
#     logger.info(f"Columns: {len(df.columns)}")

#     if "hostname" in df.columns:
#         groups = df["hostname"].astype(str).values
#     else:
#         ra_bin = (df["ra"]*1000).round().astype("Int64") if "ra" in df.columns else 0
#         dec_bin = (df["dec"]*1000).round().astype("Int64") if "dec" in df.columns else 0
#         groups = (ra_bin.astype(str) + "_" + dec_bin.astype(str)).values

#     X_all, y_all, target_encoder, numeric_features, categorical_features = preprocess_features(df, "K2")
#     if X_all is None or (hasattr(X_all, "empty") and X_all.empty):
#         logger.warning("No preprocessing steps applied for K2. Skipping training.")
#         return
#     if y_all is None:
#         logger.error("Target variable 'disposition' is missing!")
#         return
#     if numeric_features is None or categorical_features is None:
#         logger.error("Feature lists are missing!")
#         return

#     #%% [Block 6: Group-aware train/test split]
#     from sklearn.model_selection import GroupShuffleSplit
#     gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
#     train_idx, test_idx = next(gss.split(X_all, y_all, groups=groups))
#     X_train, X_test = X_all.iloc[train_idx].copy(), X_all.iloc[test_idx].copy()
#     y_train, y_test = y_all[train_idx], y_all[test_idx]
#     groups_train, groups_test = groups[train_idx], groups[test_idx]
#     logger.info("   Data split completed:")
#     logger.info(f"   Training set: {X_train.shape[0]} samples")
#     logger.info(f"   Test set: {X_test.shape[0]} samples")

#     #%% [Block 7: Preprocessing & class weights]
#     from sklearn.compose import ColumnTransformer
#     from sklearn.impute import SimpleImputer
#     from sklearn.preprocessing import OneHotEncoder
#     from collections import Counter
#     from sklearn.model_selection import GroupKFold
#     from sklearn.feature_selection import SelectKBest, f_classif
#     numeric_transformer = Pipeline(steps=[
#         ("impute", SimpleImputer(strategy="median")),
#         ("scale", StandardScaler(with_mean=True, with_std=True))
#     ])

#     categorical_transformer = Pipeline(steps=[
#         ("impute", SimpleImputer(strategy="most_frequent")),
#         ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=MIN_ONEHOT_FREQ))
#     ])

#     preproc = ColumnTransformer(
#         transformers=[
#             ("num", numeric_transformer, numeric_features),
#             ("cat", categorical_transformer, categorical_features),
#         ],
#         remainder="drop",
#         verbose_feature_names_out=False
#     )

#     counts = Counter(y_train)
#     n_classes = len(set(y_train))
#     total = sum(counts.values())
#     class_weight = {cls: total/(n_classes*cnt) for cls, cnt in counts.items()}

#     def make_select_k(k_cap, preprocessor, X_sample):
#         # Fit a small preprocessor to get transformed feature count
#         Z = preprocessor.fit_transform(X_sample.iloc[:200].copy())  # sample for speed
#         feat_count = Z.shape[1]
#         k = min(max(10, k_cap), feat_count)  # ensure 10 <= k <= feat_count
#         logger.info("SelectKBest | Transformed features=%d | Using k=%d", feat_count, k)
#         return SelectKBest(score_func=f_classif, k=k)
    
#     def make_pipe(clf):
#         return Pipeline([
#             ("pre", preproc),
#             ("sel", make_select_k(SELECT_K, preproc, X_train)),
#             ("clf", clf)
#         ])
#     model = get_model(model_type=model_type)
#     cv_pipe = make_pipe(model)
#     cv_scores = cross_val_score(
#         cv_pipe,
#         X_train, y_train,
#         cv=GroupKFold(n_splits=N_SPLITS).split(X_train, y_train, groups=groups_train),
#         scoring="f1_macro",
#         n_jobs=-1
#     )
#     cv_mean = float(cv_scores.mean())
#     cv_std = float(cv_scores.std())
#     logger.info(f"Training {model_type} model for {satellite}")
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)

#     # AUC (macro OVR)
#     from sklearn.metrics import roc_auc_score, f1_score
#     auc_macro = None
#     try:
#         if hasattr(model, "predict_proba"):
#             proba = model.predict_proba(X_test)
#             if isinstance(proba, np.ndarray) and proba.ndim == 2:
#                 auc_macro = float(roc_auc_score(y_test, proba, multi_class="ovr", average="macro"))
#         elif hasattr(model, "decision_function"):
#             s = model.decision_function(X_test)
#             if isinstance(s, np.ndarray) and s.ndim == 2:
#                 auc_macro = float(roc_auc_score(y_test, s, multi_class="ovr", average="macro"))
#     except Exception:
#         auc_macro = None

#     acc = float((y_pred == y_test).mean())
#     f1m = float(f1_score(y_test, y_pred, average="macro"))
#     report_txt = classification_report(y_test, y_pred, target_names=target_encoder.classes_, digits=4)
#     cm = confusion_matrix(y_test, y_pred)

#     logger.info("\n=== MODEL MATRIX ===")
#     logger.info(f"Test Accuracy: {acc:.4f}")
#     logger.info(f"Cross-validation F1-macro: {cv_mean:.4f} ± {cv_std:.4f}")
#     if auc_macro is not None:
#         logger.info(f"ROC-AUC (macro OVR): {auc_macro:.4f}")
#     logger.info(f"F1-macro: {f1m:.4f}")
#     logger.info("\n=== VALIDATION REPORT ===")
#     logger.info("\n" + report_txt)
#     logger.info("\n=== CONFUSION MATRIX ===")
#     try:
#         cm_df = pd.DataFrame(cm, index=target_encoder.classes_, columns=target_encoder.classes_)
#         logger.info(f"\n{cm_df}")
#     except Exception:
#         logger.info(f"\n{pd.DataFrame(cm)}")
#     # Save model under MEDIA_ROOT/models
#     model_path = MODELS_DIR / f"{satellite}_model_{model_type}.joblib"
#     target_encoder_path = MODELS_DIR / f"{satellite}_target_encoder_{model_type}.joblib"
#     feature_scaler_path = MODELS_DIR / f"{satellite}_feature_scaler_{model_type}.joblib"    
#     joblib.dump(model, model_path)
#     joblib.dump(target_encoder, target_encoder_path)
#     joblib.dump(preproc, feature_scaler_path)

#     logger.info(f"{satellite} model saved as {model_path}")
#     logger.info(f"{satellite} target encoder saved as {target_encoder_path}")
#     logger.info(f"{satellite} feature scaler saved as {feature_scaler_path}")
#     # Save CM image under MEDIA_ROOT/plots
#     timestamp = time.strftime("%Y%m%d-%H%M%S")
#     cm_img_path = PLOTS_DIR / f"cm_{satellite}_{model_type}_{timestamp}.png"
#     plt.figure(figsize=(6, 5), dpi=160)
#     plt.imshow(cm, interpolation="nearest")
#     plt.title(f"Confusion Matrix – {satellite} / {model_type}")
#     plt.colorbar()
#     ticks = np.arange(cm.shape[0])
#     plt.xticks(ticks, target_encoder.classes_, rotation=45, ha="right")
#     plt.yticks(ticks, target_encoder.classes_)
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center", fontsize=10)
#     plt.ylabel("True label")
#     plt.xlabel("Predicted label")
#     plt.tight_layout()
#     plt.savefig(cm_img_path, bbox_inches="tight")
#     plt.close()

#     # cm normalized
#     with np.errstate(all="ignore"):
#         cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
#     cm_normalized = np.nan_to_num(cm_normalized)  # replace NaN for rows with 0 samples
    
#     logger.info(f"Saved confusion matrix image at: {cm_img_path}")
#     return {
#         "accuracy": acc,
#         "cv_mean": cv_mean,
#         "cv_std": cv_std,
#         "auc_score": float(auc_macro) if auc_macro is not None else float("nan"),
#         "cm_image_path": str(cm_img_path),
#         "cm_image_url": _to_media_url(cm_img_path),
#         "cm_norm_image_path": str(cm_normalized),
#         "cm_norm_image_url": _to_media_url(cm_normalized),
#         "model_path": str(model_path),
#         "model_url": _to_media_url(model_path),
#     }

def process_k22(data_path, model_type, satellite):
    """
    Train/eval on K2 and return:
    {
      "accuracy": float,
      "cv_mean": float,
      "cv_std": float,
      "auc_score": float or NaN,
      "cm_image_path": str,
      "cm_image_url": str,
      "cm_norm_image_path": str,
      "cm_norm_image_url": str,
      "model_path": str,
      "model_url": str
    }
    """
    import time
    import numpy as np
    import pandas as pd
    import joblib
    import matplotlib.pyplot as plt
    from collections import Counter

    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.metrics import (
        classification_report, confusion_matrix, roc_auc_score,
        accuracy_score, f1_score
    )
    from sklearn.utils.validation import has_fit_parameter
    SELECT_K = 40
    MIN_ONEHOT_FREQ = 10
    TEST_SIZE = 0.2
    N_SPLITS = 5
    RANDOM_STATE = 42
    logger.info("Processing K2 dataset...")
    df = load_data(data_path, "K2")
    if df is None:
        logger.error("Dataset for K2 not found.")
        return
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {len(df.columns)}")

    # # groups 
    # if "hostname" in df.columns:
    #     groups = df["hostname"].astype(str).values
    # else:
    #     ra_bin = (df["ra"] * 1000).round().astype("Int64") if "ra" in df.columns else 0
    #     dec_bin = (df["dec"] * 1000).round().astype("Int64") if "dec" in df.columns else 0
    #     groups = (ra_bin.astype(str) + "_" + dec_bin.astype(str)).values

    # X_all: raw features (not yet transformed); y_all: numeric labels; target_encoder: LabelEncoder-like
    X_all, y_all, groups, target_encoder, numeric_features, categorical_features = preprocess_features(df, "K2")
    if X_all is None or (hasattr(X_all, "empty") and X_all.empty):
        logger.warning("No preprocessing steps applied for K2. Skipping training.")
        return
    if y_all is None:
        logger.error("Target variable 'disposition' is missing!")
        return
    if numeric_features is None or categorical_features is None:
        logger.error("Feature lists are missing!")
        return

    #split (group-safe)
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X_all, y_all, groups=groups))
    X_train, X_test = X_all.iloc[train_idx].copy(), X_all.iloc[test_idx].copy()
    y_train, y_test = y_all[train_idx], y_all[test_idx]
    groups_train = groups[train_idx]
    logger.info("   Data split completed:")
    logger.info(f"   Training set: {X_train.shape[0]} samples")
    logger.info(f"   Test set: {X_test.shape[0]} samples")

    # preprocessing
    numeric_transformer = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=True, with_std=True)),
    ])

    # Dense output so SelectKBest(f_classif) works nicely
    # For sklearn>=1.2 use 'sparse_output=False'; for older, use 'sparse=False'
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=MIN_ONEHOT_FREQ, sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=MIN_ONEHOT_FREQ, sparse=False)

    categorical_transformer = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe),
    ])

    preproc = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # class weights
    counts = Counter(y_train)
    n_classes = len(set(y_train))
    total = sum(counts.values())
    class_weight = {cls: total / (n_classes * cnt) for cls, cnt in counts.items()}
    logger.info(f"Class weights: {class_weight}")

    # model & pipeline
    # get_model must return an sklearn classifier (e.g., XGBClassifier, RandomForestClassifier, etc.)
    clf = get_model(model_type=model_type)

    # Try to set estimator class_weight directly if supported
    if hasattr(clf, "class_weight"):
        try:
            clf.set_params(class_weight=class_weight)
        except Exception:
            pass

    # SelectKBest must know how many features; safest is to place it in pipeline (it will see transformed shape)
    pipe = Pipeline([
        ("pre", preproc),
        ("sel", SelectKBest(score_func=f_classif, k=SELECT_K)),
        ("clf", clf),
    ])

    # If estimator doesn't support class_weight, fall back to sample_weight
    # (Pipeline.fit supports **fit_params; will be sliced inside CV automatically)
    sample_weight = np.vectorize(class_weight.get)(y_train)
    supports_sw = has_fit_parameter(pipe.named_steps["clf"], "sample_weight")

    gkf = GroupKFold(n_splits=N_SPLITS)
    cv_scores = []
    for tr_idx, va_idx in gkf.split(X_train, y_train, groups=groups_train):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        pipe_cv = clone(pipe)
        if supports_sw:
            sw_tr = np.vectorize(class_weight.get)(y_tr)
            pipe_cv.fit(X_tr, y_tr, clf__sample_weight=sw_tr)
        else:
            pipe_cv.fit(X_tr, y_tr)

        y_va_pred = pipe_cv.predict(X_va)
        cv_scores.append(f1_score(y_va, y_va_pred, average="macro"))

    cv_scores = np.array(cv_scores, dtype=float)
    cv_mean = float(cv_scores.mean())
    cv_std  = float(cv_scores.std())
    logger.info(f"CV F1-macro: {cv_mean:.4f} ± {cv_std:.4f} (GroupKFold={N_SPLITS})")

    #fit final (on full TRAIN)
    if supports_sw:
        pipe.fit(X_train, y_train, clf__sample_weight=np.vectorize(class_weight.get)(y_train))
    else:
        pipe.fit(X_train, y_train)

    #predict/eval
    y_pred = pipe.predict(X_test)

    # Get probabilities for AUC if available
    auc_macro = np.nan
    try:
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X_test)
            if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] == n_classes:
                auc_macro = float(roc_auc_score(y_test, proba, multi_class="ovr", average="macro"))
        elif hasattr(pipe, "decision_function"):
            s = pipe.decision_function(X_test)
            if isinstance(s, np.ndarray) and s.ndim == 2 and s.shape[1] == n_classes:
                auc_macro = float(roc_auc_score(y_test, s, multi_class="ovr", average="macro"))
    except Exception:
        pass

    acc = float(accuracy_score(y_test, y_pred))
    f1m = float(f1_score(y_test, y_pred, average="macro"))
    report_txt = classification_report(y_test, y_pred, target_names=target_encoder.classes_, digits=4)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(n_classes)))

    logger.info("\n=== MODEL METRICS ===")
    logger.info(f"Test Accuracy: {acc:.4f}")
    logger.info(f"Cross-validation F1-macro: {cv_mean:.4f} ± {cv_std:.4f}")
    if not np.isnan(auc_macro):
        logger.info(f"ROC-AUC (macro OVR): {auc_macro:.4f}")
    logger.info(f"F1-macro: {f1m:.4f}")
    logger.info("\n=== TEST REPORT ===")
    logger.info("\n" + report_txt)
    logger.info("\n=== CONFUSION MATRIX ===")
    cm_df = pd.DataFrame(cm, index=target_encoder.classes_, columns=target_encoder.classes_)
    logger.info(f"\n{cm_df}")

    # save artifacts
    # model
    model_path = MODELS_DIR / f"{satellite}_model_{model_type}.joblib"
    joblib.dump(pipe, model_path)
    logger.info(f"{satellite} pipeline (preproc+sel+clf) saved as {model_path}")

    # label encoder (so you can map back index->label later)
    target_encoder_path = MODELS_DIR / f"{satellite}_target_encoder_{model_type}.joblib"
    joblib.dump(target_encoder, target_encoder_path)
    logger.info(f"{satellite} target encoder saved as {target_encoder_path}")
    # feature scaler (so you can preprocess new data the same way)
    feature_scaler_path = MODELS_DIR / f"{satellite}_feature_scaler_{model_type}.joblib"
    joblib.dump(preproc, feature_scaler_path)

    import json
    # Save metadata.json
    metadata = {
        "best_model": model_type,
        "label_set": target_encoder.classes_.tolist(),
        "cv_f1_macro": cv_mean,
        "test_accuracy": acc,
        "test_f1_macro": f1m,
        "test_auc_ovr_macro": auc_macro,
        "n_selected_features": len(numeric_features) + len(categorical_features),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "created_at": time.strftime("%Y%m%d-%H%M%S"),
    }

    meta_path = MODELS_DIR / f"{satellite}_{model_type}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata JSON saved → {meta_path}")

    # Save selected_features.json 
    selected_features = numeric_features + categorical_features
    feat_path = MODELS_DIR / f"{satellite}_{model_type}_selected_features.json"
    with open(feat_path, "w") as f:
        json.dump(selected_features, f, indent=2)
    logger.info(f"Selected features JSON saved → {feat_path}")
    # confusion matrices (raw + normalized)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    cm_img_path = PLOTS_DIR / f"cm_{satellite}_{model_type}_{timestamp}.png"
    cm_norm_img_path = PLOTS_DIR / f"cm_norm_{satellite}_{model_type}_{timestamp}.png"

    # raw
    plt.figure(figsize=(6, 5), dpi=160)
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix – {satellite} / {model_type}")
    plt.colorbar()
    ticks = np.arange(cm.shape[0])
    plt.xticks(ticks, target_encoder.classes_, rotation=45, ha="right")
    plt.yticks(ticks, target_encoder.classes_)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center", fontsize=10)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(cm_img_path, bbox_inches="tight")
    plt.close()

    # normalized
    with np.errstate(all="ignore"):
        cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-12)
    cm_norm = np.nan_to_num(cm_norm)

    plt.figure(figsize=(6, 5), dpi=160)
    plt.imshow(cm_norm, interpolation="nearest")
    plt.title(f"Normalized Confusion Matrix – {satellite} / {model_type}")
    plt.colorbar()
    plt.xticks(ticks, target_encoder.classes_, rotation=45, ha="right")
    plt.yticks(ticks, target_encoder.classes_)
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            plt.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=10)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(cm_norm_img_path, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved confusion matrix images at: {cm_img_path} and {cm_norm_img_path}")

    return {
        "accuracy": acc,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "auc_score": float(auc_macro) if not np.isnan(auc_macro) else float("nan"),
        "cm_image_path": str(cm_img_path),
        "cm_image_url": _to_media_url(cm_img_path),
        "cm_norm_image_path": str(cm_norm_img_path),
        "cm_norm_image_url": _to_media_url(cm_norm_img_path),
        "model_path": str(model_path),
        "model_url": _to_media_url(model_path),
    }


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
    logger.info("\n=== CONFUSION MATRIX (raw counts) ===")
    try:
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        logger.info(f"\n{cm_df}")
    except Exception:
        logger.info(f"\n{pd.DataFrame(cm)}")

    # Normalized CM (row-wise)
    with np.errstate(all="ignore"):
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # replace NaN for rows with 0 samples

    logger.info("\n=== CONFUSION MATRIX (normalized) ===")
    try:
        cm_norm_df = pd.DataFrame(
            cm_normalized.round(3), index=labels, columns=labels
        )
        logger.info(f"\n{cm_norm_df}")
    except Exception:
        logger.info(f"\n{pd.DataFrame(cm_normalized)}")

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
    cm_norm_img_path = PLOTS_DIR / f"cm_{satellite}_{model_type}_{timestamp}_normalized.png"

    #Raw CM Plot
    plt.figure(figsize=(6, 5), dpi=160)
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix – {satellite}/{model_type}")
    plt.colorbar()
    ticks = np.arange(cm.shape[0])
    tick_labels = labels if labels is not None else [str(i) for i in ticks]
    plt.xticks(ticks, tick_labels, rotation=45, ha="right")
    plt.yticks(ticks, tick_labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center", fontsize=10)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(cm_img_path, bbox_inches="tight")
    plt.close()

    # Normalized CM Plot
    plt.figure(figsize=(6.5, 6.2), dpi=180)
    ax = plt.gca()
    im = ax.imshow(cm_normalized, vmin=0.0, vmax=1.0, interpolation="nearest")

    # title & axes
    ax.set_title(f"Normalized Confusion Matrix – {satellite}/{model_type}", pad=12)
    ax.set_xticks(ticks, tick_labels, rotation=45, ha="right")
    ax.set_yticks(ticks, tick_labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    # light gridlines between cells
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
            v = cm_normalized[i, j]
            # skip near-zero to reduce clutter (optional)
            if v < 0.005:
                continue
            txt_color = "white" if v >= 0.6 else "#111"
            # show percent; add counts in parentheses if you like
            label = f"{v:.2%}\n({cm[i, j]})"  # or use f"{v:.2%}" for percent only
            ax.text(j, i, label, ha="center", va="center", fontsize=9, color=txt_color, linespacing=1.1)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(cm_norm_img_path, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved confusion matrix images at:\n - Raw: {cm_img_path}\n - Normalized: {cm_norm_img_path}")

    return {
    "accuracy": accuracy,
    "cv_mean": cv_mean,
    "cv_std": cv_std,
    "auc_score": auc_score,
    "cm_image_path": str(cm_img_path),
    "cm_image_url": _to_media_url(cm_img_path),
    "cm_norm_image_path": str(cm_norm_img_path),
    "cm_norm_image_url": _to_media_url(cm_norm_img_path),
    "model_path": str(model_path),
    "model_url": _to_media_url(model_path),
    }



def main(data_path, satellite="K2", model_type="rf"):
    """Main function to load data, preprocess, train model, and evaluate."""
    if satellite == "K2":
        return process_k22(data_path, model_type, satellite)
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
    parser.add_argument("--model", type=str, choices=["rf", "gb", "xgb", "dt", "grdb", "logreg", "svm"], default="rf", help="Model type (rf, xgb, dt)")

    args = parser.parse_args()

    main(args.data_path, args.satellite, args.model)
