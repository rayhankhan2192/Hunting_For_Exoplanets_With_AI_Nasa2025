import logging
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from dataloader import load_data
from preprocess import preprocess_features, create_pipeline
from model import get_model
import os
# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def process_k2(data_path, model_type, satellite):
    """Process K2 dataset, train and evaluate the model."""
    logger.info("Processing K2 dataset...")
    
    # Load data for K2
    df = load_data(data_path, "K2")
    if df is None:
        logger.error("Dataset for K2 not found.")
        return
    
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {len(df.columns)}")

    # Strip any whitespace from column names
    df.columns = df.columns.str.strip()
    logger.info(f"Columns after stripping: {df.columns.tolist()}")

    # Preprocess K2 data
    X, y = preprocess_features(df, "K2")
    
    if X.empty:
        logger.warning("No preprocessing steps applied for K2. Skipping training.")
        return
    
    # Check if target 'y' is available
    if y is None:
        logger.error("Target variable 'disposition' is missing!")
        return

    CLASS_ORDER = ["FALSE POSITIVE", "CANDIDATE", "CONFIRMED"]
    CLASS_TO_ID = {c: i for i, c in enumerate(CLASS_ORDER)}
    
    logger.info(f"Disposition counts:\n{y.value_counts(dropna=False)}")

    # Check/encode target
    y = y.map(CLASS_TO_ID)
    keep = y.notna()
    X, y = X.loc[keep].copy(), y.loc[keep].astype(int)

    numeric_cols = [c for c in X.columns if X[c].dtype != "object"]
    categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Create preprocessing pipeline
    preprocessor = create_pipeline(numeric_cols, categorical_cols)
    
    # Get the model
    model = get_model(model_type=model_type)
    
    # Create full pipeline with preprocessing and model
    pipe = Pipeline([("preprocessor", preprocessor), ("clf", model)])

    # Train the model
    logger.info(f"Training {model_type} model for K2")
    pipe.fit(X_train, y_train)

    # Evaluate the model
    logger.info("Evaluating K2 model")
    preds = pipe.predict(X_val)
    logger.info("\n=== VALIDATION REPORT ===")
    logger.info(classification_report(y_val, preds, target_names=CLASS_ORDER, digits=4))

    cm = confusion_matrix(y_val, preds)
    logger.info("\n=== CONFUSION MATRIX ===")
    logger.info(f"\n{pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER)}")

    model_dir = '../savedmodel'
    if not os.path.exists(model_dir):
        logger.info(f"Creating directory {model_dir}")
        os.makedirs(model_dir)

    # Save the model
    model_path = os.path.join(model_dir, f"{satellite}_model_{model_type}.joblib")
    joblib.dump(pipe, model_path)
    logger.info(f"K2 model saved as K2_model_{model_type}.joblib")

def main(data_path, satellite="K2", model_type="rf"):
    """Main function to load data, preprocess, train model, and evaluate."""
    if satellite == "K2":
        process_k2(data_path, model_type, satellite)
    else:
        logger.error(f"Unknown satellite: {satellite}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train exoplanet model")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the dataset CSV file")
    parser.add_argument("--satellite", type=str, default="K2", help="Satellite name (K2, TOI, KOI)")
    parser.add_argument("--model", type=str, choices=["rf", "xgb", "dt"], default="rf", help="Model type (rf, xgb, dt)")

    args = parser.parse_args()

    main(args.data_path, args.satellite, args.model)
