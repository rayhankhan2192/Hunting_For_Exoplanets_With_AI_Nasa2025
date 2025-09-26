import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

# logging stays the same
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def coalesce_units(frame, prefer, fallback, factor):
    """Coalesce columns in the preferred units or convert using fallback and factor."""
    a = frame[prefer] if prefer in frame.columns else pd.Series(np.nan, index=frame.index)
    b = frame[fallback] if fallback in frame.columns else pd.Series(np.nan, index=frame.index)
    out = a.copy()
    use_b = out.isna() & (~b.isna())
    out[use_b] = b[use_b] * factor
    return out

def log1p_safe(s): 
    """Log transform with safety against non-positive values."""
    return np.log1p(s.clip(lower=0))

def spec_first_letter(s):
    """Extract first letter of spectral type."""
    return s.astype(str).str.strip().str.upper().str[0].replace({"N": np.nan, "": np.nan})

def preprocess_features(df, satellite):
    """Preprocess features for the dataset."""
    if satellite == "K2":
        
        # Save target column separately
        target_col = "disposition"
        if target_col in df.columns:
            target = df[target_col].copy()
            df.drop(columns=[target_col], inplace=True)

        DROP = [
            "pl_name", "hostname", "disp_refname", "pl_refname", "st_refname", "sy_refname",
            "pl_pubdate", "releasedate", "rowupdate", "rastr", "decstr", "discoverymethod", 
            "disc_facility", "default_flag", "pl_controv_flag", "ttv_flag", "soltype", "disc_year"
        ]
        
        df.drop(columns=[c for c in DROP if c in df.columns], inplace=True, errors="ignore")

        # Coalesce radius and mass into Earth units
        df["pl_radius_earth"] = coalesce_units(df, "pl_rade", "pl_radj", 11.21)
        df["pl_mass_earth"]   = coalesce_units(df, "pl_bmasse", "pl_bmassj", 317.8)

        # Density proxy (mass / radius^3)
        density_proxy = df["pl_mass_earth"] / (df["pl_radius_earth"] ** 3)
        density_proxy = density_proxy.replace([np.inf, -np.inf], np.nan)
        df["pl_density_proxy"] = density_proxy

        # Log transformation for skewed variables
        for c in ["pl_orbper", "pl_orbsmax", "sy_dist", "pl_eqt", "pl_insol"]:
            if c in df.columns:
                df[f"log_{c}"] = log1p_safe(df[c])

        # Stellar colors and spectral classes
        if all(c in df.columns for c in ["sy_vmag", "sy_kmag"]):
            df["color_v_minus_k"] = df["sy_vmag"] - df["sy_kmag"]
        if all(c in df.columns for c in ["sy_gaiamag", "sy_vmag"]):
            df["color_g_minus_v"] = df["sy_gaiamag"] - df["sy_vmag"]

        if "st_spectype" in df.columns:
            df["st_specclass"] = spec_first_letter(df["st_spectype"])

        # Final list of numeric and categorical columns
        numeric_cols = [c for c in [
            "pl_orbper", "pl_orbsmax", "pl_orbeccen", "pl_rade", "pl_radj", "pl_bmasse", "pl_bmassj",
            "pl_insol", "pl_eqt", "st_teff", "st_rad", "st_mass", "st_met", "st_logg", "sy_snum", 
            "sy_pnum", "ra", "dec", "sy_dist", "sy_vmag", "sy_kmag", "sy_gaiamag", "pl_radius_earth", 
            "pl_mass_earth", "pl_density_proxy", "log_pl_orbper", "log_pl_orbsmax", "log_sy_dist", 
            "log_pl_eqt", "log_pl_insol", "color_v_minus_k", "color_g_minus_v"
        ] if c in df.columns]
        categorical_cols = ["st_specclass"] if "st_specclass" in df.columns else []

        X = df[numeric_cols + categorical_cols].copy()

        return X, target
    

    elif satellite == "KOI":
        df_processed = df.copy()

        feature_columns = [
            col for col in df_processed.columns if col not in [
                'kepid', 
                'kepoi_name', 
                'kepler_name', 
                'koi_disposition', 
                'koi_pdisposition']]
        logger.info(f"Feature columns identified: {len(feature_columns)} features")
        logger.info(f"Features: {feature_columns}")
        logger.info("\n Handling missing values...")

        # Separate numeric and categorical features
        numeric_features = df_processed[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = df_processed[feature_columns].select_dtypes(include=['object']).columns.tolist()

        logger.info(f"Numeric features: {len(numeric_features)}")
        logger.info(f"Categorical features: {len(categorical_features)}")

        # Impute missing values for numeric features
        if len(numeric_features) > 0:
            logger.info("Imputing numeric features...")
            for col in numeric_features:
                if df_processed[col].isnull().sum() > 0:
                    median_val = df_processed[col].median()
                    df_processed[col] = df_processed[col].fillna(median_val)
                    logger.info(f"  - {col}: filled {df_processed[col].isnull().sum()} missing values")

        # Impute missing values for categorical features
        if len(categorical_features) > 0:
            logger.info("Imputing categorical features...")
            for col in categorical_features:
                if df_processed[col].isnull().sum() > 0:
                    mode_val = df_processed[col].mode().iloc[0] if not df_processed[col].mode().empty else 'Unknown'
                    df_processed[col] = df_processed[col].fillna(mode_val)
                    logger.info(f"  - {col}: filled {df_processed[col].isnull().sum()} missing values")

        logger.info("Missing values handled")

        # Prepare target variable
        target_encoder = LabelEncoder()
        df_processed['target_encoded'] = target_encoder.fit_transform(df_processed['koi_disposition'])

        logger.info(f"\nTarget encoding:")
        for i, label in enumerate(target_encoder.classes_):
            logger.info(f"{i}: {label}")
        
        logger.info("Block 5: Feature Engineering")

        df_engineered = df_processed.copy()

        # Feature engineering based on physical relationships
        if 'koi_period' in numeric_features and 'koi_prad' in numeric_features:
            # Planet density proxy (assuming circular orbit)
            df_engineered['period_to_radius_ratio'] = df_engineered['koi_period'] / (df_engineered['koi_prad'] ** 3)

        if 'koi_depth' in numeric_features and 'koi_prad' in numeric_features:
            # Transit depth should be related to planet radius squared
            df_engineered['depth_to_radius_sq'] = df_engineered['koi_depth'] / (df_engineered['koi_prad'] ** 2)

        if 'koi_duration' in numeric_features and 'koi_period' in numeric_features:
            # Transit duration to period ratio
            df_engineered['duration_period_ratio'] = df_engineered['koi_duration'] / df_engineered['koi_period']

        if 'koi_teq' in numeric_features and 'koi_steff' in numeric_features:
            # Temperature ratio (planet to star)
            df_engineered['temp_ratio'] = df_engineered['koi_teq'] / df_engineered['koi_steff']

        # Update feature list
        engineered_features = [col for col in df_engineered.columns 
                            if col not in ['kepid', 'kepoi_name', 'kepler_name', 'koi_disposition', 
                                        'koi_pdisposition', 'target_encoded'] and 
                            df_engineered[col].dtype in ['float64', 'int64']]

        logger.info(f"Feature engineering completed")
        logger.info(f"Total features after engineering: {len(engineered_features)}")
        logger.info(f"New engineered features: {[col for col in engineered_features if col not in numeric_features]}")

        logger.info("\nBlock 6: Data Splitting and Scaling")

        # Prepare features and target
        X = df_engineered[engineered_features].copy()
        y = df_engineered['target_encoded'].copy()

        # Check for any remaining missing or infinite values
        logger.info(f"  Checking data quality:")
        logger.info(f"  NaN values in X: {X.isnull().sum().sum()}")
        logger.info(f"  Infinite values in X: {np.isinf(X.values).sum()}")

        # Remove any remaining infinite or NaN values
        X = X.replace([np.inf, -np.inf], np.nan)

        # Fill remaining NaN values with column medians
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                median_val = X[col].median()
                if pd.isna(median_val):  # If median is also NaN, use 0
                    median_val = 0
                X[col] = X[col].fillna(median_val)
                logger.info(f"  Filled {col} with median value: {median_val}")

        # Final check
        logger.info(f"\nAfter cleaning:")
        logger.info(f"  NaN values in X: {X.isnull().sum().sum()}")
        logger.info(f"  Infinite values in X: {np.isinf(X.values).sum()}")

        logger.info(f"Final feature matrix shape: {X.shape}")
        logger.info(f"Target distribution: {np.bincount(y)}")
        return X, y, target_encoder
    
    else:
        return pd.DataFrame(), pd.Series()  # Placeholder for other datasets

def create_pipeline(numeric_cols, categorical_cols):
    """Create the pipeline for preprocessing numeric and categorical columns."""
    num_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler(with_centering=True, with_scaling=True)),
    ])

    cat_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_tf, numeric_cols),
            ("cat", cat_tf, categorical_cols)
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    return preprocessor
