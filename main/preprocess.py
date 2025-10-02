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
    if satellite == "K22":
        
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
    
    # elif satellite == "KOI":
    #     df = df.copy()
    #     SELECT_K = 40  
    #     MIN_ONEHOT_FREQ = 10 
    #     TEST_SIZE = 0.2
    #     N_SPLITS = 5
    #     RANDOM_STATE = 42

    #     LABEL_SET = ["CANDIDATE", "CONFIRMED", "FALSE POSITIVE"]
    #     df["disposition_norm"] = (
    #         df["disposition"].astype(str).str.strip().str.upper().str.replace("_", " ")
    #     )
    #     df = df[df["disposition_norm"].isin(LABEL_SET)].copy()
    #     df["y"] = df["disposition_norm"].map({lab: i for i, lab in enumerate(LABEL_SET)}).astype(int)

    #     class_counts = df["disposition_norm"].value_counts().to_dict()
    #     logger.info("Class distribution: %s", class_counts)

    #     def add_k2_features(frame: pd.DataFrame) -> pd.DataFrame:
    #         out = frame.copy()
    #         if {"pl_rade","pl_radj"}.issubset(out.columns):
    #             denom = (out["pl_radj"] * 11.209).replace(0, np.nan)
    #             out["radius_consistency"] = out["pl_rade"] / denom
    #         if {"pl_bmasse","pl_rade"}.issubset(out.columns):
    #             denom = (out["pl_rade"]**3).replace(0, np.nan)
    #             out["bulk_density_proxy"] = out["pl_bmasse"] / denom
    #         if {"pl_orbper","pl_orbsmax"}.issubset(out.columns):
    #             denom = (out["pl_orbsmax"]**3).replace(0, np.nan)
    #             out["kepler_ratio"] = (out["pl_orbper"]**2) / denom
    #         if {"pl_eqt","st_teff"}.issubset(out.columns):
    #             denom = out["st_teff"].replace(0, np.nan)
    #             out["temp_ratio"] = out["pl_eqt"] / denom
    #         if "pl_insol" in out.columns:
    #             out["log_insol"] = np.log10(out["pl_insol"].clip(lower=0) + 1)
    #         if {"st_mass","st_rad"}.issubset(out.columns):
    #             denom = (out["st_rad"]**3).replace(0, np.nan)
    #             out["stellar_density_proxy"] = out["st_mass"] / denom
    #         if "sy_dist" in out.columns:
    #             out["log_distance"] = np.log10(out["sy_dist"].clip(lower=0) + 1)
    #         if "pl_orbper" in out.columns:
    #             out["log_period"] = np.log10(out["pl_orbper"].clip(lower=0) + 1)
    #         return out.replace([np.inf, -np.inf], np.nan)

    #     df = add_k2_features(df)
    #     logger.info("Feature engineering complete. Current columns: %d", df.shape[1])

    #     #%% [Block 5: Feature lists & grouping]
    #     base_numeric = [
    #         'sy_snum','sy_pnum','pl_orbper','pl_orbsmax','pl_rade','pl_radj','pl_bmasse','pl_bmassj',
    #         'pl_orbeccen','pl_insol','pl_eqt','st_teff','st_rad','st_mass','st_met','st_logg',
    #         'ra','dec','sy_dist','sy_vmag','sy_kmag','sy_gaiamag',
    #         'radius_consistency','bulk_density_proxy','kepler_ratio','temp_ratio',
    #         'log_insol','stellar_density_proxy','log_distance','log_period'
    #     ]
    #     context_num = [c for c in ["default_flag","pl_controv_flag","ttv_flag","disc_year"] if c in df.columns]
    #     context_cat = [c for c in ["discoverymethod","disc_facility","st_spectype","pl_bmassprov"] if c in df.columns]

    #     numeric_features = [c for c in base_numeric + context_num if c in df.columns]
    #     categorical_features = [c for c in context_cat if c in df.columns]

    #     if "hostname" in df.columns:
    #         groups = df["hostname"].astype(str).values
    #     else:
    #         ra_bin = (df["ra"]*1000).round().astype("Int64") if "ra" in df.columns else 0
    #         dec_bin = (df["dec"]*1000).round().astype("Int64") if "dec" in df.columns else 0
    #         groups = (ra_bin.astype(str) + "_" + dec_bin.astype(str)).values

    #     X_all = df[numeric_features + categorical_features].copy()
    #     y_all = df["y"].values
    #     logger.info("Numeric features: %d | Categorical: %d", len(numeric_features), len(categorical_features))

    #     return X_all, y_all, target_encoder

    elif satellite == "K2":
        df = df.copy()

        # Define canonical label order (kept for clarity)
        LABEL_SET = ["CANDIDATE", "CONFIRMED", "FALSE POSITIVE"]

        # Normalize label text
        df["disposition_norm"] = (
            df["disposition"]
            .astype(str).str.strip().str.upper().str.replace("_", " ")
        )
        df = df[df["disposition_norm"].isin(LABEL_SET)].copy()

        # === target encoder ===
        target_encoder = LabelEncoder()
        # Note: LabelEncoder sorts classes alphabetically; with these labels
        # it becomes ['CANDIDATE','CONFIRMED','FALSE POSITIVE'] which matches LABEL_SET.
        y = target_encoder.fit_transform(df["disposition_norm"])
        df["target_encoder"] = y

        class_counts = df["disposition_norm"].value_counts().to_dict()
        logger.info("Class distribution: %s", class_counts)

        # ---- feature engineering (unchanged) ----
        def add_k2_features(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame.copy()
            if {"pl_rade","pl_radj"}.issubset(out.columns):
                denom = (out["pl_radj"] * 11.209).replace(0, np.nan)
                out["radius_consistency"] = out["pl_rade"] / denom
            if {"pl_bmasse","pl_rade"}.issubset(out.columns):
                denom = (out["pl_rade"]**3).replace(0, np.nan)
                out["bulk_density_proxy"] = out["pl_bmasse"] / denom
            if {"pl_orbper","pl_orbsmax"}.issubset(out.columns):
                denom = (out["pl_orbsmax"]**3).replace(0, np.nan)
                out["kepler_ratio"] = (out["pl_orbper"]**2) / denom
            if {"pl_eqt","st_teff"}.issubset(out.columns):
                denom = out["st_teff"].replace(0, np.nan)
                out["temp_ratio"] = out["pl_eqt"] / denom
            if "pl_insol" in out.columns:
                out["log_insol"] = np.log10(out["pl_insol"].clip(lower=0) + 1)
            if {"st_mass","st_rad"}.issubset(out.columns):
                denom = (out["st_rad"]**3).replace(0, np.nan)
                out["stellar_density_proxy"] = out["st_mass"] / denom
            if "sy_dist" in out.columns:
                out["log_distance"] = np.log10(out["sy_dist"].clip(lower=0) + 1)
            if "pl_orbper" in out.columns:
                out["log_period"] = np.log10(out["pl_orbper"].clip(lower=0) + 1)
            return out.replace([np.inf, -np.inf], np.nan)

        df = add_k2_features(df)
        logger.info("Feature engineering complete. Current columns: %d", df.shape[1])

        # ---- feature lists & grouping (unchanged) ----
        base_numeric = [
            'sy_snum','sy_pnum','pl_orbper','pl_orbsmax','pl_rade','pl_radj','pl_bmasse','pl_bmassj',
            'pl_orbeccen','pl_insol','pl_eqt','st_teff','st_rad','st_mass','st_met','st_logg',
            'ra','dec','sy_dist','sy_vmag','sy_kmag','sy_gaiamag',
            'radius_consistency','bulk_density_proxy','kepler_ratio','temp_ratio',
            'log_insol','stellar_density_proxy','log_distance','log_period'
        ]
        context_num = [c for c in ["default_flag","pl_controv_flag","ttv_flag","disc_year"] if c in df.columns]
        context_cat = [c for c in ["discoverymethod","disc_facility","st_spectype","pl_bmassprov"] if c in df.columns]

        numeric_features = [c for c in base_numeric + context_num if c in df.columns]
        categorical_features = [c for c in context_cat if c in df.columns]
        X = df[numeric_features + categorical_features].copy()
        logger.info("Numeric features: %d | Categorical: %d", len(numeric_features), len(categorical_features))
        logger.info(f"Numeric features: {numeric_features}")
        logger.info(f"Categorical features: {categorical_features}")
        logger.info(f"Final feature matrix shape: {X.shape}")
        logger.info(f"Target distribution: {np.bincount(y)}")

        if "hostname" in df.columns:
            groups = df["hostname"].astype(str).fillna("UNK").values
        else:
            ra_bin = (df["ra"]*1000).round().astype("Int64") if "ra" in df.columns else pd.Series(0, index=df.index, dtype="Int64")
            dec_bin = (df["dec"]*1000).round().astype("Int64") if "dec" in df.columns else pd.Series(0, index=df.index, dtype="Int64")
            groups = (ra_bin.astype(str) + "_" + dec_bin.astype(str)).values

        # Final sanity check: all same length
        assert len(X) == len(y) == len(groups), f"Length mismatch: X={len(X)}, y={len(y)}, groups={len(groups)}"

        return X, y, groups, target_encoder, numeric_features, categorical_features

    

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
