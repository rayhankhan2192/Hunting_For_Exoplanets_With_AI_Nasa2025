import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

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
