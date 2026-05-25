"""
Handles all tabular data operations:
  - Loading raw CSVs
  - Dropping high-null columns
  - Filling missing values
  - Encoding categoricals
  - Scaling numerics
  - Saving/loading processed artifacts
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ── Constants 

NULL_THRESHOLD   = 0.40          # drop columns with >40% missing values
TARGET_COL       = "SalePrice"
QUAL_COL         = "OverallQual" # used for image pairing
PROCESSED_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# Columns to always drop (IDs or heavy leakage)
DROP_COLS = ["Id"]


# ── Helpers 

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ── Core Functions 

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"[load_data] Loaded {df.shape[0]} rows × {df.shape[1]} cols from '{csv_path}'")
    return df


def drop_high_null_cols(df: pd.DataFrame,
                        threshold: float = NULL_THRESHOLD) -> pd.DataFrame:
    null_frac = df.isnull().mean()
    cols_to_drop = null_frac[null_frac > threshold].index.tolist()
    # Never drop the target or the quality column used for pairing
    cols_to_drop = [c for c in cols_to_drop if c not in [TARGET_COL, QUAL_COL]]
    df = df.drop(columns=cols_to_drop + DROP_COLS, errors="ignore")
    print(f"[drop_high_null_cols] Dropped {len(cols_to_drop) + len(DROP_COLS)} cols → {df.shape[1]} remain")
    return df


from pandas.api.types import is_numeric_dtype

def fill_missing(df):

    for col in df.columns:

        print(col, df[col].dtype)

        if col == TARGET_COL:
            continue

        if is_numeric_dtype(df[col]):

            print(f"NUMERIC → {col}")

            df[col] = df[col].fillna(
                df[col].median()
            )

        else:

            print(f"CATEGORICAL → {col}")

            mode_val = df[col].mode()

            df[col] = df[col].fillna(
                mode_val[0] if len(mode_val) else "Unknown"
            )

    return df


def get_column_types(df: pd.DataFrame):
    exclude = [TARGET_COL]
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in num_cols if c not in exclude]
    cat_cols = [c for c in cat_cols if c not in exclude]
    return num_cols, cat_cols


def encode_categoricals(df: pd.DataFrame,
                         cat_cols: list,
                         encoders: dict = None) -> tuple:
    
    if encoders is None:
        encoders = {}
        fit = True
    else:
        fit = False

    for col in cat_cols:
        le = encoders.get(col, LabelEncoder())
        if fit:
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            # Handle unseen labels gracefully
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in known else le.classes_[0]
            )
            df[col] = le.transform(df[col])
    print(f"[encode_categoricals] Encoded {len(cat_cols)} categorical cols")
    return df, encoders


def scale_numerics(df: pd.DataFrame,
                   num_cols: list,
                   scaler: StandardScaler = None) -> tuple:
    
    if scaler is None:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        print(f"[scale_numerics] Fitted and scaled {len(num_cols)} numeric cols")
    else:
        df[num_cols] = scaler.transform(df[num_cols])
        print(f"[scale_numerics] Scaled {len(num_cols)} numeric cols with existing scaler")
    return df, scaler


def preprocess_pipeline(csv_path: str,
                         encoders: dict = None,
                         scaler: StandardScaler = None) -> dict:
    
    df = load_data(csv_path)
    df = drop_high_null_cols(df)
    df = fill_missing(df)

    # Separate target
    y_raw = df[TARGET_COL].values.astype(np.float32)
    y     = np.log1p(y_raw)                     # log-transform for regression stability
    qual  = df[QUAL_COL].values.astype(int)

    df = df.drop(columns=[TARGET_COL])

    num_cols, cat_cols = get_column_types(df)
    df, encoders       = encode_categoricals(df, cat_cols, encoders)
    df, scaler         = scale_numerics(df, num_cols, scaler)

    X = df.values.astype(np.float32)
    print(f"[preprocess_pipeline] Final feature matrix: {X.shape}")

    return {
        "X":           X,
        "y":           y.astype(np.float32),
        "y_raw":       y_raw,
        "df":          df,
        "num_cols":    num_cols,
        "cat_cols":    cat_cols,
        "scaler":      scaler,
        "encoders":    encoders,
        "qual_values": qual,
    }


# ── Persistence ───────────────────────────────────────────────────────────────

def save_artifacts(scaler, encoders, out_dir: str = PROCESSED_DIR) -> None:
    
    _ensure_dir(out_dir)
    with open(os.path.join(out_dir, "scaler.pkl"),   "wb") as f:
        pickle.dump(scaler,   f)
    with open(os.path.join(out_dir, "encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)
    print(f"[save_artifacts] Scaler + encoders saved to '{out_dir}'")


def load_artifacts(out_dir: str = PROCESSED_DIR) -> tuple:
    with open(os.path.join(out_dir, "scaler.pkl"),   "rb") as f:
        scaler   = pickle.load(f)
    with open(os.path.join(out_dir, "encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)
    print(f"[load_artifacts] Loaded scaler + encoders from '{out_dir}'")
    return scaler, encoders


def scale_target(y: np.ndarray,
                 scaler=None) -> tuple:
    
    from sklearn.preprocessing import StandardScaler as SS
    y2d = y.reshape(-1, 1)
    if scaler is None:
        scaler = SS()
        y_scaled = scaler.fit_transform(y2d).ravel()
        print(f"[scale_target] Fitted  mean={scaler.mean_[0]:.4f}  std={scaler.scale_[0]:.4f}")
    else:
        y_scaled = scaler.transform(y2d).ravel()
        print(f"[scale_target] Transformed with existing scaler")
    return y_scaled.astype(np.float32), scaler


def inverse_scale_target(y_scaled: np.ndarray, scaler) -> np.ndarray:
    return scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel().astype(np.float32)