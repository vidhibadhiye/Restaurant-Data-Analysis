"""Helper functions for the Restaurant Rating Prediction dashboard.

Everything here is plain Python/Pandas so it can be tested without Streamlit.
Nothing in this module retrains a model or modifies the original files.
"""
import json
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "enhanced_zomato_dataset_clean.csv"
BACKEND_DIR = BASE_DIR / "backend"
NOTEBOOK_PATH = BACKEND_DIR / "SI_PROJECT.ipynb"
MODEL_PATH = BACKEND_DIR / "best_restaurant_model.pkl"
SCALER_PATH = BACKEND_DIR / "restaurant_scaler.pkl"
MODEL_NAME_PATH = BACKEND_DIR / "best_restaurant_model_name.pkl"

# Taken from the notebook ("Feature Selection for Multiple Regression" cell).
FEATURES = [
    "Dining_Rating",
    "Delivery_Rating",
    "Dining_Votes",
    "Delivery_Votes",
    "Avg_Price_Restaurant",
]
TARGET = "Avg_Rating_Restaurant"
# Notebook: tree/forest models were trained on unscaled data, the rest on scaled data.
UNSCALED_MODELS = ["Decision Tree Regression", "Random Forest Regression"]

# Notebook split settings: train_test_split(test_size=0.2, random_state=42)
TEST_SIZE = 0.2
RANDOM_STATE = 42


# --------------------------------------------------------------------------- data
def load_raw_data() -> pd.DataFrame:
    """Load the CSV exactly as provided (no changes)."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def build_restaurant_df(raw: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the notebook's restaurant-level table.

    Notebook steps: drop duplicate rows, then keep one row per
    (Restaurant_Name, City). This is the table the models were trained on.
    """
    df = raw.drop_duplicates().copy()
    return df.drop_duplicates(subset=["Restaurant_Name", "City"]).reset_index(drop=True)


# ------------------------------------------------------------------------- models
def load_artifacts():
    """Load the saved model, scaler and model name.

    Returns (artifacts_dict, errors_list). Missing/broken files are reported,
    never replaced by anything invented.
    """
    artifacts, errors, version_warnings = {}, [], []
    for key, path in [("model", MODEL_PATH), ("scaler", SCALER_PATH), ("name", MODEL_NAME_PATH)]:
        if not path.exists():
            errors.append(f"Missing file: backend/{path.name}")
            continue
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                artifacts[key] = joblib.load(path)
            version_warnings += [str(w.message).split(". Use at")[0] for w in caught
                                 if "version" in str(w.message).lower()]
        except Exception as exc:  # noqa: BLE001 - show any load problem to the user
            errors.append(f"Could not load backend/{path.name}: {exc}")
    artifacts["version_warnings"] = sorted(set(version_warnings))

    scaler = artifacts.get("scaler")
    model = artifacts.get("model")
    if scaler is not None and hasattr(scaler, "feature_names_in_"):
        if list(scaler.feature_names_in_) != FEATURES:
            errors.append(
                f"Scaler was fitted on {list(scaler.feature_names_in_)}, "
                f"which differs from the notebook features {FEATURES}."
            )
    if model is not None and getattr(model, "n_features_in_", len(FEATURES)) != len(FEATURES):
        errors.append(f"Model expects {model.n_features_in_} features, not {len(FEATURES)}.")
    return artifacts, errors


def predict_rating(artifacts: dict, values: dict) -> float:
    """Predict with the saved model using the notebook's exact preprocessing."""
    missing = [f for f in FEATURES if f not in values]
    if missing:
        raise ValueError(f"Missing inputs: {missing}")
    row = pd.DataFrame([{f: float(values[f]) for f in FEATURES}], columns=FEATURES)
    if artifacts["name"] in UNSCALED_MODELS:
        x = row
    else:
        x = artifacts["scaler"].transform(row)
    return float(artifacts["model"].predict(x)[0])


def validate_inputs(values: dict, reference: pd.DataFrame):
    """Return (errors, warnings) for the prediction form."""
    errors, warns = [], []
    for col in ("Dining_Rating", "Delivery_Rating"):
        v = values[col]
        if v is None or not np.isfinite(v) or not (0 <= v <= 5):
            errors.append(f"{col.replace('_', ' ')} must be between 0 and 5.")
    for col in ("Dining_Votes", "Delivery_Votes"):
        v = values[col]
        if v is None or not np.isfinite(v) or v < 0:
            errors.append(f"{col.replace('_', ' ')} cannot be negative.")
    p = values["Avg_Price_Restaurant"]
    if p is None or not np.isfinite(p) or p <= 0:
        errors.append("Average price must be greater than 0.")
    if not errors:
        for col in FEATURES:
            lo, hi = reference[col].min(), reference[col].max()
            if values[col] < lo or values[col] > hi:
                warns.append(
                    f"{col.replace('_', ' ')} = {values[col]:g} is outside the range the model "
                    f"was trained on ({lo:g} to {hi:g}); the prediction is an extrapolation."
                )
    return errors, warns


# ------------------------------------------------------------------------ metrics
def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mse = float(np.mean((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "R2": 1 - float(np.sum((y_true - y_pred) ** 2)) / ss_tot if ss_tot else float("nan"),
        "MAE": float(np.mean(np.abs(y_true - y_pred))),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
    }


def notebook_saved_metrics():
    """Parse the model-comparison table printed in the notebook's saved output.

    Returns a DataFrame (Model, R2, MAE, MSE, RMSE) or None if it is not found.
    These numbers are read from the notebook, not calculated here.
    """
    if not NOTEBOOK_PATH.exists():
        return None
    try:
        nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            text = "".join(out.get("text", []))
            if "Model Comparison" not in text:
                continue
            rows = []
            for line in text.splitlines():
                m = re.match(r"\s*(.+?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", line)
                if m and "Regression" in m.group(1):
                    mae, mse, rmse, r2 = map(float, m.groups()[1:])
                    rows.append({"Model": m.group(1), "R2": r2, "MAE": mae, "MSE": mse, "RMSE": rmse})
            if rows:
                return pd.DataFrame(rows)
    return None


def recompute_saved_model_metrics(artifacts: dict, restaurant_df: pd.DataFrame):
    """Evaluate the SAVED model on the notebook's test split (no retraining).

    Uses the notebook's split: test_size=0.2, random_state=42, rows with
    missing values dropped. The saved scaler was fitted on the training part
    of this split, so it is applied as-is.
    """
    from sklearn.model_selection import train_test_split  # local import: only needed here

    data = restaurant_df[FEATURES + [TARGET]].dropna()
    _, X_test, _, y_test = train_test_split(
        data[FEATURES], data[TARGET], test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    x = X_test if artifacts["name"] in UNSCALED_MODELS else artifacts["scaler"].transform(X_test)
    pred = artifacts["model"].predict(x)
    return regression_metrics(y_test, pred), len(X_test), y_test.to_numpy(), pred
