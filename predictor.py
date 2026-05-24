# =========================================================
# predictor.py
# HYBRID: Exact dataset lookup first.
# If the input matches a row in the dataset exactly,
# returns the stored CL/CD/XCP values with zero difference.
# Falls back to RandomForest ML only for unseen inputs.
# =========================================================

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# =========================================================
# COLUMN MAPPING  –  GUI key  →  Dataset column name
# =========================================================
PARAM_TO_COL = {
    'nose_len'    : 'nose length',
    'body_len'    : 'body_length',
    'wing_le'     : 'wing LE',
    'root_chord'  : 'root chord',
    'tip_chord'   : 'tip chord',
    'semi_span'   : 'semi-span',
    'root_th'     : 'root th',
    'tip_th'      : 'tip th',
    'wing_sweep'  : 'wing sweep',
    'tail_le'     : 'tail LE',
    'root_chord1' : 'root chord.1',
    'tip_chord1'  : 'tip chord.1',
    'semi_span1'  : 'semi-span.1',
    'root_th1'    : 'root th.1',
    'tip_th1'     : 'tip th.1',
    'mach'        : 'MACH',
    'alpha'       : 'ALPHA',
    'alt'         : 'ALT',
}

INPUT_COLS = [
    'nose length', 'body_length', 'wing LE',
    'root chord',  'tip chord',   'semi-span',
    'root th',     'tip th',      'wing sweep',
    'tail LE',     'root chord.1','tip chord.1',
    'semi-span.1', 'root th.1',   'tip th.1',
    'MACH',        'ALPHA',       'ALT',
]

OUTPUT_COLS = ['CL', 'CD', 'X-C.P.']

# Tolerance for float matching (wing_sweep, MACH are floats)
# All integer columns match exactly; floats matched within 1e-9
FLOAT_TOL = 1e-9

# =========================================================
# FILE PATHS
# =========================================================
_HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_CSV   = os.path.join(_HERE, 'DRDL_aero_data_final.csv')
MODEL_PKL  = os.path.join(_HERE, 'aero_model.pkl')

# =========================================================
# CACHE  (populated once, reused every call)
# =========================================================
_cache = {}


def _load_or_train():
    """Load dataset + model into memory on first call."""

    if _cache:
        return _cache

    # ── Load raw dataset for exact lookup ─────────────────
    df = pd.read_csv(DATA_CSV)
    _cache['df'] = df

    # ── Try loading saved ML model ─────────────────────────
    if os.path.exists(MODEL_PKL):
        with open(MODEL_PKL, 'rb') as f:
            saved = pickle.load(f)
        _cache.update(saved)
        return _cache

    # ── Train ML model from scratch ────────────────────────
    X = df[INPUT_COLS].values
    y = df[OUTPUT_COLS].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    )
    model.fit(X_train_sc, y_train)

    y_pred = model.predict(X_test_sc)

    metrics = {}
    for i, col in enumerate(OUTPUT_COLS):
        metrics[col] = {
            'MAE'  : round(mean_absolute_error(y_test[:, i], y_pred[:, i]), 4),
            'RMSE' : round(float(np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))), 4),
            'R2'   : round(r2_score(y_test[:, i], y_pred[:, i]), 4),
        }

    payload = {
        'model'  : model,
        'scaler' : scaler,
        'metrics': metrics,
    }

    with open(MODEL_PKL, 'wb') as f:
        pickle.dump(payload, f)

    _cache.update(payload)
    return _cache


# =========================================================
# EXACT LOOKUP
# =========================================================

def _exact_lookup(params: dict, df: pd.DataFrame):
    """
    Try to find an exact matching row in the dataset.
    Returns (CL, CD, XCP) if found, else None.
    Integer columns matched exactly.
    Float columns (wing_sweep, MACH) matched within 1e-9.
    """

    mask = pd.Series([True] * len(df), index=df.index)

    for gui_key, col in PARAM_TO_COL.items():
        # Resolve value from params dict
        if gui_key in params:
            val = float(params[gui_key])
        elif col in params:
            val = float(params[col])
        else:
            return None   # missing key — can't do exact lookup

        col_vals = df[col].astype(float)

        # Float columns: use tolerance
        if df[col].dtype == float:
            mask &= (col_vals - val).abs() <= FLOAT_TOL
        else:
            # Integer-stored columns: round both sides before compare
            mask &= (col_vals.round(6) == round(val, 6))

    matches = df[mask]

    if len(matches) == 1:
        row = matches.iloc[0]
        return (
            round(float(row['CL']),      4),
            round(float(row['CD']),      4),
            round(float(row['X-C.P.']), 4),
        )

    if len(matches) > 1:
        # Multiple matches (shouldn't happen — dataset is unique)
        row = matches.iloc[0]
        return (
            round(float(row['CL']),      4),
            round(float(row['CD']),      4),
            round(float(row['X-C.P.']), 4),
        )

    return None   # no match found


# =========================================================
# ML FALLBACK
# =========================================================

def _ml_predict(params: dict, cache: dict):
    """Use the trained RandomForest for inputs not in dataset."""

    model  = cache['model']
    scaler = cache['scaler']

    row = []
    for col in INPUT_COLS:
        gui_key = next(
            (k for k, v in PARAM_TO_COL.items() if v == col), col
        )
        if col in params:
            row.append(float(params[col]))
        elif gui_key in params:
            row.append(float(params[gui_key]))
        else:
            row.append(0.0)

    X    = np.array(row).reshape(1, -1)
    X_sc = scaler.transform(X)
    pred = model.predict(X_sc)[0]

    return (
        round(float(pred[0]), 4),
        round(float(pred[1]), 4),
        round(float(pred[2]), 4),
    )


# =========================================================
# PUBLIC API
# =========================================================

def aerodynamic_prediction(params: dict) -> dict:
    """
    Predict CL, CD, XCP for given geometry + flight params.

    Strategy
    --------
    1. Search dataset for an exact match on all 18 inputs.
       If found  → return the stored CL/CD/XCP exactly (zero error).
    2. If not found → use RandomForest ML model to predict.

    Parameters
    ----------
    params : dict   GUI keys (nose_len, body_len … mach, alpha, alt)

    Returns
    -------
    dict:
        CL, CD, XCP   – aerodynamic coefficients
        source        – 'dataset' (exact) or 'ml_model' (predicted)
        metrics       – MAE, RMSE, R2 from ML test set
    """

    cache = _load_or_train()
    df    = cache['df']
    glob_metrics = cache.get('metrics', {})

    # ── Step 1: exact lookup ───────────────────────────────
    exact = _exact_lookup(params, df)

    if exact is not None:
        cl, cd, xcp = exact
        source = 'dataset'

    else:
        # ── Step 2: ML fallback ────────────────────────────
        cl, cd, xcp = _ml_predict(params, cache)
        source = 'ml_model'

    # ── Build metrics summary ──────────────────────────────
    if glob_metrics:
        metrics = {
            'MAE'   : round(sum(glob_metrics[c]['MAE']  for c in OUTPUT_COLS) / 3, 4),
            'RMSE'  : round(sum(glob_metrics[c]['RMSE'] for c in OUTPUT_COLS) / 3, 4),
            'R2'    : round(sum(glob_metrics[c]['R2']   for c in OUTPUT_COLS) / 3, 4),
            'detail': glob_metrics,
        }
    else:
        metrics = {'MAE': 'N/A', 'RMSE': 'N/A', 'R2': 'N/A'}

    return {
        'CL'     : cl,
        'CD'     : cd,
        'XCP'    : xcp,
        'source' : source,
        'metrics': metrics,
    }