"""Quick tabular baseline for CIMMYT survey data.

Trains a RandomForestRegressor to predict `yield_kg_ha` by default.
Usage:
    python scripts/baseline_tabular.py --input cimmyt_data/NUE_survey_dataset_v2.csv
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib


def load_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def prepare_features(df: pd.DataFrame, target: str = 'yield_kg_ha'):
    # Attempt to coerce useful numeric columns. Many columns are mixed-type
    # in the CSV, so we convert each column to numeric when possible.
    candidates = []
    df_local = df.copy()
    for col in df_local.columns:
        if col == target:
            continue
        coerced = pd.to_numeric(df_local[col], errors='coerce')
        # retain column if at least one numeric value exists
        if coerced.notna().sum() > 0:
            df_local[col] = coerced
            candidates.append(col)

    # drop identifier-like columns from features
    for drop in ['Merged_ID', 'ID', 'index', 'Region', 'Dataset']:
        if drop in candidates:
            candidates.remove(drop)

    X = df_local[candidates].copy()
    y = pd.to_numeric(df_local[target], errors='coerce')

    # remove rows where target is missing/inf
    mask = y.replace([np.inf, -np.inf], np.nan).notna()
    removed = (~mask).sum()
    if removed > 0:
        print(f'Removing {removed} rows with invalid target values')
    X = X[mask]
    y = y[mask]

    # replace infinities and drop columns that are entirely non-numeric
    X = X.replace([np.inf, -np.inf], np.nan)
    med = X.median()
    keep_cols = med[med.notna()].index.tolist()
    if len(keep_cols) < X.shape[1]:
        dropped = set(X.columns) - set(keep_cols)
        print(f'Dropping {len(dropped)} non-numeric feature columns: {list(dropped)[:10]}')
    X = X[keep_cols]

    # simple imputation: median (now med has values for keep_cols)
    X = X.fillna(med[keep_cols])
    y = y.replace([np.inf, -np.inf], np.nan)
    y = y.fillna(y.median())

    return X, y


def run_baseline(input_csv: str, out_dir: str, target: str = 'yield_kg_ha'):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_table(input_csv)
    X, y = prepare_features(df, target=target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = r2_score(y_test, preds)

    metrics = {'rmse': float(rmse), 'r2': float(r2), 'n_train': len(X_train), 'n_test': len(X_test)}

    model_path = out_dir / 'baseline_rf.joblib'
    joblib.dump({'model': model, 'features': X.columns.tolist(), 'metrics': metrics}, model_path)

    print('Saved model to', model_path)
    print('Metrics:', metrics)
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='cimmyt_data/NUE_survey_dataset_v2.csv')
    p.add_argument('--out', default='outputs/baseline')
    p.add_argument('--target', default='yield_kg_ha')
    args = p.parse_args()

    run_baseline(args.input, args.out, target=args.target)


if __name__ == '__main__':
    main()
