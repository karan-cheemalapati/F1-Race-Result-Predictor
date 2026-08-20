import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score
)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb

DATA_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
MODELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')

# ── features we feed into the model ──────────────────────────────────────────
FEATURE_COLS = [
    'grid',
    'driver_age',
    'driver_experience',
    'driver_standing_pts',
    'driver_standing_pos',
    'driver_wins_so_far',
    'constructor_standing_pts',
    'constructor_standing_pos',
    'constructor_wins_so_far',
    'quali_gap_to_pole',
    'quali_position',
    'circuit_win_rate',
    'circuit_podium_rate',
    'circuit_avg_finish',
    'circuit_races_count',
    'driver_avg_finish_last3',
    'driver_avg_finish_last5',
    'constructor_avg_pts_last3',
    'constructor_avg_pts_last5',
    'grid_vs_quali',
    'season_progress',
    'pts_gap_to_leader',
]

TARGET = 'podium'


def load_featured_data():
    """Load the feature-engineered dataset from Phase 2."""
    path = os.path.join(DATA_PATH, 'f1_features.csv')
    df = pd.read_csv(path)
    print(f"  Loaded f1_features.csv  →  {df.shape[0]:,} rows  {df.shape[1]} cols")
    return df


def prepare_data(df):
    """
    - Keep only rows where we have the core features
    - Use data from 1994+ (qualifying data available)
    - Split into train (pre-2020) and test (2020-2024)
    """
    # 1994+ because qualifying data starts there
    df = df[df['year'] >= 1994].copy()

    # drop rows missing critical features
    critical = ['grid', 'driver_standing_pts', 'constructor_standing_pts']
    df = df.dropna(subset=critical)

    # fill remaining nulls with median
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # temporal split — train on pre-2020, test on 2020-2024
    train_df = df[df['year'] < 2020]
    test_df  = df[df['year'] >= 2020]

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET]
    X_test  = test_df[FEATURE_COLS]
    y_test  = test_df[TARGET]

    print(f"\n  Train: {X_train.shape[0]:,} rows  ({train_df['year'].min()}–{train_df['year'].max()})")
    print(f"  Test:  {X_test.shape[0]:,} rows  ({test_df['year'].min()}–{test_df['year'].max()})")
    print(f"\n  Train podium rate: {y_train.mean():.1%}")
    print(f"  Test  podium rate: {y_test.mean():.1%}")

    return X_train, X_test, y_train, y_test, train_df, test_df


def train_random_forest(X_train, y_train):
    """Baseline Random Forest model."""
    print("\n  Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("  Done!")
    return model


def train_xgboost(X_train, y_train):
    """XGBoost classifier."""
    print("\n  Training XGBoost...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("  Done!")
    return model


def train_lightgbm(X_train, y_train):
    """LightGBM classifier."""
    print("\n  Training LightGBM...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(X_train, y_train)
    print("  Done!")
    return model


def evaluate_model(model, X_test, y_test, model_name):
    """Print evaluation metrics for a model."""
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    auc       = roc_auc_score(y_test, y_pred_prob)
    f1        = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)

    print(f"\n  {'='*40}")
    print(f"  {model_name}")
    print(f"  {'='*40}")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Podium', 'Podium']))

    return {
        'model_name': model_name,
        'roc_auc':    auc,
        'f1':         f1,
        'precision':  precision,
        'recall':     recall,
        'y_pred':     y_pred,
        'y_pred_prob': y_pred_prob
    }


def save_model(model, filename):
    """Save trained model to models/ folder."""
    os.makedirs(MODELS_PATH, exist_ok=True)
    path = os.path.join(MODELS_PATH, filename)
    joblib.dump(model, path)
    print(f"  Saved → models/{filename}")


if __name__ == '__main__':
    print("=== Phase 3: Model Training ===\n")

    df = load_featured_data()
    X_train, X_test, y_train, y_test, train_df, test_df = prepare_data(df)

    rf  = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)
    lgb_model = train_lightgbm(X_train, y_train)

    rf_results  = evaluate_model(rf,        X_test, y_test, 'Random Forest')
    xgb_results = evaluate_model(xgb_model, X_test, y_test, 'XGBoost')
    lgb_results = evaluate_model(lgb_model, X_test, y_test, 'LightGBM')

    save_model(rf,        'random_forest.pkl')
    save_model(xgb_model, 'xgboost.pkl')
    save_model(lgb_model, 'lightgbm.pkl')