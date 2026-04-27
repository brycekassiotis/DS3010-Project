import pickle
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load splits
X_train = pd.read_csv('X_train.csv').drop(columns=['Country Name', 'Year'])
y_train = pd.read_csv('y_train.csv').squeeze()

# Train and save each model
models = {
    'lgbm_model': lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, num_leaves=31, random_state=42, verbose=-1),
    'xgb_model': xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, min_child_weight=13, subsample=0.8, colsample_bytree=0.8, random_state=42),
    'gbr_model': GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, min_samples_leaf=5, subsample=0.8, random_state=42),
    'rf_model': RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=10, n_jobs=-1, random_state=42),
    'ridge_model': RidgeCV(alphas=np.logspace(-3, 6, 100), cv=5),
}

for name, model in models.items():
    # fix column names for lgbm/xgb
    X = X_train.copy()
    X.columns = X.columns.str.replace(r'[^\w\s]', '_', regex=True)
    model.fit(X, y_train)
    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f'Saved {name}.pkl')

from sklearn.metrics import r2_score

X_val = pd.read_csv('X_val.csv').drop(columns=['Country Name', 'Year'])
y_val = pd.read_csv('y_val.csv').squeeze()
X_test = pd.read_csv('X_test.csv').drop(columns=['Country Name', 'Year'])
y_test = pd.read_csv('y_test.csv').squeeze()

for name, model in models.items():
    X = X_train.copy()
    X_v = X_val.copy()
    X_te = X_test.copy()
    X.columns = X.columns.str.replace(r'[^\w\s]', '_', regex=True)
    X_v.columns = X_v.columns.str.replace(r'[^\w\s]', '_', regex=True)
    X_te.columns = X_te.columns.str.replace(r'[^\w\s]', '_', regex=True)
    
    model.fit(X, y_train)
    
    train_r2 = r2_score(y_train, model.predict(X))
    val_r2 = r2_score(y_val, model.predict(X_v))
    test_r2 = r2_score(y_test, model.predict(X_te))
    
    print(f"{name:15s}  train_r2={train_r2:.4f}  val_r2={val_r2:.4f}  test_r2={test_r2:.4f}")
    
    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved {name}.pkl")