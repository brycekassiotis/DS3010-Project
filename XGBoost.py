import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd

# Load splits
X_train = pd.read_csv('X_train.csv')
X_val = pd.read_csv('X_val.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').squeeze()
y_val = pd.read_csv('y_val.csv').squeeze()
y_test = pd.read_csv('y_test.csv').squeeze()

# Drop non-numeric columns
X_train = X_train.drop(columns=['Country Name', 'Year'])
X_val = X_val.drop(columns=['Country Name', 'Year'])
X_test = X_test.drop(columns=['Country Name', 'Year'])

# Train model
for mcw in range(1, 20):
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=mcw,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='rmse'
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    val_r2 = r2_score(y_val, xgb_model.predict(X_val))
    test_r2 = r2_score(y_test, xgb_model.predict(X_test))
    print(f"min_child_weight={mcw:3d}  val_r2={val_r2:.4f}  test_r2={test_r2:.4f}")

# Evaluate on validation set
y_val_pred = xgb_model.predict(X_val)
print("=== Validation Performance ===")
print(f"RMSE: {np.sqrt(mean_squared_error(y_val, y_val_pred)):.4f}")
print(f"MAE:  {mean_absolute_error(y_val, y_val_pred):.4f}")
print(f"R²:   {r2_score(y_val, y_val_pred):.4f}")

# Evaluate on test set
y_test_pred = xgb_model.predict(X_test)
print("\n=== Test Performance ===")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}")
print(f"MAE:  {mean_absolute_error(y_test, y_test_pred):.4f}")
print(f"R²:   {r2_score(y_test, y_test_pred):.4f}")

# Feature importance
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n=== Feature Importances ===")
print(importance_df.to_string(index=False))


# RESULTS

# === Validation Performance ===
# RMSE: 61.8699
# MAE:  43.6228
# R²:   0.7702

# === Test Performance ===
# RMSE: 1549.7768
# MAE:  558.2240
# R²:   -0.9033

# === Feature Importances ===
#                                                        Feature  Importance
#                                               Urban population    0.483446
#             Net trade in goods and services (BoP, current US$)    0.141509
#     Agriculture, forestry, and fishing, value added (% of GDP)    0.126458
#                                             GDP (constant LCU)    0.090617
#       Electricity production from nuclear sources (% of total)    0.049572
#          Electricity production from coal sources (% of total)    0.037180
#                        Access to electricity (% of population)    0.017689
#                    Electric power consumption (kWh per capita)    0.016340
# Electricity production from hydroelectric sources (% of total)    0.011222
#                    Fossil fuel energy consumption (% of total)    0.010473
#                          Energy imports, net (% of energy use)    0.008613
#   Electricity production from natural gas sources (% of total)    0.006880
# PS C:\Users\bryce_8eiabba\DS3010-Project> 