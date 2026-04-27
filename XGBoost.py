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
# RMSE: 1.4660
# MAE:  1.0537
# R²:   0.9489

# === Test Performance ===
# RMSE: 0.9938
# MAE:  0.5773
# R²:   0.9431

# === Feature Importances ===
#                                                            Feature  Importance
#                        Electric power consumption (kWh per capita)    0.319712
#       Electricity production from natural gas sources (% of total)    0.146261
#         Agriculture, forestry, and fishing, value added (% of GDP)    0.118649
#          Industry (including construction), value added (% of GDP)    0.116413
#                              Energy imports, net (% of energy use)    0.079632
# Renewable energy consumption (% of total final energy consumption)    0.078530
#           Electricity production from nuclear sources (% of total)    0.052556
#                           Urban population (% of total population)    0.028094
#     Electricity production from hydroelectric sources (% of total)    0.022090
#                                                   Trade (% of GDP)    0.019814
#                        Fossil fuel energy consumption (% of total)    0.018247
