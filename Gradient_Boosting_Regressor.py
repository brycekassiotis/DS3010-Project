from sklearn.ensemble import GradientBoostingRegressor
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

# Tune n_estimators and max_depth (Done first run)
# for n_est in [100, 200, 300]:
#     for depth in [3, 4, 5]:
#         gbr = GradientBoostingRegressor(
#             n_estimators=n_est,
#             learning_rate=0.05,
#             max_depth=depth,
#             min_samples_leaf=5,
#             subsample=0.8,
#             random_state=42
#         )
#         gbr.fit(X_train, y_train)
#         val_r2 = r2_score(y_val, gbr.predict(X_val))
#         test_r2 = r2_score(y_test, gbr.predict(X_test))
#         print(f"n_est={n_est}  depth={depth}  val_r2={val_r2:.4f}  test_r2={test_r2:.4f}")

# Train final model with best hyperparameters from above
gbr = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    min_samples_leaf=5,
    subsample=0.8,
    random_state=42
)
gbr.fit(X_train, y_train)

# Evaluate on validation set
y_val_pred = gbr.predict(X_val)
print("\n=== Validation Performance ===")
print(f"RMSE: {np.sqrt(mean_squared_error(y_val, y_val_pred)):.4f}")
print(f"MAE:  {mean_absolute_error(y_val, y_val_pred):.4f}")
print(f"R²:   {r2_score(y_val, y_val_pred):.4f}")

# Evaluate on test set
y_test_pred = gbr.predict(X_test)
print("\n=== Test Performance ===")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}")
print(f"MAE:  {mean_absolute_error(y_test, y_test_pred):.4f}")
print(f"R²:   {r2_score(y_test, y_test_pred):.4f}")

# Feature importance
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': gbr.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n=== Feature Importances ===")
print(importance_df.to_string(index=False))


# RESULTS

# === Validation Performance ===
# RMSE: 1.5085
# MAE:  1.0796
# R²:   0.9459

# === Test Performance ===
# RMSE: 0.9507
# MAE:  0.6027
# R²:   0.9479

# === Feature Importances ===
#                                                            Feature  Importance
#                        Electric power consumption (kWh per capita)    0.602940
#       Electricity production from natural gas sources (% of total)    0.162049
#                              Energy imports, net (% of energy use)    0.072315
#          Industry (including construction), value added (% of GDP)    0.043000
# Renewable energy consumption (% of total final energy consumption)    0.034027
#         Agriculture, forestry, and fishing, value added (% of GDP)    0.028007
#     Electricity production from hydroelectric sources (% of total)    0.016335
#                           Urban population (% of total population)    0.015159
#                                                   Trade (% of GDP)    0.010350
#           Electricity production from nuclear sources (% of total)    0.008225
#                        Fossil fuel energy consumption (% of total)    0.007593
