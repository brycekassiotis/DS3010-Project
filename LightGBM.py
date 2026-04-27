import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

# load each split
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')
X_val = pd.read_csv('X_val.csv')
y_val = pd.read_csv('y_val.csv')
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')

# drop non-numeric columns (Country Name, Year)
X_train = X_train.drop(columns=['Country Name', 'Year'])
X_val = X_val.drop(columns=['Country Name', 'Year'])
X_test = X_test.drop(columns=['Country Name', 'Year'])

# remove special characters from column names
X_train.columns = X_train.columns.str.replace(r'[^\w\s]', '_', regex=True)
X_val.columns = X_val.columns.str.replace(r'[^\w\s]', '_', regex=True)
X_test.columns = X_test.columns.str.replace(r'[^\w\s]', '_', regex=True)

# train model using AI suggested hyperparameters
lgbm = lgb.LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    num_leaves=31,
    random_state=42
)
lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)])

# evaluate on validation set
y_val_pred = lgbm.predict(X_val)
mse_val = mean_squared_error(y_val, y_val_pred)
mae_val = mean_absolute_error(y_val, y_val_pred)
r2_val = r2_score(y_val, y_val_pred)

print(f'Validation MSE: {mse_val:.4f}')
print(f'Validation MAE: {mae_val:.4f}')
print(f'Validation R^2: {r2_val:.4f}')

# evaluate on test set
y_test_pred = lgbm.predict(X_test)
mse_test = mean_squared_error(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f'Test MSE: {mse_test:.4f}')
print(f'Test MAE: {mae_test:.4f}')
print(f'Test R^2: {r2_test:.4f}')

# find feature importance
importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': lgbm.feature_importances_
}).sort_values(by='importance', ascending=False)

print(importances)


# RESULTS

# Validation MSE: 652209.6857
# Validation MAE: 209.7356
# Validation R^2: 0.4642
# Test MSE: 43535.0095
# Test MAE: 95.7107
# Test R^2: 0.7027
#                                               feature  importance
# 11                                   Urban population         559
# 3   Electricity production from coal sources __ of...         157
# 10  Net trade in goods and services _BoP_ current ...         123
# 2         Electric power consumption _kWh per capita_          97
# 0             Access to electricity __ of population_          82
# 7               Energy imports_ net __ of energy use_          59
# 9                                  GDP _constant LCU_          57
# 8         Fossil fuel energy consumption __ of total_          45
# 1   Agriculture_ forestry_ and fishing_ value adde...          43
# 5   Electricity production from natural gas source...          42
# 4   Electricity production from hydroelectric sour...          22
# 6   Electricity production from nuclear sources __...          18