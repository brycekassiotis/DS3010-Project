import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

# scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# find optimal k
k_results = []
for k in range(1, 21):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train_scaled, y_train.values.ravel())
    val_r2 = r2_score(y_val, knn.predict(X_val_scaled))
    k_results.append({'k': k, 'val_r2': val_r2})

k_results_df = pd.DataFrame(k_results)
best_k = k_results_df.loc[k_results_df['val_r2'].idxmax(), 'k']
print(f'Best k: {best_k}')
print(k_results_df.to_string(index=False))

# train final model with best k
knn = KNeighborsRegressor(n_neighbors=best_k)
knn.fit(X_train_scaled, y_train.values.ravel())

# evaluate on validation set
y_val_pred = knn.predict(X_val_scaled)
print(f'KNN Regressor Validation Set Performance:')
print(f'RMSE: {np.sqrt(mean_squared_error(y_val, y_val_pred)):.4f}')
print(f'MAE: {mean_absolute_error(y_val, y_val_pred):.4f}')
print(f'R^2: {r2_score(y_val, y_val_pred):.4f}')

# evaluate on test set
y_test_pred = knn.predict(X_test_scaled)
print(f'KNN Regressor Test Set Performance:')
print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}')
print(f'MAE: {mean_absolute_error(y_test, y_test_pred):.4f}')
print(f'R^2: {r2_score(y_test, y_test_pred):.4f}')


# RESULTS

# KNN Regressor Validation Set Performance:
# RMSE: 2.8113
# MAE: 2.0451
# R^2: 0.8121
# KNN Regressor Test Set Performance:
# RMSE: 2.4821
# MAE: 1.4257
# R^2: 0.6452