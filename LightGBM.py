import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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

# actual vs predicted plot for the test set
y_test_actual = y_test.squeeze()
plot_df = pd.DataFrame({
    'Actual': y_test_actual,
    'Predicted': y_test_pred
})

min_value = min(plot_df['Actual'].min(), plot_df['Predicted'].min())
max_value = max(plot_df['Actual'].max(), plot_df['Predicted'].max())

fig = px.scatter(
    plot_df,
    x='Actual',
    y='Predicted',
    title='LightGBM: Actual vs Predicted',
    labels={'Actual': 'Actual Values', 'Predicted': 'Predicted Values'},
    color_discrete_sequence=['blue']
)
fig.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    width=800,
    height=800,
    font=dict(size=18),
    title_font=dict(size=24)
)
fig.update_xaxes(showgrid=False, title_font=dict(size=20), tickfont=dict(size=16))
fig.update_yaxes(showgrid=False, title_font=dict(size=20), tickfont=dict(size=16))
fig.add_trace(
    go.Scatter(
        x=[min_value, max_value],
        y=[min_value, max_value],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='black', dash='dash')
    )
)
fig.show()

# find feature importance
importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': lgbm.feature_importances_
}).sort_values(by='importance', ascending=False)

print(importances)


# RESULTS

# Validation MSE: 2.2056
# Validation MAE: 1.0881
# Validation R^2: 0.9476
# Test MSE: 0.7551
# Test MAE: 0.5396
# Test R^2: 0.9565
#                                               feature  importance
# 1         Electric power consumption _kWh per capita_         339
# 3   Electricity production from natural gas source...         250
# 8   Renewable energy consumption __ of total final...         146
# 5               Energy imports_ net __ of energy use_         143
# 7   Industry _including construction__ value added...         131
# 2   Electricity production from hydroelectric sour...         126
# 10           Urban population __ of total population_         125
# 0   Agriculture_ forestry_ and fishing_ value adde...         120
# 6         Fossil fuel energy consumption __ of total_         112
# 4   Electricity production from nuclear sources __...          38
# 9                                    Trade __ of GDP_          30
