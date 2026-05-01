import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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

# Tune on validation only so the test set remains a strict final check.
best_mcw = None
best_val_r2 = float('-inf')
best_model = None

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
    print(f"min_child_weight={mcw:3d}  val_r2={val_r2:.4f}")

    if val_r2 > best_val_r2:
        best_val_r2 = val_r2
        best_mcw = mcw
        best_model = xgb_model

print(f"\nSelected min_child_weight={best_mcw} based on validation R²={best_val_r2:.4f}")

# Evaluate on validation set
y_val_pred = best_model.predict(X_val)
print("=== Validation Performance ===")
print(f"RMSE: {np.sqrt(mean_squared_error(y_val, y_val_pred)):.4f}")
print(f"MAE:  {mean_absolute_error(y_val, y_val_pred):.4f}")
print(f"R²:   {r2_score(y_val, y_val_pred):.4f}")

# Evaluate on test set once, after tuning is complete
y_test_pred = best_model.predict(X_test)
print("\n=== Test Performance ===")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}")
print(f"MAE:  {mean_absolute_error(y_test, y_test_pred):.4f}")
print(f"R²:   {r2_score(y_test, y_test_pred):.4f}")

# Actual vs predicted plot for the test set
plot_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_test_pred
})

min_value = min(plot_df['Actual'].min(), plot_df['Predicted'].min())
max_value = max(plot_df['Actual'].max(), plot_df['Predicted'].max())

fig = px.scatter(
    plot_df,
    x='Actual',
    y='Predicted',
    title='XGBoost: Actual vs Predicted',
    labels={'Actual': 'Actual Values', 'Predicted': 'Predicted Values'},
    color_discrete_sequence=['purple']
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

# Feature importance
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n=== Feature Importances ===")
print(importance_df.to_string(index=False))


# RESULTS

# === Validation Performance ===
# RMSE: 1.3632
# MAE:  1.0054
# R²:   0.9558

# === Test Performance ===
# RMSE: 1.0221
# MAE:  0.6051
# R²:   0.9398

# === Feature Importances ===
#                                                            Feature  Importance
#                        Electric power consumption (kWh per capita)    0.345826
#       Electricity production from natural gas sources (% of total)    0.148123
#         Agriculture, forestry, and fishing, value added (% of GDP)    0.120605
#          Industry (including construction), value added (% of GDP)    0.096680
#                              Energy imports, net (% of energy use)    0.072666
# Renewable energy consumption (% of total final energy consumption)    0.065631
#           Electricity production from nuclear sources (% of total)    0.056313
#                           Urban population (% of total population)    0.032127
#     Electricity production from hydroelectric sources (% of total)    0.025685
#                        Fossil fuel energy consumption (% of total)    0.018692
#                                                   Trade (% of GDP)    0.017651
