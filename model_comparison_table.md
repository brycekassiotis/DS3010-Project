# Model Comparison Table

Test-set metrics for the requested models:

| Model | R2 | MAE | RMSE |
| --- | ---: | ---: | ---: |
| XGBoost | 0.9398 | 0.6051 | 1.0221 |
| Gradient Boosting | 0.9479 | 0.6027 | 0.9507 |
| LightGBM | 0.9565 | 0.5396 | 0.8690 |
| Random Forest | 0.8888 | 0.7960 | 1.3900 |
| Ridge | 0.7294 | 1.6053 | 2.1677 |

LightGBM RMSE was derived from its reported test MSE of `0.7551` as `sqrt(0.7551) = 0.8690`.
