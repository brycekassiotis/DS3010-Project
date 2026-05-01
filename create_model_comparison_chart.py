import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


data = pd.DataFrame(
    [
        {"Model": "LightGBM", "R2": 0.9565, "MAE": 0.5396, "RMSE": 0.8690},
        {"Model": "Gradient Boosting", "R2": 0.9479, "MAE": 0.6027, "RMSE": 0.9507},
        {"Model": "XGBoost", "R2": 0.9398, "MAE": 0.6051, "RMSE": 1.0221},
        {"Model": "Random Forest", "R2": 0.8888, "MAE": 0.7960, "RMSE": 1.3900},
        {"Model": "Ridge", "R2": 0.7294, "MAE": 1.6053, "RMSE": 2.1677},
    ]
)

metric_specs = [
    ("R2", "R2 Comparison", "#2E86DE"),
    ("MAE", "MAE Comparison", "#E74C3C"),
    ("RMSE", "RMSE Comparison", "#8E44AD"),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 8))
fig.patch.set_facecolor("white")

for ax, (metric, title, color) in zip(axes, metric_specs):
    ranked = data.sort_values(metric, ascending=(metric != "R2"))
    bars = ax.barh(ranked["Model"], ranked[metric], color=color, alpha=0.9)
    ax.set_title(title, fontsize=18, weight="bold")
    ax.set_xlabel(metric, fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.invert_yaxis()

    max_value = ranked[metric].max()
    for bar, value in zip(bars, ranked[metric]):
        ax.text(
            value + (max_value * 0.02),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            fontsize=11,
        )

fig.suptitle("Model Performance Comparison", fontsize=24, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("model_comparison_chart.png", dpi=300, bbox_inches="tight", facecolor="white")
