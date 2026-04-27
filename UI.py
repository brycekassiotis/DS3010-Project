import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="CO2 Emissions Predictor",
    page_icon=":earth_americas:",
    layout="wide",
)


MODEL_FILES = {
    "LightGBM": "lgbm_model.pkl",
    "XGBoost": "xgb_model.pkl",
    "Gradient Boosting": "gbr_model.pkl",
    "Random Forest": "rf_model.pkl",
    "Ridge": "ridge_model.pkl",
}

MODEL_R2_RANKING = {
    "Gradient Boosting": 0.947,
    "LightGBM": 0.943,
    "XGBoost": 0.943,
    "Random Forest": 0.890,
    "Ridge": 0.874,
}

BEST_MODEL_METRICS = {
    "name": "LightGBM",
    "r2": 0.957,
    "mae": 0.540,
    "rmse": 0.869,
}

GBR_IMPORTANCES = {
    "Electric power consumption (kWh per capita)": 0.602940,
    "Electricity production from natural gas sources (% of total)": 0.162049,
    "Energy imports, net (% of energy use)": 0.072315,
    "Industry (including construction), value added (% of GDP)": 0.043000,
    "Renewable energy consumption (% of total final energy consumption)": 0.034027,
    "Agriculture, forestry, and fishing, value added (% of GDP)": 0.028007,
    "Electricity production from hydroelectric sources (% of total)": 0.016335,
    "Urban population (% of total population)": 0.015159,
    "Trade (% of GDP)": 0.010350,
    "Electricity production from nuclear sources (% of total)": 0.008225,
    "Fossil fuel energy consumption (% of total)": 0.007593,
}


def clean_feature_names(columns):
    return pd.Index(columns).str.replace(r"[^\w\s]", "_", regex=True)


@st.cache_data
def load_feature_columns():
    train_df = pd.read_csv("X_train.csv")
    return train_df.drop(columns=["Country Name", "Year"]).columns.tolist()


@st.cache_data
def load_data():
    data = pd.read_csv("pre_split.csv")
    feature_columns = load_feature_columns()
    feature_means = data[feature_columns].mean(numeric_only=True)
    feature_ranges = {}

    for feature in feature_columns:
        series = pd.to_numeric(data[feature], errors="coerce")
        valid = series.dropna()
        mean_value = float(feature_means[feature])

        if valid.empty:
            min_value = max(mean_value - 1.0, 0.0)
            max_value = mean_value + 1.0
        else:
            min_value = float(valid.min())
            max_value = float(valid.max())
            if np.isclose(min_value, max_value):
                padding = max(abs(min_value) * 0.1, 1.0)
                min_value -= padding
                max_value += padding

        feature_ranges[feature] = {
            "min": min_value,
            "max": max_value,
            "mean": mean_value,
        }

    return data, feature_columns, feature_means, feature_ranges


@st.cache_resource
def load_models():
    models = {}
    for name, file_name in MODEL_FILES.items():
        with Path(file_name).open("rb") as file:
            model = pickle.load(file)
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
        models[name] = model
    return models


def get_row_for_selection(data, country, year):
    selected = data[(data["Country Name"] == country) & (data["Year"] == year)]
    if selected.empty:
        return None
    return selected.iloc[0]


def get_feature_defaults(row, feature_columns, feature_means):
    defaults = {}
    for feature in feature_columns:
        value = feature_means[feature] if row is None else row.get(feature, feature_means[feature])
        if pd.isna(value):
            value = feature_means[feature]
        defaults[feature] = float(value)
    return defaults


def sync_feature_state(defaults):
    for feature, value in defaults.items():
        st.session_state[f"feature::{feature}"] = float(value)


def build_prediction_frame(feature_values, feature_columns):
    ordered = [feature_values[feature] for feature in feature_columns]
    frame = pd.DataFrame([ordered], columns=feature_columns)
    frame.columns = clean_feature_names(frame.columns)
    return frame


def predict_value(model, feature_values, feature_columns):
    prediction_frame = build_prediction_frame(feature_values, feature_columns)
    prediction = float(model.predict(prediction_frame)[0])
    return max(prediction, 0.0)


def get_model_feature_order(model_name, model, feature_columns):
    cleaned = clean_feature_names(feature_columns).tolist()

    if hasattr(model, "feature_importances_"):
        scores = pd.Series(model.feature_importances_, index=cleaned)
    elif hasattr(model, "coef_"):
        scores = pd.Series(np.abs(np.ravel(model.coef_)), index=cleaned)
    else:
        fallback = {feature: len(feature_columns) - idx for idx, feature in enumerate(feature_columns)}
        return sorted(feature_columns, key=lambda feature: fallback[feature], reverse=True)

    mapping = dict(zip(cleaned, feature_columns))
    ranked_cleaned = scores.sort_values(ascending=False).index.tolist()
    return [mapping[name] for name in ranked_cleaned if name in mapping]


def prediction_badge(prediction):
    if prediction < 2:
        return "badge-green", "Low"
    if prediction <= 8:
        return "badge-yellow", "Moderate"
    return "badge-red", "High"


def build_distribution_chart(data, prediction, country_name):
    global_min = float(data["carbon_emissions"].min())
    global_max = float(data["carbon_emissions"].max())

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[global_min, global_max],
            y=[0, 0],
            mode="lines",
            line=dict(color="#334155", width=12),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[prediction],
            y=[0],
            mode="markers+text",
            marker=dict(size=18, color="#38bdf8", line=dict(color="#f8fafc", width=2)),
            text=[country_name],
            textposition="top center",
            showlegend=False,
            hovertemplate="Prediction: %{x:.2f} t<extra></extra>",
        )
    )
    fig.update_layout(
        title="Prediction Within Global CO2 Distribution",
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
        font=dict(color="#e2e8f0"),
        xaxis=dict(title="CO2 per capita (t)", range=[global_min, global_max], zeroline=False),
        yaxis=dict(visible=False),
        height=180,
        margin=dict(l=10, r=10, t=55, b=10),
    )
    return fig


def build_model_ranking_chart():
    ranking_df = (
        pd.DataFrame(
            {
                "Model": list(MODEL_R2_RANKING.keys()),
                "Test R2": list(MODEL_R2_RANKING.values()),
            }
        )
        .sort_values("Test R2", ascending=True)
    )

    fig = go.Figure(
        go.Bar(
            x=ranking_df["Test R2"],
            y=ranking_df["Model"],
            orientation="h",
            marker=dict(
                color=["#475569", "#475569", "#475569", "#64748b", "#38bdf8"],
                line=dict(color="#94a3b8", width=0),
            ),
            text=[f"{value:.3f}" for value in ranking_df["Test R2"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Model Ranking by Test R²",
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
        font=dict(color="#e2e8f0"),
        xaxis=dict(title="Test R²", range=[0.82, 0.97]),
        yaxis_title="",
        height=340,
        margin=dict(l=10, r=30, t=55, b=10),
    )
    return fig


def build_gbr_importance_chart():
    importance_df = (
        pd.DataFrame(
            {
                "Feature": list(GBR_IMPORTANCES.keys()),
                "Importance": list(GBR_IMPORTANCES.values()),
            }
        )
        .sort_values("Importance", ascending=True)
    )

    fig = go.Figure(
        go.Bar(
            x=importance_df["Importance"],
            y=importance_df["Feature"],
            orientation="h",
            marker_color="#38bdf8",
            text=[f"{value:.3f}" for value in importance_df["Importance"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Gradient Boosting Feature Importance",
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
        font=dict(color="#e2e8f0"),
        xaxis_title="Importance",
        yaxis_title="",
        height=460,
        margin=dict(l=10, r=30, t=55, b=10),
    )
    return fig


def build_prediction_table(year_df, model, feature_columns, feature_means):
    if year_df.empty:
        return pd.DataFrame(
            columns=[
                "Country",
                "Actual CO2 per capita",
                "Predicted CO2 per capita",
                "Delta",
            ]
        )

    frame = year_df.copy()
    features = frame[feature_columns].fillna(feature_means)
    features.columns = clean_feature_names(features.columns)
    predictions = np.clip(model.predict(features), 0, None)

    table = pd.DataFrame(
        {
            "Country": frame["Country Name"].values,
            "Actual CO2 per capita": frame["carbon_emissions"].round(3).values,
            "Predicted CO2 per capita": np.round(predictions, 3),
            "Delta": np.round(predictions - frame["carbon_emissions"].values, 3),
        }
    ).sort_values("Predicted CO2 per capita", ascending=False)
    return table


def build_choropleth(table_df, year, model_name):
    if table_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"No country data available for {year}",
            paper_bgcolor="#0b1220",
            plot_bgcolor="#0b1220",
            font=dict(color="#e2e8f0"),
            height=620,
            margin=dict(l=0, r=0, t=60, b=0),
            annotations=[
                dict(
                    text="Try a different year to explore predictions.",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=18, color="#94a3b8"),
                )
            ],
        )
        return fig

    fig = go.Figure(
        go.Choropleth(
            locations=table_df["Country"],
            locationmode="country names",
            z=table_df["Predicted CO2 per capita"],
            text=table_df["Country"],
            colorscale=[
                [0.0, "#0f172a"],
                [0.25, "#0f3d5e"],
                [0.5, "#0369a1"],
                [0.75, "#0ea5e9"],
                [1.0, "#93c5fd"],
            ],
            marker_line_color="#0b1220",
            colorbar=dict(title="Predicted t"),
            hovertemplate="%{text}<br>Predicted: %{z:.2f} t<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Predicted CO2 per Capita by Country • {model_name} • {year}",
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type="natural earth",
            bgcolor="#0b1220",
        ),
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
        font=dict(color="#e2e8f0"),
        height=620,
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(56, 189, 248, 0.08), transparent 32%),
            linear-gradient(180deg, #04070f 0%, #0b1220 48%, #0a0f1d 100%);
        color: #f8fafc;
    }
    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 100%;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #08101e 0%, #0b1220 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.12);
    }
    [data-testid="stHeader"] {
        background: rgba(0, 0, 0, 0);
    }
    .card {
        background: rgba(10, 15, 29, 0.86);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 22px;
        padding: 1.4rem 1.3rem;
        box-shadow: 0 20px 70px rgba(0, 0, 0, 0.24);
    }
    .hero {
        text-align: center;
        padding: 4.8rem 1rem 2.6rem 1rem;
    }
    .hero h1 {
        font-size: 4rem;
        line-height: 1;
        margin-bottom: 0.8rem;
        font-weight: 800;
        letter-spacing: -0.04em;
    }
    .hero p {
        max-width: 780px;
        margin: 0 auto;
        color: #94a3b8;
        font-size: 1.05rem;
    }
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(148, 163, 184, 0.28), transparent);
        margin: 2rem 0 2.2rem 0;
    }
    .metric-card {
        background: rgba(15, 23, 42, 0.82);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 20px;
        padding: 1.25rem 1rem;
        min-height: 120px;
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .metric-value {
        font-size: 2.4rem;
        font-weight: 800;
        margin-top: 0.5rem;
    }
    .metric-note {
        color: #64748b;
        font-size: 0.88rem;
        margin-top: 0.3rem;
    }
    .prediction-value {
        text-align: center;
        font-size: 4rem;
        font-weight: 800;
        line-height: 1;
        margin: 0.6rem 0 0.8rem 0;
    }
    .subtle {
        color: #94a3b8;
        text-align: center;
        font-size: 0.95rem;
    }
    .badge {
        display: inline-block;
        padding: 0.32rem 0.8rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.02em;
    }
    .badge-green {
        background: rgba(34, 197, 94, 0.16);
        color: #86efac;
        border: 1px solid rgba(34, 197, 94, 0.22);
    }
    .badge-yellow {
        background: rgba(250, 204, 21, 0.16);
        color: #fde68a;
        border: 1px solid rgba(250, 204, 21, 0.22);
    }
    .badge-red {
        background: rgba(59, 130, 246, 0.18);
        color: #bfdbfe;
        border: 1px solid rgba(59, 130, 246, 0.26);
    }
    .mini-label {
        color: #94a3b8;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.35rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        margin-top: 0.1rem;
        margin-bottom: 0.8rem;
        align-items: flex-start;
        padding-top: 0.2rem;
        padding-bottom: 0.5rem;
        overflow: visible;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(15, 23, 42, 0.68);
        border-radius: 999px;
        padding: 0.55rem 1.1rem;
        border: 1px solid rgba(148, 163, 184, 0.10);
        margin-top: -0.1rem;
        color: #cbd5e1 !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(56, 189, 248, 0.14);
        border-color: rgba(56, 189, 248, 0.30);
        box-shadow: 0 8px 24px rgba(56, 189, 248, 0.12);
        color: #7dd3fc !important;
    }
    .stTabs [aria-selected="true"] p {
        color: #7dd3fc !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background: #38bdf8 !important;
        height: 3px !important;
        bottom: -0.15rem !important;
        border-radius: 999px !important;
    }
    .stSlider label,
    .stSlider p,
    .stSlider span,
    .stSlider div[data-testid="stWidgetLabel"] *,
    .stSlider div[data-testid="stWidgetLabel"],
    div[data-testid="stWidgetLabel"] label,
    div[data-testid="stWidgetLabel"] p,
    section[data-testid="stSidebar"] div[data-testid="stWidgetLabel"] *,
    section[data-testid="stSidebar"] div[data-testid="stWidgetLabel"] {
        color: #7dd3fc !important;
    }
    .stSlider [data-testid="stSliderValue"],
    .stSlider [data-testid="stSliderValue"] *,
    .stSlider div[data-baseweb="slider"] + div,
    .stSlider div[data-baseweb="slider"] + div * {
        color: #38bdf8 !important;
    }
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #38bdf8 !important;
        box-shadow: 0 0 0 4px rgba(56, 189, 248, 0.18) !important;
    }
    .stSlider [data-baseweb="slider"] > div > div > div {
        background: linear-gradient(90deg, #0ea5e9 0%, #38bdf8 100%) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


data, feature_columns, feature_means, feature_ranges = load_data()
models = load_models()
countries = sorted(data["Country Name"].dropna().unique())

results_tab, predict_tab, explore_tab = st.tabs(["Results", "Predict", "Explore"])

with st.sidebar:
    st.header("Predict Controls")
    selected_country = st.selectbox("Country", countries, key="predict_country")
    selected_year = st.slider("Year", min_value=2001, max_value=2022, value=2022, key="predict_year")
    selected_model_name = st.selectbox("Model", list(MODEL_FILES.keys()), key="predict_model")

selected_row = get_row_for_selection(data, selected_country, selected_year)
current_selection = (selected_country, selected_year)
if st.session_state.get("predict_last_selection") != current_selection:
    defaults = get_feature_defaults(selected_row, feature_columns, feature_means)
    sync_feature_state(defaults)
    st.session_state["predict_last_selection"] = current_selection

with results_tab:
    st.markdown(
        """
        <div class="hero">
            <h1>CO2 Emissions Predictor</h1>
            <p>
                A machine learning dashboard for exploring carbon emissions per capita across countries,
                comparing model performance, and testing how economic and energy features shift predicted outcomes.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(3, gap="large")
    metric_content = [
        ("Best Model Test R²", f"{BEST_MODEL_METRICS['r2']:.3f}", BEST_MODEL_METRICS["name"]),
        ("Best Model MAE", f"{BEST_MODEL_METRICS['mae']:.3f}", "tons per capita"),
        ("Best Model RMSE", f"{BEST_MODEL_METRICS['rmse']:.3f}", "tons per capita"),
    ]

    for col, (label, value, note) in zip(metric_cols, metric_content):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    hero_left, hero_right = st.columns([1.05, 1.2], gap="large")

    with hero_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.plotly_chart(build_model_ranking_chart(), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with hero_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.plotly_chart(build_gbr_importance_chart(), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

with predict_tab:
    model = models[selected_model_name]
    ordered_features = get_model_feature_order(selected_model_name, model, feature_columns)

    left_col, center_col, right_col = st.columns([1.05, 1.2, 0.95], gap="large")

    with left_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="mini-label">Selected Profile</div>', unsafe_allow_html=True)
        st.subheader(f"{selected_country} • {selected_year}")
        if selected_row is None:
            st.info("This country-year is missing from the cleaned dataset, so feature means are being used where needed.")
        else:
            st.metric("Actual CO2 per capita", f"{float(selected_row['carbon_emissions']):.2f} t")
        st.caption("Feature sliders are ordered by the selected model's importance ranking.")
        st.markdown("</div>", unsafe_allow_html=True)

    feature_values = {}
    with center_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="mini-label">Feature Inputs</div>', unsafe_allow_html=True)
        for feature in ordered_features:
            bounds = feature_ranges[feature]
            current_value = st.session_state.get(f"feature::{feature}", bounds["mean"])
            clamped_value = float(np.clip(current_value, bounds["min"], bounds["max"]))
            feature_values[feature] = st.slider(
                feature,
                min_value=float(bounds["min"]),
                max_value=float(bounds["max"]),
                value=clamped_value,
                key=f"feature::{feature}",
            )
        st.markdown("</div>", unsafe_allow_html=True)

    prediction = predict_value(model, feature_values, feature_columns)
    actual_value = None if selected_row is None else float(selected_row["carbon_emissions"])
    delta_value = None if actual_value is None else prediction - actual_value
    badge_class, badge_text = prediction_badge(prediction)

    with right_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="mini-label">Prediction</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="prediction-value">{prediction:.2f} t</div>', unsafe_allow_html=True)
        st.markdown(
            f'<p style="text-align:center;"><span class="badge {badge_class}">{badge_text} Emissions</span></p>',
            unsafe_allow_html=True,
        )
        if delta_value is None:
            st.markdown('<div class="subtle">Actual value unavailable for this country-year.</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="subtle">Delta from actual: {delta_value:+.2f} t</div>',
                unsafe_allow_html=True,
            )
        st.plotly_chart(
            build_distribution_chart(data, prediction, selected_country),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

with explore_tab:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    explore_controls = st.columns([0.18, 0.18, 0.64], gap="large")
    with explore_controls[0]:
        map_year = st.slider("Explore Year", min_value=2001, max_value=2022, value=2022, key="map_year")
    with explore_controls[1]:
        map_model_name = st.selectbox("Explore Model", list(MODEL_FILES.keys()), key="map_model")

    year_df = data[data["Year"] == map_year].copy()
    table_df = build_prediction_table(year_df, models[map_model_name], feature_columns, feature_means)

    st.plotly_chart(build_choropleth(table_df, map_year, map_model_name), use_container_width=True)
    if table_df.empty:
        st.info("No rows are available for this year in `pre_split.csv`.")
    else:
        st.dataframe(
            table_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Actual CO2 per capita": st.column_config.NumberColumn(format="%.3f"),
                "Predicted CO2 per capita": st.column_config.NumberColumn(format="%.3f"),
                "Delta": st.column_config.NumberColumn(format="%.3f"),
            },
        )
    st.markdown("</div>", unsafe_allow_html=True)
