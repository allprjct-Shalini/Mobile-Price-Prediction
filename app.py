import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Mobile Price Prediction", page_icon="📱", layout="wide")

from model_repair_agent import (
    load_model_and_scaler,
    train_and_save_model,
    FEATURES,
    TARGET,
)

# Load images
banner = Image.open("images/banner.jpg")
logo = Image.open("images/logo.png")
mobile_ai = Image.open("images/mobile_ai.png")

# Sidebar logo
st.sidebar.image(logo, width=180)

# Main banner
st.image(banner, use_container_width=True)

FEATURE_LABELS = [
    ("battery", "Battery Capacity (mAh)"),
    ("ram", "RAM (GB)"),
    ("ppi", "Screen PPI"),
    ("resoloution", "Screen Resolution"),
    ("weight", "Weight (g)"),
    ("internal mem", "Internal Memory (GB)"),
    ("cpu freq", "CPU Frequency (GHz)"),
    ("Front_Cam", "Front Camera (MP)"),
    ("RearCam", "Rear Camera (MP)"),
]

st.sidebar.title("Controls")
st.sidebar.write("Use this panel to refresh the model and get help.")
if st.sidebar.button("Rebuild model"):
    model, scaler = train_and_save_model()
    st.sidebar.success("Model and scaler rebuilt successfully.")
else:
    model, scaler = load_model_and_scaler()

st.sidebar.markdown("---")
st.sidebar.write("**Version:** 1.0")
st.sidebar.write("**Dataset:** Cellphone.csv")
st.sidebar.write("**Model:** Ridge Regression")


def load_dataset():
    return pd.read_csv("Cellphone.csv")


def evaluate_model(model, scaler, df):
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    metrics = {
        "R2 Score": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
    }
    feature_importance = pd.Series(model.coef_, index=FEATURES).abs().sort_values(ascending=False)
    return metrics, feature_importance

st.title("Mobile Price Prediction")
st.write("Predict the estimated price of a mobile device from its hardware features.")
st.write("Use the tabs below to switch between prediction, analytics, and dataset preview.")

tab_predict, tab_analytics, tab_data = st.tabs(["Predict", "Analytics", "Dataset"])

with tab_predict:
    with st.container():
        left, right = st.columns([2, 1])

        with left:
            st.subheader("Input Features")
            inputs = []
            cols = st.columns(3)
            for index, (_, label) in enumerate(FEATURE_LABELS):
                with cols[index % 3]:
                    value = st.number_input(label, min_value=0.0, value=0.0, step=1.0)
                    inputs.append(value)

            if st.button("Predict"):
                input_data = np.array([inputs])
                try:
                    input_data_scaled = scaler.transform(input_data)
                    prediction = model.predict(input_data_scaled)
                    st.success(f"Predicted Price: {prediction[0]:.2f}")
                    st.metric("Estimated Price", f"{prediction[0]:.2f}")
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    st.write("Run the Rebuild model button if the error persists.")

        with right:
            st.subheader("Phone preview")
            st.image(mobile_ai, width=280)
            st.write(
                "This app predicts price using device specifications. "
                "Set values in the left panel and tap Predict."
            )

    st.subheader("Dataset price distribution")
    try:
        df = load_dataset()
        st.line_chart(df["Price"].sort_values().reset_index(drop=True))
    except FileNotFoundError:
        st.error("Cellphone.csv not found in the app folder.")

with tab_analytics:
    st.subheader("Model summary")
    st.markdown(
        "- Ridge regression model\n"
        "- Features normalized with StandardScaler\n"
        "- Current app version: 1.0"
    )
    st.info("Model evaluation is based on the current dataset and saved model.")

    try:
        df = load_dataset()
        metrics, feature_importance = evaluate_model(model, scaler, df)
        metric_cols = st.columns(3)
        metric_cols[0].metric("R² Score", f"{metrics['R2 Score']:.3f}")
        metric_cols[1].metric("MAE", f"{metrics['MAE']:.2f}")
        metric_cols[2].metric("RMSE", f"{metrics['RMSE']:.2f}")

        st.subheader("Feature importance")
        st.bar_chart(feature_importance)

        st.subheader("Input order")
        cols = st.columns(3)
        for index, (_, label) in enumerate(FEATURE_LABELS):
            cols[index % 3].write(f"• {label}")

        st.subheader("Price distribution")
        price_data = df["Price"].sort_values().reset_index(drop=True)
        st.line_chart(price_data)
    except Exception as e:
        st.error(f"Could not evaluate model: {e}")

with tab_data:
    with st.expander("Dataset preview"):
        try:
            df = load_dataset()
            st.dataframe(df.head())
            st.write(f"Dataset shape: {df.shape}")
        except FileNotFoundError:
            st.error("Cellphone.csv not found in the app folder.")

