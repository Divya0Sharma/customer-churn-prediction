"""
Customer Churn Prediction — Streamlit App
==========================================
Loads the trained model + preprocessing bundle (saved with joblib from
notebooks/01_churn_prediction.ipynb), collects a customer's profile
through a form, applies the exact same preprocessing used at training
time, and displays a churn prediction with a confidence score.

Run with:
    streamlit run app/streamlit_app.py
"""

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "..", "models", "churn_model.joblib")

# Options for each categorical field, in the same spelling used by the
# original Telco dataset (must match what the saved LabelEncoders were
# fitted on).
YES_NO = ["Yes", "No"]
INTERNET_DEPENDENT = ["No internet service", "No", "Yes"]
PHONE_DEPENDENT = ["No phone service", "No", "Yes"]
INTERNET_SERVICE_OPTIONS = ["DSL", "Fiber optic", "No"]
CONTRACT_OPTIONS = ["Month-to-month", "One year", "Two year"]
PAYMENT_METHOD_OPTIONS = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]


# ---------------------------------------------------------------------------
# Model loading (cached so the file is only read once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model_bundle(path: str):
    """Load the joblib bundle saved by the training notebook.

    Returns None if the file doesn't exist, so the caller can show a
    friendly message instead of crashing the whole app.
    """
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


bundle = load_model_bundle(MODEL_PATH)


# ---------------------------------------------------------------------------
# Sidebar — model info & instructions
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Model Info")

    if bundle is not None:
        st.success("Model loaded successfully")
        st.metric("Algorithm", bundle.get("model_name", "Unknown"))

        metrics = bundle.get("test_metrics", {})
        if metrics:
            st.caption("Performance on held-out test data")
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                if "Accuracy" in metrics:
                    st.metric("Accuracy", f"{metrics['Accuracy']:.1%}")
                if "Recall" in metrics:
                    st.metric("Recall", f"{metrics['Recall']:.1%}")
            with m_col2:
                if "F1" in metrics:
                    st.metric("F1 Score", f"{metrics['F1']:.1%}")
                if "ROC-AUC" in metrics:
                    st.metric("ROC-AUC", f"{metrics['ROC-AUC']:.1%}")
    else:
        st.error("Model file not found")
        st.caption(
            "Run `notebooks/01_churn_prediction.ipynb` first to train "
            "and save `models/churn_model.joblib`."
        )

    st.divider()

    st.header("How to use this app")
    st.markdown(
        """
        1. Fill in the customer's details in the form.
        2. Click **Predict Churn**.
        3. Review the prediction, confidence score, and the key
           factors driving churn overall.

        **Tip:** Fields like *Online Security* or *Multiple Lines*
        only make sense when the customer has the related base
        service (Internet / Phone) — the options are provided as
        they appear in the original dataset.
        """
    )

    st.divider()
    st.caption(
        "Built with Python, scikit-learn, XGBoost, and Streamlit — "
        "trained on the Telco Customer Churn dataset."
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Customer Churn Prediction")
st.markdown(
    """
    Predict whether a telecom customer is likely to **churn** (cancel
    their subscription) based on their account, billing, and service
    details. Fill out the form below with a customer's profile and get
    an instant prediction with a confidence score.
    """
)
st.divider()

# Stop early with a clear message if the model isn't available —
# no point rendering a form that can't do anything.
if bundle is None:
    st.error(
        "⚠️ Could not find a trained model at `models/churn_model.joblib`. "
        "Please run the training notebook (`notebooks/01_churn_prediction.ipynb`) "
        "first — it will create this file for you."
    )
    st.stop()

model = bundle["model"]
encoders = bundle["encoders"]
scaler = bundle.get("scaler")
feature_names = bundle["feature_names"]
categorical_cols = bundle.get("categorical_cols", list(encoders.keys()))


def options_for(col: str, fallback: list) -> list:
    """Return the exact category values the saved LabelEncoder learned
    for this column, so the dropdown can never offer a value the model
    wasn't trained on. Falls back to a hardcoded list if the encoder
    bundle doesn't include this column for some reason.
    """
    encoder = encoders.get(col)
    if encoder is not None and hasattr(encoder, "classes_"):
        return list(encoder.classes_)
    return fallback


# Note: fields like "Multiple Lines" or the internet add-ons deliberately
# show all categories the model was trained on (Yes / No / "No X service")
# rather than narrowing them based on other answers in this same form —
# Streamlit form widgets don't rerun until the whole form is submitted, so
# options that depend on a sibling widget inside the same form can get out
# of sync with what's actually selected. Showing the full set is simpler
# and always valid.

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
st.subheader("👤 Customer Profile")

with st.form("churn_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics**")
        gender = st.selectbox("Gender", options_for("gender", ["Female", "Male"]))
        senior_citizen_label = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has a Partner?", options_for("Partner", YES_NO))
        dependents = st.selectbox("Has Dependents?", options_for("Dependents", YES_NO))

        st.markdown("**Account**")
        tenure = st.slider("Tenure (months with the company)", 0, 72, 12)
        contract = st.selectbox("Contract Type", options_for("Contract", CONTRACT_OPTIONS))
        paperless_billing = st.selectbox(
            "Paperless Billing?", options_for("PaperlessBilling", YES_NO)
        )

    with col2:
        st.markdown("**Phone Services**")
        phone_service = st.selectbox("Phone Service?", options_for("PhoneService", YES_NO))
        multiple_lines = st.selectbox(
            "Multiple Lines?", options_for("MultipleLines", PHONE_DEPENDENT)
        )
        st.caption("Choose \"No phone service\" if Phone Service is No.")

        st.markdown("**Internet Services**")
        internet_service = st.selectbox(
            "Internet Service", options_for("InternetService", INTERNET_SERVICE_OPTIONS)
        )
        online_security = st.selectbox(
            "Online Security?", options_for("OnlineSecurity", INTERNET_DEPENDENT)
        )
        online_backup = st.selectbox(
            "Online Backup?", options_for("OnlineBackup", INTERNET_DEPENDENT)
        )
        st.caption("Internet add-ons: choose \"No internet service\" if there's no internet.")

    with col3:
        st.markdown("**More Internet Services**")
        device_protection = st.selectbox(
            "Device Protection?", options_for("DeviceProtection", INTERNET_DEPENDENT)
        )
        tech_support = st.selectbox(
            "Tech Support?", options_for("TechSupport", INTERNET_DEPENDENT)
        )
        streaming_tv = st.selectbox(
            "Streaming TV?", options_for("StreamingTV", INTERNET_DEPENDENT)
        )
        streaming_movies = st.selectbox(
            "Streaming Movies?", options_for("StreamingMovies", INTERNET_DEPENDENT)
        )

        st.markdown("**Billing**")
        payment_method = st.selectbox(
            "Payment Method", options_for("PaymentMethod", PAYMENT_METHOD_OPTIONS)
        )
        monthly_charges = st.number_input(
            "Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=1.0
        )
        total_charges = st.number_input(
            "Total Charges ($)", min_value=0.0, max_value=10000.0, value=840.0, step=10.0
        )

    submitted = st.form_submit_button("Predict Churn", use_container_width=True)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if submitted:
    raw_input = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen_label == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    try:
        input_df = pd.DataFrame([raw_input])

        # Apply the exact same label encoders used at training time
        for col, encoder in encoders.items():
            try:
                input_df[col] = encoder.transform(input_df[col])
            except ValueError as exc:
                raise ValueError(
                    f"Unexpected value for '{col}': {input_df[col].iloc[0]!r}. "
                    f"Known values: {list(encoder.classes_)}"
                ) from exc

        # Ensure the exact column order the model was trained on
        input_df = input_df[feature_names]

        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]
        churn_probability = float(probabilities[1])
        stay_probability = float(probabilities[0])

        st.divider()
        st.subheader("Prediction Result")

        result_col, prob_col = st.columns([1, 1])

        with result_col:
            if prediction == 1:
                st.error("⚠️ **This customer is likely to CHURN**")
            else:
                st.success("✅ **This customer is likely to STAY**")

        with prob_col:
            st.metric(
                label="Churn Probability",
                value=f"{churn_probability:.1%}",
                delta=f"{(churn_probability - 0.5):+.1%} vs. 50/50",
                delta_color="inverse",
            )

        # Confidence visual — colored progress bar
        st.markdown("**Confidence Breakdown**")
        st.progress(churn_probability, text=f"Churn risk: {churn_probability:.1%}")

        b_col1, b_col2 = st.columns(2)
        with b_col1:
            st.info(f"🟢 Probability of staying: **{stay_probability:.1%}**")
        with b_col2:
            st.warning(f"🔴 Probability of churning: **{churn_probability:.1%}**")

        # Simple risk-level callout
        if churn_probability >= 0.7:
            st.error(
                "🚨 **High risk** — consider proactive retention outreach "
                "(discounts, contract upgrade offers, service check-ins)."
            )
        elif churn_probability >= 0.4:
            st.warning(
                "⚠️ **Moderate risk** — this customer may benefit from "
                "engagement or loyalty incentives."
            )
        else:
            st.success("👍 **Low risk** — this customer looks stable for now.")

    except Exception as exc:
        st.error(f"❌ Prediction failed: {exc}")


# ---------------------------------------------------------------------------
# Feature importance section
# ---------------------------------------------------------------------------
st.divider()
st.subheader("What Drives Churn? (Feature Importance)")

if hasattr(model, "feature_importances_"):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False)

    top_n = st.slider(
        "Number of top features to show", min_value=5, max_value=len(importances), value=10
    )

    st.bar_chart(importances.head(top_n))
    st.caption(
        "Higher values indicate the feature had a bigger influence on the "
        f"model's ({bundle.get('model_name', 'this model')}) churn predictions overall."
    )
else:
    st.info(
        "Feature importance is not available for this model type "
        f"({bundle.get('model_name', 'unknown')})."
    )

st.divider()
st.caption(
    "Disclaimer: predictions are estimates from a machine learning model "
    "trained on historical data and should support, not replace, human "
    "judgment in retention decisions."
)
