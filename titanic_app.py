import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

# ==========================================
# LOAD SCALER
# ==========================================

scaler = joblib.load("scaler.pkl")

# ==========================================
# MANUAL TRAINED WEIGHTS
# (Paste your trained weights here)
# ==========================================

# Hidden layer weights (3 × 2)
W1 = np.array([
    [0.45, 0.38],
    [0.62, 0.29],
    [0.81, 0.73]
])

# Hidden layer bias
b1 = np.array([0.12, 0.15])

# Output layer weights (2 × 1)
W2 = np.array([
    [0.88],
    [0.67]
])

# Output bias
b2 = np.array([0.20])

# ==========================================
# SIGMOID FUNCTION
# ==========================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ==========================================
# HEADER
# ==========================================

st.markdown(
    """
    <h1 style='text-align:center;color:#1f77b4;'>
    👨 🚢 Titanic Survival Prediction System
    </h1>
    <h4 style='text-align:center;color:gray;'>
    Deep Learning Based Passenger Survival Prediction
    </h4>
    """,
    unsafe_allow_html=True
)

st.divider()

# ==========================================
# PROJECT DESCRIPTION
# ==========================================

st.subheader("📌 Project Description")

st.info("""
This application predicts whether a Titanic passenger
would survive using an Artificial Neural Network (ANN).

The system uses:
• Passenger Class
• Age
• Fare

The inputs are normalized using Min-Max Scaling and
processed using a trained ANN model.
""")

st.divider()

# ==========================================
# INPUT FORM
# ==========================================

st.subheader("🧾 Passenger Details")

col1, col2, col3 = st.columns(3)

with col1:
    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

with col2:
    age = st.slider(
        "Age",
        1,
        80,
        24
    )

with col3:
    fare = st.number_input(
        "Fare",
        min_value=0.0,
        value=120.0
    )

st.divider()

# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button("Predict Survival"):

    # ===============================
    # PREPROCESS INPUT
    # ===============================

    user_input = np.array([
        [pclass, age, fare]
    ])

    user_scaled = scaler.transform(
        user_input
    )

    # ===============================
    # FORWARD PROPAGATION
    # ===============================

    hidden_input = (
        np.dot(user_scaled, W1)
        + b1
    )

    hidden_output = sigmoid(
        hidden_input
    )

    output_input = (
        np.dot(hidden_output, W2)
        + b2
    )

    prediction = sigmoid(
        output_input
    )[0][0]

    survive_prob = float(
        prediction
    )

    nonsurvive_prob = (
        1 - survive_prob
    )

    # ===============================
    # PREDICTION LOGIC
    # ===============================

    if prediction > 0.5:
        result = "✅ SURVIVED"
    else:
        result = "❌ NOT SURVIVED"

    confidence = max(
        survive_prob,
        nonsurvive_prob
    )

    st.divider()

    # ===============================
    # OUTPUT AREA
    # ===============================

    st.subheader("📊 Prediction Output")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Prediction Result",
            result
        )

    with c2:
        st.metric(
            "Survival Probability",
            f"{survive_prob*100:.2f}%"
        )

    with c3:
        st.metric(
            "Confidence Score",
            f"{confidence*100:.2f}%"
        )

    st.divider()

    # ===============================
    # VISUALIZATION
    # ===============================

    st.subheader(
        "📈 Probability Visualization"
    )

    fig, ax = plt.subplots()

    labels = [
        "Survived",
        "Not Survived"
    ]

    values = [
        survive_prob,
        nonsurvive_prob
    ]

    ax.bar(labels, values)

    ax.set_ylabel(
        "Probability"
    )

    st.pyplot(fig)

    fig2, ax2 = plt.subplots()

    ax2.pie(
        values,
        labels=labels,
        autopct="%1.1f%%"
    )

    st.pyplot(fig2)
