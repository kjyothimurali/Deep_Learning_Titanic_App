import streamlit as st
import tensorflow as tf
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
# LOAD MODEL + SCALER
# ==========================================
# ==========================================
# BUILD MODEL (FIXED VERSION)
# ==========================================

model = tf.keras.Sequential()

# Input + Hidden layer
model.add(
    tf.keras.layers.Dense(
        2,
        activation="sigmoid",
        input_dim=3
    )
)

# Output layer
model.add(
    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
)

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.SGD(
        learning_rate=0.1
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ==========================================
# LOAD TRAINED WEIGHTS
# ==========================================

model.load_weights(
    "titanic_weights.weights.h5"
)

# ==========================================
# LOAD SCALER
# ==========================================

import joblib

scaler = joblib.load(
    "scaler.pkl"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main-title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#1f77b4;
}

.subtitle{
    text-align:center;
    color:gray;
    font-size:20px;
}

.card{
    background-color:#f8f9fa;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 2px 8px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# SECTION 1 — HEADER AREA
# ==========================================

st.markdown(
    "<h1 class='main-title'>🚢 Titanic Survival Prediction System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='subtitle'>Deep Learning Based Passenger Survival Prediction</p>",
    unsafe_allow_html=True
)

st.image(
    "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
    width=120
)

st.divider()

# ==========================================
# SECTION 2 — PROJECT DESCRIPTION
# ==========================================

st.subheader(" Project Description")

st.markdown("""
<div class='card'>

This web application predicts whether a Titanic passenger
would survive during an emergency situation using an
Artificial Neural Network (ANN).

The ANN model is trained using TensorFlow on passenger data.

The application:
- accepts passenger information
- preprocesses inputs using Min-Max normalization
- loads a trained TensorFlow model
- predicts survival probability

</div>
""", unsafe_allow_html=True)

st.divider()

# ==========================================
# SECTION 3 — PASSENGER INPUT FORM
# ==========================================

st.subheader("🧾 Passenger Information")

col1, col2, col3 = st.columns(3)

with col1:
    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

with col2:
    age = st.slider(
        "Age",
        min_value=1,
        max_value=80,
        value=24
    )

with col3:
    fare = st.number_input(
        "Fare",
        min_value=0.0,
        value=120.0
    )

st.divider()

# ==========================================
# SECTION 4 — PREDICTION BUTTON
# ==========================================

if st.button("Predict Survival"):

    # ======================================
    # TASK 4 — PREPROCESSING
    # ======================================

    user_input = np.array([
        [pclass, age, fare]
    ])

    user_scaled = scaler.transform(
        user_input
    )

    # ======================================
    # TASK 5 — MODEL INFERENCE
    # ======================================

    prediction = model.predict(
        user_scaled
    )[0][0]

    survive_prob = float(prediction)
    non_survive_prob = 1 - survive_prob

    # ======================================
    # TASK 6 — PREDICTION LOGIC
    # ======================================

    if prediction > 0.5:
        result = "✅ SURVIVED"
    else:
        result = "❌ NOT SURVIVED"

    confidence = max(
        survive_prob,
        non_survive_prob
    )

    st.divider()

    # ======================================
    # SECTION 5 — OUTPUT AREA
    # ======================================

    st.subheader(" Prediction Output")

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

    # ======================================
    # SECTION 6 — VISUALIZATION
    # ======================================

    st.subheader("📈 Probability Visualization")

    col4, col5 = st.columns(2)

    with col4:

        fig, ax = plt.subplots()

        labels = [
            "Survived",
            "Not Survived"
        ]

        values = [
            survive_prob,
            non_survive_prob
        ]

        ax.bar(labels, values)

        ax.set_ylabel(
            "Probability"
        )

        st.pyplot(fig)

    with col5:

        fig2, ax2 = plt.subplots()

        ax2.pie(
            values,
            labels=labels,
            autopct="%1.1f%%"
        )

        st.pyplot(fig2)