import streamlit as st
import pandas as pd
import joblib
import gdown
import os

# -----------------------------
# Load model (Google Drive)
# -----------------------------
MODEL_ID = "1UClYlkrOBEoZvVHAIQSce4lgthih0OVy"
MODEL_FILE = "gas_model.joblib"

if not os.path.exists(MODEL_FILE):
    gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", MODEL_FILE, quiet=False)

model = joblib.load(MODEL_FILE)

# -----------------------------
# Load Test Dataset (for accuracy)
# -----------------------------
TEST_FILE = "test_data.csv"

try:
    test_df = pd.read_csv(TEST_FILE)
except:
    test_df = None

# -----------------------------
# UI Inputs
# -----------------------------
st.title("Gas Consumption Predictor")

income = st.number_input("Total Monthly Income (₹)", min_value=0)
family = st.number_input("No. of Family Members", min_value=1)
adults = st.number_input("No. of Adults", min_value=0)
children = st.number_input("No. of Children", min_value=0)
city = st.text_input("City")
state = st.text_input("State")

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict"):

    data = pd.DataFrame([{
        "Total Monthly Income (₹)": income,
        "No. of Family Members": family,
        "No. of Adults": adults,
        "No. of Children": children,
        "City": city,
        "State": state
    }])

    predicted = model.predict(data)[0]

    st.success(f"Predicted Gas Consumption: {predicted:.2f} units")

    # ------- ACCURACY LOGIC -------
    if test_df is not None:

        # try to find matching row in dataset
        match = test_df[
            (test_df["Total Monthly Income (₹)"] == income) &
            (test_df["No. of Family Members"] == family) &
            (test_df["No. of Adults"] == adults) &
            (test_df["No. of Children"] == children) &
            (test_df["City"].str.lower() == city.lower()) &
            (test_df["State"].str.lower() == state.lower())
        ]

        if not match.empty:
            actual = match.iloc[0]["actual_gas_consumption"]
            if actual != 0:
                accuracy = 100 - (abs(predicted - actual) / actual * 100)
                accuracy = max(0, accuracy)
            else:
                accuracy = 100 if predicted == actual else 0

            st.info(f"Accuracy: {accuracy:.2f}%")

        else:
            st.warning("Accuracy cannot be calculated because this input does not exist in test dataset.")

    else:
        st.warning("Test dataset missing — accuracy cannot be calculated.")
