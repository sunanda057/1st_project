import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gdown

# --------------------------------------
# Load Model from Google Drive
# --------------------------------------
@st.cache_resource
def load_model():
    try:
        url = "https://drive.google.com/uc?id=1UClYlkrOBEoZvVHAIQSce4lgthih0OVy"
        output = "gas_model.joblib"
        gdown.download(url, output, quiet=False)
        model = joblib.load(output)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()

# --------------------------------------
# Load Model Accuracy (Static)
# --------------------------------------
# You MUST create accuracy.txt (from your training script)
try:
    with open("accuracy.txt", "r") as f:
        model_accuracy = float(f.read())
except:
    model_accuracy = None

# --------------------------------------
# APP UI
# --------------------------------------
st.title("Gas Consumption Prediction App")

st.write("Enter the household details below to predict monthly gas consumption.")

income = st.number_input("Total Monthly Income (₹)", min_value=0)
family_members = st.number_input("No. of Family Members", min_value=1)
adults = st.number_input("No. of Adults", min_value=0)
children = st.number_input("No. of Children", min_value=0)
city = st.text_input("City")
state = st.text_input("State")

if model is not None and st.button("Predict Gas Consumption"):
    
    input_data = pd.DataFrame({
        'Total Monthly Income (₹)': [income],
        'No. of Family Members': [family_members],
        'No. of Adults': [adults],
        'No. of Children': [children],
        'City': [city],
        'State': [state]
    })

    try:
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Monthly Gas Consumption: {prediction:.2f} units")

        # -------- SHOW ACCURACY BELOW PREDICTION --------
        if model_accuracy is not None:
            with st.expander("📊 Show Model Accuracy"):
                st.info(f"Model Accuracy: **{model_accuracy:.2f}%**")
        else:
            st.warning("Accuracy file not found. Please generate accuracy.txt.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
