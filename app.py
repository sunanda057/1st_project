# app.py — Dynamic per-input accuracy (nearest-neighbor fallback)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gdown
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

st.set_page_config(layout="centered")

# -------- CONFIG --------
MODEL_GDRIVE_ID = "1UClYlkrOBEoZvVHAIQSce4lgthih0OVy"  # your model file id (joblib)
MODEL_LOCAL = "gas_model.joblib"
TEST_DATA_LOCAL = "test_data.csv"  # saved during training
ACTUAL_COL = "actual_gas_consumption"  # column name in test_data.csv with true gas value
NUM_NEIGHBORS = 5  # k for nearest neighbors
NUMERIC_FEATURES = [
    "Total Monthly Income (₹)",
    "No. of Family Members",
    "No. of Adults",
    "No. of Children"
]
# ------------------------

@st.cache_resource
def download_and_load_model():
    # download if not exists
    try:
        if not os.path.exists(MODEL_LOCAL):
            url = f"https://drive.google.com/uc?id={MODEL_GDRIVE_ID}"
            gdown.download(url, MODEL_LOCAL, quiet=False)
        model = joblib.load(MODEL_LOCAL)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def load_test_data():
    # Try to load test_data.csv from local repo
    try:
        df = pd.read_csv(TEST_DATA_LOCAL)
        return df
    except FileNotFoundError:
        st.warning(f"Test data file '{TEST_DATA_LOCAL}' not found in app folder.")
        return None

# For file/path check
import os

# Load model & test set
with st.spinner("Loading model..."):
    model = None
    try:
        # If model is already present locally, load directly (joblib)
        if os.path.exists(MODEL_LOCAL):
            model = joblib.load(MODEL_LOCAL)
        else:
            # download from Drive
            url = f"https://drive.google.com/uc?id={MODEL_GDRIVE_ID}"
            gdown.download(url, MODEL_LOCAL, quiet=False)
            model = joblib.load(MODEL_LOCAL)
    except Exception as e:
        st.error(f"Model load error: {e}")
        model = None

test_df = load_test_data()

# App UI
st.title("Gas Consumption Prediction — Dynamic Accuracy")
st.markdown("Enter household details, click **Predict**, and the app will compute prediction **and** a dynamic accuracy score.")
st.info("If an exact matching record exists in the test dataset, the app shows exact per-input accuracy. Otherwise it computes an approximate accuracy using nearest neighbors.")

# Input fields (match your dataset column names)
income = st.number_input("Total Monthly Income (₹)", min_value=0)
family_members = st.number_input("No. of Family Members", min_value=1, step=1)
adults = st.number_input("No. of Adults", min_value=0, step=1)
children = st.number_input("No. of Children", min_value=0, step=1)
city = st.text_input("City", "")
state = st.text_input("State", "")

def build_input_df():
    return pd.DataFrame([{
        "Total Monthly Income (₹)": income,
        "No. of Family Members": family_members,
        "No. of Adults": adults,
        "No. of Children": children,
        "City": city,
        "State": state
    }])

if st.button("Predict Gas Consumption"):
    if model is None:
        st.error("Model not loaded. Check the logs and model path.")
    else:
        input_df = build_input_df()

        # Try predicting (pipeline should handle preprocessing)
        try:
            pred = model.predict(input_df)[0]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        st.success(f"Predicted Monthly Gas Consumption: {pred:.2f} units")

        # ---------- Dynamic accuracy logic ----------
        if test_df is None:
            st.warning("No test_data.csv available in app folder — dynamic accuracy cannot be computed.")
        else:
            # First try exact match on all columns (including city/state if present)
            match_cols = ["Total Monthly Income (₹)", "No. of Family Members", "No. of Adults", "No. of Children", "City", "State"]
            # Ensure those columns exist in test_df
            missing_cols = [c for c in match_cols if c not in test_df.columns]
            if missing_cols:
                st.warning(f"Test data is missing columns required for matching: {missing_cols}. Using numeric neighbors only.")
                exact_match = pd.DataFrame()
            else:
                # Normalize city/state case for matching
                temp_test = test_df.copy()
                temp_test["City"] = temp_test["City"].astype(str).str.strip().str.lower()
                temp_test["State"] = temp_test["State"].astype(str).str.strip().str.lower()
                key_city = str(city).strip().lower()
                key_state = str(state).strip().lower()

                exact_match = temp_test[
                    (temp_test["Total Monthly Income (₹)"] == income) &
                    (temp_test["No. of Family Members"] == family_members) &
                    (temp_test["No. of Adults"] == adults) &
                    (temp_test["No. of Children"] == children) &
                    (temp_test["City"] == key_city) &
                    (temp_test["State"] == key_state)
                ]

            if not exact_match.empty:
                # Use first matched row
                actual = exact_match.iloc[0][ACTUAL_COL]
                # compute percent accuracy (based on relative error)
                if actual != 0:
                    pct = max(0.0, 100.0 * (1.0 - abs(pred - actual) / abs(actual)))
                else:
                    pct = 100.0 if pred == actual else 0.0

                st.success(f"Exact match found in test set. Per-input Accuracy: {pct:.2f}%")
                st.write(f"Actual value (from test set): {actual:.2f} units")
                # Visualize
                fig, ax = plt.subplots()
                ax.bar(["Predicted", "Actual"], [pred, actual], color=["#1f77b4", "#ff7f0e"])
                ax.set_ylabel("Gas Consumption (units)")
                st.pyplot(fig)

            else:
                # No exact match — use nearest neighbors approach on numeric features
                st.info("No exact match found. Computing approximate accuracy using nearest neighbors on numeric features...")

                # Ensure numeric features exist
                numeric_missing = [c for c in NUMERIC_FEATURES if c not in test_df.columns]
                if numeric_missing:
                    st.warning(f"Test data missing numeric columns required for neighbors: {numeric_missing}. Cannot compute approximate accuracy.")
                else:
                    # optionally filter by same state if provided
                    candidate_df = test_df.copy()
                    if state.strip() != "":
                        # do case-insensitive filtering if column exists
                        if "State" in candidate_df.columns:
                            candidate_df = candidate_df[candidate_df["State"].astype(str).str.strip().str.lower() == state.strip().lower()]
                        # if filtering results in empty, fall back to full df
                        if candidate_df.empty:
                            candidate_df = test_df.copy()

                    # Compute numeric distances (simple Euclidean) — scale numeric columns to unit range to balance
                    num_df = candidate_df[NUMERIC_FEATURES].astype(float).copy()
                    # min-max scaling
                    mins = num_df.min()
                    maxs = num_df.max()
                    denom = (maxs - mins).replace(0, 1.0)
                    num_scaled = (num_df - mins) / denom

                    # scale input numeric similarly
                    inp = np.array([income, family_members, adults, children], dtype=float)
                    inp_scaled = (inp - mins.values) / denom.values

                    # Euclidean distances
                    dists = np.linalg.norm(num_scaled.values - inp_scaled, axis=1)
                    candidate_df = candidate_df.copy()
                    candidate_df["__dist__"] = dists
                    candidate_df = candidate_df.sort_values("__dist__").reset_index(drop=True)

                    neighbors = candidate_df.head(NUM_NEIGHBORS)
                    if neighbors.empty:
                        st.warning("No neighbors found in test set to compute approximate accuracy.")
                    else:
                        mean_actual = neighbors[ACTUAL_COL].astype(float).mean()
                        # compute approximate percent accuracy relative to mean_actual
                        if mean_actual != 0:
                            approx_pct = max(0.0, 100.0 * (1.0 - abs(pred - mean_actual) / abs(mean_actual)))
                        else:
                            approx_pct = 100.0 if pred == mean_actual else 0.0

                        st.success(f"Approximate Accuracy (based on {len(neighbors)} nearest neighbors): {approx_pct:.2f}%")
                        st.markdown("**Note:** This is an *approximate* accuracy computed from the mean actual consumption of nearest neighbors (numeric features).")
                        st.write(f"Mean Actual (neighbors): {mean_actual:.2f} units")

                        # show neighbors table (top 5)
                        show_table = neighbors[[*NUMERIC_FEATURES, ACTUAL_COL, "__dist__"]].copy()
                        show_table = show_table.rename(columns={ACTUAL_COL: "Actual_gas", "__dist__": "Distance"})
                        st.dataframe(show_table.head(10))

                        # bar chart predicted vs mean_actual
                        fig, ax = plt.subplots()
                        ax.bar(["Predicted", "Neighbors Mean Actual"], [pred, mean_actual], color=["#2ca02c", "#d62728"])
                        ax.set_ylabel("Gas Consumption (units)")
                        st.pyplot(fig)

        # ---------- END dynamic accuracy ----------
