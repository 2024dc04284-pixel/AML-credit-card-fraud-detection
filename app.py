


import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ------------------------------------------------------------
# Load Model + Scaler
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("credit_card_fraud_model_lrl2.pkl")
    scaler = joblib.load("scaler_lrl2.pkl")
    return model, scaler

model, scaler = load_model()

# ------------------------------------------------------------
# UI Title
# ------------------------------------------------------------
st.title("💳 Credit Card Fraud Detection App Using LRL2")
st.write("Choose an input method below:")

# ------------------------------------------------------------
# Input Method Selection
# ------------------------------------------------------------
method = st.selectbox(
    "Select Input Method",
    [
        "Enter comma-separated values (single row)",
        "Choose a sample transaction",
        "Upload CSV (multiple rows allowed)"
    ]
)

# ------------------------------------------------------------
# Predefined sample transactions
# ------------------------------------------------------------
# Predefined samples
sample_transactions = {
    "Legit Transaction": [
        58864,11.48,-0.187085067174817,0.966631780184178,2.00782498417892,0.467856301280053,0.129698901697207,-0.562195406862846,0.876019865695857,-0.32857345251613,-0.443224822479481,-0.301762940133195,-0.153269211854297,0.653095593480255,1.46015965487421,-0.386643555410452,0.827201197364987,0.135250101277895,-0.725095538615815,-0.231571269691444,-0.23315455067351,0.100600750717467,-0.151524905797562,-0.247543138155553,-0.0747848842761469,0.375177812261376,-0.112288178958045,-0.696803708696841,-0.0934689753361401,-0.179294311110525
    ],
    "Fraud Transaction Example": [
        406,0,-2.3122265423263,1.95199201064158,-1.60985073229769,3.9979055875468,-0.522187864667764,-1.42654531920595,-2.53738730624579,1.39165724829804,-2.77008927719433,-2.77227214465915,3.20203320709635,-2.89990738849473,-0.595221881324605,-4.28925378244217,0.389724120274487,-1.14074717980657,-2.83005567450437,-0.0168224681808257,0.416955705037907,0.126910559061474,0.517232370861764,-0.0350493686052974,-0.465211076182388,0.320198198514526,0.0445191674731724,0.177839798284401,0.261145002567677,-0.143275874698919

    ]
}

data_values = None  # store values for prediction

# ------------------------------------------------------------
# Input Options
# ------------------------------------------------------------

# -------- OPTION 1: ENTER COMMA-SEPARATED VALUES --------
if method == "Enter comma-separated values (single row)":
    st.write("Enter **30 comma-separated values** (Time, Amount, V1–V28):")
    user_input = st.text_area("Input:")
    if st.button("Use these values"):
        try:
            values = [float(x.strip()) for x in user_input.split(",")]
            if len(values) != 30:
                st.error("❌ Please enter exactly 30 numeric values.")
            else:
                data_values = values
                st.success("Values processed!")
        except:
            st.error("❌ Invalid format! Ensure values are numbers separated by commas.")


# -------- OPTION 2: SELECT SAMPLE TRANSACTION --------
elif method == "Choose a sample transaction":
    choice = st.selectbox("Choose a sample:", list(sample_transactions.keys()))
    data_values = sample_transactions[choice]
    st.success("Sample loaded!")


# -------- OPTION 3: UPLOAD MULTI-ROW CSV --------
elif method == "Upload CSV (multiple rows allowed)":
    file = st.file_uploader("Upload your CSV", type=["csv"])
    if file:
        df_csv = pd.read_csv(file)

        if df_csv.shape[1] != 30:
            st.error("❌ CSV must contain exactly 30 columns: Time, Amount, V1–V28")
        else:
            data_values = df_csv
            st.success(f"CSV loaded with **{df_csv.shape[0]} rows**!")


# ------------------------------------------------------------
# Prediction Logic
# ------------------------------------------------------------
if data_values is not None:

    # Detect if single row or multiple rows
    if isinstance(data_values, list):
        # Single row from comma-separated input or sample
        columns = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
        df_input = pd.DataFrame([data_values], columns=columns)
    else:
        # Multi-row CSV
        df_input = data_values.copy()

    # ---------------------------------------------------
    # Feature Engineering
    # ---------------------------------------------------
    df_input["Amount_log"] = np.log1p(df_input["Amount"])
    df_input["Hour"] = (df_input["Time"] // 3600) % 24

    # Remove unused original columns
    df_model = df_input.drop(["Time", "Amount"], axis=1)

    # Scale
    df_scaled = scaler.transform(df_model)

    # Predict
    predictions = model.predict(df_scaled)
    probabilities = model.predict_proba(df_scaled)[:, 1]

    # ---------------------------------------------------
    # Results Output
    # ---------------------------------------------------
    st.subheader("🔍 Prediction Results")

    # Attach predictions to output table
    df_results = df_input.copy()
    df_results["Prediction"] = predictions
    df_results["Fraud_Probability"] = probabilities

    st.dataframe(df_results)

    # Summary
    fraud_count = int(sum(predictions))
    st.write(f"### 📌 Total Rows: {len(predictions)}")
    st.write(f"### ⚠️ Fraudulent Transactions Detected: **{fraud_count}**")

    # Fraud probability chart
    st.line_chart(df_results["Fraud_Probability"])

