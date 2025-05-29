import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Fraud Detection System", page_icon="🔎")

# Load trained model
try:
    model = joblib.load('decision_tree_model.pkl')
    st.success("✅ Model loaded successfully.")
except FileNotFoundError:
    st.error("❌ Model file not found. Please make sure 'decision_tree_model.pkl' exists.")
    st.stop()

# App Title
st.title('🔎 Fraud Detection System')

st.write("Input the transaction details below to predict whether it's **Fraud** or **Not Fraud**:")

# Input fields

# Time Feature

time = st.number_input('⏱️ Time (seconds since first transaction)', min_value=0.0, max_value=200000.0, value=0.0, key="unique_time_input")


# V1 through V28 Features
features = [time]
for i in range(1, 29):
    features.append(st.number_input(f'V{i}', value=0.0, key=f'v{i}'))

# Amount feature
amount = st.number_input('💰 Amount', value=0.0, key="amount")
features.append(amount)

# Convert features into model input shape
input_data = np.array([features])

# Show input shape for debugging
st.caption(f"🧪 Input shape: {input_data.shape} | Model expects: {model.n_features_in_} features")

# debug statement
# st.write("🧪 Features sent to model:", features)

# debug statement
# st.caption(f"Shape: {input_data.shape} | Expected: {model.n_features_in_} features")

# === Predict Button with Probability Output ===
if st.button('Predict'):
    try:
        prediction = model.predict(input_data)
        proba = model.predict_proba(input_data)[0] # Get [legit, fraud] probabilities

        st.write("### 🔍 Model Prediction Probability")
        st.write(f"- Legitimate: `{proba[0]*100:.2f}%`")
        st.write(f"- Fraudulent: `{proba[1]*100:.2f}%`")

        # Show results
        if prediction[0] == 1:
            st.error('🚨 Fraud Detected!')
        else:
            st.success('✅ Transaction is legitimate.')

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# === CSV Upload + Insights Section ===
st.write("---")
st.subheader("📂 Upload CSV to Analyze Fraud Distribution")

uploaded_file = st.file_uploader("Upload a CSV file (must include a 'Class' column)", type="csv")

if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)

        if 'Class' in uploaded_df.columns:
            st.success("✅ File loaded successfully.")

            # Display Fraud vs. Non-Fraud count
            st.write("### 📊 Fraud vs. Non-Fraud Count")
            class_counts = uploaded_df['Class'].value_counts()
            st.bar_chart(class_counts)

            # Show basic stats
            st.write("### 🧮 Descriptive Statistics")
            st.write(uploaded_df.describe())

        else:
            st.error("❌ Column 'Class' not found in the uploaded CSV.")
    except Exception as e:
        st.error(f"Error reading file: {e}") 
