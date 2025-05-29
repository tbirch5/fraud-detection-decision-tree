import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Fraud Detection System", page_icon="🔎")

# Load model
try:
    model = joblib.load('decision_tree_model.pkl')
    st.success("✅ Model loaded successfully.")
except FileNotFoundError:
    st.error("❌ Model file not found. Please make sure 'decision_tree_model.pkl' exists.")
    st.stop()

st.title('🔎 Fraud Detection System')
st.write("Input the transaction details below to predict whether it's **Fraud** or **Not Fraud**:")

# Time (make sure this shows up!)
time = st.number_input('⏱️ Time (seconds since first transaction)', min_value=0.0, max_value=200000.0, value=0.0, key="unique_time_input")


# V1 to V28
features = [time]
for i in range(1, 29):
    features.append(st.number_input(f'V{i}', value=0.0, key=f'v{i}'))

# Amount
amount = st.number_input('💰 Amount', value=0.0, key="amount")
features.append(amount)

# Convert to input array
input_data = np.array([features])

# Show input shape for debugging
st.caption(f"🧪 Input shape: {input_data.shape} | Model expects: {model.n_features_in_} features")

# debug statement
st.write("🧪 Features sent to model:", features)

# debug statement
st.caption(f"Shape: {input_data.shape} | Expected: {model.n_features_in_} features")


# Predict
if st.button('Predict'):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error('🚨 Fraud Detected!')
    else:
        st.success('✅ Transaction is legitimate.')
