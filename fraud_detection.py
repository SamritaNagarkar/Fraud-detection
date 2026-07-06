import streamlit as st
import pandas as pd
import joblib

model = joblib.load('fraud_detection_model.pkl')

st.title("Fraud Detection App")

st.markdown("Please enter the transaction details below to predict whether it is fraudulent or not, Use prediction button to predict.")
st.divider()

transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"])
amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)
OldBalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=10000.0)
NewBalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, value=9000.0)
OldBalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0)
NewBalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, value=0.0)

if st.button("Predict"):
    input_data = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": OldBalanceOrg,
        "newbalanceOrig": NewBalanceOrig,
        "oldbalanceDest": OldBalanceDest,
        "newbalanceDest": NewBalanceDest
        }])
    
    prediction = model.predict(input_data)[0]
    st.subheader(f"prediction:'{int(prediction)}' ")

    if prediction == 1:
        st.error("The transaction can be fraud.")
    else:
        st.success("The transaction looks like it's not a fraud.")