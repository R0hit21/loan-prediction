import streamlit as st
import pandas as pd
import pickle

# Loading the pickled model
with open('loan_prediction_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Defining the Streamlit app
def main():
    st.title('Loan Prediction Web App')
    st.write('Enter customer details to predict loan eligibility:')

    # User input fields
    gender = st.selectbox('Gender', ['Male', 'Female'])
    married = st.selectbox('Marital Status', ['Yes', 'No'])
    education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
    self_employed = st.selectbox('Self Employed', ['Yes', 'No'])
    applicant_income = st.number_input('Applicant Income')
    coapplicant_income = st.number_input('Coapplicant Income')
    loan_amount = st.number_input('Loan Amount')
    loan_term = st.number_input('Loan Term')
    credit_history = st.selectbox('Credit History', [0, 1])

    # Predicting loan status based on user input
    if st.button('Predict'):
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Married': [married],
            'Education': [education],
            'Self_Employed': [self_employed],
            'ApplicantIncome': [applicant_income],
            'CoapplicantIncome': [coapplicant_income],
            'LoanAmount': [loan_amount],
            'Loan_Amount_Term': [loan_term],
            'Credit_History': [credit_history]
        })
        prediction = model.predict(input_data)
        st.write('Loan Status:', prediction[0])

# Running the Streamlit app
if __name__ == '__main__':
    main()
