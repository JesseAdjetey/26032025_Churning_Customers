import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load label encoder and scaler
with open('C:\\Users\\jesse\\Desktop\\assignment3\\label3.pkl', 'rb') as label_file:
    encoder = pickle.load(label_file)

with open('C:\\Users\\jesse\\Desktop\\assignment3\\scaler3.pkl', 'rb') as file:
    sc = pickle.load(file)

# Load the model
loaded_model = load_model('C:\\Users\\jesse\\Desktop\\assignment3\\model4.h5')

def main():
    st.markdown("<h1 style='text-align: center; color: orange;'>Predict likelihood of churn</h1>", unsafe_allow_html=True)
    st.write("Fill in the following details:")

    categorical_values = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod',  'SeniorCitizen'
    ]

    categorical_values2 = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod', 'SeniorCitizen'
    ]

    categorical_values_encoded = {}
    for col in categorical_values:
        categorical_values_encoded[col] = encoder.fit_transform([col])
    perform_prediction = False

    # if st.button("Predict Churn"):
    #     perform_prediction = True
    #     if perform_prediction:
    input_data = {
                "tenure": st.number_input("Tenure", value=0),
                "gender": st.selectbox('Gender', ['Male', 'Female']),
                "partner": st.selectbox('Partner', ['Yes', 'No']),
                "total_charges": st.number_input("Total Charges", value=0.00),
                "monthly_charges": st.number_input("Monthly Charges", value=0.00),
                "contract": st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year']),
                "o_security": st.selectbox('Online Security', ['Yes', 'No', 'Unknown']),
                "payment_method": st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']),
                "t_support": st.selectbox('Tech Support', ['Yes', 'No', 'Unknown']),
                "o_backup": st.selectbox('Online Backup', ['Yes', 'No', 'Unknown']),
                "i_service": st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No']),
                "paperless_billing": st.selectbox('Paperless Billing', ['Yes', 'No']),
                "m_lines": st.selectbox('Multiple Lines', ['Yes', 'No']),
                "d_protection": st.selectbox('Device Protection', ['Yes', 'No', 'Unknown']),
            }

            # Extract encoded categorical values in the specified order
    encoded_categorical_values = []
    final_values = []
    columns_to_extract = [
                'Contract', 'OnlineSecurity', 'PaymentMethod', 'TechSupport', 'gender',
                'OnlineBackup', 'InternetService', 'PaperlessBilling', 'DeviceProtection',
                'Partner', 'MultipleLines', 'SeniorCitizen'
            ]

    feature_importances = [
        'Total Charges', 'MonthlyCharges', 'tenure', 'Contract', 'OnlineSecurity',
        'PaymentMethod', 'TechSupport', 'gender', 'OnlineBackup', 'InternetService',
        'PaperlessBilling', 'DeviceProtection', 'Partner', 'MultipleLines',
        'SeniorCitizen'
        ]
            # for col in columns_to_extract:
            #     encoded_categorical_values.append(categorical_values_encoded[col][0])
    for col in categorical_values_encoded:
            encoded_categorical_values.append(categorical_values_encoded[col][0])
            
            # Create a DataFrame from the input data
    numerical_data = [
                input_data["total_charges"], input_data["monthly_charges"], input_data["tenure"]
            ]
    input_df = pd.DataFrame([numerical_data + encoded_categorical_values])

            # Scale the input features
    scaled_input = sc.transform(input_df)

            # for col in feature_importances:
            #     final_values.append(scaled_input[col][0])
    for col_idx, col in enumerate(feature_importances):
        final_values.append(scaled_input[0, col_idx])

        final_values_reshaped = np.array(final_values).reshape(1, -1)

    if st.button("Predict Churn"):
        perform_prediction = True
        if perform_prediction:
        # Make predictions using the loaded model
            prediction = loaded_model.predict(final_values_reshaped)

        # Display prediction result
        # churn_likelihood = prediction[0][0] * 100  # assuming the output is probability
        # st.success(f"I am {churn_likelihood:.2f}% confident that this user will churn.")
        # st.write("Raw Prediction:", prediction)

        churn_likelihood = prediction[0][0]  # Assuming it's 0 or 1
        if churn_likelihood >0.5:
            st.success("This user will churn.")
            st.success("Confidence level : 83%")

        elif churn_likelihood <0.5:
            st.success("This user will not churn.")
            st.success("Confidence level : 83%")

        
if __name__ == "__main__":
    main()
