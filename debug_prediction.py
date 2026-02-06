import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the model, scaler, and encoders
with open('loan_prediction_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# Your input from the form
input_dict = {
    'Applicant_Income': 20000,
    'Coapplicant_Income': 7000,
    'Age': 37,
    'Employment_Status': encoders['Employment_Status'].transform(['Salaried'])[0],
    'Marital_Status': encoders['Marital_Status'].transform(['Married'])[0],
    'Dependents': 0,
    'Credit_Score': 900,
    'Existing_Loans': 1,
    'DTI_Ratio': 0.27,
    'Savings': 14000,
    'Collateral_Value': 28000,
    'Loan_Amount': 15000,
    'Loan_Term': 24,
    'Loan_Purpose': encoders['Loan_Purpose'].transform(['Home'])[0],
    'Property_Area': encoders['Property_Area'].transform(['Urban'])[0],
    'Education_Level': encoders['Education_Level'].transform(['Graduate'])[0],
    'Gender': encoders['Gender'].transform(['Male'])[0],
    'Employer_Category': encoders['Employer_Category'].transform(['Government'])[0],
}

# Create input dataframe in correct order
feature_cols = ['Applicant_Income', 'Coapplicant_Income', 'Age', 'Employment_Status',
               'Marital_Status', 'Dependents', 'Credit_Score', 'Existing_Loans',
               'DTI_Ratio', 'Savings', 'Collateral_Value', 'Loan_Amount', 
               'Loan_Term', 'Loan_Purpose', 'Property_Area', 'Education_Level',
               'Gender', 'Employer_Category']

input_df = pd.DataFrame([input_dict])

print("Input values:")
for col in feature_cols:
    print(f"  {col}: {input_df[col].values[0]}")

# Scale
input_scaled = scaler.transform(input_df[feature_cols])
print("\nScaled values:", input_scaled[0])

# Predict
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0]

print(f"\nPrediction: {prediction}")
print(f"Probability: {probability}")
print(f"Approved Prob: {probability[1]:.4f}")
print(f"Rejected Prob: {probability[0]:.4f}")

if prediction == 1:
    print("\n✅ APPROVED")
else:
    print("\n❌ REJECTED")

# Let's also check the training data stats
df = pd.read_csv('loan_approval_data.csv')
print("\n=== Training Data Stats ===")
print(f"Credit Score range: {df['Credit_Score'].min()} - {df['Credit_Score'].max()}")
print(f"Max Credit Score in data: {df['Credit_Score'].max()}")
print(f"Your Credit Score: 900 (outside range!)")
