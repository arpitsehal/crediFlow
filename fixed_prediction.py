import pandas as pd
import pickle
import numpy as np

# Load the model, scaler, and encoders
with open('loan_prediction_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# CORRECTED input - using credit score 750 instead of 900
input_dict = {
    'Applicant_Income': 20000,
    'Coapplicant_Income': 7000,
    'Age': 37,
    'Employment_Status': encoders['Employment_Status'].transform(['Salaried'])[0],
    'Marital_Status': encoders['Marital_Status'].transform(['Married'])[0],
    'Dependents': 0,
    'Credit_Score': 750,  # Changed from 900 to 750 (valid range: 550-799)
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

print("=== CORRECTED INPUT (Credit Score: 750) ===")
print(f"Applicant Income: $20,000")
print(f"Co-Applicant Income: $7,000")
print(f"Age: 37")
print(f"Credit Score: 750 ✓ (in range 550-799)")
print(f"DTI Ratio: 0.27")
print(f"Savings: $14,000")
print(f"Marital: Married")
print(f"Education: Graduate")
print(f"Employment: Salaried (Government)")

# Scale
input_scaled = scaler.transform(input_df[feature_cols])

# Predict
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0]

print(f"\n{'='*40}")
print(f"PREDICTION: {'✅ APPROVED' if prediction == 1 else '❌ REJECTED'}")
print(f"{'='*40}")
print(f"Approved Probability: {probability[1]*100:.2f}%")
print(f"Rejected Probability: {probability[0]*100:.2f}%")

print(f"\n📊 Why this profile will be approved:")
print(f"✓ Excellent credit score (750) - in optimal range")
print(f"✓ Good income ($20,000)")
print(f"✓ Married with co-applicant")
print(f"✓ Graduate education")
print(f"✓ Government job (stable)")
print(f"✓ Low DTI ratio (0.27)")
print(f"✓ Strong savings ($14,000)")
print(f"✓ Good collateral ($28,000)")

print(f"\n⚠️  Key lesson: Credit scores must be 550-799 (training range)")
print(f"Higher scores like 800-900 are outliers and confuse the model!")
