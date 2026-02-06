import pandas as pd
import numpy as np

df = pd.read_csv('loan_approval_data.csv')

# Get approved loans
approved = df[df['Loan_Approved'] == 'Yes']
rejected = df[df['Loan_Approved'] == 'No']

print('=== APPROVED LOANS STATISTICS ===')
print(f'Count: {len(approved)}')
print(f'Avg Income: ${approved["Applicant_Income"].mean():,.0f}')
print(f'Avg Credit Score: {approved["Credit_Score"].mean():.0f}')
print(f'Avg DTI Ratio: {approved["DTI_Ratio"].mean():.2f}')
print(f'Avg Savings: ${approved["Savings"].mean():,.0f}')
print(f'Avg Collateral: ${approved["Collateral_Value"].mean():,.0f}')
print(f'Avg Loan Amount: ${approved["Loan_Amount"].mean():,.0f}')
print()
print('=== REJECTED LOANS STATISTICS ===')
print(f'Count: {len(rejected)}')
print(f'Avg Income: ${rejected["Applicant_Income"].mean():,.0f}')
print(f'Avg Credit Score: {rejected["Credit_Score"].mean():.0f}')
print(f'Avg DTI Ratio: {rejected["DTI_Ratio"].mean():.2f}')
print()
print('=== BEST APPROVED LOAN (By Income) ===')
top_approved = approved.nlargest(1, 'Applicant_Income').iloc[0]
print(f'Gender: {top_approved["Gender"]}')
print(f'Employment: {top_approved["Employment_Status"]}')
print(f'Marital: {top_approved["Marital_Status"]}')
print(f'Education: {top_approved["Education_Level"]}')
print(f'Employer: {top_approved["Employer_Category"]}')
print(f'Income: ${top_approved["Applicant_Income"]:,.0f}')
print(f'Co-Income: ${top_approved["Coapplicant_Income"]:,.0f}')
print(f'Age: {top_approved["Age"]:.0f}')
print(f'Credit Score: {top_approved["Credit_Score"]:.0f}')
print(f'Savings: ${top_approved["Savings"]:,.0f}')
print(f'Collateral: ${top_approved["Collateral_Value"]:,.0f}')
print(f'DTI Ratio: {top_approved["DTI_Ratio"]:.2f}')
print(f'Loan Amount: ${top_approved["Loan_Amount"]:,.0f}')
print(f'Loan Term: {top_approved["Loan_Term"]:.0f} months')
print()
print('=== GOOD MIDDLE-CLASS APPROVED LOAN ===')
mid_approved = approved[
    (approved["Applicant_Income"] > 40000) & 
    (approved["Applicant_Income"] < 80000) &
    (approved["Credit_Score"] > 650) &
    (approved["DTI_Ratio"] < 0.4)
].iloc[0] if len(approved[
    (approved["Applicant_Income"] > 40000) & 
    (approved["Applicant_Income"] < 80000) &
    (approved["Credit_Score"] > 650) &
    (approved["DTI_Ratio"] < 0.4)
]) > 0 else None

if mid_approved is not None:
    print(f'Gender: {mid_approved["Gender"]}')
    print(f'Employment: {mid_approved["Employment_Status"]}')
    print(f'Marital: {mid_approved["Marital_Status"]}')
    print(f'Education: {mid_approved["Education_Level"]}')
    print(f'Employer: {mid_approved["Employer_Category"]}')
    print(f'Income: ${mid_approved["Applicant_Income"]:,.0f}')
    print(f'Co-Income: ${mid_approved["Coapplicant_Income"]:,.0f}')
    print(f'Credit Score: {mid_approved["Credit_Score"]:.0f}')
    print(f'DTI Ratio: {mid_approved["DTI_Ratio"]:.2f}')
    print(f'Savings: ${mid_approved["Savings"]:,.0f}')
    print(f'Loan Amount: ${mid_approved["Loan_Amount"]:,.0f}')
