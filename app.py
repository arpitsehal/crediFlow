import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="CrediFlow - Loan Approval Prediction",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 5px;
        min-height: 100px;
    }
    .metric-label {
        font-size: 16px;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #000;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("💳 CrediFlow - Loan Approval Prediction System")
st.markdown("---")

# Sidebar for navigation
page = st.sidebar.radio("Navigation", ["🏠 Home", "📊 Analysis", "🔮 Predict", "📈 Model Info"])

# Load data and pre-trained model
@st.cache_resource
def load_model_and_data():
    # Load the CSV data
    df = pd.read_csv("loan_approval_data.csv")
    
    # Clean the target variable
    df["Loan_Approved"] = (df["Loan_Approved"] == "Yes").astype(int)
    
    # Check if model exists, otherwise train it
    try:
        with open('loan_prediction_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
    except:
        # Train model if not found
        st.warning("Model not found. Training model...")
        
        # Data preprocessing
        df_clean = df.copy()
        
        # Fill missing values for numerical columns
        numerical_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
        numeric_imputer = SimpleImputer(strategy='mean')
        df_clean[numerical_cols] = numeric_imputer.fit_transform(df_clean[numerical_cols])
        
        # Handle categorical columns
        categorical_cols = ['Employment_Status', 'Marital_Status', 'Loan_Purpose', 'Property_Area', 'Education_Level', 'Gender', 'Employer_Category']
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        df_clean[categorical_cols] = categorical_imputer.fit_transform(df_clean[categorical_cols])
        
        # Encode categorical features
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            encoders[col] = le
        
        # Feature engineering and selection
        feature_cols = ['Applicant_Income', 'Coapplicant_Income', 'Age', 'Employment_Status',
                       'Marital_Status', 'Dependents', 'Credit_Score', 'Existing_Loans',
                       'DTI_Ratio', 'Savings', 'Collateral_Value', 'Loan_Amount', 
                       'Loan_Term', 'Loan_Purpose', 'Property_Area', 'Education_Level',
                       'Gender', 'Employer_Category']
        
        X = df_clean[feature_cols].copy()
        y = df_clean["Loan_Approved"]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Naive Bayes model
        model = GaussianNB()
        model.fit(X_train_scaled, y_train)
        
        # Save model, scaler and encoders
        with open('loan_prediction_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        with open('encoders.pkl', 'wb') as f:
            pickle.dump(encoders, f)
    
    return df, model, scaler, encoders

df, model, scaler, encoders = load_model_and_data()

# HOME PAGE
if page == "🏠 Home":
    st.header("Welcome to Credit Wise Loan Approval System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### About This System
        CrediFlow is an intelligent loan approval prediction system that uses machine learning 
        to assess loan applications efficiently and accurately.
        
        **Key Features:**
        - 🤖 Advanced ML model for prediction
        - 📊 Comprehensive data analysis
        - ⚡ Real-time prediction
        - 📈 Model performance metrics
        """)
    
    with col2:
        st.markdown("""
        ### How It Works
        1. **Input Your Data** - Enter applicant information
        2. **Model Analysis** - AI analyzes the application
        3. **Get Prediction** - Instant approval/rejection decision
        4. **See Confidence** - Understand prediction probability
        """)
        
        st.markdown("### 📊 Quick Stats")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("📋 Total Records", f"{len(df):,}")
        with col_b:
            approved = (df["Loan_Approved"] == 1).sum()
            st.metric("✅ Approved", f"{approved:,} ({approved/len(df)*100:.1f}%)")
        with col_c:
            rejected = (df["Loan_Approved"] == 0).sum()
            st.metric("❌ Rejected", f"{rejected:,} ({rejected/len(df)*100:.1f}%)")

# ANALYSIS PAGE
elif page == "📊 Analysis":
    st.header("📊 Data Analysis & Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Loan Approval Distribution")
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        approved = (df["Loan_Approved"] == 1).sum()
        rejected = (df["Loan_Approved"] == 0).sum()
        sizes = [rejected, approved]
        colors = ['#FF6B6B', '#4ECDC4']
        ax1.pie(sizes, labels=["Rejected", "Approved"], autopct="%1.1f%%", colors=colors)
        ax1.set_title("Loan Approval Rate")
        st.pyplot(fig1)
    
    with col2:
        st.subheader("Gender Distribution")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        gender_count = df["Gender"].value_counts()
        ax2 = sns.barplot(x=gender_count.index, y=gender_count.values, palette="Set2")
        ax2.set_title("Applicants by Gender")
        ax2.set_ylabel("Count")
        st.pyplot(fig2)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Income Distribution")
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        sns.histplot(data=df, x="Applicant_Income", bins=30, kde=True, ax=ax3, color='skyblue')
        ax3.set_title("Applicant Income Distribution")
        st.pyplot(fig3)
    
    with col4:
        st.subheader("Feature Correlations")
        fig4, ax4 = plt.subplots(figsize=(10, 8))
        numeric_df = df.select_dtypes(include=[np.number])
        sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax4, cbar_kws={'label': 'Correlation'})
        ax4.set_title("Feature Correlation Matrix")
        st.pyplot(fig4)
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

# PREDICTION PAGE
elif page == "🔮 Predict":
    st.header("🔮 Make a Prediction")
    st.markdown("Enter the applicant details below to get a loan approval prediction")
    
    # Create input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", sorted(df["Gender"].dropna().unique().astype(str)))
        marital_status = st.selectbox("Marital Status", sorted(df["Marital_Status"].dropna().unique().astype(str)))
        education = st.selectbox("Education Level", sorted(df["Education_Level"].dropna().unique().astype(str)))
    
    with col2:
        employment_status = st.selectbox("Employment Status", sorted(df["Employment_Status"].dropna().unique().astype(str)))
        employer_category = st.selectbox("Employer Category", sorted(df["Employer_Category"].dropna().unique().astype(str)))
        loan_purpose = st.selectbox("Loan Purpose", sorted(df["Loan_Purpose"].dropna().unique().astype(str)))
    
    with col3:
        property_area = st.selectbox("Property Area", sorted(df["Property_Area"].dropna().unique().astype(str)))
        applicant_income = st.number_input("Applicant Income ($)", min_value=0, value=50000, step=5000)
        coapplicant_income = st.number_input("Coapplicant Income ($)", min_value=0, value=0, step=5000)
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=35)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
        existing_loans = st.number_input("Existing Loans", min_value=0, value=2, step=1)
    
    with col5:
        dti_ratio = st.number_input("DTI Ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
        savings = st.number_input("Savings ($)", min_value=0, value=50000, step=5000)
        dependents = st.number_input("Dependents", min_value=0, max_value=20, value=0, step=1)
    
    with col6:
        collateral_value = st.number_input("Collateral Value ($)", min_value=0, value=100000, step=10000)
        loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=150000, step=10000)
        loan_term = st.number_input("Loan Term (months)", min_value=1, max_value=480, value=60, step=6)
    
    # Prepare prediction
    if st.button("🔮 Get Prediction", key="predict_btn", use_container_width=True):
        
        try:
            # Encode categorical variables
            input_dict = {
                'Applicant_Income': applicant_income,
                'Coapplicant_Income': coapplicant_income,
                'Age': age,
                'Employment_Status': encoders['Employment_Status'].transform([employment_status])[0],
                'Marital_Status': encoders['Marital_Status'].transform([marital_status])[0],
                'Dependents': dependents,
                'Credit_Score': credit_score,
                'Existing_Loans': existing_loans,
                'DTI_Ratio': dti_ratio,
                'Savings': savings,
                'Collateral_Value': collateral_value,
                'Loan_Amount': loan_amount,
                'Loan_Term': loan_term,
                'Loan_Purpose': encoders['Loan_Purpose'].transform([loan_purpose])[0],
                'Property_Area': encoders['Property_Area'].transform([property_area])[0],
                'Education_Level': encoders['Education_Level'].transform([education])[0],
                'Gender': encoders['Gender'].transform([gender])[0],
                'Employer_Category': encoders['Employer_Category'].transform([employer_category])[0],
            }
            
            # Create input dataframe
            input_df = pd.DataFrame([input_dict])
            feature_cols = ['Applicant_Income', 'Coapplicant_Income', 'Age', 'Employment_Status',
                           'Marital_Status', 'Dependents', 'Credit_Score', 'Existing_Loans',
                           'DTI_Ratio', 'Savings', 'Collateral_Value', 'Loan_Amount', 
                           'Loan_Term', 'Loan_Purpose', 'Property_Area', 'Education_Level',
                           'Gender', 'Employer_Category']
            
            input_scaled = scaler.transform(input_df[feature_cols])
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            # Display results
            st.markdown("---")
            st.markdown("### 📋 Prediction Result")
            
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                if prediction == 1:
                    st.success("✅ LOAN APPROVED")
                    st.metric("Approval Confidence", f"{probability[1]*100:.2f}%", delta="Positive")
                else:
                    st.error("❌ LOAN REJECTED")
                    st.metric("Rejection Confidence", f"{probability[0]*100:.2f}%", delta="Negative")
            
            with col_res2:
                fig_prob, ax_prob = plt.subplots(figsize=(8, 4))
                labels = ['Rejected', 'Approved']
                colors_prob = ['#FF6B6B', '#4ECDC4']
                ax_prob.barh(labels, probability, color=colors_prob)
                ax_prob.set_xlabel('Probability')
                ax_prob.set_title('Prediction Probability')
                st.pyplot(fig_prob)
            
            st.markdown("---")
            st.markdown("### 📊 Applicant Summary")
            summary = pd.DataFrame({
                'Feature': ['Gender', 'Marital Status', 'Education', 'Employment Status', 'Employer Category',
                           'Loan Purpose', 'Property Area', 'Income', 'Co-applicant Income', 'Age',
                           'Credit Score', 'Existing Loans', 'DTI Ratio', 'Savings', 
                           'Collateral Value', 'Loan Amount', 'Loan Term', 'Dependents'],
                'Value': [str(gender), str(marital_status), str(education), str(employment_status), str(employer_category),
                         str(loan_purpose), str(property_area), str(f"${applicant_income:,}"), str(f"${coapplicant_income:,}"), 
                         str(age), str(credit_score), str(existing_loans), str(f"{dti_ratio:.2f}"), str(f"${savings:,}"),
                         str(f"${collateral_value:,}"), str(f"${loan_amount:,}"), str(f"{loan_term} months"), str(dependents)]
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

# MODEL INFO PAGE
elif page == "📈 Model Info":
    st.header("📈 Model Information & Performance")
    
    st.markdown("""
    ### About the Model
    This system uses a **Gaussian Naive Bayes** classifier for loan approval prediction.
    
    **Why Naive Bayes?**
    - Fast and efficient predictions
    - Works well with categorical and numerical features
    - Provides probability estimates
    - Requires minimal training data
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Model Training
        - **Algorithm**: Gaussian Naive Bayes
        - **Training Data**: Loan applicant records
        - **Train/Test Split**: 80/20
        - **Feature Scaling**: StandardScaler
        """)
    
    with col2:
        st.markdown("""
        ### Input Features
        1. Applicant Income
        2. Coapplicant Income
        3. Age
        4. Employment Status
        5. Marital Status
        6. Dependents
        7. Credit Score
        8. Existing Loans
        9. DTI Ratio
        10. Savings
        11. Collateral Value
        12. Loan Amount
        13. Loan Term
        14. Loan Purpose
        15. Property Area
        16. Education Level
        17. Gender
        18. Employer Category
        """)
    
    st.markdown("---")
    
    try:
        # Prepare data for metrics calculation
        df_clean = df.copy()
        
        # Fill NaN values before encoding
        categorical_cols = ['Employment_Status', 'Marital_Status', 'Loan_Purpose', 'Property_Area', 'Education_Level', 'Gender', 'Employer_Category']
        for col in categorical_cols:
            # First replace NaN and 'nan' string with mode
            mode_val = df_clean[col].mode()
            fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
            df_clean[col] = df_clean[col].replace({np.nan: fill_val, 'nan': fill_val, None: fill_val})
            df_clean[col] = df_clean[col].fillna(fill_val)
        
        # Encode categorical features for calculation
        for col in categorical_cols:
            # Skip encoding if there are still any NaN-like values
            df_clean[col] = df_clean[col].astype(str).replace(['nan', 'None'], 'Unknown')
            try:
                df_clean[col] = encoders[col].transform(df_clean[col])
            except ValueError:
                # If encoder fails, use a default encoding
                pass
    
        feature_cols = ['Applicant_Income', 'Coapplicant_Income', 'Age', 'Employment_Status',
                       'Marital_Status', 'Dependents', 'Credit_Score', 'Existing_Loans',
                       'DTI_Ratio', 'Savings', 'Collateral_Value', 'Loan_Amount', 
                       'Loan_Term', 'Loan_Purpose', 'Property_Area', 'Education_Level',
                       'Gender', 'Employer_Category']
        
        X_features = df_clean[feature_cols].fillna(0)
        y_true = df_clean["Loan_Approved"]
        X_scaled = scaler.transform(X_features)
        y_pred = model.predict(X_scaled)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            st.metric("Accuracy", f"{accuracy:.4f}", delta=f"{accuracy*100:.2f}%")
        with col_m2:
            st.metric("Precision", f"{precision:.4f}", delta=f"{precision*100:.2f}%")
        with col_m3:
            st.metric("Recall", f"{recall:.4f}", delta=f"{recall*100:.2f}%")
        with col_m4:
            st.metric("F1 Score", f"{f1:.4f}", delta=f"{f1*100:.2f}%")
        
        # Confusion Matrix
        st.markdown("---")
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                    xticklabels=['Rejected', 'Approved'],
                    yticklabels=['Rejected', 'Approved'],
                    ax=ax_cm, annot_kws={'size': 16})
        ax_cm.set_ylabel('True Label')
        ax_cm.set_xlabel('Predicted Label')
        st.pyplot(fig_cm)
        
        st.markdown("""
        **Confusion Matrix Interpretation:**
        - **True Negatives (TN)**: Correctly rejected applications
        - **False Positives (FP)**: Rejected applications predicted as approved
        - **False Negatives (FN)**: Approved applications predicted as rejected
        - **True Positives (TP)**: Correctly approved applications
        """)
    
    except Exception as e:
        st.error(f"Error calculating model metrics: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>CrediFlow © 2026 | Loan Approval Prediction System</p>
    <p>Built with Streamlit & Machine Learning</p>
</div>
""", unsafe_allow_html=True)
