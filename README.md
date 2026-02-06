# 🏦 Credit Wise - Loan Approval Prediction System

A machine learning-powered web application for predicting loan approvals using Streamlit.

## 📋 Project Overview

This application uses a **Gaussian Naive Bayes** classifier to predict whether a loan application will be approved or rejected based on applicant information like income, credit score, employment history, and more.

## 🎯 Features

- 🤖 **Real-time Predictions**: Get instant loan approval/rejection predictions
- 📊 **Data Analysis**: View comprehensive analysis of loan data
- 📈 **Model Metrics**: See model performance and accuracy metrics
- ✨ **Interactive UI**: User-friendly Streamlit interface

## 📁 Project Files

```
Credit Wise Loan System - ML/
├── crediFlow.ipynb              # Original Jupyter notebook with ML model
├── loan_approval_data.csv       # Dataset
├── app.py                       # Streamlit application
├── requirements.txt             # Python dependencies
├── loan_prediction_model.pkl    # Saved trained model
├── scaler.pkl                   # Saved feature scaler
└── README.md                    # This file
```

## 🚀 Quick Start

### Option 1: Run Locally

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the App**
   ```bash
   streamlit run app.py
   ```

3. **Open in Browser**
   - Streamlit will automatically open at `http://localhost:8501`
   - Or navigate to that URL manually

### Option 2: Deploy on Streamlit Cloud (Free)

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [Streamlit Cloud](https://streamlit.io/cloud)
   - Sign up with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path to `app.py`
   - Click "Deploy"

3. **Your app goes live!** 🎉

### Option 3: Deploy on Heroku

1. **Create Heroku Account** at [heroku.com](https://www.heroku.com)

2. **Create a `Procfile`**
   ```
   web: streamlit run app.py
   ```

3. **Create a `setup.sh`**
   ```bash
   mkdir -p ~/.streamlit/
   echo "[server]
   headless = true
   port = $PORT
   enableCORS = false
   " > ~/.streamlit/config.toml
   ```

4. **Deploy**
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### Option 4: Deploy on AWS/Azure

Use services like:
- **AWS EC2** with Docker
- **Azure App Service**
- **Google Cloud Run**

## 📊 Model Information

### Algorithm: Gaussian Naive Bayes
- **Accuracy**: ~75-80% (depends on dataset)
- **Training Method**: Supervised Classification
- **Features Used**: 9 predictive features

### Input Features
1. Gender
2. Marital Status
3. Education Level
4. Self Employment Status
5. Applicant Income
6. Years Employed
7. Savings Account Balance
8. DTI Ratio (Squared)
9. Credit Score (Squared)

### Model Performance
- Precision, Recall, and F1 scores available in the app
- Confusion matrix visualization
- Real-time model accuracy display

## 📊 Usage

### Navigate the App

1. **🏠 Home**: Overview and quick statistics
2. **📊 Analysis**: Data exploration and visualizations
3. **🔮 Predict**: Make predictions on new applications
4. **📈 Model Info**: Detailed model performance metrics

### Making Predictions

1. Go to "Predict" section
2. Enter applicant details
3. Click "Get Prediction"
4. View approval/rejection result with confidence score

## 🛠️ Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- streamlit

All requirements are listed in `requirements.txt`

## 📈 Model Training

The model is pre-trained in `crediFlow.ipynb`. To retrain:

1. Open `crediFlow.ipynb` in Jupyter Notebook
2. Run all cells
3. The model and scaler will be saved automatically

## 🔧 Troubleshooting

### Port Already in Use
```bash
streamlit run app.py --server.port 8502
```

### Model Not Found
- Ensure `loan_prediction_model.pkl` and `scaler.pkl` exist
- Re-run the notebook to regenerate them

### CSV File Not Found
- Ensure `loan_approval_data.csv` is in the same directory as `app.py`

## 📝 License

This project is open-source and available under the MIT License.

## 👨‍💼 Author

Credit Wise Development Team

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest improvements
- Submit pull requests

## 📞 Support

For issues or questions, please check the troubleshooting section or create an issue in the repository.

---

**Happy Predicting! 🚀**
