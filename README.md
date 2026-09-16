# 📉 Customer Churn Prediction — Logistic Regression vs ANN

A machine learning project predicting customer churn for a telecom company using the Telco Customer Churn dataset. The project walks through full EDA, data cleaning, and preprocessing, then compares a classical **Logistic Regression** baseline against a **Keras Artificial Neural Network (ANN)**.

## 📊 Dataset

- **Telco Customer Churn dataset** (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)
- **7,043 customers**, 21 columns — demographic info, account details (tenure, contract type, payment method), subscribed services (internet, phone, streaming, tech support), billing (`MonthlyCharges`, `TotalCharges`), and the target `Churn` (Yes/No)
- Class distribution: **5,174 No-churn** vs **1,869 Churn** (~26.5% churn rate — moderately imbalanced)

## 🛠️ Workflow

1. **Import Libraries** — NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, TensorFlow/Keras
2. **Load Dataset**
3. **Exploratory Data Analysis (EDA)**
   - Basic understanding: shape, dtypes, `describe()`, class balance
   - Missing value and duplicate checks
   - Fixed `TotalCharges` (loaded as text, converted to numeric)
   - **Univariate analysis** — churn count distribution
   - **Bivariate analysis** — churn vs. contract type, tenure, monthly charges, internet service, and tech support
4. **Data Cleaning**
   - Dropped rows with missing values (11 blank `TotalCharges` entries)
   - Dropped the non-predictive `customerID` column
5. **Data Preprocessing**
   - Train-test split (67/33, `random_state=42`)
   - **One-Hot Encoding** for 15 categorical columns (gender, contract, payment method, service subscriptions, etc.)
   - **Standard scaling** for numerical columns (`SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`)
   - Combined scaled numerical + encoded categorical features into the final training matrix
   - Mapped target labels `Churn`: `No → 0`, `Yes → 1`
6. **Model Selection**
   - **Logistic Regression** — baseline classifier
   - **ANN** — Keras Sequential model: `Dense(16, relu) → Dense(8, relu) → Dense(1, sigmoid)`, trained with SGD optimizer and binary cross-entropy loss for 50 epochs

## 📈 Results

| Model | Test Accuracy |
|---|---|
| **Logistic Regression** | **80.05%** |
| ANN (Keras) | 79.62% |

> The Logistic Regression baseline slightly edges out the ANN here. On tabular data with a modest number of engineered features, simpler linear models often match or beat small neural networks — the ANN's extra capacity doesn't pay off without more data, tuning, or regularization. This makes a good case for always benchmarking against a simple baseline before reaching for deep learning.

## 🧰 Tech Stack

- **Python 3**
- **Pandas** & **NumPy** — data handling
- **Matplotlib** & **Seaborn** — EDA visualizations
- **Scikit-learn** — preprocessing (OneHotEncoder, StandardScaler), Logistic Regression, and metrics
- **TensorFlow / Keras** — ANN model

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

### Data

Download the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the working directory, or update the file path in the notebook.

### Run

```bash
jupyter notebook ANN_Customer_Churn_Project.ipynb
```

Run all cells sequentially to reproduce the EDA, preprocessing, and model results.

## 📁 Project Structure

```
.
├── ANN_Customer_Churn_Project.ipynb   # Main notebook: EDA, preprocessing, LR & ANN training/comparison
└── README.md
```

## 🔮 Future Improvements

- Address class imbalance (SMOTE, class weighting) to improve recall on the churn class
- Add precision/recall/F1 and confusion matrix reporting for both models (metrics were imported but not fully used)
- Hyperparameter tuning for the ANN (optimizer, layer sizes, dropout, epochs) and Logistic Regression (`C`, regularization)
- Try tree-based models (Random Forest, XGBoost, LightGBM) which typically perform strongly on tabular churn data
- Feature importance / SHAP analysis to identify key churn drivers

## 👤 Author

**Muhammad Ibrahim**

- 📧 Email: [mibrahim.seng@gmail.com](mailto:mibrahim.seng@gmail.com)
- 💼 LinkedIn: [muhammad-ibrahim-python](https://www.linkedin.com/in/muhammad-ibrahim-python)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
