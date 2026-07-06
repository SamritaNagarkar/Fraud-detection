# Fraud-detection
Fraud Detection — Logistic Regression Pipeline

A supervised machine learning pipeline that classifies financial transactions as fraudulent or legitimate, deployed as an interactive Streamlit app.

Overview-
This project builds an end-to-end fraud detection system on a large-scale, heavily imbalanced financial transactions dataset. The pipeline handles preprocessing, class imbalance, model training, and evaluation, and is deployed as a Streamlit app for real-time predictions.
Dataset: Fraud Detection Dataset (Kaggle) — 6.3M+ transaction records, with fraudulent transactions making up roughly 0.13% of the data.

Problem-
Fraud detection is a classic extreme class-imbalance problem: with fraud cases this rare, a model can score 99%+ accuracy by predicting "not fraud" every time while catching zero actual fraud. The goal here was to build a model that actually identifies fraud, and to evaluate it with metrics that reflect that goal honestly rather than a headline accuracy number.

Approach-
EDA: Explored transaction type distributions, fraud rate by transaction type, amount distributions, and balance-difference patterns to understand what separates fraudulent from legitimate transactions.
Feature engineering: Used transaction type, amount, and sender/receiver account balances (before and after the transaction) as model inputs.
Preprocessing: StandardScaler for numerical features, OneHotEncoder for transaction type, combined via a ColumnTransformer.
Class imbalance: Handled using class_weight='balanced' in LogisticRegression, rather than under/oversampling, to keep the full dataset in play.
Split: 70/30 train/test split, stratified on the target to preserve the fraud ratio in both sets.
Deployment: Trained pipeline exported with joblib and served through a Streamlit app for interactive predictions.


Results-
MetricScoreROC-AUC0.9897Recall (fraud class)0.95Precision (fraud class)0.02Accuracy0.95
Why these numbers, read together, and not accuracy alone:
Accuracy is not a meaningful metric on data this imbalanced — a model predicting "not fraud" for every transaction would already score ~99.87%. ROC-AUC (0.9897) is the more reliable single-number metric here, since it measures the model's ability to rank fraud vs. legitimate transactions independent of the class imbalance.
The model catches 95% of actual fraud cases (recall), which is the right design priority for fraud detection — missing fraud is typically far costlier than a false alarm. The tradeoff is low precision (2%): most transactions flagged as fraud are false positives. In a real deployment, this model is better suited to feeding a second-stage review queue than to auto-blocking transactions outright. A natural next step would be threshold tuning to balance precision and recall against the actual cost of a false positive vs. a missed fraud case.

Repository Structure-

├── analysis_model.ipynb          # EDA, preprocessing, training, and evaluation
├── fraud_detection.py            # Streamlit app for interactive predictions
├── fraud_detection_model.pkl     # Trained pipeline (preprocessing + logistic regression)
├── requirements.txt              # Dependencies
└── README.md

Running the App Locally-
Clone the repository-
bash   git clone https://github.com/SamritaNagarkar/Fraud-detection.git
   cd fraud-detection-logistic-regression

Install dependencies-
bash   pip install -r requirements.txt

Run the Streamlit app-
bash   streamlit run fraud_detection.py

Enter transaction details in the browser UI and click Predict.


Tech Stack-
Python, pandas, NumPy, scikit-learn (Pipeline, ColumnTransformer, LogisticRegression), Matplotlib, Seaborn, Streamlit, Joblib.

Limitations & Next Steps-
Precision on the fraud class is low; a production version would need threshold tuning or a second-stage model to reduce false positives.
The dataset is simulated (PaySim-style), not real transaction data, so real-world performance may differ.
Next steps: try tree-based models (XGBoost, Random Forest) for comparison, add SHAP-based explainability for flagged transactions, and tune the classification threshold against a defined cost matrix.
