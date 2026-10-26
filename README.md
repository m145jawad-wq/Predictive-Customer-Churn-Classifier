Predictive Customer Churn Classifier
Task 3 — Random Forest + XGBoost + SMOTE + ROC-AUC
Overview
This project predicts whether a telecommunications customer is likely to leave a company (customer churn). It trains and compares two machine-learning models:
- Random Forest
- XGBoost
The workflow includes data cleaning, preprocessing, handling class imbalance with SMOTE, model evaluation, cross-validation, feature-importance analysis, and saving the best-performing model.
Features
- Loads the Telco Customer Churn CSV dataset.
- Converts TotalCharges to a numeric column and handles invalid or blank values as missing data.
- Removes the customer ID column when present.
- Converts the Churn target (Yes/No) to binary labels (1/0).
- Splits the data into training and test sets using stratification.
- Imputes missing values, scales numeric features, and one-hot encodes categorical features.
- Applies SMOTE to the training data within each model pipeline.
- Compares models using accuracy, precision, recall, F1-score, and ROC-AUC.
- Displays ROC curves, confusion matrices, classification reports, and top-10 feature-importance charts.
- Runs stratified 5-fold cross-validation using ROC-AUC.
- Saves the best model and CSV evaluation outputs.
Requirements
- Python 3.9 or later recommended
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- imbalanced-learn
- XGBoost
- joblib
Install dependencies:
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost joblib
Dataset
The script looks for one of these files in the same folder:
- WA_Fn-UseC_-Telco-Customer-Churn.csv
- Telco-Customer-Churn.csv
If you do not already have the dataset, the script's error message provides this download source:

https://raw.githubusercontent.com/IBM/Customer-Churn-Data/main/Telco-Customer-Churn.csv

Save the CSV in the same folder as the Python script before running it. The dataset should include a Churn column containing Yes and No values.
Run the Project
1. Download or clone the project files.
2. Put the dataset CSV in the same folder as task3_customer_churn_classifier.py.
3. Install the dependencies listed above.
4. Open a terminal in the project folder.
5. Run:
   python task3_customer_churn_classifier.py
The program prints the dataset overview, model metrics, cross-validation results, and feature importances. It also displays several charts.
Evaluation Metrics
- Accuracy: Overall proportion of correct predictions.
- Precision: Of customers predicted to churn, the proportion who actually churn.
- Recall: Of customers who actually churn, the proportion correctly identified.
- F1-score: Harmonic mean of precision and recall.
- ROC-AUC: Measures how well the model distinguishes churners from non-churners across classification thresholds. A higher score generally indicates better discrimination.
The best model is selected by test-set ROC-AUC. Actual scores depend on the dataset and execution environment; no performance result is guaranteed in advance.
Output Files
After a successful run, the script saves these files in the current working directory:
File	Description
best_churn_model.joblib	Saved pipeline for the model with the higher test ROC-AUC
model_evaluation_results.csv	Test-set metrics for both models
random_forest_top10_features.csv	Top 10 Random Forest features by importance
xgboost_top10_features.csv	Top 10 XGBoost features by importance


Project Structure
project-folder/
├── task3_customer_churn_classifier.py
├── Telco-Customer-Churn.csv
└── README.md
The four output files are created after the script runs successfully.
Methodology
1. Load and clean data: Read the CSV, clean TotalCharges, remove the ID field, and encode the target.
2. Prepare features: Identify numeric and categorical columns, then split the data into training and test sets.
3. Preprocess: Impute missing values, standardize numeric features, and one-hot encode categorical features.
4. Handle class imbalance: Apply SMOTE within each imbalanced-learn pipeline so synthetic samples are generated during training, not before the train/test split.
5. Train models: Fit Random Forest and XGBoost.
6. Evaluate: Compare test-set metrics, ROC curves, confusion matrices, and classification reports.
7. Validate: Use stratified 5-fold cross-validation on the training data.
8. Interpret and save: Report top feature importances and save the model with the higher test ROC-AUC.
Important Notes
- SMOTE is applied after preprocessing within the training pipeline to avoid applying it directly to the held-out test set.
- The selected model is based on test-set ROC-AUC. For a rigorous final estimate, use a separate untouched test set or nested validation when tuning models; repeatedly choosing models based on test performance can make the test score optimistic.
- Feature importance indicates how the fitted model uses features; it does not establish that a feature causes churn.
- This is a predictive project, not a guarantee that an individual customer will churn.
- The saved model expects input data with the same feature columns and compatible data types as the training dataset, excluding Churn and customerID.
License and Attribution
Use the dataset according to its source's terms and cite the dataset source in academic or public work. Add a project-specific license if you intend to distribute this code.
Author
M Jawad Mazhar
