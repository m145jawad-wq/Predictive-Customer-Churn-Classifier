# ============================================================
# TASK 3: PREDICTIVE CUSTOMER CHURN CLASSIFIER
# Random Forest + XGBoost + SMOTE + ROC-AUC
# ============================================================
# Install dependencies in your terminal if needed:
# pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost joblib

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
RANDOM_STATE = 42

# ============================================================
# 1. LOAD DATASET
# ============================================================
possible_files = [
    "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "Telco-Customer-Churn.csv"
]
file_path = next((f for f in possible_files if os.path.exists(f)), None)

if file_path is None:
    raise FileNotFoundError(
        "Telco Customer Churn CSV not found. Download it from:\n"
        "https://raw.githubusercontent.com/IBM/Customer-Churn-Data/main/Telco-Customer-Churn.csv\n"
        "Save the CSV in the same folder as this Python file."
    )

df = pd.read_csv(file_path)
print("Dataset loaded:", df.shape)
print(df.head())
print("\nDataset information:")
df.info()

# ============================================================
# 2. DATA CLEANING
# ============================================================
df = df.copy()

if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip().replace("", np.nan),
        errors="coerce"
    )

df.drop(columns=["customerID"], errors="ignore", inplace=True)

if "Churn" not in df.columns:
    raise ValueError("The dataset must contain a 'Churn' target column.")

df["Churn"] = df["Churn"].astype(str).str.strip().map({"Yes": 1, "No": 0})
df.dropna(subset=["Churn"], inplace=True)
df["Churn"] = df["Churn"].astype(int)

print("\nMissing values:")
print(df.isnull().sum().sort_values(ascending=False).head(10))
print("\nChurn distribution:")
print(df["Churn"].value_counts())
print("\nChurn percentages:")
print((df["Churn"].value_counts(normalize=True) * 100).round(2))

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Churn", hue="Churn", legend=False)
plt.title("Customer Churn Distribution")
plt.xticks([0, 1], ["No Churn", "Churn"])
plt.tight_layout()
plt.show()

# ============================================================
# 3. FEATURES AND TARGET
# ============================================================
X = df.drop(columns=["Churn"])
y = df["Churn"]

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()
numerical_features = X.select_dtypes(include=np.number).columns.tolist()

print("\nNumerical features:", numerical_features)
print("Categorical features:", categorical_features)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ============================================================
# 4. PREPROCESSING
# ============================================================
try:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:  # Compatibility with older scikit-learn versions
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", encoder)
])
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
], remainder="drop", verbose_feature_names_out=False)

# ============================================================
# 5. RANDOM FOREST MODEL WITH SMOTE
# ============================================================
rf_model = RandomForestClassifier(
    n_estimators=300, min_samples_split=5, min_samples_leaf=2,
    class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
)
rf_pipeline = ImbPipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=RANDOM_STATE)),
    ("classifier", rf_model)
])

# ============================================================
# 6. XGBOOST MODEL WITH SMOTE
# ============================================================
xgb_model = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    objective="binary:logistic", eval_metric="logloss",
    random_state=RANDOM_STATE, n_jobs=-1
)
xgb_pipeline = ImbPipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=RANDOM_STATE)),
    ("classifier", xgb_model)
])

# ============================================================
# 7. TRAIN BOTH MODELS
# ============================================================
print("\nTraining Random Forest...")
rf_pipeline.fit(X_train, y_train)
print("Training XGBoost...")
xgb_pipeline.fit(X_train, y_train)
print("Both models trained successfully!")

models = {"Random Forest": rf_pipeline, "XGBoost": xgb_pipeline}

# ============================================================
# 8. MODEL EVALUATION
# ============================================================
results = []
predictions = {}

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    predictions[name] = {"pred": y_pred, "prob": y_prob}
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    })

results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
print("\nMODEL PERFORMANCE COMPARISON")
print(results_df.round(4).to_string(index=False))

# ============================================================
# 9. ROC CURVES
# ============================================================
plt.figure(figsize=(9, 7))
for name, result in predictions.items():
    fpr, tpr, _ = roc_curve(y_test, result["prob"])
    auc = roc_auc_score(y_test, result["prob"])
    plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc:.4f})")

plt.plot([0, 1], [0, 1], linestyle="--", color="gray",
         label="Random Classifier (AUC = 0.50)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: Random Forest vs XGBoost")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# 10. CONFUSION MATRICES
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (name, result) in zip(axes, predictions.items()):
    cm = confusion_matrix(y_test, result["pred"], labels=[0, 1])
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"], ax=ax
    )
    ax.set_title(f"{name} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.show()

# ============================================================
# 11. CLASSIFICATION REPORTS
# ============================================================
for name, result in predictions.items():
    print("\n" + "=" * 60)
    print(name.upper())
    print("=" * 60)
    print(classification_report(
        y_test, result["pred"], labels=[0, 1],
        target_names=["No Churn", "Churn"], zero_division=0
    ))

# ============================================================
# 12. STRATIFIED 5-FOLD CROSS-VALIDATION
# ============================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_results = []

for name, model in models.items():
    print(f"\nCross-validating {name}...")
    scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1
    )
    cv_results.append({
        "Model": name,
        "Mean CV ROC-AUC": scores.mean(),
        "Std CV ROC-AUC": scores.std()
    })

cv_df = pd.DataFrame(cv_results)
print("\nCROSS-VALIDATION RESULTS")
print(cv_df.round(4).to_string(index=False))

# ============================================================
# 13. FEATURE IMPORTANCE: TOP 10 FEATURES
# ============================================================
def get_feature_importances(model_pipeline):
    fitted_preprocessor = model_pipeline.named_steps["preprocessor"]
    fitted_classifier = model_pipeline.named_steps["classifier"]
    feature_names = fitted_preprocessor.get_feature_names_out()
    importances = fitted_classifier.feature_importances_
    return pd.DataFrame({
        "Feature": feature_names, "Importance": importances
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

rf_importances = get_feature_importances(rf_pipeline)
xgb_importances = get_feature_importances(xgb_pipeline)

print("\nTOP 10 RANDOM FOREST FEATURES")
print(rf_importances.head(10).round(5).to_string(index=False))
print("\nTOP 10 XGBOOST FEATURES")
print(xgb_importances.head(10).round(5).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
sns.barplot(
    data=rf_importances.head(10), x="Importance", y="Feature",
    hue="Feature", legend=False, palette="Blues_r", ax=axes[0]
)
axes[0].set_title("Top 10 Random Forest Features")
axes[0].set_xlabel("Importance")
sns.barplot(
    data=xgb_importances.head(10), x="Importance", y="Feature",
    hue="Feature", legend=False, palette="Greens_r", ax=axes[1]
)
axes[1].set_title("Top 10 XGBoost Features")
axes[1].set_xlabel("Importance")
plt.tight_layout()
plt.show()

# ============================================================
# 14. SELECT AND SAVE BEST MODEL
# ============================================================
best_model_name = results_df.iloc[0]["Model"]
best_auc = results_df.iloc[0]["ROC-AUC"]
best_model = models[best_model_name]

print("\n" + "=" * 60)
print("FINAL PROJECT SUMMARY")
print("=" * 60)
print("Best model:", best_model_name)
print(f"Test ROC-AUC: {best_auc:.4f}")
if best_auc >= 0.80:
    print("ROC-AUC target of 0.80 achieved.")
else:
    print("ROC-AUC target not achieved; further tuning may help.")

joblib.dump(best_model, "best_churn_model.joblib")
results_df.to_csv("model_evaluation_results.csv", index=False)
rf_importances.head(10).to_csv("random_forest_top10_features.csv", index=False)
xgb_importances.head(10).to_csv("xgboost_top10_features.csv", index=False)

print("\nFiles saved successfully:")
print("1. best_churn_model.joblib")
print("2. model_evaluation_results.csv")
print("3. random_forest_top10_features.csv")
print("4. xgboost_top10_features.csv")
print("\nProject completed successfully!")
