# Import Libraries 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt

# Task 1: Load cleaned_data.csv. Define labels & Features
# Load cleaned dataset
df = pd.read_csv("part1/cleaned_data.csv")

# If any NaN values present fill it with Median
columns_with_nan = df.columns[df.isna().any()]
df[columns_with_nan] = df[columns_with_nan].fillna(df[columns_with_nan].median())

# Regression target (continuous)
y_reg = df["Glucose"]

# Classification target (binary)
y_clf = df["Class"]

# Feature matrix
X = df.drop(columns=["Glucose", "Class", "Diabetes_Status"])

print("Feature Matrix Shape:", X.shape)
print("Regression Label Shape:", y_reg.shape)
print("Classification Label Shape:", y_clf.shape)

print("\nFeature Columns:")
print(X.columns.tolist())

# Task 2: Encode categorical columns

# Display categorical columns
categorical_cols = X.select_dtypes(include=["object", "category"]).columns

print("Categorical Columns:")
print(categorical_cols)

# One-Hot Encode categorical columns
X = pd.get_dummies(
    X,
    columns=categorical_cols,
    drop_first=True
)

print("Encoded Feature Matrix:")
print(X.head())

print("Encoded Columns:")
print(X.columns.tolist())

# Task 3: Leak-free train-test split and scaling
#Train-Test Split for Regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X,
    y_reg,
    test_size=0.20,
    random_state=42
)

#Train-Test Split for Classification
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X,
    y_clf,
    test_size=0.20,
    random_state=42,
    stratify=y_clf
)

#Standardize the Regression Features
scaler_reg = StandardScaler()

X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

#Standardize the Classification Features
scaler_clf = StandardScaler()

X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)

#Verifying the shapes
print("Regression")
print("Training Features:", X_train_reg_scaled.shape)
print("Testing Features :", X_test_reg_scaled.shape)
print("Training Labels  :", y_train_reg.shape)
print("Testing Labels   :", y_test_reg.shape)

print("\nClassification")
print("Training Features:", X_train_clf_scaled.shape)
print("Testing Features :", X_test_clf_scaled.shape)
print("Training Labels  :", y_train_clf.shape)
print("Testing Labels   :", y_test_clf.shape)

# Task 4: Regression model — Linear Regression
# Create the model
lr_model = LinearRegression()

# Train the model
lr_model.fit(X_train_reg_scaled, y_train_reg)

# Predict on the test data
y_pred_reg = lr_model.predict(X_test_reg_scaled)

# Evaluation metrics
mse_lr = mean_squared_error(y_test_reg, y_pred_reg)
r2_lr = r2_score(y_test_reg, y_pred_reg)

print("Linear Regression Results")
print("-"*35)
print(f"MSE : {mse_lr:.2f}")
print(f"R²  : {r2_lr:.3f}")

# Print Coefficients
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": lr_model.coef_
})

print("\nRegression Coefficients")
print(coef_df)

# Print top 3 features
coef_df["Absolute Coefficient"] = coef_df["Coefficient"].abs()

top3 = coef_df.sort_values(
    by="Absolute Coefficient",
    ascending=False
).head(3)

print("\nTop 3 Most Important Features")
print(top3)

# Train Ridge regression
ridge_model = Ridge(alpha=1.0)

ridge_model.fit(
    X_train_reg_scaled,
    y_train_reg
)

y_pred_ridge = ridge_model.predict(
    X_test_reg_scaled
)

# Evaluate Ridge Regresssion

mse_ridge = mean_squared_error(
    y_test_reg,
    y_pred_ridge
)

r2_ridge = r2_score(
    y_test_reg,
    y_pred_ridge
)

print("\nRidge Regression Results")
print("-"*35)
print(f"MSE : {mse_ridge:.2f}")
print(f"R²  : {r2_ridge:.3f}")

# Comparison Table

comparison = pd.DataFrame({
    "Model":[
        "Linear Regression",
        "Ridge Regression"
    ],
    "MSE":[
        mse_lr,
        mse_ridge
    ],
    "R²":[
        r2_lr,
        r2_ridge
    ]
})

print("\nModel Comparison")
print(comparison)

# Task 5a: Classification model — Logistic Regression

# Check Class Distribution
print("Training Class Distribution")
print("-" * 35)
print(y_train_clf.value_counts())

print("\nPercentage Distribution")
print((y_train_clf.value_counts(normalize=True) * 100).round(2))

log_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

log_model.fit(
    X_train_clf_scaled,
    y_train_clf
)

# Predicted class labels
y_pred = log_model.predict(X_test_clf_scaled)

# Predicted probabilities
y_prob = log_model.predict_proba(X_test_clf_scaled)[:, 1]

cm = confusion_matrix(
    y_test_clf,
    y_pred
)

print("Confusion Matrix")
print(cm)

# Classification Report

print(classification_report(
    y_test_clf,
    y_pred
))

# ROC curve

fpr, tpr, thresholds = roc_curve(
    y_test_clf,
    y_prob
)

auc = roc_auc_score(
    y_test_clf,
    y_prob
)

print("AUC =", round(auc,3))

# Plot ROC curve

plt.figure(figsize=(7,6))
plt.plot(
    fpr,
    tpr,
    color="blue",
    linewidth=2,
    label=f"AUC = {auc:.3f}"
)
plt.plot(
    [0,1],
    [0,1],
    linestyle="--",
    color="red"
)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("part2/plots/ROC_Curve.png", dpi=300, bbox_inches="tight")
plt.show()

# Task 5b: Decision-threshold sensitivity

# Predicted probabilities from Logistic Regression
y_prob = log_model.predict_proba(X_test_clf_scaled)[:, 1]

# Thresholds to evaluate
thresholds = np.arange(0.30, 0.71, 0.10)

results = []

for threshold in thresholds:

    # Convert probabilities into class predictions
    y_pred_threshold = (y_prob >= threshold).astype(int)

    # Calculate metrics
    precision = precision_score(y_test_clf, y_pred_threshold)
    recall = recall_score(y_test_clf, y_pred_threshold)
    f1 = f1_score(y_test_clf, y_pred_threshold)

    results.append([
        threshold,
        precision,
        recall,
        f1
    ])

# Display results
threshold_table = pd.DataFrame(
    results,
    columns=[
        "Threshold",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

print("\nDecision Threshold Comparison")
print("-"*50)
print(threshold_table)

# Threshold with highest F1 score
best = threshold_table.loc[threshold_table["F1 Score"].idxmax()]

print("\nBest Threshold Based on F1 Score")
print(best)

# Task 6: Regularization experiment on Logistic Regression:
# Train a Second Logistic Regression Model (C=0.01)
# Strongly regularized Logistic Regression
log_model_reg = LogisticRegression(
    C=0.01,
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

# Train the model
log_model_reg.fit(X_train_clf_scaled, y_train_clf)

# Make Predictions
# Predicted class labels
y_pred_reg = log_model_reg.predict(X_test_clf_scaled)

# Predicted probabilities
y_prob_reg = log_model_reg.predict_proba(X_test_clf_scaled)[:, 1]

# Compute Evaluation Metrics
# Baseline model (C=1.0)
precision_base = precision_score(y_test_clf, y_pred)
recall_base = recall_score(y_test_clf, y_pred)
auc_base = roc_auc_score(y_test_clf, y_prob)

# Regularized model (C=0.01)
precision_reg = precision_score(y_test_clf, y_pred_reg)
recall_reg = recall_score(y_test_clf, y_pred_reg)
auc_reg = roc_auc_score(y_test_clf, y_prob_reg)

# Comparision Table:

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression (C=1.0)",
        "Logistic Regression (C=0.01)"
    ],
    "Precision": [
        precision_base,
        precision_reg
    ],
    "Recall": [
        recall_base,
        recall_reg
    ],
    "AUC": [
        auc_base,
        auc_reg
    ]
})

print("\nLogistic Regression Regularization Comparison")
print(comparison)


# Task 7: Bootstrap confidence interval for AUC difference

# Number of bootstrap iterations
n_bootstrap = 500

# Store AUC differences
auc_differences = []

np.random.seed(42)

# Bootstrap sampling
for i in range(n_bootstrap):

    # Sample test indices with replacement
    indices = np.random.choice(
        len(y_test_clf),
        size=len(y_test_clf),
        replace=True
    )

    # Select bootstrap samples
    y_sample = y_test_clf.iloc[indices]

    prob_base_sample = y_prob[indices]       # C=1.0 model
    prob_reg_sample = y_prob_reg[indices]    # C=0.01 model

    # Calculate AUC for both models
    auc_base = roc_auc_score(
        y_sample,
        prob_base_sample
    )

    auc_reg = roc_auc_score(
        y_sample,
        prob_reg_sample
    )

    # Difference: C=1.0 - C=0.01
    auc_difference = auc_base - auc_reg

    auc_differences.append(auc_difference)


# Convert to numpy array
auc_differences = np.array(auc_differences)

# Calculate statistics
mean_difference = np.mean(auc_differences)

lower_bound = np.percentile(
    auc_differences,
    2.5
)

upper_bound = np.percentile(
    auc_differences,
    97.5
)
print("Bootstrap AUC Difference Results")
print("-"*40)
print(f"Mean AUC Difference : {mean_difference:.4f}")
print(f"95% CI Lower Bound : {lower_bound:.4f}")
print(f"95% CI Upper Bound : {upper_bound:.4f}")

