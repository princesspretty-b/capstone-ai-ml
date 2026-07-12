# Import Libraries 
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedKFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import joblib
import numpy as np
import pandas as pd

# Load cleaned dataset from part 1 and test and train data from part 2
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

print("\nFeature Columns:")
print(X.columns.tolist())

# Display categorical columns
categorical_cols = X.select_dtypes(include=["object", "category"]).columns

print("\nCategorical Columns:")
print(categorical_cols)

# One-Hot Encode categorical columns
X = pd.get_dummies(
    X,
    columns=categorical_cols,
    drop_first=True
)

#Train-Test Split for Regression
# Regression split
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X,
    y_reg,
    test_size=0.20,
    random_state=42
)

#Train-Test Split for Classification
# Classification split
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X,
    y_clf,
    test_size=0.20,
    random_state=42,
    stratify=y_clf
)

#Standardize the Regression Features
# Regression scaling
scaler_reg = StandardScaler()

X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

#Standardize the Classification Features
# Classification scaling
scaler_clf = StandardScaler()

X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)

# Verify Shapes

print("\nRegression")
print("Training Features :", X_train_reg_scaled.shape)
print("Testing Features  :", X_test_reg_scaled.shape)

print("\nClassification")
print("Training Features :", X_train_clf_scaled.shape)
print("Testing Features  :", X_test_clf_scaled.shape)

print("\nTraining Class Distribution")
print(y_train_clf.value_counts())

# Task 1: Decision Tree baseline:
# Decision Tree with default parameters (max_depth=None)
dt_model = DecisionTreeClassifier(random_state=42)

# Train the model
dt_model.fit(X_train_clf_scaled, y_train_clf)

# Predictions
y_train_pred = dt_model.predict(X_train_clf_scaled)
y_test_pred = dt_model.predict(X_test_clf_scaled)

# Accuracy
train_accuracy = accuracy_score(y_train_clf, y_train_pred)
test_accuracy = accuracy_score(y_test_clf, y_test_pred)

print("\nDecision Tree (Default Parameters)")
print("-" * 40)
print(f"Training Accuracy : {train_accuracy:.3f}")
print(f"Test Accuracy     : {test_accuracy:.3f}")

# Task 2: Controlled Decision Tree:

# Controlled Decision Tree
dt_controlled = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=20,
    random_state=42
)

# Train the model
dt_controlled.fit(X_train_clf_scaled, y_train_clf)

# Predictions
y_train_pred_control = dt_controlled.predict(X_train_clf_scaled)
y_test_pred_control = dt_controlled.predict(X_test_clf_scaled)

# Accuracy
train_accuracy_control = accuracy_score(y_train_clf, y_train_pred_control)
test_accuracy_control = accuracy_score(y_test_clf, y_test_pred_control)

print("\nControlled Decision Tree")
print("-" * 40)
print(f"Training Accuracy : {train_accuracy_control:.3f}")
print(f"Test Accuracy     : {test_accuracy_control:.3f}")

# Task 3: Gini vs Entropy comparison

# Decision Tree using Gini Index
dt_gini = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)

dt_gini.fit(X_train_clf_scaled, y_train_clf)

y_pred_gini = dt_gini.predict(X_test_clf_scaled)

gini_accuracy = accuracy_score(y_test_clf, y_pred_gini)

# Decision Tree using Entropy
dt_entropy = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    random_state=42
)

dt_entropy.fit(X_train_clf_scaled, y_train_clf)

y_pred_entropy = dt_entropy.predict(X_test_clf_scaled)

entropy_accuracy = accuracy_score(y_test_clf, y_pred_entropy)

# Comparison Table
comparison = pd.DataFrame({
    "Criterion": ["Gini", "Entropy"],
    "Test Accuracy": [gini_accuracy, entropy_accuracy]
})

print("\nDecision Tree: Gini vs Entropy")
print("-" * 40)
print(comparison)

# Task 4: Random Forest

# Random Forest Model
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

# Train
rf_model.fit(X_train_clf_scaled, y_train_clf)

# Predictions
y_train_pred_rf = rf_model.predict(X_train_clf_scaled)
y_test_pred_rf = rf_model.predict(X_test_clf_scaled)

# Predicted probabilities
y_prob_rf = rf_model.predict_proba(X_test_clf_scaled)[:, 1]

# Accuracy
train_acc = accuracy_score(y_train_clf, y_train_pred_rf)
test_acc = accuracy_score(y_test_clf, y_test_pred_rf)

# ROC-AUC
auc = roc_auc_score(y_test_clf, y_prob_rf)

print("\nRandom Forest Results")
print("-" * 40)
print(f"Training Accuracy : {train_acc:.3f}")
print(f"Test Accuracy     : {test_acc:.3f}")
print(f"ROC-AUC           : {auc:.3f}")

# -------------------------------
# Feature Importance
# -------------------------------

feature_importance = pd.DataFrame({
    "Feature": X_train_clf.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 5 Most Important Features")
print("-" * 40)
print(feature_importance.head(5))

# Task 4a: Gradient Boosting

gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

# Train model
gb_model.fit(X_train_clf_scaled, y_train_clf)

# Predictions
y_train_pred_gb = gb_model.predict(X_train_clf_scaled)
y_test_pred_gb = gb_model.predict(X_test_clf_scaled)

# Probability predictions for AUC
y_prob_gb = gb_model.predict_proba(X_test_clf_scaled)[:, 1]

# Accuracy
train_accuracy_gb = accuracy_score(y_train_clf, y_train_pred_gb)
test_accuracy_gb = accuracy_score(y_test_clf, y_test_pred_gb)

# ROC-AUC
auc_gb = roc_auc_score(y_test_clf, y_prob_gb)

print("\nGradient Boosting Results")
print("-" * 40)
print(f"Training Accuracy : {train_accuracy_gb:.3f}")
print(f"Test Accuracy     : {test_accuracy_gb:.3f}")
print(f"ROC-AUC           : {auc_gb:.3f}")

# Task 4b: Feature ablation study

# Full Random Forest AUC
# Existing full model predictions
y_prob_full = rf_model.predict_proba(X_test_clf_scaled)[:, 1]

full_auc = roc_auc_score(
    y_test_clf,
    y_prob_full
)

print("\nFull Random Forest ROC-AUC:", round(full_auc, 3))

# Identify 5 Least Important Features

feature_importance = pd.DataFrame({
    "Feature": X_train_clf.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=True
)

lowest_features = feature_importance.head(5)["Feature"].tolist()

print("\n5 Least Important Features")
print("-" * 40)
print(feature_importance.head(5))

# Remove Lowest Importance Features

X_train_reduced = X_train_clf.drop(
    columns=lowest_features
)

X_test_reduced = X_test_clf.drop(
    columns=lowest_features
)

# Train Reduced Random Forest

rf_reduced = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

rf_reduced.fit(
    X_train_reduced,
    y_train_clf
)


# Probability prediction
y_prob_reduced = rf_reduced.predict_proba(
    X_test_reduced
)[:,1]


reduced_auc = roc_auc_score(
    y_test_clf,
    y_prob_reduced
)


# Results

ablation_results = pd.DataFrame({
    "Model": [
        "Full Random Forest",
        "Reduced Random Forest"
    ],
    "ROC-AUC": [
        full_auc,
        reduced_auc
    ]
})

print("\nFeature Ablation Results")
print("-" * 40)
print(ablation_results)

# Task 5: Cross-validated comparison:

# 
logistic_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

logistic_model.fit(
    X_train_clf_scaled,
    y_train_clf
)

# Predicted class labels
y_pred = logistic_model.predict(X_test_clf_scaled)

# Predicted probabilities
y_prob = logistic_model.predict_proba(X_test_clf_scaled)[:, 1]

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

models = {
    "Logistic Regression": logistic_model,
    "Controlled Decision Tree": dt_controlled,
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model
}

cv_results = []

for name, model in models.items():

    scores = cross_val_score(
        model,
        X_train_clf_scaled,
        y_train_clf,
        cv=cv,
        scoring="roc_auc"
    )

    cv_results.append({
        "Model": name,
        "Mean AUC": scores.mean(),
        "Std AUC": scores.std()
    })

cv_table = pd.DataFrame(cv_results)

print(f"\n Cross validation comaprision\n", cv_table)

# Task 6: Hyperparameter tuning with GridSearchCV:

# Create Pipeline

rf_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    RandomForestClassifier(random_state=42)
)

# Parameter Grid

param_grid = {
    'randomforestclassifier__n_estimators': [50, 100, 200],
    'randomforestclassifier__max_depth': [5, 10, None],
    'randomforestclassifier__min_samples_leaf': [1, 5]
}

# Stratified 5-Fold CV

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Grid Search

grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)


# Train Grid Search
grid_search.fit(
    X_train_clf,
    y_train_clf
)

# Results

print("\nBest Parameters")
print("-"*40)
print(grid_search.best_params_)


print("\nBest Cross Validation ROC-AUC")
print("-"*40)
print(round(grid_search.best_score_, 4))


# Best model
best_pipeline = grid_search.best_estimator_

cv_auc_scores = cross_val_score(
    best_pipeline,
    X_train_clf,
    y_train_clf,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)


print("\nGridSearchCV Best Pipeline - 5 Fold CV Results")
print("-"*50)

print("AUC Scores:")
print(cv_auc_scores)

print("\nMean CV AUC:",
      round(cv_auc_scores.mean(), 4))

print("Std CV AUC:",
      round(cv_auc_scores.std(), 4))


# Task 7: Manual learning curve

# Training fractions
fractions = [0.2, 0.4, 0.6, 0.8, 1.0]

learning_curve_results = []


for f in fractions:

    # Select first f fraction of training data
    train_size = int(f * len(X_train_clf))

    X_subset = X_train_clf.iloc[:train_size]
    y_subset = y_train_clf.iloc[:train_size]


    # Refit best pipeline on subset
    best_pipeline.fit(
        X_subset,
        y_subset
    )


    # Training predictions
    train_prob = best_pipeline.predict_proba(X_subset)[:,1]

    train_auc = roc_auc_score(
        y_subset,
        train_prob
    )


    # Test predictions using fixed test set
    test_prob = best_pipeline.predict_proba(X_test_clf)[:,1]

    test_auc = roc_auc_score(
        y_test_clf,
        test_prob
    )


    learning_curve_results.append({
        "Training Fraction": f,
        "Training AUC": round(train_auc,4),
        "Test AUC": round(test_auc,4)
    })


# Display results
learning_curve_df = pd.DataFrame(
    learning_curve_results
)

print("\nManual Learning Curve Results")
print("-"*50)
print(learning_curve_df)

# Task 8: Serialize the best model

# Save the best GridSearchCV pipeline
joblib.dump(best_pipeline, "part3/best_model.pkl")

print("\nBest model saved successfully as best_model.pkl")

# Load saved pipeline
loaded_model = joblib.load("part3/best_model.pkl")


# Create two sample patient records
test_rows = pd.DataFrame({
    "Pregnant": [2, 6],
    "Diastolic_BP": [70, 80],
    "Skin_Fold": [25, 35],
    "Serum_Insulin": [100, 250],
    "BMI": [25.5, 35.2],
    "Diabetes_Pedigree": [0.3, 0.8],
    "Age": [30, 55],

    # One-hot encoded Hospital columns
    "Hospital_Fortis": [0, 1],
    "Hospital_Gleneagals": [0, 0],
    "Hospital_Kavery": [0, 0],
    "Hospital_MMM": [0, 0]
})


# Generate predictions
predictions = loaded_model.predict(test_rows)


# Generate probabilities
probabilities = loaded_model.predict_proba(test_rows)[:,1]


print("\nPredicted Classes:")
print(predictions)

print("\nDiabetes Probabilities:")
print(probabilities)
