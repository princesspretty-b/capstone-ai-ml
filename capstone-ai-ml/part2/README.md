**Supervised Machine Learning Model - Build, Train, and Evaluate**

**Overview:**

This project is to produce two predictive models: one for regression (predicting a continuous output) and one for binary classification. The program should preprocess the cleaned data correctly to avoid data leakage, handle class imbalance where applicable, and evaluate both models rigorously

**Task 1: Load cleaned_data.csv. Define labels & Features**

The cleaned dataset from part 1 is loaded into a dataframe for further processing. Below labels and features are defined.

- **Regression Label (y_reg)**

The regression target is **Glucose**, which is a continuous numeric variable representing a patient's blood glucose concentration. The objective of the regression model is to predict glucose levels from the remaining patient features.

- **Classification Label (y_clf)**

The classification target is **Class**, which is already present in the dataset as a natural binary variable:

- **0** = Non-Diabetic
- **1** = Diabetic

Since the dataset already includes a clinically meaningful binary outcome, it was used directly instead of creating a new binary label by splitting the regression target at its median.

- **Feature Matrix (X)**

The feature matrix consists of all remaining predictor variables after removing the regression target (Glucose) and the classification target (Class). If the derived column Diabetes_Status is present, it is also excluded because it contains the same information as Class and would introduce **data leakage** into the classification model.

**Task 2: Encode categorical columns**

The feature matrix contained one categorical variable:

- **Hospital**

The hospital names represent different hospitals and **do not have a natural order**. Therefore, **One-Hot Encoding** was used to convert this variable into multiple binary indicator columns using pd.get_dummies(drop_first=True).

Each new column represents whether a patient belongs to a particular hospital (1) or not (0). The option drop_first=True was used to remove one dummy variable (the reference category) and avoid **multicollinearity**, also known as the **dummy variable trap**. Without dropping one category, the dummy variables would be perfectly correlated because one category can always be inferred from the others.

**Why One-Hot Encoding Instead of Label Encoding?**

Label encoding assigns integers to categories (for example, Apollo = 0, Fortis = 1, Kavery = 2), which introduces a **false ordinal relationship**. This incorrectly suggests that one hospital is greater than or less than another, even though hospital names are simply distinct categories with no inherent ranking. One-hot encoding avoids this problem by representing each category independently as a binary feature. As a result, the model treats each hospital as a separate category without assuming any order or numerical relationship between them. This makes one-hot encoding the appropriate choice for nominal categorical variables such as **Hospital**.

**Task 3: Leak-free train-test split and scaling**

Since we have **two prediction tasks** (regression and classification), creating **separate train-test splits** for each target while using the same feature matrix X

**Train-Test Split and Feature Scaling :** The cleaned dataset was divided into **training (80%)** and **testing (20%)** subsets using train_test_split() with random_state=42 to ensure reproducibility. For the classification task, **stratified sampling** (stratify=y_clf) was used to preserve the proportion of diabetic and non-diabetic samples in both the training and testing datasets. Before training the models, the numeric features were standardized using **StandardScaler**. The scaler was **fitted only on the training features** using scaler.fit(X_train). The fitted scaler was then applied to both the training and test feature sets using X_train_scaled = scaler.transform(X_train) and X_test_scaled = scaler.transform(X_test).

**Why Fit the Scaler Only on the Training Data?**

The scaler computes the **mean** and **standard deviation** of each feature during the fit() step. If it were fitted on the **entire dataset**, these statistics would include information from the test set. This would introduce **data leakage**, because the model would indirectly gain knowledge about the test data during training.

Data leakage leads to overly optimistic evaluation results, as the model benefits from information that would not be available in a real-world prediction scenario. By fitting the scaler only on the training data and applying the learned scaling parameters to the test data, the evaluation remains fair and accurately reflects the model's ability to generalize to unseen data.

**Task 4: Regression model - Linear Regression**

**Linear Regression**

A **Linear Regression** model was trained using the standardized training data to predict the continuous target variable **Glucose**. Model performance was evaluated on the test set using Mean Squared Error (MSE) and the coefficient of determination (R²).

**Model Performance**

| **Metric**               | **Value**  |
| ------------------------ | ---------- |
| Mean Squared Error (MSE) | **764.79** |
| R² Score                 | **0.143**  |

The model achieved an **R² score of 0.143**, indicating that approximately **14.3% of the variation in glucose levels** is explained by the selected predictor variables. The relatively low R² suggests that glucose levels are influenced by additional factors not included in the dataset or by complex non-linear relationships that a simple linear model cannot capture.

**Feature Coefficients**

The regression coefficients indicate how the predicted glucose level changes when a feature increases by **one standard deviation** (because the features were standardized), while all other variables remain constant.

**Top Three Features by Absolute Coefficient**

| **Rank** | **Feature**   | **Coefficient** |
| -------- | ------------- | --------------- |
| 1        | Serum_Insulin | 11.09           |
| 2        | Age           | 6.45            |
| 3        | Diastolic_BP  | 3.55            |

**Interpretation**

- **Serum_Insulin (+11.09)** has the largest positive coefficient. This indicates that, holding all other variables constant, a **one standard deviation increase in Serum Insulin is associated with an increase of approximately 11.09 units in the predicted glucose level**.
- **Age (+6.45)** indicates that older patients tend to have higher predicted glucose levels. A one standard deviation increase in age is associated with an increase of approximately **6.45 glucose units**.
- **Diastolic_BP (+3.55)** suggests that higher diastolic blood pressure is associated with moderately higher predicted glucose levels.

A **large positive coefficient** means that increasing the standardized feature increases the predicted glucose value. Conversely, a **large negative coefficient** means that increasing the standardized feature decreases the predicted glucose value. For example, the negative coefficient for **Hospital_Fortis (-1.91)** suggests a slight decrease in predicted glucose relative to the reference hospital category after accounting for all other variables. However, these hospital coefficients should not be interpreted as causal effects because the hospital assignment was randomly generated for this exercise.

**Ridge Regression**

A **Ridge Regression** model with **α = 1.0** was trained using the same scaled training and testing datasets.

**Model Comparison**

| **Model**         | **MSE**    | **R²**     |
| ----------------- | ---------- | ---------- |
| Linear Regression | **764.79** | **0.1430** |
| Ridge Regression  | **764.78** | **0.1430** |

**Interpretation**

The Ridge Regression model produced performance that was almost identical to the standard Linear Regression model. The MSE decreased only marginally (764.79 to 764.78), while the R² score increased from **0.143002** to **0.143010**. This indicates that applying L2 regularization had **very little effect** on predictive performance for this dataset.

Ridge Regression differs from Ordinary Least Squares (OLS) because it adds an **L2 regularization penalty** to the loss function, shrinking large coefficient values toward zero. This helps reduce overfitting and improves model stability when predictors are highly correlated (multicollinearity). The **alpha (α)** parameter controls the strength of this penalty. A larger alpha results in stronger coefficient shrinkage, whereas a smaller alpha makes Ridge behave more like ordinary Linear Regression. Since the Ridge and OLS results are nearly identical here, the dataset does not appear to suffer from severe multicollinearity, and the chosen regularization strength (α = 1.0) had only a minimal impact on the fitted model.

Both Linear Regression and Ridge Regression explain only a modest proportion of the variability in glucose levels (approximately **14%**). This suggests that either additional predictive features or more flexible, non-linear models may be required to improve regression performance on this dataset. The similarity between the two models also indicates that regularization was not a major factor affecting model performance for these predictors.

**Task 5a: Classification model - Logistic Regression**

**Classification Label**

The classification target was the **Class** column from the cleaned diabetes dataset, where:

- **0 = Non-Diabetic**
- **1 = Diabetic**

This natural binary target was selected because the objective is to classify whether a patient has diabetes based on the available medical measurements.

**Class Imbalance Handling**

The distribution of the training data was examined before training the classification model.

| **Class**        | **Count** | **Percentage** |
| ---------------- | --------- | -------------- |
| Non-Diabetic (0) | 400       | 65.15%         |
| Diabetic (1)     | 214       | 34.85%         |

The minority class (Diabetic) represented **34.85%** of the training samples, which is below the assignment threshold of **35%**. Therefore, class imbalance was addressed using the class_weight='balanced' option in the Logistic Regression model.

This approach automatically assigns larger weights to the minority class during training, increasing the penalty for misclassifying diabetic patients. Compared with SMOTE, this method is simpler to implement, avoids generating synthetic samples, and preserves the original distribution of the training data while helping the model learn from both classes more effectively.

**Logistic Regression Model**

A Logistic Regression classifier was trained using the standardized training data with the following settings:

- **Model:** Logistic Regression
- **Class Weight:** balanced
- **Maximum Iterations:** 1000

The trained model was then used to predict both class labels and class probabilities for the unseen test dataset.

**Confusion Matrix**

| **Actual / Predicted** | **Non-Diabetic** | **Diabetic** |
| ---------------------- | ---------------- | ------------ |
| **Non-Diabetic**       | 73               | 27           |
| **Diabetic**           | 18               | 36           |

**Interpretation**

The confusion matrix shows that:

- **True Negatives (TN):** 73 patients were correctly classified as non-diabetic.
- **False Positives (FP):** 27 non-diabetic patients were incorrectly classified as diabetic.
- **False Negatives (FN):** 18 diabetic patients were incorrectly classified as non-diabetic.
- **True Positives (TP):** 36 diabetic patients were correctly identified.

**Classification Performance**

| **Metric**           | **Value** |
| -------------------- | --------- |
| Accuracy             | **0.71**  |
| Precision (Diabetic) | **0.57**  |
| Recall (Diabetic)    | **0.67**  |
| F1-Score (Diabetic)  | **0.62**  |

The model achieved an **accuracy of 71%**, correctly classifying approximately seven out of every ten patients.

The **precision** for the diabetic class is **0.57**, meaning that 57% of patients predicted as diabetic were actually diabetic.

The **recall** for the diabetic class is **0.67**, indicating that the model successfully identified 67% of all diabetic patients while missing 33%.

The **F1-score** of **0.62** reflects a reasonable balance between precision and recall.

**Precision and Recall**

The evaluation metrics are defined as follows:

**Precision**
"Precision"=TP/(TP+FP)
Precision measures the proportion of patients predicted as diabetic who are actually diabetic.

**Recall**
"Recall"=TP/(TP+FN)
Recall measures the proportion of actual diabetic patients that are correctly identified by the model.

**Most Important Evaluation Metric**

For diabetes prediction, **Recall** is more important than Precision.

A **false negative** occurs when a patient with diabetes is incorrectly classified as non-diabetic. Missing a diabetic patient can delay diagnosis and treatment, increasing the risk of serious health complications. In contrast, a **false positive** usually results in additional medical tests, which are generally less harmful than failing to detect a true diabetic case.

Therefore, maximizing recall is preferred for this healthcare classification problem, even if it leads to a slight increase in false positives.

**ROC Curve and AUC**

The Receiver Operating Characteristic (ROC) curve was generated using the predicted probabilities from the Logistic Regression model.

The model achieved an **Area Under the ROC Curve (AUC) of 0.782**.

An AUC value of **0.782** indicates that the model has **good discriminatory ability**, meaning it can effectively distinguish between diabetic and non-diabetic patients. Specifically, there is approximately a **78.2% probability** that the model will assign a higher predicted probability to a randomly selected diabetic patient than to a randomly selected non-diabetic patient.

Since the AUC is substantially higher than **0.50**, the classifier performs considerably better than random guessing and demonstrates good overall classification capability.

**Inference**

The Logistic Regression model, trained with balanced class weights, achieved **71% accuracy** and an **AUC of 0.782**, indicating good overall predictive performance. The model correctly identified **67% of diabetic patients**, making it suitable for diabetes screening where detecting positive cases is particularly important. Although the precision for diabetic predictions is moderate, prioritizing recall is appropriate in this application because the consequences of failing to detect diabetes are generally more serious than conducting additional follow-up tests for false positive cases.

**Task 5b: Decision-threshold sensitivity**

**Decision Threshold Analysis**

The Logistic Regression model predicts the probability that a patient belongs to the diabetic class. By default, a patient is classified as diabetic when the predicted probability is **0.50 or higher**. To evaluate how the decision threshold affects classification performance, the threshold was varied from **0.30 to 0.70** in increments of **0.10**. At each threshold, Precision, Recall, and F1-score were computed.

**Precision**
"Precision"=TP/(TP+FP)
Precision measures the proportion of patients predicted as diabetic who are actually diabetic.

**Recall**
"Recall"=TP/(TP+FN)
Recall measures the proportion of actual diabetic patients that are correctly identified.

**Threshold Comparison**

| **Threshold** | **Precision** | **Recall** | **F1-Score** |
| ------------- | ------------- | ---------- | ------------ |
| **0.30**      | 0.471         | **0.907**  | 0.620        |
| **0.40**      | 0.529         | **0.852**  | **0.652**    |
| **0.50**      | 0.571         | 0.667      | 0.615        |
| **0.60**      | 0.652         | 0.556      | 0.600        |
| **0.70**      | **0.607**     | 0.315      | 0.415        |

**Threshold that Maximizes F1-Score**

The highest **F1-score (0.652)** was achieved at a **decision threshold of 0.40**.

At this threshold:

- **Precision = 0.529**
- **Recall = 0.852**
- **F1-score = 0.652**

This threshold provides the best balance between precision and recall for this dataset.

The threshold comparison illustrates the trade-off between precision and recall:

- At a **lower threshold (0.30)**, the classifier predicts more patients as diabetic. This results in a very high **recall (0.907)**, meaning that over 90% of diabetic patients are correctly identified. However, precision decreases because more non-diabetic patients are incorrectly classified as diabetic.
- At the **default threshold (0.50)**, the model achieves a better balance, with a precision of **0.571** and a recall of **0.667**.
- At a **higher threshold (0.70)**, the classifier becomes more conservative. Precision improves compared with lower thresholds, but recall falls sharply to **0.315**, meaning that many diabetic patients are missed.

For diabetes prediction, **Recall** is the more important evaluation metric. A **false negative** means that a patient with diabetes is incorrectly classified as non-diabetic. Missing a diabetic patient can delay diagnosis and treatment, increasing the risk of severe health complications. In contrast, a **false positive** generally leads to additional medical examinations, which are less harmful than failing to detect an actual diabetic patient.

Therefore, maximizing recall is preferred in this healthcare application.

**Recommended Decision Threshold**

Based on the results, a **threshold of 0.40** is recommended because it produces the highest F1-score while maintaining a high recall of **85.2%**. Lowering the threshold from the default value of **0.50** allows the model to identify substantially more diabetic patients, reducing the number of false negatives.

The trade-off is a reduction in precision, meaning that more non-diabetic patients may be incorrectly classified as diabetic. In a medical screening context, this trade-off is generally acceptable because additional diagnostic tests are typically less costly than failing to detect a patient with diabetes.

**Task 6: Regularization experiment on Logistic Regression:**

**Regularization Experiment**

To investigate the effect of regularization on model performance, a second Logistic Regression model was trained using a stronger L2 regularization penalty by setting **C = 0.01**. All other training settings remained unchanged, including the use of class_weight='balanced' and max_iter=1000.

The performance of this model was compared with the baseline Logistic Regression model (C = 1.0) using Precision, Recall, and Area Under the ROC Curve (AUC).

**Model Comparison**

| **Model**                          | **Precision** | **Recall** | **AUC**   |
| ---------------------------------- | ------------- | ---------- | --------- |
| Logistic Regression (**C = 1.0**)  | **0.571**     | **0.667**  | **0.782** |
| Logistic Regression (**C = 0.01**) | **0.593**     | **0.648**  | **0.780** |

**What Does the C Parameter Control?**

The **C** parameter controls the strength of **L2 regularization** in Logistic Regression. It is the inverse of the regularization strength.

- A **larger value of C** (e.g., **1.0**) applies **weaker regularization**, allowing the model coefficients to fit the training data more closely.
- A **smaller value of C** (e.g., **0.01**) applies **stronger regularization**, shrinking the model coefficients toward zero. This can reduce overfitting by producing a simpler model, but excessive regularization may lead to underfitting if important relationships are suppressed.

**Performance Comparison**

The strongly regularized model (**C = 0.01**) achieved a **slightly higher precision (0.593 vs. 0.571)**, indicating that a greater proportion of patients predicted as diabetic were actually diabetic. However, this improvement came at the expense of **lower recall (0.648 vs. 0.667)**, meaning that fewer diabetic patients were correctly identified.

The **AUC** also decreased slightly from **0.782** to **0.780**, indicating a small reduction in the model's overall ability to distinguish between diabetic and non-diabetic patients.

**Interpretation**

Reducing the value of **C** from **1.0** to **0.01** resulted in a model with stronger regularization. Although this slightly improved precision, it reduced recall and produced a marginally lower AUC. Since diabetes screening prioritizes identifying as many true diabetic patients as possible, the decrease in recall is undesirable.

Overall, the baseline Logistic Regression model (**C = 1.0**) provided the better balance between precision and recall while also achieving the highest AUC. Therefore, the baseline model is the preferred choice for this dataset because it offers better overall classification performance and is more suitable for a healthcare application where missing diabetic patients has greater consequences than producing additional false positives.

The experiment shows that stronger regularization did **not improve** the overall performance of the Logistic Regression model on this dataset. While the model became slightly more precise, it was less sensitive to diabetic cases and exhibited a marginal decrease in discrimination ability. Consequently, **C = 1.0** remains the recommended setting for this dataset.

**Task 7: Bootstrap confidence interval for AUC difference**

**Bootstrap Analysis**

A bootstrap confidence interval analysis was performed to quantify the reliability of the performance difference between the baseline Logistic Regression model (**C = 1.0**) and the strongly regularized Logistic Regression model (**C = 0.01**).

The analysis used **500 bootstrap samples** generated from the test dataset by sampling rows with replacement. For each bootstrap sample, the AUC was calculated for both models, and the difference was computed as: "AUC Difference"=〖"AUC" 〗_(C=1.0)-〖"AUC" 〗_(C=0.01)

A positive value indicates that the baseline model performs better, while a negative value indicates that the regularized model performs better.

**Bootstrap Results**

| **Statistic**       | **Value**    |
| ------------------- | ------------ |
| Mean AUC Difference | **0.0018**   |
| 2.5th Percentile    | **\-0.0122** |
| 97.5th Percentile   | **0.0164**   |

The estimated 95% bootstrap confidence interval for the AUC difference is: [-0.0122, 0.0164]

**Interpretation**

The mean AUC difference was **0.0018**, indicating that the baseline Logistic Regression model (**C = 1.0**) achieved a slightly higher AUC than the strongly regularized model (**C = 0.01**).

However, the 95% confidence interval ranges from **\-0.0122 to 0.0164**, which **includes zero**. This means that the observed difference in AUC is small and may be caused by random variation in the test samples rather than a consistent performance advantage.

Because the confidence interval includes zero, there is **not enough statistical evidence to conclude that the C=1.0 model consistently outperforms the C=0.01 model** across different samples of the data.

**Final Model Selection Interpretation**

Although the baseline model had a slightly higher AUC (**0.782**) compared with the regularized model (**0.780**), the bootstrap analysis shows that this difference is not statistically significant. Both models provide very similar classification performance.

The **C=1.0 Logistic Regression model** remains the preferred model because it achieved slightly better recall (**0.667 vs. 0.648**) and AUC while maintaining competitive precision. Since diabetes prediction prioritizes identifying as many true diabetic cases as possible, the slightly higher recall of the baseline model makes it more suitable for this application.

The bootstrap analysis confirms that the performance difference between the two models is very small, and the choice between them has minimal impact on predictive performance for this dataset.
